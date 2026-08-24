"""Blender main-thread controller for the queue-only sidecar client."""

from __future__ import annotations

import base64
from collections import deque
import math
import os
import queue
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import bpy

from .live_stream import (
    LiveStreamError,
    get_live_stream_controller,
    unregister_live_stream,
    validate_stream_frame,
)
from .model_inputs import (
    ModelInputError,
    validate_model_engines,
    validate_model_pair,
)
from .model_optimize import (
    ModelOptimizationCancelled,
    ModelOptimizationError,
    OptimizationProgress,
    optimize_models,
)
from .path_contract import require_unaliased_path
from .preferences import get_preferences
from .protocol import WORKER_PROFILE
from .properties import apply_model_schema, inference_settings
from .runtime_bundle import (
    BundleError,
    RuntimeModelSpec,
    resolve_runtime_bundle,
)
from .sidecar import (
    ClientDiagnostic,
    ControlMessage,
    Lifecycle,
    ProcessExited,
    SidecarClient,
    SidecarError,
)
from .wav_stream import WavStreamSource


def _model_emotion_channels(model_schema: dict[str, Any]) -> list[str]:
    return [descriptor["name"] for descriptor in model_schema["emotion_channels"]]


@dataclass(frozen=True, slots=True)
class SetupStatus:
    ready: bool
    message: str

    def __post_init__(self) -> None:
        if type(self.ready) is not bool:
            raise TypeError("setup status readiness must be an exact bool")
        if type(self.message) is not str or not self.message:
            raise ValueError("setup status message must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RuntimeSetupSnapshot:
    runtime_status: SetupStatus
    model_status: SetupStatus
    engine_status: SetupStatus
    model_spec: RuntimeModelSpec | None

    def __post_init__(self) -> None:
        model_ready = self.runtime_status.ready and self.model_status.ready
        if (self.model_spec is not None) != model_ready:
            raise ValueError("runtime-model specification contradicts its statuses")
        if self.engine_status.ready and not model_ready:
            raise ValueError("engine readiness requires a valid runtime and model pair")

    def require_optimization_spec(self) -> RuntimeModelSpec:
        if not self.runtime_status.ready:
            raise SidecarError(self.runtime_status.message)
        if not self.model_status.ready:
            raise SidecarError(self.model_status.message)
        return cast(RuntimeModelSpec, self.model_spec)

    def require_inference_spec(self) -> RuntimeModelSpec:
        spec = self.require_optimization_spec()
        if not self.engine_status.ready:
            raise SidecarError(self.engine_status.message)
        return spec


@dataclass(frozen=True, slots=True)
class PendingRequest:
    method: str
    scene_name: str
    model_signature: tuple[str, str] | None
    operation_id: str | None

    def __post_init__(self) -> None:
        if self.method == "load_model":
            if self.model_signature is None:
                raise ValueError("load_model pending state requires a model signature")
        elif self.model_signature is not None:
            raise ValueError(
                "only load_model pending state may carry a model signature"
            )
        stream_methods = {
            "stream_start",
            "stream_chunk",
            "stream_settings",
            "stream_end",
            "cancel",
        }
        if self.method in stream_methods and self.operation_id is None:
            raise ValueError(f"{self.method} pending state requires an operation ID")
        if self.method not in stream_methods and self.operation_id is not None:
            raise ValueError(f"{self.method} pending state cannot carry an operation ID")


@dataclass(slots=True)
class PCMIngress:
    """Thread-safe pending input whose first chunk starts one live stream."""

    scene_name: str
    chunks: deque[bytes]
    ending: bool = False


@dataclass(slots=True)
class SelectedWavSource:
    """State that exists only when a worker stream is fed from a selected WAV."""

    audio_path: Path
    start_position: float
    cancel: threading.Event
    playing: threading.Event
    playback_started: threading.Event
    thread: threading.Thread | None = None
    timestamp_offset: int | None = None


@dataclass(slots=True)
class ActiveStream:
    """The worker's single canonical stream and its local presentation state."""

    operation_id: str
    scene_name: str
    wav_source: SelectedWavSource | None
    chunk_credit: threading.Event = field(default_factory=threading.Event)
    prebuffer_samples: int | None = None
    end_sent: bool = False
    stop_requested: bool = False
    worker_ended: bool = False
    refresh_deadline: float | None = None


@dataclass(slots=True)
class _StatusNotice:
    """Presentation timing for one operational status, separate from RNA state."""

    status: str
    started_at: float
    visible: bool = False


POLL_INTERVAL_SECONDS = 0.10
PLAYBACK_INTERVAL_SECONDS = 1.0 / 60.0
INFERENCE_REFRESH_DELAY_SECONDS = 0.05
_STATUS_NOTICE_DELAY_SECONDS = 0.25
SHUTDOWN_TIMEOUT_SECONDS = 2.0
HANDSHAKE_TIMEOUT_SECONDS = 15.0
MAX_STREAM_CHUNK_BYTES = 256 * 1024
MAX_PENDING_STREAM_CHUNKS = 64
_STATUS_NOTICE_VALUES = frozenset(
    {
        "STARTING",
        "LOADING_MODEL",
        "STREAM_STARTING",
        "STREAM_ENDING",
        "STOPPING",
    }
)


class RuntimeController:
    """Translate protocol messages into RNA changes on Blender's main thread."""

    def __init__(self) -> None:
        self.client = SidecarClient()
        self.pending: dict[str, PendingRequest] = {}
        self.pending_lock = threading.Lock()
        self.negotiated = False
        self.startup_scene: str | None = None
        self.handshake_spec: RuntimeModelSpec | None = None
        # One sidecar owns exactly one selected model pair, so its signature is global rather
        # than attached to a Blender scene.
        self.loaded_signature: tuple[str, str] | None = None
        self.model_sample_rate: int | None = None
        self.model_schema: dict[str, Any] | None = None
        self.rejected_reason: str | None = None
        self.optimization_thread: threading.Thread | None = None
        self.optimization_cancel: threading.Event | None = None
        self.optimization_commit_lock = threading.Lock()
        self.optimization_events: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self.optimization_progress_lock = threading.Lock()
        self.optimization_latest_progress: OptimizationProgress | None = None
        self.optimization_progress = 0.0
        self.optimization_message = ""
        self.optimization_failed = False
        self.handshake_deadline: float | None = None
        self.last_worker_diagnostic = ""
        self.expected_worker_exit = False
        self.stream_source_events: queue.Queue[tuple[str, str, str | None]] = queue.Queue()
        self.active_stream: ActiveStream | None = None
        self.selected_restart: tuple[str, float, bool] | None = None
        self.pcm_ingress: PCMIngress | None = None
        self._status_notices: dict[int, _StatusNotice] = {}

    def _scene(self, name: str | None) -> bpy.types.Scene | None:
        return bpy.data.scenes.get(name) if name else None

    @staticmethod
    def _editable_scenes() -> tuple[bpy.types.Scene, ...]:
        """Return local/override scenes whose RNA state may be changed safely."""

        return tuple(
            scene
            for scene in bpy.data.scenes
            if scene.is_editable
        )

    @staticmethod
    def _tag_runtime_setup_redraw() -> None:
        """Refresh Preferences and the compact sidebar setup status."""

        window_manager = getattr(bpy.context, "window_manager", None)
        if window_manager is None:
            return
        for window in window_manager.windows:
            screen = getattr(window, "screen", None)
            if screen is None:
                continue
            for area in screen.areas:
                if area.type in {"PREFERENCES", "VIEW_3D"}:
                    area.tag_redraw()

    def _queue_optimization_progress(self, event: OptimizationProgress) -> None:
        """Publish the latest optimizer snapshot without touching :mod:`bpy`."""

        with self.optimization_progress_lock:
            self.optimization_latest_progress = event

    def _take_optimization_progress(self) -> OptimizationProgress | None:
        with self.optimization_progress_lock:
            event = self.optimization_latest_progress
            self.optimization_latest_progress = None
        return event

    @staticmethod
    def _require_editable_scene(scene: bpy.types.Scene) -> None:
        if not scene.is_editable:
            raise SidecarError("Audio2Face requires an editable local or library-override scene")

    def _set_status(
        self,
        scene: bpy.types.Scene,
        status: str,
        message: str,
    ) -> None:
        if not scene.is_editable:
            return
        settings = scene.audio2face
        previous_status = settings.status
        changed = settings.status != status or settings.status_message != message
        settings.status = status
        settings.status_message = message
        if changed:
            scene_key = int(scene.as_pointer())
            current = self._status_notices.get(scene_key)
            if status in _STATUS_NOTICE_VALUES:
                if previous_status != status or current is None:
                    self._status_notices[scene_key] = _StatusNotice(
                        status=status,
                        started_at=time.monotonic(),
                    )
            else:
                self._status_notices.pop(scene_key, None)
            self._tag_runtime_setup_redraw()

    def status_notice(
        self,
        scene: bpy.types.Scene,
    ) -> tuple[str, str] | None:
        """Return only an error or an informational status that has persisted."""

        settings = scene.audio2face
        if settings.status == "ERROR":
            return settings.status, settings.status_message
        notice = self._status_notices.get(int(scene.as_pointer()))
        if (
            notice is None
            or not notice.visible
            or notice.status != settings.status
        ):
            return None
        return settings.status, settings.status_message

    def _poll_status_notices(self) -> None:
        """Reveal a still-current informational status after one stable delay."""

        scenes = {
            int(scene.as_pointer()): scene
            for scene in self._editable_scenes()
        }
        now = time.monotonic()
        redraw = False
        for scene_key, notice in tuple(self._status_notices.items()):
            scene = scenes.get(scene_key)
            if scene is None or scene.audio2face.status != notice.status:
                redraw = redraw or notice.visible
                del self._status_notices[scene_key]
            elif (
                not notice.visible
                and now - notice.started_at >= _STATUS_NOTICE_DELAY_SECONDS
            ):
                notice.visible = True
                redraw = True
        if redraw:
            self._tag_runtime_setup_redraw()

    @staticmethod
    def _selected_path(value: str, description: str) -> Path:
        if value.startswith("//"):
            raise SidecarError(f"{description} must be one canonical absolute path")
        return require_unaliased_path(
            value,
            description=description,
            error_type=SidecarError,
        )

    @staticmethod
    def _selected_directory_path(value: str, description: str) -> Path:
        """Remove only the terminal separator emitted by Blender ``DIR_PATH``."""

        if value.startswith("//"):
            raise SidecarError(f"{description} must be one canonical absolute path")
        canonical = os.path.abspath(value)
        separators = os.sep if os.altsep is None else os.sep + os.altsep
        if value != canonical and value.rstrip(separators) != canonical:
            raise SidecarError(f"{description} must be one canonical absolute path")
        return require_unaliased_path(
            canonical,
            description=description,
            error_type=SidecarError,
        )

    @staticmethod
    def _extension_data_directory(name: str) -> Path:
        description = f"Audio2Face {name} directory"
        try:
            value = bpy.utils.extension_path_user(
                __package__,
                path=name,
                create=True,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise SidecarError(f"cannot access the {description}: {exc}") from exc
        path = require_unaliased_path(
            value,
            description=description,
            error_type=SidecarError,
        )
        if not path.is_dir():
            raise SidecarError(f"{description} is not a directory: {path}")
        return path

    @staticmethod
    def log_directory() -> Path:
        return RuntimeController._extension_data_directory("logs")

    def setup_snapshot(self) -> RuntimeSetupSnapshot:
        """Inspect the bundled runtime and selected model pair exactly once."""

        try:
            bundle = resolve_runtime_bundle()
        except BundleError as exc:
            bundle = None
            runtime_status = SetupStatus(False, str(exc))
        else:
            runtime_status = SetupStatus(
                True,
                f"The bundled {bundle.platform} GPU runtime is valid",
            )

        try:
            audio2face_directory, audio2emotion_directory = self._model_selections()
            audio2face_model, audio2emotion_model = validate_model_pair(
                audio2face_directory,
                audio2emotion_directory,
            )
        except (ModelInputError, SidecarError) as exc:
            models = None
            model_status = SetupStatus(False, str(exc))
            engine_status = SetupStatus(
                False,
                "Both model folders must be valid before engine status can be checked",
            )
        else:
            models = (audio2face_model, audio2emotion_model)
            model_status = SetupStatus(
                True,
                "Both selected model folders contain the required files",
            )
            try:
                validate_model_engines(audio2face_model, audio2emotion_model)
            except ModelInputError:
                engine_status = SetupStatus(
                    False,
                    "Click Optimize Models to generate the GPU-specific "
                    "TensorRT engines from the downloaded ONNX models",
                )
            else:
                engine_status = SetupStatus(
                    True,
                    "Selected models are optimized for this GPU",
                )

        model_spec = (
            RuntimeModelSpec(
                runtime=bundle,
                audio2face_model=models[0],
                audio2emotion_model=models[1],
            )
            if bundle is not None and models is not None
            else None
        )
        return RuntimeSetupSnapshot(
            runtime_status=runtime_status,
            model_status=model_status,
            engine_status=engine_status,
            model_spec=model_spec,
        )

    def _model_selections(self) -> tuple[Path, Path]:
        """Return the two configured folder paths without validating model files."""

        preferences = get_preferences()
        if preferences is None:
            raise SidecarError("Audio2Face Add-on Preferences are unavailable")
        selections: list[Path] = []
        for value, label in (
            (preferences.audio2face_model_directory, "Audio2Face"),
            (preferences.audio2emotion_model_directory, "Audio2Emotion"),
        ):
            if not value:
                raise SidecarError(
                    f"select the complete downloaded {label} model folder "
                    "in Add-on Preferences"
                )
            selections.append(
                self._selected_directory_path(
                    value,
                    f"selected {label} model directory",
                )
            )
        return selections[0], selections[1]

    def optimization_eligibility(
        self,
        setup: RuntimeSetupSnapshot,
    ) -> tuple[bool, str]:
        """Return whether both selected model engines can be rebuilt now."""

        if self.optimization_in_progress:
            return False, "model optimization is already running"
        if not setup.runtime_status.ready:
            return False, setup.runtime_status.message
        if not setup.model_status.ready:
            return False, setup.model_status.message
        if self.client.state not in {Lifecycle.STOPPED, Lifecycle.FAILED}:
            return False, "stop the Audio2Face worker before optimizing models"
        preferences = get_preferences()
        if preferences is None:
            return False, "Audio2Face Add-on Preferences are unavailable"
        if not preferences.nvidia_terms_accepted:
            return False, "accept the NVIDIA terms first"
        return True, "The bundled GPU runtime and selected model inputs are ready"

    @property
    def optimization_in_progress(self) -> bool:
        return self.optimization_thread is not None

    @property
    def operation_in_progress(self) -> bool:
        if (
            self.active_stream is not None
            or self.selected_restart is not None
            or self.client.state == Lifecycle.STOPPING
        ):
            return True
        with self.pending_lock:
            return any(
                pending.method in {"hello", "load_model"}
                for pending in self.pending.values()
            )

    def _require_operation_idle(self) -> None:
        if self.operation_in_progress:
            raise SidecarError("wait for the current Audio2Face operation to finish")

    def optimize_models(self) -> None:
        """Optimize both selected models without blocking Blender's UI."""

        setup = self.setup_snapshot()
        can_optimize, reason = self.optimization_eligibility(setup)
        if not can_optimize:
            raise SidecarError(reason)
        spec = setup.require_optimization_spec()
        log_directory = self.log_directory()

        canceled = threading.Event()
        with self.optimization_progress_lock:
            self.optimization_latest_progress = None
        self.optimization_cancel = canceled
        self.optimization_progress = 0.0
        self.optimization_message = "Preparing both NVIDIA models"
        self.optimization_failed = False
        self._tag_runtime_setup_redraw()

        def progress(event: OptimizationProgress) -> None:
            self._queue_optimization_progress(event)

        def run_optimization() -> None:
            try:
                optimize_models(
                    spec,
                    log_directory=log_directory,
                    progress=progress,
                    canceled=canceled,
                    commit_lock=self.optimization_commit_lock,
                )
            except ModelOptimizationCancelled:
                self.optimization_events.put(("canceled", None))
            except (ModelOptimizationError, OSError, ValueError) as exc:
                self.optimization_events.put(("error", str(exc)))
            except Exception as exc:  # Never let a background exception disappear.
                self.optimization_events.put(("error", f"model optimization failed: {exc}"))
            else:
                self.optimization_events.put(("complete", None))

        self.optimization_thread = threading.Thread(
            name="audio2face-model-optimization",
            target=run_optimization,
            daemon=True,
        )
        try:
            self.optimization_thread.start()
        except RuntimeError as exc:
            self.optimization_thread = None
            self.optimization_cancel = None
            self.optimization_progress = 0.0
            self.optimization_message = f"could not start model optimization: {exc}"
            self._tag_runtime_setup_redraw()
            raise SidecarError(f"could not start model optimization: {exc}") from exc

    def cancel_model_optimization(self) -> None:
        if not self.optimization_in_progress or self.optimization_cancel is None:
            raise SidecarError("model optimization is not running")
        with self.optimization_commit_lock:
            self.optimization_cancel.set()
        self.optimization_message = "Canceling model optimization"
        self._tag_runtime_setup_redraw()

    def start(self, scene: bpy.types.Scene) -> None:
        self._require_editable_scene(scene)
        if self.optimization_in_progress:
            raise SidecarError("wait for model optimization to finish")
        if self.client.state == Lifecycle.RUNNING:
            raise SidecarError("worker is already running")
        if self.client.state == Lifecycle.STOPPING:
            raise SidecarError("worker is still shutting down")
        spec = self.setup_snapshot().require_inference_spec()
        self.last_worker_diagnostic = ""
        self.client.start(
            spec.runtime.executable,
            cwd=spec.runtime.root,
            env=spec.runtime.env,
        )
        self.handshake_spec = spec
        with self.pending_lock:
            self.pending.clear()
        self.selected_restart = None
        self._release_active_stream()
        with self.pending_lock:
            self.pcm_ingress = None
        self.negotiated = False
        self._clear_model_state()
        self.rejected_reason = None
        self.expected_worker_exit = False
        self.startup_scene = scene.name
        self.handshake_deadline = time.monotonic() + HANDSHAKE_TIMEOUT_SECONDS
        self._send_hello(scene)

    def _require_worker_ready(self) -> None:
        if self.client.state != Lifecycle.RUNNING or not self.negotiated:
            raise SidecarError("start the Audio2Face worker first")

    def _request(
        self,
        scene: bpy.types.Scene,
        method: str,
        params: dict[str, Any],
        *,
        model_signature: tuple[str, str] | None,
        operation_id: str | None,
    ) -> str:
        with self.pending_lock:
            # Store correlation state before the main-thread poller can consume
            # a fast worker response.  This also covers calls made by an audio
            # source thread.
            return self._request_locked(
                scene.name,
                method,
                params,
                model_signature=model_signature,
                operation_id=operation_id,
            )

    def _request_locked(
        self,
        scene_name: str,
        method: str,
        params: dict[str, Any],
        *,
        model_signature: tuple[str, str] | None,
        operation_id: str | None,
    ) -> str:
        """Submit and correlate one request while ``pending_lock`` is held."""

        request_id = self.client.request(method, params)
        self.pending[request_id] = PendingRequest(
            method=method,
            scene_name=scene_name,
            model_signature=model_signature,
            operation_id=operation_id,
        )
        return request_id

    def _send_hello(self, scene: bpy.types.Scene) -> None:
        self._request(
            scene,
            "hello",
            {},
            model_signature=None,
            operation_id=None,
        )
        self._set_status(scene, "STARTING", "Starting bundled Audio2Face GPU worker")

    @staticmethod
    def _model_signature(spec: RuntimeModelSpec) -> tuple[str, str]:
        return (
            str(spec.audio2face_model),
            str(spec.audio2emotion_model),
        )

    def _clear_model_state(self) -> None:
        self.loaded_signature = None
        self.model_sample_rate = None
        self.model_schema = None

    def _ensure_scene_model_schema(self, scene: bpy.types.Scene) -> None:
        """Populate model-derived emotion channels for the target scene."""

        model_schema = self.model_schema
        if self.loaded_signature is None or model_schema is None:
            raise SidecarError("loaded worker model metadata is unavailable")
        try:
            apply_model_schema(
                scene.audio2face,
                model_schema,
                self.loaded_signature,
            )
        except ValueError as exc:
            raise SidecarError(str(exc)) from exc

    def _submit_model_load(
        self,
        scene: bpy.types.Scene,
        spec: RuntimeModelSpec,
    ) -> None:
        signature = self._model_signature(spec)
        self._clear_model_state()
        self._request(
            scene,
            "load_model",
            {
                "audio2face_model_path": str(spec.audio2face_model),
                "audio2emotion_model_path": str(spec.audio2emotion_model),
            },
            model_signature=signature,
            operation_id=None,
        )
        self._set_status(
            scene,
            "LOADING_MODEL",
            "Loading Audio2Face 3.0 and Audio2Emotion 3.0 models",
        )

    def _submit_stream_start(
        self,
        scene: bpy.types.Scene,
        *,
        audio_path: Path | None,
        audio_start_position: float = 0.0,
        start_paused: bool = False,
    ) -> str:
        sample_rate = self.model_sample_rate
        if sample_rate is None:
            raise SidecarError("worker model did not report its sampling rate")
        model_schema = self.model_schema
        if model_schema is None:
            raise SidecarError("worker model did not report its output channels")
        if self.active_stream is not None:
            raise SidecarError("another Audio2Face stream is already active")
        self._ensure_scene_model_schema(scene)
        operation_id = uuid.uuid4().hex
        wav_source = (
            SelectedWavSource(
                audio_path=audio_path,
                start_position=audio_start_position,
                cancel=threading.Event(),
                playing=threading.Event(),
                playback_started=threading.Event(),
            )
            if audio_path is not None
            else None
        )
        if wav_source is not None and not start_paused:
            wav_source.playing.set()
        try:
            get_live_stream_controller().prepare(
                scene,
                operation_id,
                sample_rate,
                model_schema["channels"],
                _model_emotion_channels(model_schema),
                audio_path=audio_path,
                audio_start_position=audio_start_position,
                start_paused=start_paused,
                playback_started=(
                    wav_source.playback_started.set
                    if wav_source is not None
                    else None
                ),
                playback_paused=(
                    wav_source.playing.clear if wav_source is not None else None
                ),
                playback_resumed=(
                    wav_source.playing.set if wav_source is not None else None
                ),
                playback_seeked=(
                    (
                        lambda position, paused: self.seek_selected_audio(
                            scene,
                            position,
                            paused=paused,
                        )
                    )
                    if wav_source is not None
                    else None
                ),
                playback_stopped=lambda natural: self._finish_stream_presentation(
                    scene.name,
                    operation_id,
                    natural=natural,
                ),
            )
        except LiveStreamError as exc:
            raise SidecarError(str(exc)) from exc
        try:
            self._request(
                scene,
                "stream_start",
                {
                    "operation_id": operation_id,
                    "sample_rate": sample_rate,
                    "settings": inference_settings(scene.audio2face),
                },
                model_signature=None,
                operation_id=operation_id,
            )
        except Exception:
            get_live_stream_controller().stop(reset=False, notify=False)
            raise

        scene.audio2face.stream_time = 0.0
        stream = ActiveStream(
            operation_id=operation_id,
            scene_name=scene.name,
            wav_source=wav_source,
        )
        stream.chunk_credit.set()
        self.active_stream = stream
        self._set_status(scene, "STREAM_STARTING", "Preparing audio inference")
        return operation_id

    def start_selected_audio(
        self,
        scene: bpy.types.Scene,
        *,
        position: float = 0.0,
        paused: bool = False,
    ) -> str | None:
        """Start selected-WAV inference as the direct result of pressing Play."""

        self._require_editable_scene(scene)
        self._require_operation_idle()
        self._require_worker_ready()
        if type(position) is not float or not math.isfinite(position) or position < 0.0:
            raise SidecarError("selected audio position must be a finite non-negative float")
        if type(paused) is not bool:
            raise SidecarError("paused must be an exact bool")
        settings = scene.audio2face
        audio_path = self._selected_path(settings.audio_path, "selected WAV file")
        if not audio_path.is_file():
            raise SidecarError(f"audio file does not exist: {audio_path}")
        spec = self.setup_snapshot().require_inference_spec()
        if self.loaded_signature != self._model_signature(spec):
            self.selected_restart = (scene.name, position, paused)
            self._submit_model_load(
                scene,
                spec,
            )
            return None
        return self._submit_stream_start(
            scene,
            audio_path=audio_path,
            audio_start_position=position,
            start_paused=paused,
        )

    def _stream_scene(self, operation_id: str) -> bpy.types.Scene | None:
        stream = self.active_stream
        if stream is None or stream.operation_id != operation_id:
            return None
        scene = self._scene(stream.scene_name)
        if scene is None or not scene.is_editable:
            return None
        return scene

    @staticmethod
    def _validate_f32le_chunk(audio_f32le: bytes) -> bytes:
        if type(audio_f32le) is not bytes:
            raise SidecarError("stream audio must be an exact bytes mono f32le payload")
        payload = audio_f32le
        if not payload:
            raise SidecarError("stream audio chunk must not be empty")
        if len(payload) > MAX_STREAM_CHUNK_BYTES:
            raise SidecarError(
                f"stream audio chunk exceeds {MAX_STREAM_CHUNK_BYTES} bytes"
            )
        if len(payload) % 4:
            raise SidecarError("stream f32le payload length must be divisible by four")
        if any(not math.isfinite(value[0]) for value in struct.iter_unpack("<f", payload)):
            raise SidecarError("stream audio samples must be finite float32 values")
        return payload

    def _send_stream_audio(
        self,
        audio_f32le: bytes,
        *,
        operation_id: str,
    ) -> str:
        """Send one validated mono-f32le chunk to an accepted worker stream."""

        stream = self.active_stream
        if stream is None or stream.operation_id != operation_id:
            raise SidecarError("the requested PCM stream is not active")
        with self.pending_lock:
            if not stream.chunk_credit.is_set():
                raise SidecarError("wait for the worker's next PCM chunk credit")
            stream.chunk_credit.clear()
            try:
                return self._request_locked(
                    stream.scene_name,
                    "stream_chunk",
                    {
                        "operation_id": operation_id,
                        "audio_f32le_base64": base64.b64encode(audio_f32le).decode("ascii"),
                    },
                    model_signature=None,
                    operation_id=operation_id,
                )
            except Exception:
                stream.chunk_credit.set()
                raise

    def queue_pcm_audio(self, audio_f32le: bytes, *, scene_name: str) -> None:
        """Buffer live PCM; the main-thread poller opens the stream on first audio."""

        if type(scene_name) is not str:
            raise TypeError("scene_name must be an exact string")
        if not scene_name:
            raise SidecarError("scene_name must not be empty")
        payload = self._validate_f32le_chunk(audio_f32le)
        sample_rate = self.model_sample_rate
        if sample_rate is None:
            raise SidecarError("start the Audio2Face worker before sending PCM audio")
        maximum_bytes = min(MAX_STREAM_CHUNK_BYTES, sample_rate * 4)
        if len(payload) > maximum_bytes:
            raise SidecarError(
                f"stream audio chunk exceeds one model-rate second ({maximum_bytes} bytes)"
            )
        with self.pending_lock:
            ingress = self.pcm_ingress
            if ingress is None:
                ingress = PCMIngress(scene_name=scene_name, chunks=deque())
                self.pcm_ingress = ingress
            elif ingress.scene_name != scene_name:
                raise SidecarError("another Blender scene already owns live PCM input")
            if ingress.ending:
                raise SidecarError("live PCM input has already ended")
            in_flight = sum(
                pending.method == "stream_chunk" for pending in self.pending.values()
            )
            if len(ingress.chunks) + in_flight >= MAX_PENDING_STREAM_CHUNKS:
                raise SidecarError(
                    "stream audio queue is full; the source is outrunning inference"
                )
            ingress.chunks.append(payload)

    def finish_pcm_audio(self, *, scene_name: str) -> None:
        """Mark live PCM complete after every already-buffered chunk."""

        if type(scene_name) is not str:
            raise TypeError("scene_name must be an exact string")
        with self.pending_lock:
            ingress = self.pcm_ingress
            if ingress is None or ingress.scene_name != scene_name:
                raise SidecarError("there is no live PCM input for this scene")
            if ingress.ending:
                raise SidecarError("live PCM input has already ended")
            ingress.ending = True

    def _queue_stream_end(
        self,
        operation_id: str,
    ) -> str:
        stream = self.active_stream
        if stream is None or stream.operation_id != operation_id:
            raise SidecarError("the requested PCM stream is not active")
        with self.pending_lock:
            if stream.end_sent:
                raise SidecarError("the active PCM stream has already ended")
            request_id = self._request_locked(
                stream.scene_name,
                "stream_end",
                {"operation_id": operation_id},
                model_signature=None,
                operation_id=operation_id,
            )
            stream.end_sent = True
            return request_id

    def _queue_stream_settings(
        self,
        stream: ActiveStream,
        settings: dict[str, Any],
    ) -> bool:
        """Serialize one settings snapshot against EOF and cancellation."""

        with self.pending_lock:
            if (
                stream.end_sent
                or stream.worker_ended
                or stream.stop_requested
                or any(
                    pending.operation_id == stream.operation_id
                    and pending.method == "stream_settings"
                    for pending in self.pending.values()
                )
            ):
                return False
            self._request_locked(
                stream.scene_name,
                "stream_settings",
                {"operation_id": stream.operation_id, "settings": settings},
                model_signature=None,
                operation_id=stream.operation_id,
            )
        return True

    def _start_wav_stream_source(
        self,
        scene: bpy.types.Scene,
        operation_id: str,
        wav_source: SelectedWavSource,
        sample_rate: int,
        prebuffer_samples: int,
    ) -> None:
        prediction_delay = float(scene.audio2face.prediction_delay)
        if not math.isfinite(prediction_delay):
            raise SidecarError("prediction delay must be finite")
        start_sample = round(wav_source.start_position * sample_rate)
        source_start_sample = max(0, start_sample - prebuffer_samples)
        wav_source.timestamp_offset = source_start_sample
        prediction_lead = max(0, math.ceil(prediction_delay * sample_rate))
        required_prebuffer = (
            start_sample
            - source_start_sample
            + prebuffer_samples
            + prediction_lead
        )
        stream = self.active_stream
        if stream is None or stream.operation_id != operation_id:
            raise SidecarError("the selected-WAV stream is no longer active")
        chunk_credit = stream.chunk_credit

        def run_source() -> None:
            try:
                chunk_frames = max(1, min(sample_rate // 10, 65_536))
                with WavStreamSource(
                    wav_source.audio_path,
                    output_sample_rate=sample_rate,
                    chunk_frames=chunk_frames,
                    start_frame=source_start_sample,
                ) as wav_reader:
                    samples_sent = 0
                    playback_clock: float | None = None
                    initial_lead_samples = 0
                    pause_started: float | None = None
                    for chunk in wav_reader:
                        if wav_source.cancel.is_set():
                            return
                        if (
                            playback_clock is None
                            and samples_sent >= required_prebuffer
                        ):
                            while not wav_source.playback_started.wait(0.05):
                                if wav_source.cancel.is_set():
                                    return
                            playback_clock = time.monotonic()
                            initial_lead_samples = samples_sent
                        if playback_clock is not None:
                            while not wav_source.playing.is_set():
                                if pause_started is None:
                                    pause_started = time.monotonic()
                                if wav_source.cancel.wait(0.05):
                                    return
                            if pause_started is not None:
                                playback_clock += time.monotonic() - pause_started
                                pause_started = None
                            target = playback_clock + (
                                samples_sent - initial_lead_samples
                            ) / sample_rate
                            while True:
                                delay = target - time.monotonic()
                                if delay <= 0.0:
                                    break
                                if wav_source.cancel.wait(min(0.05, delay)):
                                    return
                        self._send_stream_audio(chunk, operation_id=operation_id)
                        while not chunk_credit.wait(0.05):
                            if wav_source.cancel.is_set():
                                return
                        if wav_source.cancel.is_set():
                            return
                        samples_sent += len(chunk) // 4
                if wav_source.cancel.is_set():
                    return
                self._queue_stream_end(operation_id)
                self.stream_source_events.put(("ending", operation_id, None))
            except (OSError, SidecarError, ValueError) as exc:
                if not wav_source.cancel.is_set():
                    self.stream_source_events.put(("error", operation_id, str(exc)))
            except Exception as exc:
                if not wav_source.cancel.is_set():
                    self.stream_source_events.put(
                        ("error", operation_id, f"selected-WAV stream failed: {exc}")
                    )

        wav_source.thread = threading.Thread(
            name="a2f-selected-wav-stream",
            target=run_source,
            daemon=True,
        )
        try:
            wav_source.thread.start()
        except RuntimeError as exc:
            wav_source.thread = None
            raise SidecarError(f"could not start selected-WAV stream source: {exc}") from exc

        self._set_status(scene, "STREAMING", "Streaming selected WAV as incremental PCM")

    def _poll_stream_source_events(self) -> None:
        while True:
            try:
                kind, operation_id, message = self.stream_source_events.get_nowait()
            except queue.Empty:
                return
            stream = self.active_stream
            if stream is None or stream.operation_id != operation_id:
                continue
            if stream.stop_requested:
                continue
            scene = self._stream_scene(operation_id)
            if scene is None:
                continue
            if kind == "ending":
                self._set_status(scene, "STREAM_ENDING", "Draining final streamed frames")
                continue
            self._fail_stream(
                scene,
                operation_id,
                message,
                cancel_worker=True,
            )

    def _request_stream_cancel(
        self,
        scene: bpy.types.Scene,
        stream: ActiveStream,
    ) -> None:
        """Cancel one media stream without unloading its model."""

        stream.stop_requested = True
        stream.refresh_deadline = None
        if stream.wav_source is not None:
            stream.wav_source.cancel.set()
        try:
            self._request(
                scene,
                "cancel",
                {"operation_id": stream.operation_id},
                model_signature=None,
                operation_id=stream.operation_id,
            )
        except Exception:
            stream.stop_requested = False
            raise

    def _reconcile_input_media(self) -> None:
        """Detach media that no longer matches its owning scene selection."""

        stream = self.active_stream
        if stream is None:
            return
        scene = self._scene(stream.scene_name)
        if scene is None or not scene.is_editable:
            return
        stream_mode = "SELECTED" if stream.wav_source is not None else "STREAM"
        media_matches = scene.audio2face.input_mode == stream_mode
        if media_matches and stream.wav_source is not None:
            media_matches = Path(scene.audio2face.audio_path) == stream.wav_source.audio_path
        if media_matches:
            return

        get_live_stream_controller().stop(reset=False, notify=False)
        if stream.worker_ended:
            self._release_active_stream(stream.operation_id)
            if scene.audio2face.status not in {"ERROR", "IDLE", "STOPPING"}:
                self._set_status(
                    scene,
                    "MODEL_READY",
                    "Input media detached; model remains ready",
                )
            return
        if stream.stop_requested:
            return

        try:
            self._request_stream_cancel(scene, stream)
        except (OSError, SidecarError, ValueError) as exc:
            self._reject_worker_contract(f"could not detach input media: {exc}")
            return
        self._set_status(scene, "STREAM_ENDING", "Switching input media")

    def _clear_pcm_ingress(self) -> None:
        with self.pending_lock:
            self.pcm_ingress = None

    def _poll_pcm_ingress(self) -> None:
        """Turn the first buffered live chunk into one internal worker stream."""

        with self.pending_lock:
            ingress = self.pcm_ingress
            if ingress is None:
                return
            scene_name = ingress.scene_name

        scene = self._scene(scene_name)
        if scene is None or not scene.is_editable:
            self._clear_pcm_ingress()
            return
        settings = scene.audio2face
        if settings.input_mode != "STREAM":
            self._clear_pcm_ingress()
            return

        stream = self.active_stream
        if stream is None:
            try:
                self._require_worker_ready()
                spec = self.setup_snapshot().require_inference_spec()
                if self.loaded_signature != self._model_signature(spec):
                    raise SidecarError(
                        "configured models changed; restart the worker before sending PCM audio"
                    )
                if self.operation_in_progress:
                    raise SidecarError(
                        "live PCM cannot start while another audio operation is active"
                    )
                self._submit_stream_start(scene, audio_path=None)
            except (OSError, SidecarError, ValueError) as exc:
                self._clear_pcm_ingress()
                self._set_status(scene, "ERROR", str(exc))
                return
            return

        if stream.scene_name != scene_name:
            self._clear_pcm_ingress()
            self._set_status(
                scene,
                "ERROR",
                "live PCM cannot run while another audio operation is active",
            )
            return
        if stream.stop_requested:
            return
        if stream.wav_source is not None:
            self._clear_pcm_ingress()
            self._set_status(
                scene,
                "ERROR",
                "live PCM cannot run while another audio operation is active",
            )
            return
        if stream.prebuffer_samples is None:
            return
        operation_id = stream.operation_id

        with self.pending_lock:
            current = self.pcm_ingress
            if current is not ingress:
                return
            if not stream.chunk_credit.is_set():
                return
            chunk = current.chunks.popleft() if current.chunks else None
            should_end = (
                chunk is None
                and current.ending
                and not stream.end_sent
            )
        if chunk is not None:
            try:
                self._send_stream_audio(chunk, operation_id=operation_id)
            except SidecarError as exc:
                with self.pending_lock:
                    current = self.pcm_ingress
                    if current is ingress:
                        current.chunks.appendleft(chunk)
                self._fail_stream(
                    scene,
                    operation_id,
                    str(exc),
                    cancel_worker=True,
                )
            return
        if should_end:
            try:
                self._queue_stream_end(operation_id)
            except SidecarError as exc:
                self._fail_stream(
                    scene,
                    operation_id,
                    str(exc),
                    cancel_worker=True,
                )
            else:
                self._set_status(
                    scene,
                    "STREAM_ENDING",
                    "Draining final live audio frames",
                )

    def _release_active_stream(self, operation_id: str | None = None) -> None:
        stream = self.active_stream
        if stream is None or (
            operation_id is not None and stream.operation_id != operation_id
        ):
            return
        if stream.wav_source is not None:
            stream.wav_source.cancel.set()
        else:
            self._clear_pcm_ingress()
        self.active_stream = None

    def pcm_stream_requirements(
        self,
        scene: bpy.types.Scene,
    ) -> tuple[int, int | None]:
        """Return the model rate and exact prebuffer when live input is accepted."""

        self._require_editable_scene(scene)
        sample_rate = self.model_sample_rate
        if sample_rate is None or sample_rate <= 0:
            raise SidecarError("start the Audio2Face worker before requesting PCM format")
        stream = self.active_stream
        if (
            stream is None
            or stream.scene_name != scene.name
            or stream.prebuffer_samples is None
        ):
            return sample_rate, None
        return sample_rate, stream.prebuffer_samples

    def _finish_stream_presentation(
        self,
        scene_name: str,
        operation_id: str,
        *,
        natural: bool,
    ) -> None:
        stream = self.active_stream
        if (
            stream is None
            or stream.operation_id != operation_id
            or stream.scene_name != scene_name
        ):
            return
        scene = self._scene(scene_name)
        if scene is None or not scene.is_editable:
            self._release_active_stream(operation_id)
            return
        self._release_active_stream(operation_id)
        if scene.audio2face.status not in {"ERROR", "IDLE", "STOPPING"}:
            self._set_status(scene, "MODEL_READY", "PCM stream ended; model remains ready")
        if (
            natural
            and scene.audio2face.input_mode == "SELECTED"
            and scene.audio2face.playback_loop
        ):
            self.selected_restart = (scene.name, 0.0, False)

    def pause_selected_audio(self, scene: bpy.types.Scene) -> None:
        self._require_editable_scene(scene)
        if scene.audio2face.input_mode != "SELECTED":
            raise SidecarError("only selected audio can be paused")
        try:
            get_live_stream_controller().pause()
        except LiveStreamError as exc:
            raise SidecarError(str(exc)) from exc

    def resume_selected_audio(self, scene: bpy.types.Scene) -> None:
        self._require_editable_scene(scene)
        if scene.audio2face.input_mode != "SELECTED":
            raise SidecarError("only selected audio can be resumed")
        try:
            get_live_stream_controller().resume()
        except LiveStreamError as exc:
            raise SidecarError(str(exc)) from exc

    def seek_selected_audio(
        self,
        scene: bpy.types.Scene,
        position: float,
        *,
        paused: bool | None = None,
    ) -> None:
        """Restart only the current inference stream at one selected-audio time."""

        self._require_editable_scene(scene)
        if type(position) is not float or not math.isfinite(position) or position < 0.0:
            raise SidecarError("selected audio position must be a finite non-negative float")
        settings = scene.audio2face
        stream = self.active_stream
        if (
            settings.input_mode != "SELECTED"
            or stream is None
            or stream.scene_name != scene.name
            or stream.wav_source is None
        ):
            raise SidecarError("selected audio playback is not active")
        if paused is None:
            paused = settings.playback_state == "PAUSED"
        if type(paused) is not bool:
            raise SidecarError("paused must be an exact bool")
        if stream.worker_ended:
            live = get_live_stream_controller()
            if live.operation_id != stream.operation_id or not live.can_seek:
                raise SidecarError("active stream is not selected audio")
            self.selected_restart = (scene.name, position, paused)
            try:
                live.stop_for_seek(position, paused=paused)
            except LiveStreamError as exc:
                self.selected_restart = None
                raise SidecarError(str(exc)) from exc
            self._release_active_stream(stream.operation_id)
            self._set_status(scene, "STREAM_ENDING", "Seeking selected audio")
            return
        self._restart_selected_audio(
            scene,
            stream,
            position,
            paused=paused,
        )

    def _restart_selected_audio(
        self,
        scene: bpy.types.Scene,
        stream: ActiveStream,
        position: float,
        *,
        paused: bool,
    ) -> None:
        """Cancel one selected-WAV stream and retain the loaded GPU model."""

        self.selected_restart = (scene.name, position, paused)
        try:
            self._request_stream_cancel(scene, stream)
        except Exception:
            self.selected_restart = None
            raise
        live = get_live_stream_controller()
        try:
            live.stop_for_seek(position, paused=paused)
        except LiveStreamError:
            live.stop(reset=False, notify=False)
        self._set_status(scene, "STREAM_ENDING", "Seeking selected audio")

    def refresh_inference_settings(self, scene: bpy.types.Scene) -> None:
        """Queue one active-stream settings refresh without touching transport."""

        if not scene.is_editable:
            return
        stream = self.active_stream
        if (
            stream is None
            or stream.scene_name != scene.name
            or self.selected_restart is not None
            or stream.end_sent
            or stream.stop_requested
            or stream.worker_ended
        ):
            return
        stream.refresh_deadline = time.monotonic() + INFERENCE_REFRESH_DELAY_SECONDS

    def _poll_inference_refresh(self) -> None:
        stream = self.active_stream
        if stream is None or stream.refresh_deadline is None:
            return
        if time.monotonic() < stream.refresh_deadline:
            return
        scene = self._stream_scene(stream.operation_id)
        if scene is None:
            stream.refresh_deadline = None
            return
        if stream.end_sent or stream.stop_requested or stream.worker_ended:
            stream.refresh_deadline = None
            return
        try:
            queued = self._queue_stream_settings(
                stream,
                inference_settings(scene.audio2face),
            )
        except (OSError, SidecarError, ValueError) as exc:
            stream.refresh_deadline = None
            self._set_status(scene, "ERROR", str(exc))
            return
        if queued:
            stream.refresh_deadline = None

    def _poll_selected_restart(self) -> None:
        restart = self.selected_restart
        if restart is None:
            return
        scene_name, position, paused = restart
        scene = self._scene(scene_name)
        if scene is None or not scene.is_editable:
            self.selected_restart = None
            return
        if scene.audio2face.input_mode != "SELECTED":
            self.selected_restart = None
            scene.audio2face.playback_state = "IDLE"
            return
        if self.active_stream is not None:
            return
        with self.pending_lock:
            if any(
                pending.method in {"load_model", "stream_start", "stream_end", "cancel"}
                for pending in self.pending.values()
            ):
                return
        self.selected_restart = None
        try:
            self.start_selected_audio(scene, position=position, paused=paused)
        except (OSError, SidecarError, ValueError) as exc:
            scene.audio2face.playback_state = "IDLE"
            self._set_status(scene, "ERROR", str(exc))

    def _fail_stream(
        self,
        scene: bpy.types.Scene,
        operation_id: str,
        message: str,
        *,
        cancel_worker: bool,
    ) -> None:
        self.selected_restart = None
        stream = self.active_stream
        if (
            stream is not None
            and stream.operation_id == operation_id
            and stream.wav_source is not None
        ):
            stream.wav_source.cancel.set()
        if cancel_worker and stream is not None and stream.operation_id == operation_id:
            stream.stop_requested = True
            try:
                self._request(
                    scene,
                    "cancel",
                    {"operation_id": operation_id},
                    model_signature=None,
                    operation_id=operation_id,
                )
            except (OSError, SidecarError, ValueError) as exc:
                self._reject_worker_contract(
                    f"{message}; failed to cancel the active stream: {exc}"
                )
                return
        get_live_stream_controller().stop(reset=False, notify=False)
        self._clear_pcm_ingress()
        if not cancel_worker:
            self._release_active_stream(operation_id)
        self._set_status(scene, "ERROR", message)

    def stop(self, scene: bpy.types.Scene) -> None:
        self._require_editable_scene(scene)
        stream = self.active_stream
        if stream is not None and stream.wav_source is not None:
            stream.wav_source.cancel.set()
        self.selected_restart = None
        get_live_stream_controller().stop(reset=False, notify=False)
        if self.client.state in {Lifecycle.STOPPED, Lifecycle.FAILED}:
            self._clear_model_state()
            self._release_active_stream()
            self._set_status(scene, "IDLE", "Worker is already stopped")
            return
        if self.client.state == Lifecycle.STOPPING:
            self._set_status(scene, "STOPPING", "Worker shutdown is already in progress")
            return
        if stream is not None:
            stream.stop_requested = True
            owner = self._scene(stream.scene_name)
            if owner is not None and owner.is_editable:
                self._set_status(owner, "STOPPING", "Worker shutdown requested")
        request_id = self.client.begin_shutdown(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        self.expected_worker_exit = True
        if request_id:
            with self.pending_lock:
                self.pending[request_id] = PendingRequest(
                    "shutdown",
                    scene.name,
                    model_signature=None,
                    operation_id=None,
                )
        self._set_status(scene, "STOPPING", "Worker shutdown requested")

    def _reject_worker_contract(self, message: str) -> None:
        """Record one terminal protocol violation and stop the worker."""

        if self.rejected_reason is not None:
            return
        self.rejected_reason = message
        self.negotiated = False
        self.handshake_deadline = None
        self.handshake_spec = None
        self._clear_model_state()
        scenes = self._editable_scenes()
        self.selected_restart = None
        get_live_stream_controller().stop(reset=False, notify=False)
        self._release_active_stream()
        for scene in scenes:
            self._set_status(scene, "ERROR", message)
        with self.pending_lock:
            self.pending.clear()
            self.pcm_ingress = None
        self.client.begin_shutdown(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        self.expected_worker_exit = True

    def _handle_response(self, envelope: dict[str, Any]) -> None:
        with self.pending_lock:
            request_id = envelope["id"]
            pending = (
                self.pending.pop(request_id)
                if request_id in self.pending
                else None
            )
        if pending is None:
            message = "worker returned a response for an unknown request ID"
            self._reject_worker_contract(message)
            return
        scene = self._scene(pending.scene_name)
        if scene is None or not scene.is_editable:
            if pending.method == "hello":
                self.handshake_spec = None
                self.client.begin_shutdown(timeout=SHUTDOWN_TIMEOUT_SECONDS)
            return
        settings = scene.audio2face
        result = envelope["result"]

        # Once shutdown starts, late model or inference responses must not revive the UI.
        if self.client.state == Lifecycle.STOPPING and pending.method != "shutdown":
            return

        if pending.method == "hello":
            expected_fields = {"worker_profile", "worker_version"}
            valid = (
                set(result) == expected_fields
                and result["worker_profile"] == WORKER_PROFILE
                and isinstance(result["worker_version"], str)
                and bool(result["worker_version"])
            )
            if not valid:
                self.negotiated = False
                self.handshake_spec = None
                self._reject_worker_contract(
                    f"worker does not implement the exact {WORKER_PROFILE} contract"
                )
                return

            spec = self.handshake_spec
            if spec is None:
                self.negotiated = False
                self._reject_worker_contract(
                    "worker startup specification is unavailable"
                )
                return
            self.negotiated = True
            self.handshake_deadline = None
            self.handshake_spec = None
            self._submit_model_load(
                scene,
                spec,
            )
            return

        if pending.method == "load_model":
            expected_fields = {"model_schema", "sample_rate"}
            if set(result) != expected_fields:
                message = "worker returned a noncanonical model response"
                self._clear_model_state()
                self._reject_worker_contract(message)
                return
            model_schema = result["model_schema"]
            sample_rate = result["sample_rate"]
            if (
                isinstance(sample_rate, bool)
                or not isinstance(sample_rate, int)
                or sample_rate <= 0
            ):
                self._clear_model_state()
                self._reject_worker_contract(
                    "worker returned an invalid model sample rate"
                )
                return
            try:
                apply_model_schema(
                    settings,
                    model_schema,
                    pending.model_signature,
                )
            except ValueError as exc:
                self._clear_model_state()
                self._reject_worker_contract(
                    f"worker returned an invalid model schema: {exc}"
                )
                return
            self.loaded_signature = pending.model_signature
            self.model_schema = model_schema
            self.model_sample_rate = sample_rate
            self._set_status(
                scene,
                "MODEL_READY",
                "Loaded Audio2Face 3.0 and Audio2Emotion 3.0 models",
            )
        elif pending.method == "cancel":
            if result:
                self._reject_worker_contract(
                    "worker returned an invalid stream-cancel response",
                )
                return
            stream = self.active_stream
            if stream is None or stream.operation_id != pending.operation_id:
                return
            if settings.status not in {"ERROR", "STOPPING"}:
                self._set_status(scene, "STREAM_ENDING", "Worker accepted stream stop")
        elif pending.method == "stream_start":
            stream = self.active_stream
            if (
                stream is None
                or stream.operation_id != pending.operation_id
                or stream.scene_name != scene.name
            ):
                return
            if set(result) != {"sample_rate", "prebuffer_samples"}:
                self._reject_worker_contract(
                    "worker returned a noncanonical stream response",
                )
                return
            response_rate = result["sample_rate"]
            prebuffer_samples = result["prebuffer_samples"]
            if (
                isinstance(response_rate, bool)
                or not isinstance(response_rate, int)
                or response_rate <= 0
                or response_rate != self.model_sample_rate
                or isinstance(prebuffer_samples, bool)
                or not isinstance(prebuffer_samples, int)
                or prebuffer_samples < 0
            ):
                self._reject_worker_contract(
                    "worker returned a noncanonical stream response",
                )
                return
            self._set_status(scene, "STREAMING", "PCM stream is ready")
            stream.prebuffer_samples = prebuffer_samples
            wav_source = stream.wav_source
            if wav_source is not None:
                self._start_wav_stream_source(
                    scene,
                    stream.operation_id,
                    wav_source,
                    response_rate,
                    prebuffer_samples,
                )
        elif pending.method == "stream_chunk":
            if result:
                self._reject_worker_contract(
                    "worker returned an invalid stream-chunk response",
                )
        elif pending.method == "stream_settings":
            if result:
                self._reject_worker_contract(
                    "worker returned an invalid stream-settings response",
                )
        elif pending.method == "stream_end":
            if result:
                self._reject_worker_contract(
                    "worker returned an invalid stream-end response",
                )
            elif settings.status not in {"ERROR", "STOPPING"}:
                self._set_status(scene, "STREAM_ENDING", "Worker is draining final frames")
        elif pending.method == "shutdown":
            if result:
                self._reject_worker_contract(
                    "worker returned an invalid shutdown response"
                )
            elif self.rejected_reason is None:
                self._set_status(scene, "STOPPING", "Worker is exiting")
        else:
            self._reject_worker_contract(
                f"worker returned a response for unsupported state {pending.method!r}"
            )

    def _handle_error(self, envelope: dict[str, Any]) -> None:
        error = envelope["error"]
        message = f"{error['code']}: {error['message']}"
        if "id" not in envelope:
            self._reject_worker_contract(message)
            return
        with self.pending_lock:
            request_id = envelope["id"]
            pending = self.pending.pop(request_id) if request_id in self.pending else None
        if pending is None:
            self._reject_worker_contract(
                "worker returned an error for an unknown request ID"
            )
            return
        if pending.method == "load_model":
            self._clear_model_state()
        if pending.method == "hello":
            self.handshake_spec = None
        scene = self._scene(pending.scene_name)
        if scene is None or not scene.is_editable:
            return
        if pending.operation_id is not None:
            stream = self.active_stream
            if stream is None or stream.operation_id != pending.operation_id:
                return
            if (
                stream.stop_requested
                and error["code"] == "operation_not_found"
                and pending.method in {"cancel", "stream_chunk", "stream_end"}
            ):
                return
            self._fail_stream(
                scene,
                pending.operation_id,
                message,
                cancel_worker=pending.method != "cancel",
            )
            return
        self._set_status(scene, "ERROR", message)

    def _handle_event(self, envelope: dict[str, Any]) -> None:
        event = envelope["event"]
        data = envelope["data"]
        operation_id = envelope["operation_id"]
        stream = self.active_stream
        if stream is None or stream.operation_id != operation_id:
            self._reject_worker_contract(
                "worker returned an event for an unknown operation ID"
            )
            return
        scene = self._scene(stream.scene_name)
        if scene is None or not scene.is_editable:
            get_live_stream_controller().stop(reset=False, notify=False)
            self._release_active_stream(operation_id)
            return
        settings = scene.audio2face

        if event == "stream_credit":
            if data:
                self._reject_worker_contract(
                    "stream-credit event data must be empty",
                )
                return
            if not stream.stop_requested:
                stream.chunk_credit.set()
        elif event == "stream_reset":
            if data:
                self._reject_worker_contract(
                    "stream-reset event data must be empty",
                )
                return
            if stream.stop_requested:
                return
            try:
                get_live_stream_controller().reset_frames(operation_id)
            except LiveStreamError as exc:
                self._reject_worker_contract(str(exc))
        elif event == "stream_frame":
            if set(data) != {"timestamp_sample", "weights", "emotions"}:
                self._reject_worker_contract(
                    "worker returned invalid stream-frame data",
                )
                return
            if stream.stop_requested:
                model_schema = self.model_schema
                if model_schema is None:
                    self._reject_worker_contract(
                        "worker returned a stream frame without a loaded model"
                    )
                    return
                try:
                    validate_stream_frame(
                        tuple(model_schema["channels"]),
                        tuple(_model_emotion_channels(model_schema)),
                        data["timestamp_sample"],
                        data["weights"],
                        data["emotions"],
                    )
                except (LiveStreamError, TypeError, ValueError) as exc:
                    self._reject_worker_contract(str(exc))
                return
            try:
                timestamp = data["timestamp_sample"]
                if stream.wav_source is not None:
                    if stream.wav_source.timestamp_offset is None:
                        raise LiveStreamError(
                            "selected-audio frame arrived before its source was ready"
                        )
                    timestamp += stream.wav_source.timestamp_offset
                get_live_stream_controller().receive(
                    operation_id,
                    timestamp,
                    data["weights"],
                    data["emotions"],
                )
            except (LiveStreamError, TypeError, ValueError) as exc:
                self._reject_worker_contract(str(exc))
                return
            if stream.end_sent:
                settings.status_message = "Draining final Audio2Face frames"
            else:
                self._set_status(
                    scene,
                    "STREAMING",
                    "Streaming ARKit-52 and emotion values",
                )
        elif event == "stream_ended":
            if data:
                self._reject_worker_contract(
                    "stream-ended event data must be empty",
                )
                return
            explicit_stop = stream.stop_requested
            if explicit_stop:
                if self.selected_restart is None:
                    get_live_stream_controller().stop(reset=False, notify=False)
                self._release_active_stream(operation_id)
                if settings.status not in {"ERROR", "STOPPING"}:
                    self._set_status(
                        scene,
                        "MODEL_READY",
                        "PCM stream stopped; model remains ready",
                    )
            else:
                stream.worker_ended = True
                stream.refresh_deadline = None
                get_live_stream_controller().mark_terminal(operation_id)
                if get_live_stream_controller().active:
                    self._set_status(
                        scene,
                        "STREAMING",
                        "Finishing buffered streamed audio and Audio2Face values",
                    )
                else:
                    self._set_status(
                        scene,
                        "MODEL_READY",
                        "PCM stream ended; model remains ready",
                    )
        elif event == "error":
            if set(data) != {"code", "message"}:
                message = "worker returned invalid operation error data"
                self._reject_worker_contract(message)
                return
            code = data["code"]
            worker_message = data["message"]
            if (
                not isinstance(code, str)
                or not code
                or not isinstance(worker_message, str)
                or not worker_message
            ):
                message = "worker returned invalid operation error data"
                self._reject_worker_contract(message)
                return
            if stream.wav_source is not None:
                stream.wav_source.cancel.set()
            message = f"{code}: {worker_message}"
            self._fail_stream(
                scene,
                operation_id,
                message,
                cancel_worker=False,
            )
        else:
            self._reject_worker_contract(f"worker returned unsupported event {event!r}")

    def _handle_control(self, envelope: dict[str, Any]) -> None:
        if self.rejected_reason is not None:
            return
        if envelope["type"] == "response":
            self._handle_response(envelope)
        elif envelope["type"] == "error":
            self._handle_error(envelope)
        elif envelope["type"] == "event":
            self._handle_event(envelope)

    def _finish_optimization(self, kind: str, payload: str | None) -> None:
        self.optimization_failed = kind == "error"
        if kind == "complete":
            self.optimization_message = "Both NVIDIA models are optimized"
            self.optimization_progress = 1.0
        elif kind == "canceled":
            self.optimization_message = "Model optimization canceled"
            self.optimization_progress = 0.0
        else:
            if payload is None:
                raise RuntimeError("model optimizer error event has no message")
            self.optimization_message = payload

        self.optimization_thread = None
        self.optimization_cancel = None
        with self.optimization_progress_lock:
            self.optimization_latest_progress = None
        self._tag_runtime_setup_redraw()

    def _poll_optimization_events(self) -> None:
        latest_progress = self._take_optimization_progress()
        try:
            terminal = self.optimization_events.get_nowait()
        except queue.Empty:
            terminal = None

        if latest_progress is not None:
            self.optimization_message = latest_progress.message
            self.optimization_progress = latest_progress.progress
            self._tag_runtime_setup_redraw()
        if terminal is not None:
            self._finish_optimization(*terminal)

    def poll(self) -> None:
        self._poll_optimization_events()
        self._reconcile_input_media()
        self._poll_stream_source_events()
        self._poll_pcm_ingress()
        self.client.tick()
        for event in self.client.poll():
            if isinstance(event, ControlMessage):
                self._handle_control(event.envelope)
            elif isinstance(event, ClientDiagnostic):
                self.last_worker_diagnostic = event.message[-1000:]
                if self.rejected_reason is None:
                    scene = self._scene(self.startup_scene)
                    if scene is not None and scene.is_editable:
                        scene.audio2face.status_message = event.message
            elif isinstance(event, ProcessExited):
                expected_exit = self.expected_worker_exit
                self.negotiated = False
                self.handshake_deadline = None
                self.handshake_spec = None
                with self.pending_lock:
                    self.pending.clear()
                    self.pcm_ingress = None
                live_controller = get_live_stream_controller()
                live_controller.stop(reset=False, notify=False)
                self.selected_restart = None
                self._release_active_stream()
                for scene in self._editable_scenes():
                    settings = scene.audio2face
                    if self.rejected_reason:
                        self._set_status(scene, "ERROR", self.rejected_reason)
                    elif expected_exit and event.returncode == 0:
                        self._set_status(scene, "IDLE", "Worker stopped")
                    elif settings.status != "IDLE" or scene.name == self.startup_scene:
                        detail = (
                            f": {self.last_worker_diagnostic}"
                            if self.last_worker_diagnostic
                            else ""
                        )
                        message = f"Worker exited with code {event.returncode}{detail}"
                        self._set_status(scene, "ERROR", message)
                self._clear_model_state()
                self.expected_worker_exit = False

        self._poll_pcm_ingress()
        self._poll_selected_restart()
        self._poll_inference_refresh()

        if (
            self.handshake_deadline is not None
            and not self.negotiated
            and self.client.state == Lifecycle.RUNNING
            and time.monotonic() >= self.handshake_deadline
        ):
            self.handshake_deadline = None
            self.handshake_spec = None
            detail = (
                f": {self.last_worker_diagnostic}" if self.last_worker_diagnostic else ""
            )
            message = f"Audio2Face worker handshake timed out{detail}"
            self._reject_worker_contract(message)

        self._poll_status_notices()

    def close(self) -> None:
        if self.optimization_cancel is not None:
            with self.optimization_commit_lock:
                self.optimization_cancel.set()
        if self.optimization_thread is not None and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        stream = self.active_stream
        if stream is not None and stream.wav_source is not None:
            stream.wav_source.cancel.set()
        if (
            stream is not None
            and stream.wav_source is not None
            and stream.wav_source.thread is not None
            and stream.wav_source.thread.is_alive()
        ):
            stream.wav_source.thread.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        unregister_live_stream()
        self.client.close(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        with self.pending_lock:
            self.pending.clear()
            self.pcm_ingress = None
        self.negotiated = False
        self.handshake_deadline = None
        self.handshake_spec = None
        self._clear_model_state()
        self.optimization_thread = None
        self.optimization_cancel = None
        self.optimization_progress = 0.0
        self._release_active_stream()
        self.selected_restart = None
        self._status_notices.clear()
        with self.optimization_progress_lock:
            self.optimization_latest_progress = None


_CONTROLLER: RuntimeController | None = None


def get_controller() -> RuntimeController:
    global _CONTROLLER
    if _CONTROLLER is None:
        _CONTROLLER = RuntimeController()
    return _CONTROLLER


def _timer_callback() -> float | None:
    controller = _CONTROLLER
    if controller is None:
        return None
    try:
        controller.poll()
        stream_active = get_live_stream_controller().tick()
    except Exception as exc:  # Keep timer alive, but surface the main-thread failure.
        stream_active = False
        scene = bpy.context.scene
        if scene is not None and hasattr(scene, "audio2face"):
            controller._set_status(scene, "ERROR", str(exc))
    return PLAYBACK_INTERVAL_SECONDS if stream_active else POLL_INTERVAL_SECONDS


def _dispose_runtime_state() -> None:
    """Close the controller before removing the singletons it owns."""

    global _CONTROLLER
    controller = _CONTROLLER
    _CONTROLLER = None
    if controller is not None:
        controller.close()
    else:
        unregister_live_stream()


@bpy.app.handlers.persistent
def _load_pre_handler(_unused: object) -> None:
    """Drop process, thread, and playback state before Blender replaces its data."""

    try:
        _dispose_runtime_state()
    except Exception as exc:
        print(f"Audio2Face load cleanup failed: {exc}")


@bpy.app.handlers.persistent
def _load_post_handler(_unused: object) -> None:
    """Restore a fresh controller and timer for the newly loaded file."""

    try:
        register_runtime()
    except Exception as exc:
        print(f"Audio2Face load initialization failed: {exc}")


def register_runtime() -> None:
    get_controller()
    if _load_pre_handler not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(_load_pre_handler)
    if _load_post_handler not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_load_post_handler)
    if not bpy.app.timers.is_registered(_timer_callback):
        bpy.app.timers.register(
            _timer_callback,
            first_interval=POLL_INTERVAL_SECONDS,
            persistent=True,
        )


def unregister_runtime() -> None:
    if bpy.app.timers.is_registered(_timer_callback):
        bpy.app.timers.unregister(_timer_callback)
    if _load_pre_handler in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_load_pre_handler)
    if _load_post_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_load_post_handler)
    _dispose_runtime_state()
