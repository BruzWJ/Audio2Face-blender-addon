"""Blender main-thread controller for the queue-only sidecar client."""

from __future__ import annotations

import base64
import math
import os
import queue
import struct
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import bpy

from .live_stream import (
    LiveStreamError,
    get_live_stream_controller,
    unregister_live_stream,
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
from .preview import get_preview_controller, unregister_preview
from .protocol import WORKER_PROFILE
from .properties import apply_model_schema, tuning_parameters
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


ModelContinuation = Literal["ready", "generate", "stream_wav"]


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
        spec = self.model_spec
        if spec is None:
            raise RuntimeError("valid setup has no runtime-model specification")
        return spec

    def require_inference_spec(self) -> RuntimeModelSpec:
        spec = self.require_optimization_spec()
        if not self.engine_status.ready:
            raise SidecarError(self.engine_status.message)
        return spec


@dataclass(frozen=True, slots=True)
class PendingRequest:
    method: str
    scene_name: str
    continuation: ModelContinuation | None
    model_signature: tuple[str, str, int] | None
    operation_id: str | None

    def __post_init__(self) -> None:
        if self.method == "load_model":
            if self.continuation is None or self.model_signature is None:
                raise ValueError(
                    "load_model pending state requires a continuation and model signature"
                )
        elif self.continuation is not None or self.model_signature is not None:
            raise ValueError(
                "only load_model pending state may carry model continuation metadata"
            )
        stream_methods = {"stream_start", "stream_chunk", "stream_end"}
        if self.method in stream_methods and self.operation_id is None:
            raise ValueError(f"{self.method} pending state requires an operation ID")
        if self.method not in stream_methods | {"cancel"} and self.operation_id is not None:
            raise ValueError(f"{self.method} pending state cannot carry an operation ID")


POLL_INTERVAL_SECONDS = 0.10
PREVIEW_INTERVAL_SECONDS = 1.0 / 60.0
SHUTDOWN_TIMEOUT_SECONDS = 2.0
HANDSHAKE_TIMEOUT_SECONDS = 15.0
MAX_STREAM_CHUNK_BYTES = 256 * 1024
MAX_PENDING_STREAM_CHUNKS = 64


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
        self.loaded_signature: tuple[str, str, int] | None = None
        self.model_sample_rate: int | None = None
        self.model_schema: dict[str, Any] | None = None
        self.schema_scenes: set[int] = set()
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
        self.reset_scene_state_on_poll = True
        self.expected_worker_exit = False
        self.stream_source_thread: threading.Thread | None = None
        self.stream_source_cancel: threading.Event | None = None
        self.stream_playback_started: threading.Event | None = None
        self.stream_source_events: queue.Queue[tuple[str, str, str | None]] = queue.Queue()
        self.stream_audio_paths: dict[str, Path] = {}
        self.generation_scene_names: dict[str, str] = {}
        self.stream_scene_names: dict[str, str] = {}
        self.stream_stop_requests: set[str] = set()

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

    @staticmethod
    def _set_status(
        scene: bpy.types.Scene,
        status: str,
        message: str,
    ) -> None:
        if not scene.is_editable:
            return
        settings = scene.audio2face
        settings.status = status
        settings.status_message = message

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
    def result_directory() -> Path:
        return RuntimeController._extension_data_directory("results")

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
        active_methods = {
            "load_model",
            "generate",
            "stream_start",
            "stream_end",
            "cancel",
        }
        with self.pending_lock:
            pending_operation = any(
                pending.method in active_methods for pending in self.pending.values()
            )
        if pending_operation:
            return True
        return any(
            scene.audio2face.status
            in {
                "LOADING_MODEL",
                "GENERATING",
                "CANCELLING",
                "STREAM_STARTING",
                "STREAMING",
                "STREAM_ENDING",
                "STOPPING",
            }
            for scene in bpy.data.scenes
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
        self.generation_scene_names.clear()
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
        continuation: ModelContinuation | None,
        model_signature: tuple[str, str, int] | None,
        operation_id: str | None,
    ) -> str:
        with self.pending_lock:
            # Store correlation state before the main-thread poller can consume
            # a fast worker response.  This also covers calls made by an audio
            # source thread.
            request_id = self.client.request(method, params)
            self.pending[request_id] = PendingRequest(
                method=method,
                scene_name=scene.name,
                continuation=continuation,
                model_signature=model_signature,
                operation_id=operation_id,
            )
        return request_id

    def _send_hello(self, scene: bpy.types.Scene) -> None:
        self._request(
            scene,
            "hello",
            {},
            continuation=None,
            model_signature=None,
            operation_id=None,
        )
        self._set_status(scene, "STARTING", "Starting bundled Audio2Face GPU worker")

    @staticmethod
    def _model_signature(
        spec: RuntimeModelSpec,
        identity_index: int,
    ) -> tuple[str, str, int]:
        return (
            str(spec.audio2face_model),
            str(spec.audio2emotion_model),
            identity_index,
        )

    def _clear_model_state(self) -> None:
        self.loaded_signature = None
        self.model_sample_rate = None
        self.model_schema = None
        self.schema_scenes.clear()

    def _ensure_scene_model_schema(self, scene: bpy.types.Scene) -> None:
        """Populate model-derived controls for any scene using the loaded worker."""

        model_schema = self.model_schema
        if self.loaded_signature is None or model_schema is None:
            raise SidecarError("loaded worker model metadata is unavailable")
        scene_key = int(scene.as_pointer())
        if scene_key in self.schema_scenes:
            return
        try:
            apply_model_schema(
                scene.audio2face,
                model_schema,
                self.loaded_signature,
            )
        except ValueError as exc:
            raise SidecarError(str(exc)) from exc
        self.schema_scenes.add(scene_key)

    def _submit_model_load(
        self,
        scene: bpy.types.Scene,
        spec: RuntimeModelSpec,
        *,
        identity_index: int,
        continuation: ModelContinuation,
    ) -> None:
        signature = self._model_signature(spec, identity_index)
        self._clear_model_state()
        self._request(
            scene,
            "load_model",
            {
                "audio2face_model_path": str(spec.audio2face_model),
                "audio2emotion_model_path": str(spec.audio2emotion_model),
                "identity_index": identity_index,
            },
            continuation=continuation,
            model_signature=signature,
            operation_id=None,
        )
        self._set_status(
            scene,
            "LOADING_MODEL",
            "Loading Audio2Face 3.0 and Audio2Emotion 3.0 models",
        )

    def generate(self, scene: bpy.types.Scene) -> None:
        self._require_editable_scene(scene)
        self._require_operation_idle()
        self._require_worker_ready()
        settings = scene.audio2face
        spec = self.setup_snapshot().require_inference_spec()
        identity_index = int(settings.identity_index)
        if self.loaded_signature != self._model_signature(spec, identity_index):
            self._submit_model_load(
                scene,
                spec,
                identity_index=identity_index,
                continuation="generate",
            )
            return

        self._ensure_scene_model_schema(scene)

        audio_path = self._selected_path(settings.audio_path, "selected WAV file")
        if not audio_path.is_file():
            raise SidecarError(f"audio file does not exist: {audio_path}")
        output_directory = self.result_directory()
        operation_id = uuid.uuid4().hex
        result_path = output_directory / f"{operation_id}.a2f.json"
        self._request(
            scene,
            "generate",
            {
                "operation_id": operation_id,
                "audio_path": str(audio_path),
                "result_path": str(result_path),
                "settings": tuning_parameters(settings),
            },
            continuation=None,
            model_signature=None,
            operation_id=None,
        )
        settings.result_operation_id = operation_id
        self.generation_scene_names[operation_id] = scene.name
        settings.result_path = ""
        settings.result_audio_path = str(audio_path)
        settings.progress = 0.0
        self._set_status(scene, "GENERATING", "Generation request queued")

    def _submit_stream_start(
        self,
        scene: bpy.types.Scene,
        *,
        audio_path: Path | None,
    ) -> str:
        sample_rate = self.model_sample_rate
        if sample_rate is None:
            raise SidecarError("worker model did not report its sampling rate")
        model_schema = self.model_schema
        if model_schema is None:
            raise SidecarError("worker model did not report its output channels")
        self._ensure_scene_model_schema(scene)
        operation_id = uuid.uuid4().hex
        playback_started = threading.Event() if audio_path is not None else None
        get_preview_controller().stop(
            reset=bool(scene.audio2face.preview_reset_on_stop)
        )
        try:
            get_live_stream_controller().prepare(
                scene,
                operation_id,
                sample_rate,
                model_schema["channels"],
                audio_path=audio_path,
                playback_started=(
                    playback_started.set if playback_started is not None else None
                ),
                playback_stopped=lambda: self._finish_stream_presentation(
                    scene.name,
                    operation_id,
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
                    "settings": tuning_parameters(scene.audio2face),
                },
                continuation=None,
                model_signature=None,
                operation_id=operation_id,
            )
        except Exception:
            get_live_stream_controller().stop(reset=False)
            raise

        settings = scene.audio2face
        settings.stream_operation_id = operation_id
        settings.stream_sample_rate = sample_rate
        settings.stream_prebuffer_samples = 0
        settings.stream_time = 0.0
        self.stream_scene_names[operation_id] = scene.name
        self.stream_source_cancel = threading.Event()
        self.stream_playback_started = playback_started
        if audio_path is not None:
            self.stream_audio_paths[operation_id] = audio_path
        self._set_status(scene, "STREAM_STARTING", "Preparing incremental PCM stream")
        return operation_id

    def start_wav_stream(self, scene: bpy.types.Scene) -> str | None:
        """Start the built-in selected-WAV source on the incremental model path."""

        self._require_editable_scene(scene)
        self._require_operation_idle()
        self._require_worker_ready()
        settings = scene.audio2face
        audio_path = self._selected_path(settings.audio_path, "selected WAV file")
        if not audio_path.is_file():
            raise SidecarError(f"audio file does not exist: {audio_path}")
        spec = self.setup_snapshot().require_inference_spec()
        identity_index = int(settings.identity_index)
        if self.loaded_signature != self._model_signature(spec, identity_index):
            self._submit_model_load(
                scene,
                spec,
                identity_index=identity_index,
                continuation="stream_wav",
            )
            return None
        return self._submit_stream_start(scene, audio_path=audio_path)

    def start_pcm_stream(self, scene: bpy.types.Scene) -> str:
        """Begin source-agnostic mono-f32 PCM ingress for Blender integrations."""

        self._require_editable_scene(scene)
        self._require_operation_idle()
        self._require_worker_ready()
        settings = scene.audio2face
        spec = self.setup_snapshot().require_inference_spec()
        if self.loaded_signature != self._model_signature(
            spec,
            int(settings.identity_index),
        ):
            raise SidecarError(
                "model settings changed; restart the worker before opening a PCM stream"
            )
        return self._submit_stream_start(scene, audio_path=None)

    def _stream_scene(self, operation_id: str) -> bpy.types.Scene | None:
        if operation_id not in self.stream_scene_names:
            return None
        scene = self._scene(self.stream_scene_names[operation_id])
        if (
            scene is None
            or not scene.is_editable
            or scene.audio2face.stream_operation_id != operation_id
        ):
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

    def push_stream_audio(
        self,
        audio_f32le: bytes,
        *,
        operation_id: str,
    ) -> str:
        """Queue one mono-f32le chunk; safe to call from an audio-source thread."""

        if operation_id not in self.stream_scene_names:
            raise SidecarError("the requested PCM stream is not active")
        scene_name = self.stream_scene_names[operation_id]
        payload = self._validate_f32le_chunk(audio_f32le)
        sample_rate = self.model_sample_rate
        if sample_rate is None:
            raise SidecarError("worker model did not report its sampling rate")
        maximum_bytes = min(MAX_STREAM_CHUNK_BYTES, sample_rate * 4)
        if len(payload) > maximum_bytes:
            raise SidecarError(
                f"stream audio chunk exceeds one model-rate second ({maximum_bytes} bytes)"
            )
        with self.pending_lock:
            pending_chunks = sum(
                pending.method == "stream_chunk" for pending in self.pending.values()
            )
            if pending_chunks >= MAX_PENDING_STREAM_CHUNKS:
                raise SidecarError(
                    "stream audio queue is full; the source is outrunning inference"
                )
            request_id = self.client.request(
                "stream_chunk",
                {
                    "operation_id": operation_id,
                    "audio_f32le_base64": base64.b64encode(payload).decode("ascii"),
                },
            )
            self.pending[request_id] = PendingRequest(
                "stream_chunk",
                scene_name,
                continuation=None,
                model_signature=None,
                operation_id=operation_id,
            )
        return request_id

    def _queue_stream_end(
        self,
        operation_id: str,
    ) -> str:
        if operation_id not in self.stream_scene_names:
            raise SidecarError("the requested PCM stream is not active")
        scene_name = self.stream_scene_names[operation_id]
        with self.pending_lock:
            request_id = self.client.request(
                "stream_end",
                {"operation_id": operation_id},
            )
            self.pending[request_id] = PendingRequest(
                "stream_end",
                scene_name,
                continuation=None,
                model_signature=None,
                operation_id=operation_id,
            )
        return request_id

    def _start_wav_stream_source(
        self,
        scene: bpy.types.Scene,
        operation_id: str,
        prebuffer_samples: int,
    ) -> None:
        if self.stream_source_thread is not None and self.stream_source_thread.is_alive():
            raise SidecarError("a selected-WAV stream source is already running")
        audio_path = (
            self.stream_audio_paths[operation_id]
            if operation_id in self.stream_audio_paths
            else None
        )
        canceled = self.stream_source_cancel
        playback_started = self.stream_playback_started
        sample_rate = self.model_sample_rate
        if audio_path is None or canceled is None or playback_started is None:
            raise SidecarError("selected-WAV stream source state is incomplete")
        if sample_rate is None:
            raise SidecarError("worker model did not report its sampling rate")
        if (
            isinstance(prebuffer_samples, bool)
            or not isinstance(prebuffer_samples, int)
            or prebuffer_samples < 0
        ):
            raise SidecarError("worker reported an invalid stream prebuffer")

        def run_source() -> None:
            try:
                chunk_frames = max(1, min(sample_rate // 10, 65_536))
                with WavStreamSource(
                    audio_path,
                    output_sample_rate=sample_rate,
                    chunk_frames=chunk_frames,
                ) as source:
                    samples_sent = 0
                    playback_clock: float | None = None
                    initial_lead_samples = 0
                    required_prebuffer = prebuffer_samples
                    for chunk in source:
                        if canceled.is_set():
                            return
                        if (
                            playback_clock is None
                            and samples_sent >= required_prebuffer
                        ):
                            while not playback_started.wait(0.05):
                                if canceled.is_set():
                                    return
                            playback_clock = time.monotonic()
                            initial_lead_samples = samples_sent
                        if playback_clock is not None:
                            target = playback_clock + (
                                samples_sent - initial_lead_samples
                            ) / sample_rate
                            while True:
                                delay = target - time.monotonic()
                                if delay <= 0.0:
                                    break
                                if canceled.wait(min(0.05, delay)):
                                    return
                        self.push_stream_audio(chunk, operation_id=operation_id)
                        samples_sent += len(chunk) // 4
                if canceled.is_set():
                    return
                self._queue_stream_end(operation_id)
                self.stream_source_events.put(("ending", operation_id, None))
            except (OSError, SidecarError, ValueError) as exc:
                if not canceled.is_set():
                    self.stream_source_events.put(("error", operation_id, str(exc)))
            except Exception as exc:
                if not canceled.is_set():
                    self.stream_source_events.put(
                        ("error", operation_id, f"selected-WAV stream failed: {exc}")
                    )

        self.stream_source_thread = threading.Thread(
            name="a2f-selected-wav-stream",
            target=run_source,
            daemon=True,
        )
        try:
            self.stream_source_thread.start()
        except RuntimeError as exc:
            self.stream_source_thread = None
            raise SidecarError(f"could not start selected-WAV stream source: {exc}") from exc

        self._set_status(scene, "STREAMING", "Streaming selected WAV as incremental PCM")

    def _poll_stream_source_events(self) -> None:
        while True:
            try:
                kind, operation_id, message = self.stream_source_events.get_nowait()
            except queue.Empty:
                return
            scene = self._stream_scene(operation_id)
            if scene is None:
                continue
            if kind == "ending":
                self._set_status(scene, "STREAM_ENDING", "Draining final streamed frames")
                continue
            if message is None:
                message = "selected-WAV stream failed"
            if self.stream_source_cancel is not None:
                self.stream_source_cancel.set()
            try:
                self._request(
                    scene,
                    "cancel",
                    {"operation_id": operation_id},
                    continuation=None,
                    model_signature=None,
                    operation_id=operation_id,
                )
            except SidecarError:
                pass
            get_live_stream_controller().stop(reset=False)
            self._clear_stream_state(scene, operation_id=operation_id)
            self._set_status(scene, "ERROR", message)

    def end_stream(self, scene: bpy.types.Scene) -> None:
        """Close PCM input and let the worker drain the model's padded tail."""

        self._require_editable_scene(scene)
        operation_id = scene.audio2face.stream_operation_id
        if not operation_id:
            raise SidecarError("there is no active PCM stream")
        if self.stream_source_cancel is not None:
            self.stream_source_cancel.set()
        if operation_id not in self.stream_scene_names:
            raise SidecarError("stream input has already ended; stop its buffered playback instead")
        self._request(
            scene,
            "stream_end",
            {"operation_id": operation_id},
            continuation=None,
            model_signature=None,
            operation_id=operation_id,
        )
        self._set_status(scene, "STREAM_ENDING", "Draining final streamed frames")

    def stop_stream(self, scene: bpy.types.Scene) -> None:
        """Cancel one stream without stopping or unloading the GPU worker."""

        self._require_editable_scene(scene)
        operation_id = scene.audio2face.stream_operation_id
        if not operation_id:
            raise SidecarError("there is no active PCM stream")
        if operation_id not in self.stream_scene_names:
            get_live_stream_controller().stop(
                reset=bool(scene.audio2face.stream_reset_on_stop)
            )
            self._clear_stream_state(scene, operation_id=operation_id)
            self._set_status(
                scene,
                "MODEL_READY",
                "Buffered PCM stream stopped; model remains ready",
            )
            return
        if self.stream_source_cancel is not None:
            self.stream_source_cancel.set()
        self.stream_stop_requests.add(operation_id)
        try:
            self._request(
                scene,
                "cancel",
                {"operation_id": operation_id},
                continuation=None,
                model_signature=None,
                operation_id=operation_id,
            )
        except Exception:
            self.stream_stop_requests.discard(operation_id)
            raise
        self._set_status(scene, "STREAM_ENDING", "Stopping PCM stream")

    def _release_worker_stream_state(self, operation_id: str) -> None:
        if operation_id:
            self.stream_audio_paths.pop(operation_id, None)
            self.stream_scene_names.pop(operation_id, None)
            self.stream_stop_requests.discard(operation_id)
        self.stream_source_thread = None
        self.stream_source_cancel = None
        self.stream_playback_started = None

    def _release_generation_operation(
        self,
        scene: bpy.types.Scene,
        operation_id: str,
        *,
        keep_result: bool,
    ) -> None:
        self.generation_scene_names.pop(operation_id, None)
        if (
            not keep_result
            and scene.audio2face.result_operation_id == operation_id
        ):
            scene.audio2face.result_operation_id = ""

    def _clear_stream_state(
        self,
        scene: bpy.types.Scene,
        *,
        operation_id: str,
    ) -> None:
        self._release_worker_stream_state(operation_id)
        if scene.audio2face.stream_operation_id == operation_id:
            scene.audio2face.stream_operation_id = ""
            scene.audio2face.stream_sample_rate = 0
            scene.audio2face.stream_prebuffer_samples = 0

    def pcm_stream_requirements(
        self,
        scene: bpy.types.Scene,
    ) -> tuple[int, int] | None:
        """Return model-rate/prebuffer metadata once an external PCM stream is ready."""

        self._require_editable_scene(scene)
        settings = scene.audio2face
        if not settings.stream_operation_id:
            raise SidecarError("there is no active PCM stream")
        if settings.status == "STREAM_STARTING":
            return None
        sample_rate = settings.stream_sample_rate
        prebuffer_samples = settings.stream_prebuffer_samples
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
            or isinstance(prebuffer_samples, bool)
            or not isinstance(prebuffer_samples, int)
            or prebuffer_samples < 0
        ):
            raise SidecarError("active PCM stream metadata is invalid")
        return sample_rate, prebuffer_samples

    def _finish_stream_presentation(self, scene_name: str, operation_id: str) -> None:
        scene = self._scene(scene_name)
        if scene is None or not scene.is_editable:
            self._release_worker_stream_state(operation_id)
            return
        if scene.audio2face.stream_operation_id != operation_id:
            self._release_worker_stream_state(operation_id)
            return
        self._clear_stream_state(scene, operation_id=operation_id)
        if scene.audio2face.status not in {"ERROR", "IDLE", "STOPPING"}:
            self._set_status(scene, "MODEL_READY", "PCM stream ended; model remains ready")

    def _fail_stream(
        self,
        scene: bpy.types.Scene,
        operation_id: str,
        message: str,
        *,
        cancel_worker: bool,
    ) -> None:
        if self.stream_source_cancel is not None:
            self.stream_source_cancel.set()
        if cancel_worker and scene.audio2face.stream_operation_id == operation_id:
            try:
                self._request(
                    scene,
                    "cancel",
                    {"operation_id": operation_id},
                    continuation=None,
                    model_signature=None,
                    operation_id=operation_id,
                )
            except SidecarError:
                pass
        get_live_stream_controller().stop(reset=False)
        self._clear_stream_state(scene, operation_id=operation_id)
        self._set_status(scene, "ERROR", message)

    def cancel(self, scene: bpy.types.Scene) -> None:
        self._require_editable_scene(scene)
        settings = scene.audio2face
        if not settings.result_operation_id:
            raise SidecarError("there is no active generation operation")
        self._request(
            scene,
            "cancel",
            {"operation_id": settings.result_operation_id},
            continuation=None,
            model_signature=None,
            operation_id=None,
        )
        self._set_status(scene, "CANCELLING", "Cancellation requested")

    def stop(self, scene: bpy.types.Scene) -> None:
        self._require_editable_scene(scene)
        if self.stream_source_cancel is not None:
            self.stream_source_cancel.set()
        get_live_stream_controller().stop(
            reset=bool(scene.audio2face.stream_reset_on_stop)
        )
        if self.client.state in {Lifecycle.STOPPED, Lifecycle.FAILED}:
            self._clear_model_state()
            self._clear_stream_state(
                scene,
                operation_id=scene.audio2face.stream_operation_id,
            )
            self._set_status(scene, "IDLE", "Worker is already stopped")
            return
        if self.client.state == Lifecycle.STOPPING:
            self._set_status(scene, "STOPPING", "Worker shutdown is already in progress")
            return
        request_id = self.client.begin_shutdown(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        self.expected_worker_exit = True
        if request_id:
            with self.pending_lock:
                self.pending[request_id] = PendingRequest(
                    "shutdown",
                    scene.name,
                    continuation=None,
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
        if self.stream_source_cancel is not None:
            self.stream_source_cancel.set()
        get_live_stream_controller().stop(reset=False)
        for scene in scenes:
            settings = scene.audio2face
            operation_id = settings.result_operation_id
            if self.generation_scene_names.get(operation_id) == scene.name:
                settings.result_operation_id = ""
                settings.progress = 0.0
            settings.stream_operation_id = ""
            settings.stream_sample_rate = 0
            settings.stream_prebuffer_samples = 0
            self._set_status(scene, "ERROR", message)
        self.generation_scene_names.clear()
        self.stream_audio_paths.clear()
        self.stream_scene_names.clear()
        self.stream_stop_requests.clear()
        self.stream_source_thread = None
        self.stream_source_cancel = None
        self.stream_playback_started = None
        with self.pending_lock:
            self.pending.clear()
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
        if settings.status == "STOPPING" and pending.method != "shutdown":
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
            # No identity index is trustworthy until this worker has described
            # the current model. Bootstrap identity zero, then let the validated
            # schema populate the selector used by later model reloads.
            settings.identity_index = 0
            self.handshake_spec = None
            self._submit_model_load(
                scene,
                spec,
                identity_index=0,
                continuation="ready",
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
            if pending.model_signature is None:
                raise RuntimeError(
                    "load_model pending state has no exact model signature"
                )
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
            self.schema_scenes.add(int(scene.as_pointer()))
            self.model_sample_rate = sample_rate
            self._set_status(
                scene,
                "MODEL_READY",
                "Loaded Audio2Face 3.0 and Audio2Emotion 3.0 models",
            )
            if pending.continuation == "generate":
                self.generate(scene)
            elif pending.continuation == "stream_wav":
                self.start_wav_stream(scene)
        elif pending.method == "generate":
            if result:
                self._reject_worker_contract(
                    "worker returned a noncanonical generate response",
                )
            else:
                self._set_status(scene, "GENERATING", "Worker accepted generation operation")
        elif pending.method == "cancel":
            if result:
                if pending.operation_id is not None:
                    self._reject_worker_contract(
                        "worker returned an invalid stream-cancel response",
                    )
                else:
                    self._reject_worker_contract(
                        "worker returned an invalid cancel response",
                    )
                return
            if pending.operation_id is not None:
                if settings.stream_operation_id != pending.operation_id:
                    return
                self._set_status(
                    scene,
                    "STREAM_ENDING",
                    "Worker accepted stream stop",
                )
                return
            if settings.status in {"COMPLETED", "ERROR"} or (
                not settings.result_operation_id and not settings.stream_operation_id
            ):
                return
            self._set_status(scene, "CANCELLING", "Worker accepted cancellation")
        elif pending.method == "stream_start":
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
            settings.stream_prebuffer_samples = prebuffer_samples
            if settings.stream_operation_id in self.stream_audio_paths:
                self._start_wav_stream_source(
                    scene,
                    settings.stream_operation_id,
                    prebuffer_samples,
                )
        elif pending.method == "stream_chunk":
            if result:
                self._reject_worker_contract(
                    "worker returned an invalid stream-chunk response",
                )
        elif pending.method == "stream_end":
            if result:
                self._reject_worker_contract(
                    "worker returned an invalid stream-end response",
                )
            else:
                self._set_status(scene, "STREAM_ENDING", "Worker is draining final frames")
        elif pending.method == "shutdown":
            if result:
                self._reject_worker_contract(
                    "worker returned an invalid shutdown response"
                )
            elif self.rejected_reason is None:
                self._set_status(scene, "STOPPING", "Worker is exiting")

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
        if pending.method == "generate":
            operation_id = scene.audio2face.result_operation_id
            if (
                operation_id in self.generation_scene_names
                and self.generation_scene_names[operation_id] == scene.name
            ):
                self._release_generation_operation(
                    scene,
                    operation_id,
                    keep_result=False,
                )
        if pending.operation_id is not None:
            if scene.audio2face.stream_operation_id != pending.operation_id:
                return
            self._fail_stream(
                scene,
                pending.operation_id,
                message,
                cancel_worker=pending.method != "cancel",
            )
            return
        if pending.method == "cancel" and (
            scene.audio2face.status in {"COMPLETED", "ERROR"}
            or (
                not scene.audio2face.result_operation_id
                and not scene.audio2face.stream_operation_id
            )
        ):
            return
        if (
            pending.method == "cancel"
            and scene.audio2face.status
            in {"LOADING_MODEL", "GENERATING", "CANCELLING", "STOPPING"}
        ):
            # Preserve the active operation while surfacing a cancel diagnostic.
            scene.audio2face.status_message = message
            return
        self._set_status(scene, "ERROR", message)

    def _handle_event(self, envelope: dict[str, Any]) -> None:
        event = envelope["event"]
        data = envelope["data"]
        operation_id = envelope["operation_id"]
        has_generation = operation_id in self.generation_scene_names
        has_stream = operation_id in self.stream_scene_names
        if has_generation and has_stream:
            self._reject_worker_contract(
                "worker event operation ID belongs to both operation maps"
            )
            return
        if not has_generation and not has_stream:
            self._reject_worker_contract(
                "worker returned an event for an unknown operation ID"
            )
            return
        is_stream = has_stream
        scene_name = (
            self.stream_scene_names[operation_id]
            if is_stream
            else self.generation_scene_names[operation_id]
        )
        scene = self._scene(scene_name)
        if scene is None or not scene.is_editable:
            if is_stream:
                self._release_worker_stream_state(operation_id)
            else:
                self.generation_scene_names.pop(operation_id, None)
            return
        settings = scene.audio2face
        if is_stream:
            if settings.stream_operation_id != operation_id:
                self._release_worker_stream_state(operation_id)
                self._reject_worker_contract(
                    "worker stream event does not match its scene operation"
                )
                return
        elif settings.result_operation_id != operation_id:
            self.generation_scene_names.pop(operation_id, None)
            self._reject_worker_contract(
                "worker generation event does not match its scene operation"
            )
            return
        if event in {"stream_frame", "stream_ended"} and not is_stream:
            self._reject_worker_contract(
                "worker routed a stream event to a generation operation",
            )
            return
        if event in {"progress", "result", "canceled"} and is_stream:
            self._reject_worker_contract(
                "worker routed a generation event to a PCM stream",
            )
            return

        if event == "stream_frame":
            if set(data) != {"timestamp_sample", "weights"}:
                self._reject_worker_contract(
                    "worker returned invalid stream-frame data",
                )
                return
            try:
                get_live_stream_controller().receive(
                    operation_id,
                    data["timestamp_sample"],
                    data["weights"],
                )
            except (LiveStreamError, TypeError, ValueError) as exc:
                self._reject_worker_contract(str(exc))
                return
            if settings.status == "STREAM_ENDING":
                settings.status_message = "Draining final streamed ARKit-52 frames"
            else:
                self._set_status(
                    scene,
                    "STREAMING",
                    "Streaming ARKit-52 Shape Key values",
                )
        elif event == "stream_ended":
            if data:
                self._reject_worker_contract(
                    "stream-ended event data must be empty",
                )
                return
            explicit_stop = operation_id in self.stream_stop_requests
            if explicit_stop:
                get_live_stream_controller().stop(
                    reset=bool(scene.audio2face.stream_reset_on_stop)
                )
                self._clear_stream_state(scene, operation_id=operation_id)
                self._set_status(
                    scene,
                    "MODEL_READY",
                    "PCM stream stopped; model remains ready",
                )
            else:
                self._release_worker_stream_state(operation_id)
                get_live_stream_controller().mark_terminal(operation_id)
                if get_live_stream_controller().active:
                    self._set_status(
                        scene,
                        "STREAMING",
                        "Finishing buffered streamed audio and ARKit-52 values",
                    )
                else:
                    self._clear_stream_state(scene, operation_id=operation_id)
                    self._set_status(
                        scene,
                        "MODEL_READY",
                        "PCM stream ended; model remains ready",
                    )
        elif event == "progress":
            if set(data) != {"progress", "stage"}:
                self._reject_worker_contract(
                    "worker returned invalid progress data",
                )
                return
            progress = data["progress"]
            stage = data["stage"]
            if (
                type(progress) is not float
                or not math.isfinite(progress)
                or progress < 0.0
                or progress > 1.0
                or not isinstance(stage, str)
                or not stage
            ):
                self._reject_worker_contract(
                    "worker returned invalid progress data",
                )
                return
            settings.progress = progress
            self._set_status(scene, "GENERATING", stage)
        elif event == "result":
            if data:
                self._reject_worker_contract(
                    "result event data must be empty",
                )
                return
            path = self.result_directory() / f"{operation_id}.a2f.json"
            if not path.is_file():
                self._reject_worker_contract(
                    f"worker result file is missing: {path}",
                )
                return
            self._release_generation_operation(
                scene,
                operation_id,
                keep_result=True,
            )
            settings.result_path = str(path)
            settings.progress = 1.0
            self._set_status(scene, "COMPLETED", f"Result ready: {path.name}")
        elif event == "canceled":
            if data:
                self._reject_worker_contract(
                    "worker returned invalid cancellation data",
                )
                return
            self._release_generation_operation(
                scene,
                operation_id,
                keep_result=False,
            )
            settings.progress = 0.0
            self._set_status(scene, "MODEL_READY", "Generation canceled")
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
            message = f"{code}: {worker_message}"
            if is_stream:
                self._fail_stream(
                    scene,
                    operation_id,
                    message,
                    cancel_worker=False,
                )
            else:
                self._release_generation_operation(
                    scene,
                    operation_id,
                    keep_result=False,
                )
                self._set_status(scene, "ERROR", message)

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
        if self.reset_scene_state_on_poll:
            for scene in self._editable_scenes():
                settings = scene.audio2face
                settings.status = "IDLE"
                settings.status_message = "Worker is stopped"
                settings.progress = 0.0
                settings.preview_state = "IDLE"
                settings.preview_time = 0.0
                settings.preview_duration = 0.0
                settings.stream_operation_id = ""
                settings.stream_sample_rate = 0
                settings.stream_prebuffer_samples = 0
                settings.stream_time = 0.0
            self.reset_scene_state_on_poll = False
        self._poll_optimization_events()
        self._poll_stream_source_events()
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
                if self.stream_source_cancel is not None:
                    self.stream_source_cancel.set()
                live_controller = get_live_stream_controller()
                live_controller.stop(reset=live_controller.reset_on_stop)
                for scene in self._editable_scenes():
                    settings = scene.audio2face
                    operation_id = settings.result_operation_id
                    if (
                        operation_id in self.generation_scene_names
                        and self.generation_scene_names[operation_id] == scene.name
                    ):
                        self._release_generation_operation(
                            scene,
                            operation_id,
                            keep_result=False,
                        )
                    settings.stream_operation_id = ""
                    settings.stream_sample_rate = 0
                    settings.stream_prebuffer_samples = 0
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
                self.generation_scene_names.clear()
                self.stream_audio_paths.clear()
                self.stream_scene_names.clear()
                self.stream_stop_requests.clear()
                self.stream_source_thread = None
                self.stream_source_cancel = None
                self.stream_playback_started = None
                self.expected_worker_exit = False

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

    def close(self) -> None:
        if self.optimization_cancel is not None:
            with self.optimization_commit_lock:
                self.optimization_cancel.set()
        if self.optimization_thread is not None and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        if self.stream_source_cancel is not None:
            self.stream_source_cancel.set()
        if self.stream_source_thread is not None and self.stream_source_thread.is_alive():
            self.stream_source_thread.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        unregister_live_stream()
        self.client.close(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        with self.pending_lock:
            self.pending.clear()
        self.negotiated = False
        self.handshake_deadline = None
        self.handshake_spec = None
        self._clear_model_state()
        self.optimization_thread = None
        self.optimization_cancel = None
        self.optimization_progress = 0.0
        self.stream_source_thread = None
        self.stream_source_cancel = None
        self.generation_scene_names.clear()
        self.stream_playback_started = None
        self.stream_audio_paths.clear()
        self.stream_scene_names.clear()
        self.stream_stop_requests.clear()
        with self.optimization_progress_lock:
            self.optimization_latest_progress = None


_CONTROLLER: RuntimeController | None = None


def get_controller() -> RuntimeController:
    global _CONTROLLER
    if _CONTROLLER is None:
        _CONTROLLER = RuntimeController()
    return _CONTROLLER


def _timer_callback() -> float | None:
    if _CONTROLLER is None:
        return None
    try:
        _CONTROLLER.poll()
        preview_active = get_preview_controller().tick()
        stream_active = get_live_stream_controller().tick()
    except Exception as exc:  # Keep timer alive, but surface the main-thread failure.
        preview_active = False
        stream_active = False
        scene = bpy.context.scene
        if scene is not None and hasattr(scene, "audio2face"):
            RuntimeController._set_status(scene, "ERROR", str(exc))
    return (
        PREVIEW_INTERVAL_SECONDS
        if preview_active or stream_active
        else POLL_INTERVAL_SECONDS
    )


def _dispose_runtime_state() -> None:
    """Close the controller before removing the singletons it owns."""

    global _CONTROLLER
    controller = _CONTROLLER
    _CONTROLLER = None
    try:
        if controller is not None:
            controller.close()
        else:
            unregister_live_stream()
    finally:
        unregister_preview()


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
