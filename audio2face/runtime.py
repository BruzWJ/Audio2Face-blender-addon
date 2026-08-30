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
from typing import Any, Iterator, cast

import bpy

from .animation_bake import (
    AnimationBakeError,
    BakeTarget,
    bake_shape_key_actions,
    plan_bake_targets,
)
from .live_stream import (
    LiveStreamError,
    apply_model_frame,
    get_live_stream_controller,
    unregister_live_stream,
    validate_stream_frame,
)
from .frame_stream import sample_linear
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
from .selected_audio_timeline import (
    configure_selected_audio,
    frame_to_audio_sample,
    selected_audio_frame_span,
)
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


def _model_emotion_channels(model_schema: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        descriptor["name"] for descriptor in model_schema["emotion_channels"]
    )


@dataclass(frozen=True, slots=True)
class SettingsTimeline:
    """Compact worker payload and cumulative settings at each change sample."""

    payload: tuple[dict[str, object], ...]
    snapshots: tuple[tuple[int, dict[str, object]], ...]


def _object_patch(
    previous: dict[str, object],
    current: dict[str, object],
) -> dict[str, object]:
    """Return the recursive changed-leaf patch from one settings object to another."""

    patch: dict[str, object] = {}
    for name, value in current.items():
        old_value = previous[name]
        if isinstance(old_value, dict) and isinstance(value, dict):
            nested = _object_patch(old_value, value)
            if nested:
                patch[name] = nested
        elif value != old_value:
            patch[name] = value
    return patch


def _settings_timeline(
    snapshots: list[tuple[int, dict[str, object]]],
) -> SettingsTimeline:
    """Encode full per-frame snapshots as one full object followed by patches."""

    unique_samples: list[tuple[int, dict[str, object]]] = []
    for sample, settings in snapshots:
        if unique_samples and unique_samples[-1][0] == sample:
            unique_samples[-1] = (sample, settings)
        else:
            unique_samples.append((sample, settings))

    first_settings = unique_samples[0][1]
    payload: list[dict[str, object]] = [
        {"sample": 0, "settings": first_settings}
    ]
    cumulative = [(0, first_settings)]
    previous = first_settings
    for sample, settings in unique_samples[1:]:
        patch = _object_patch(previous, settings)
        if patch:
            payload.append({"sample": sample, "settings": patch})
            cumulative.append((sample, settings))
            previous = settings
    return SettingsTimeline(tuple(payload), tuple(cumulative))


def _settings_at_sample(
    timeline: SettingsTimeline,
    sample: int,
) -> dict[str, object]:
    snapshots = timeline.snapshots
    low = 0
    high = len(snapshots)
    while low < high:
        middle = (low + high) // 2
        if snapshots[middle][0] <= sample:
            low = middle + 1
        else:
            high = middle
    return snapshots[low - 1][1]


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
        operation_methods = {
            "stream_start",
            "stream_chunk",
            "stream_settings",
            "stream_end",
            "track_start",
            "track_chunk",
            "track_prepare",
            "track_render",
            "cancel",
        }
        if self.method in operation_methods and self.operation_id is None:
            raise ValueError(f"{self.method} pending state requires an operation ID")
        if self.method not in operation_methods and self.operation_id is not None:
            raise ValueError(f"{self.method} pending state cannot carry an operation ID")


@dataclass(slots=True)
class PCMIngress:
    """Thread-safe pending input whose first chunk starts one live stream."""

    scene_name: str
    chunks: deque[bytes]
    ending: bool = False


@dataclass(slots=True)
class ActiveStream:
    """One sequential external-PCM operation."""

    operation_id: str
    scene_name: str
    submitted_settings: dict[str, object]
    chunk_credit: threading.Event = field(default_factory=threading.Event)
    prebuffer_samples: int | None = None
    end_sent: bool = False
    stop_requested: bool = False
    worker_ended: bool = False


@dataclass(slots=True)
class TrackRenderStage:
    """One revision assembled from bounded worker frame batches."""

    revision: int
    total_frames: int | None = None
    timestamps: list[int] = field(default_factory=list)
    weights: list[tuple[float, ...]] = field(default_factory=list)
    effective_emotions: list[tuple[float, ...]] = field(default_factory=list)


@dataclass(slots=True)
class SelectedTrack:
    """One retained Selected WAV source and its prepared render cache."""

    operation_id: str
    scene_name: str
    path: Path
    wav_source: WavStreamSource
    chunks: Iterator[bytes]
    uploaded_samples: int = 0
    prepared: bool = False
    cancel_requested: bool = False
    restart_after_cancel: bool = False
    render_error: str | None = None
    render_revision: int = 0
    render_timeline: SettingsTimeline | None = None
    published_timeline: SettingsTimeline | None = None
    stage: TrackRenderStage | None = None
    timestamps: tuple[int, ...] = ()
    weights: tuple[tuple[float, ...], ...] = ()
    effective_emotions: tuple[tuple[float, ...], ...] = ()


@dataclass(slots=True)
class ActiveBake:
    """One Blender-side bake over the prepared Selected track."""

    scene_name: str
    frame_start: int
    frame_end: int
    targets: tuple[BakeTarget, ...]
    settings_timeline: SettingsTimeline
    frame_samples: tuple[int, ...]


POLL_INTERVAL_SECONDS = 0.10
PRESENTATION_INTERVAL_SECONDS = 1.0 / 60.0
_STREAM_ENDING_MESSAGE = "Draining final Audio2Face frames"
_STREAM_STATUSES = frozenset({"STREAM_STARTING", "STREAMING", "STREAM_ENDING"})
SHUTDOWN_TIMEOUT_SECONDS = 2.0
HANDSHAKE_TIMEOUT_SECONDS = 15.0
MAX_STREAM_CHUNK_BYTES = 256 * 1024
MAX_PENDING_STREAM_CHUNKS = 64


def _native_playback_active() -> bool:
    return any(
        window.screen.is_animation_playing
        for manager in bpy.data.window_managers
        for window in manager.windows
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
        self.active_stream: ActiveStream | None = None
        self.selected_track: SelectedTrack | None = None
        self.active_bake: ActiveBake | None = None
        self.pcm_ingress: PCMIngress | None = None
        self.evaluating_settings_timeline = False
        self.invalidated_selected_scene: str | None = None

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
        changed = settings.status != status or settings.status_message != message
        settings.status = status
        settings.status_message = message
        if changed:
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
    def log_directory() -> Path:
        description = "Audio2Face logs directory"
        try:
            value = bpy.utils.extension_path_user(
                __package__,
                path="logs",
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
        if not preferences.nvidia_terms_accepted:
            return False, "accept the NVIDIA terms first"
        return True, "The bundled GPU runtime and selected model inputs are ready"

    @property
    def optimization_in_progress(self) -> bool:
        return self.optimization_thread is not None

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
            self.optimization_failed = True
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
        self._release_active_stream()
        self._release_selected_track()
        self._release_active_bake()
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
    ) -> None:
        with self.pending_lock:
            # Store correlation state before the main-thread poller can consume
            # a fast worker response.  This also covers calls made by an audio
            # source thread.
            self._request_locked(
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
    ) -> None:
        """Submit and correlate one request while ``pending_lock`` is held."""

        request_id = self.client.request(method, params)
        self.pending[request_id] = PendingRequest(
            method=method,
            scene_name=scene_name,
            model_signature=model_signature,
            operation_id=operation_id,
        )

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
        get_live_stream_controller().stop(reset=False, notify=False)
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
        if scene.audio2face.status == "STARTING":
            self._set_status(
                scene,
                "LOADING_MODEL",
                "Loading Audio2Face 3.0 and Audio2Emotion 3.0 models",
            )

    def _submit_stream_start(
        self,
        scene: bpy.types.Scene,
    ) -> None:
        """Start one external PCM stream."""

        operation_id, sample_rate, model_schema = self._stream_start_metadata(scene)
        get_live_stream_controller().prepare_external(
            scene,
            operation_id,
            sample_rate,
            tuple(model_schema["channels"]),
            _model_emotion_channels(model_schema),
            lambda error: self._finish_stream_presentation(
                scene.name,
                operation_id,
                error,
            ),
        )
        scene.audio2face.stream_time = 0.0
        settings = inference_settings(scene.audio2face)
        self._activate_stream(
            scene,
            ActiveStream(
                operation_id=operation_id,
                scene_name=scene.name,
                submitted_settings=settings,
            ),
            sample_rate,
        )

    def _stream_start_metadata(
        self,
        scene: bpy.types.Scene,
    ) -> tuple[str, int, dict[str, Any]]:
        sample_rate = self.model_sample_rate
        if sample_rate is None:
            raise SidecarError("worker model did not report its sampling rate")
        model_schema = self.model_schema
        if model_schema is None:
            raise SidecarError("worker model did not report its output channels")
        self._ensure_scene_model_schema(scene)
        return uuid.uuid4().hex, sample_rate, model_schema

    def _activate_stream(
        self,
        scene: bpy.types.Scene,
        stream: ActiveStream,
        sample_rate: int,
    ) -> None:
        try:
            self._request(
                scene,
                "stream_start",
                {
                    "operation_id": stream.operation_id,
                    "sample_rate": sample_rate,
                    "settings": stream.submitted_settings,
                },
                model_signature=None,
                operation_id=stream.operation_id,
            )
        except Exception:
            get_live_stream_controller().stop(reset=False, notify=False)
            raise
        stream.chunk_credit.set()
        self.active_stream = stream
        self._set_status(scene, "STREAM_STARTING", "Preparing audio inference")

    def _ensure_selected_track(self, scene: bpy.types.Scene) -> None:
        """Upload the configured WAV into one retained prepared track."""

        settings = scene.audio2face
        if (
            settings.input_mode != "SELECTED"
            or not settings.audio_path
            or self.client.state != Lifecycle.RUNNING
            or not self.negotiated
            or self.loaded_signature is None
        ):
            return
        audio_path = self._selected_path(
            bpy.path.abspath(settings.audio_path),
            "selected WAV file",
        )
        track = self.selected_track
        if track is not None:
            if (
                track.scene_name == scene.name
                and track.path == audio_path
                and not track.cancel_requested
            ):
                return
            return
        if self.active_stream is not None:
            return
        sample_rate = self.model_sample_rate
        if sample_rate is None:
            raise SidecarError("worker model did not report its sampling rate")
        self._ensure_scene_model_schema(scene)
        wav_source = WavStreamSource(
            audio_path,
            output_sample_rate=sample_rate,
            chunk_frames=MAX_STREAM_CHUNK_BYTES // 4,
        )
        operation_id = uuid.uuid4().hex
        track = SelectedTrack(
            operation_id=operation_id,
            scene_name=scene.name,
            path=audio_path,
            wav_source=wav_source,
            chunks=iter(wav_source),
        )
        self.selected_track = track
        try:
            self._request(
                scene,
                "track_start",
                {"operation_id": operation_id, "sample_rate": sample_rate},
                model_signature=None,
                operation_id=operation_id,
            )
        except Exception:
            self._release_selected_track(operation_id)
            raise
        self._set_status(scene, "TRACK_UPLOADING", "Uploading the selected WAV")

    def _send_next_track_chunk(
        self,
        scene: bpy.types.Scene,
        track: SelectedTrack,
    ) -> None:
        if track.cancel_requested:
            return
        try:
            payload = next(track.chunks)
        except StopIteration:
            if track.uploaded_samples != track.wav_source.metadata.output_frames:
                raise SidecarError("selected WAV upload ended at an unexpected sample")
            self._request(
                scene,
                "track_prepare",
                {"operation_id": track.operation_id},
                model_signature=None,
                operation_id=track.operation_id,
            )
            if scene.audio2face.status == "TRACK_UPLOADING":
                self._set_status(
                    scene,
                    "TRACK_PREPARING",
                    "Preparing continuous Audio2Face inference",
                )
            return
        self._request(
            scene,
            "track_chunk",
            {
                "operation_id": track.operation_id,
                "audio_f32le_base64": base64.b64encode(payload).decode("ascii"),
            },
            model_signature=None,
            operation_id=track.operation_id,
        )
        track.uploaded_samples += len(payload) // 4

    def _release_selected_track(self, operation_id: str | None = None) -> None:
        track = self.selected_track
        if track is None or (
            operation_id is not None and track.operation_id != operation_id
        ):
            return
        track.wav_source.close()
        self.selected_track = None
        self.invalidated_selected_scene = None
        with self.pending_lock:
            stale = tuple(
                request_id
                for request_id, pending in self.pending.items()
                if pending.operation_id == track.operation_id
                and pending.method != "cancel"
            )
            for request_id in stale:
                self.pending.pop(request_id, None)

    def _cancel_selected_track(
        self,
        track: SelectedTrack,
        *,
        restart: bool = False,
    ) -> None:
        if track.cancel_requested:
            track.restart_after_cancel = track.restart_after_cancel or restart
            return
        with self.pending_lock:
            self._request_locked(
                track.scene_name,
                "cancel",
                {"operation_id": track.operation_id},
                model_signature=None,
                operation_id=track.operation_id,
            )
            track.cancel_requested = True
            track.restart_after_cancel = restart

    def _complete_selected_track_cancel(
        self,
        scene: bpy.types.Scene | None,
        track: SelectedTrack,
    ) -> None:
        """Release a canceled source and start only an authored replacement."""

        restart = track.restart_after_cancel
        self._release_active_bake()
        self._release_selected_track(track.operation_id)
        if (
            scene is None
            or self.expected_worker_exit
            or self.client.state != Lifecycle.RUNNING
        ):
            return
        if restart:
            if scene.audio2face.status == "MODEL_READY":
                self._ensure_selected_track(scene)
                if self.selected_track is not None:
                    return
        if scene.audio2face.status != "ERROR":
            self._set_status(scene, "MODEL_READY", "Selected WAV unloaded")

    def _fail_selected_track(
        self,
        scene: bpy.types.Scene,
        track: SelectedTrack,
        message: str,
    ) -> None:
        """Terminate a failed source without retrying the same input."""

        self._release_active_bake()
        track.stage = None
        self._set_status(scene, "ERROR", message)
        try:
            self._cancel_selected_track(track)
        except (OSError, RuntimeError, ValueError):
            self._release_selected_track(track.operation_id)
        else:
            track.restart_after_cancel = False

    def selected_audio_changed(self, scene: bpy.types.Scene) -> None:
        """Replace the resident source without touching Blender transport."""

        track = self.selected_track
        if track is not None and track.scene_name == scene.name:
            self._release_active_bake()
            restart = bool(scene.audio2face.audio_path)
            self._cancel_selected_track(track, restart=restart)
            self._set_status(
                scene,
                "MODEL_READY",
                "Selected WAV replacement queued" if restart else "Selected WAV unloaded",
            )
            return
        self._ensure_selected_track(scene)

    def selected_audio_failed(self, scene: bpy.types.Scene, message: str) -> None:
        """Retire Selected inference after timeline or presentation failure."""

        track = self.selected_track
        if track is not None and track.scene_name == scene.name:
            self._fail_selected_track(scene, track, message)
        else:
            self._set_status(scene, "ERROR", message)

    def input_mode_changed(self, scene: bpy.types.Scene) -> None:
        """Switch source ownership without changing Blender transport."""

        stream = self.active_stream
        if (
            stream is not None
            and stream.scene_name == scene.name
            and not stream.stop_requested
        ):
            try:
                self._request_stream_cancel(stream)
            except (OSError, RuntimeError, ValueError) as exc:
                self._set_status(scene, "ERROR", str(exc))
                return
            get_live_stream_controller().stop(reset=False, notify=False)
            if scene.audio2face.status in _STREAM_STATUSES:
                self._set_status(scene, "STREAM_ENDING", _STREAM_ENDING_MESSAGE)
        track = self.selected_track
        if track is not None and track.scene_name == scene.name:
            self._release_active_bake()
            restart = scene.audio2face.input_mode == "SELECTED"
            try:
                self._cancel_selected_track(
                    track,
                    restart=restart,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                self._set_status(scene, "ERROR", str(exc))
                return
            self._set_status(
                scene,
                "MODEL_READY",
                "Selected WAV replacement queued" if restart else "Selected WAV unloaded",
            )
        with self.pending_lock:
            if (
                self.pcm_ingress is not None
                and self.pcm_ingress.scene_name == scene.name
            ):
                self.pcm_ingress = None
        if scene.audio2face.input_mode == "SELECTED":
            try:
                self._ensure_selected_track(scene)
            except (OSError, RuntimeError, ValueError) as exc:
                self._set_status(scene, "ERROR", str(exc))

    def _track_sample(
        self,
        scene: bpy.types.Scene,
        track: SelectedTrack,
        frame: int,
        prediction_delay: float | None = None,
    ) -> int | None:
        settings = scene.audio2face
        span = selected_audio_frame_span(scene)
        if span is None:
            return None
        frame_start, frame_end = span
        sample_rate = self.model_sample_rate
        if sample_rate is None:
            raise SidecarError("worker model sampling rate is unavailable")
        if not frame_start <= frame <= frame_end:
            return None
        return frame_to_audio_sample(
            frame,
            frame_start=frame_start,
            sample_rate=sample_rate,
            fps=scene.render.fps,
            fps_base=scene.render.fps_base,
            prediction_delay=(
                settings.prediction_delay
                if prediction_delay is None
                else prediction_delay
            ),
            audio_samples=track.wav_source.metadata.output_frames,
        )

    def _evaluate_settings_timeline(
        self,
        scene: bpy.types.Scene,
        track: SelectedTrack,
    ) -> tuple[SettingsTimeline, tuple[int, ...]]:
        """Evaluate inference settings and presentation samples over the sound span."""

        span = selected_audio_frame_span(scene)
        if span is None:
            raise SidecarError("selected WAV sound strip is unavailable")
        frame_start, frame_end = span
        original_frame = int(scene.frame_current)
        original_subframe = float(scene.frame_subframe)
        snapshots: list[tuple[int, dict[str, object]]] = []
        frame_samples: list[int] = []
        self.evaluating_settings_timeline = True
        try:
            for frame in range(frame_start, frame_end + 1):
                scene.frame_set(frame, subframe=0.0)
                sample = cast(
                    int,
                    self._track_sample(
                        scene,
                        track,
                        frame,
                        prediction_delay=0.0,
                    ),
                )
                snapshots.append((sample, inference_settings(scene.audio2face)))
                frame_samples.append(
                    cast(int, self._track_sample(scene, track, frame))
                )
        finally:
            try:
                scene.frame_set(original_frame, subframe=original_subframe)
            finally:
                self.evaluating_settings_timeline = False
        return _settings_timeline(snapshots), tuple(frame_samples)

    def _settings_timeline_matches_current_frame(
        self,
        scene: bpy.types.Scene,
        track: SelectedTrack,
        timeline: SettingsTimeline | None,
    ) -> bool:
        if timeline is None:
            return False
        sample = self._track_sample(
            scene,
            track,
            int(scene.frame_current),
            prediction_delay=0.0,
        )
        return sample is not None and _settings_at_sample(
            timeline, sample
        ) == inference_settings(scene.audio2face)

    def _apply_selected_cache(
        self,
        scene: bpy.types.Scene,
        track: SelectedTrack,
    ) -> None:
        model_schema = self.model_schema
        if model_schema is None:
            return
        target_sample = self._track_sample(scene, track, int(scene.frame_current))
        if target_sample is None:
            self._apply_neutral_selected_frame(scene)
            return
        if not track.timestamps:
            return
        apply_model_frame(
            scene.audio2face,
            tuple(model_schema["channels"]),
            _model_emotion_channels(model_schema),
            sample_linear(track.timestamps, track.weights, target_sample),
            sample_linear(
                track.timestamps,
                track.effective_emotions,
                target_sample,
            ),
        )

    def _apply_neutral_selected_frame(self, scene: bpy.types.Scene) -> None:
        model_schema = self.model_schema
        if model_schema is None:
            return
        channels = tuple(model_schema["channels"])
        emotion_channels = _model_emotion_channels(model_schema)
        apply_model_frame(
            scene.audio2face,
            channels,
            emotion_channels,
            (0.0,) * len(channels),
            (0.0,) * len(emotion_channels),
        )

    def request_selected_frame(self, scene: bpy.types.Scene) -> None:
        """Apply the prepared row selected by Blender's current native frame."""

        if (
            not scene.is_editable
            or not hasattr(scene, "audio2face")
            or scene.audio2face.input_mode != "SELECTED"
            or not scene.audio2face.audio_path
        ):
            return
        track = self.selected_track
        if (
            track is None
            or track.scene_name != scene.name
            or not track.prepared
            or track.cancel_requested
        ):
            return
        try:
            self._apply_selected_cache(scene, track)
        except (LiveStreamError, RuntimeError, ValueError) as exc:
            self._fail_selected_track(scene, track, str(exc))

    def _request_track_render(
        self,
        scene: bpy.types.Scene,
        track: SelectedTrack,
        settings_timeline: SettingsTimeline,
    ) -> None:
        if track.cancel_requested or not track.prepared:
            return
        if track.stage is not None and track.render_timeline == settings_timeline:
            return
        if track.published_timeline == settings_timeline and track.stage is None:
            render_error = track.render_error
            track.render_error = None
            if render_error is not None:
                if (
                    scene.audio2face.status == "ERROR"
                    and scene.audio2face.status_message == render_error
                ):
                    self._set_status(scene, "MODEL_READY", "Selected WAV is ready")
            self.request_selected_frame(scene)
            return
        revision = track.render_revision + 1
        previous = (
            track.render_revision,
            track.render_timeline,
            track.stage,
        )
        track.render_revision = revision
        track.render_timeline = settings_timeline
        track.stage = TrackRenderStage(revision)
        try:
            self._request(
                scene,
                "track_render",
                {
                    "operation_id": track.operation_id,
                    "revision": revision,
                    "settings_timeline": list(settings_timeline.payload),
                    "preview_sample": self._track_sample(
                        scene,
                        track,
                        int(scene.frame_current),
                    ),
                },
                model_signature=None,
                operation_id=track.operation_id,
            )
        except Exception:
            (
                track.render_revision,
                track.render_timeline,
                track.stage,
            ) = previous
            raise

    def bake_selected_audio(self, scene: bpy.types.Scene) -> None:
        """Write the coherent Selected preview cache as native Shape Key curves."""

        self._require_editable_scene(scene)
        self._require_worker_ready()
        settings = scene.audio2face
        if settings.input_mode != "SELECTED":
            raise SidecarError("animation baking requires Selected Audio mode")
        audio_path = self._selected_path(
            bpy.path.abspath(settings.audio_path),
            "selected WAV file",
        )
        if not audio_path.is_file():
            raise SidecarError(f"audio file does not exist: {audio_path}")
        model_schema = self.model_schema
        if model_schema is None:
            raise SidecarError("worker model metadata is unavailable")
        self._ensure_scene_model_schema(scene)
        targets = tuple(
            item.object
            for item in scene.audio2face.target_objects
            if item.object is not None
        )
        target_plans = plan_bake_targets(tuple(model_schema["channels"]), targets)
        if not target_plans:
            raise AnimationBakeError(
                "none of the target objects has a Shape Key matching the model channels"
            )

        frame_start, frame_end = configure_selected_audio(
            scene,
            str(audio_path),
            first_frame=int(scene.audio2face.audio_first_frame),
        )
        track = self.selected_track
        if track is not None and (
            track.scene_name != scene.name
            or track.path != audio_path
            or track.cancel_requested
        ):
            raise SidecarError("wait for the selected WAV track to finish replacing")
        if track is None:
            self._ensure_selected_track(scene)
            track = self.selected_track
        if (
            track is None
            or track.scene_name != scene.name
            or track.path != audio_path
            or track.cancel_requested
        ):
            raise SidecarError("selected WAV track is not available")

        settings_timeline, frame_samples = self._evaluate_settings_timeline(
            scene, track
        )
        bake = ActiveBake(
            scene_name=scene.name,
            frame_start=frame_start,
            frame_end=frame_end,
            targets=target_plans,
            settings_timeline=settings_timeline,
            frame_samples=frame_samples,
        )
        self.active_bake = bake
        try:
            self._set_status(
                scene,
                "BAKING",
                "Rendering one continuous Audio2Face animation",
            )
            if (
                track.published_timeline == bake.settings_timeline
                and track.timestamps
            ):
                self._finish_bake(scene)
                return
            if track.prepared:
                self._request_track_render(scene, track, bake.settings_timeline)
        except Exception:
            self._release_active_bake()
            raise

    def cancel_bake(self, scene: bpy.types.Scene) -> None:
        """Cancel Action writing without touching worker or media state."""

        self._require_editable_scene(scene)
        bake = self.active_bake
        if bake is None or bake.scene_name != scene.name:
            raise SidecarError("there is no active bake for this scene")
        self._release_active_bake()
        if scene.audio2face.status == "BAKING":
            self._set_status(scene, "MODEL_READY", "Animation bake canceled")
        if self.invalidated_selected_scene == scene.name:
            self._refresh_invalidated_selected_settings()
        else:
            self.refresh_inference_settings(scene)

    def _release_active_bake(self) -> None:
        self.active_bake = None

    def _finish_bake(
        self,
        scene: bpy.types.Scene,
    ) -> None:
        bake = self.active_bake
        if bake is None:
            raise SidecarError("completed bake state is unavailable")
        track = self.selected_track
        if (
            track is None
            or not track.timestamps
            or track.published_timeline != bake.settings_timeline
        ):
            return
        frames = tuple(range(bake.frame_start, bake.frame_end + 1))
        sampled_weights = tuple(
            sample_linear(track.timestamps, track.weights, sample)
            for sample in bake.frame_samples
        )
        actions = bake_shape_key_actions(
            frames,
            sampled_weights,
            bake.targets,
            bpy.data.actions,
        )
        self._release_active_bake()
        if scene.audio2face.status == "BAKING":
            self._set_status(
                scene,
                "MODEL_READY",
                f"Baked {len(frames)} Blender frames to {len(actions)} Shape Key Action"
                f"{'s' if len(actions) != 1 else ''}",
            )
        self._refresh_invalidated_selected_settings()

    def _fail_bake(
        self,
        scene: bpy.types.Scene,
        message: str,
    ) -> None:
        if self.active_bake is None:
            return
        self._release_active_bake()
        if scene.audio2face.status == "BAKING":
            self._set_status(scene, "ERROR", message)
        self._refresh_invalidated_selected_settings()

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
    ) -> None:
        """Send one validated mono-f32le chunk to an accepted worker stream."""

        stream = self.active_stream
        if stream is None or stream.operation_id != operation_id:
            raise SidecarError("the requested PCM stream is not active")
        with self.pending_lock:
            if not stream.chunk_credit.is_set():
                raise SidecarError("wait for the worker's next PCM chunk credit")
            stream.chunk_credit.clear()
            try:
                self._request_locked(
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
    ) -> None:
        stream = self.active_stream
        if stream is None or stream.operation_id != operation_id:
            raise SidecarError("the requested PCM stream is not active")
        with self.pending_lock:
            if stream.end_sent:
                raise SidecarError("the active PCM stream has already ended")
            self._request_locked(
                stream.scene_name,
                "stream_end",
                {"operation_id": operation_id},
                model_signature=None,
                operation_id=operation_id,
            )
            stream.end_sent = True

    def _request_stream_cancel(
        self,
        stream: ActiveStream,
    ) -> None:
        """Cancel one media stream without unloading its model."""

        with self.pending_lock:
            self._request_locked(
                stream.scene_name,
                "cancel",
                {"operation_id": stream.operation_id},
                model_signature=None,
                operation_id=stream.operation_id,
            )
            stream.stop_requested = True

    def _cancel_orphaned_operation(self) -> None:
        """Cancel inference whose owning Blender scene is unavailable."""

        track = self.selected_track
        if track is not None:
            scene = self._scene(track.scene_name)
            if scene is not None and scene.is_editable:
                return
            self._release_active_bake()
            if not track.cancel_requested:
                try:
                    self._cancel_selected_track(track)
                except (OSError, RuntimeError, ValueError) as exc:
                    self._release_selected_track(track.operation_id)
                    self._reject_worker_contract(
                        f"could not cancel a track whose scene is unavailable: {exc}"
                    )
            return

        bake = self.active_bake
        if bake is not None:
            scene = self._scene(bake.scene_name)
            if scene is not None and scene.is_editable:
                return
            self._release_active_bake()
            return

        stream = self.active_stream
        if stream is None:
            return
        scene = self._scene(stream.scene_name)
        if scene is not None and scene.is_editable:
            return
        if stream.worker_ended:
            live = get_live_stream_controller()
            if live.active:
                live.stop(reset=False, notify=False)
            self._release_active_stream(stream.operation_id)
            return
        if stream.stop_requested:
            return
        live = get_live_stream_controller()
        if live.active:
            live.stop(reset=False, notify=False)
        self._clear_pcm_ingress()
        try:
            self._request_stream_cancel(stream)
        except (OSError, RuntimeError, ValueError) as exc:
            self._release_active_stream(stream.operation_id)
            self._reject_worker_contract(
                f"could not cancel a stream whose scene is unavailable: {exc}"
            )

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
                track = self.selected_track
                if track is not None:
                    self._cancel_selected_track(track)
                    return
                self._submit_stream_start(scene)
            except (OSError, RuntimeError, ValueError) as exc:
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
                if settings.status in {"STREAM_STARTING", "STREAMING"}:
                    self._set_status(
                        scene,
                        "STREAM_ENDING",
                        _STREAM_ENDING_MESSAGE,
                    )

    def _release_active_stream(self, operation_id: str | None = None) -> None:
        stream = self.active_stream
        if stream is None or (
            operation_id is not None and stream.operation_id != operation_id
        ):
            return
        self._clear_pcm_ingress()
        self.active_stream = None
        with self.pending_lock:
            stale = tuple(
                request_id
                for request_id, pending in self.pending.items()
                if pending.operation_id == stream.operation_id
            )
            for request_id in stale:
                self.pending.pop(request_id, None)

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
        error: str | None,
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
        if error is not None:
            self._fail_stream(
                scene,
                operation_id,
                error,
                cancel_worker=not stream.worker_ended,
            )
            return
        self._release_active_stream(operation_id)
        if (
            not self.expected_worker_exit
            and self.client.state == Lifecycle.RUNNING
            and scene.audio2face.status in _STREAM_STATUSES
        ):
            self._set_status(scene, "MODEL_READY", "PCM stream ended; model remains ready")
            try:
                self._ensure_selected_track(scene)
            except (OSError, RuntimeError, ValueError) as exc:
                self._set_status(scene, "ERROR", str(exc))

    def refresh_inference_settings(
        self,
        scene: bpy.types.Scene,
        *,
        rebuild_selected: bool = False,
    ) -> None:
        """Apply settings to the resident source independently of transport."""

        if not scene.is_editable or self.expected_worker_exit:
            return
        if scene.audio2face.input_mode == "SELECTED" and (
            self.active_bake is not None or _native_playback_active()
        ):
            self.invalidate_selected_settings(scene)
            return
        if scene.audio2face.input_mode == "STREAM":
            stream = self.active_stream
            if (
                stream is None
                or stream.scene_name != scene.name
                or stream.end_sent
                or stream.stop_requested
                or stream.worker_ended
            ):
                return
            settings = inference_settings(scene.audio2face)
            if settings == stream.submitted_settings:
                return
            try:
                self._request(
                    scene,
                    "stream_settings",
                    {
                        "operation_id": stream.operation_id,
                        "settings": settings,
                    },
                    model_signature=None,
                    operation_id=stream.operation_id,
                )
                stream.submitted_settings = settings
            except (OSError, RuntimeError, ValueError) as exc:
                self._fail_stream(
                    scene,
                    stream.operation_id,
                    str(exc),
                    cancel_worker=True,
                )
            return
        if scene.audio2face.input_mode != "SELECTED":
            return
        if rebuild_selected and self.invalidated_selected_scene == scene.name:
            self.invalidated_selected_scene = None
        track = self.selected_track
        if (
            track is None
            or track.scene_name != scene.name
            or not track.prepared
            or track.cancel_requested
        ):
            return
        try:
            render_timeline = track.render_timeline
            if (
                not rebuild_selected
                and render_timeline is not None
                and self._settings_timeline_matches_current_frame(
                    scene, track, render_timeline
                )
            ):
                if (
                    track.render_error is not None
                    and track.stage is None
                    and track.published_timeline == render_timeline
                ):
                    self._request_track_render(scene, track, render_timeline)
                return
            self._request_track_render(
                scene,
                track,
                self._evaluate_settings_timeline(scene, track)[0],
            )
        except (OSError, RuntimeError, ValueError) as exc:
            track.render_error = str(exc)
            self._set_status(scene, "ERROR", track.render_error)

    def refresh_frame_inference_settings(self, scene: bpy.types.Scene) -> None:
        """Forward settings evaluated by Blender for the current native frame."""

        if scene.audio2face.input_mode == "SELECTED":
            span = selected_audio_frame_span(scene)
            if span is None or not span[0] <= int(scene.frame_current) <= span[1]:
                return
            if _native_playback_active():
                track = self.selected_track
                if (
                    track is not None
                    and track.scene_name == scene.name
                    and not self._settings_timeline_matches_current_frame(
                        scene, track, track.render_timeline
                    )
                ):
                    self.invalidate_selected_settings(scene)
                return
        self.refresh_inference_settings(scene)

    def invalidate_selected_settings(self, scene: bpy.types.Scene) -> None:
        """Queue one full settings rescan after Blender edits the scene Action."""

        track = self.selected_track
        if (
            scene.is_editable
            and scene.audio2face.input_mode == "SELECTED"
            and track is not None
            and track.scene_name == scene.name
            and not track.cancel_requested
        ):
            self.invalidated_selected_scene = scene.name

    def _refresh_invalidated_selected_settings(self) -> None:
        scene_name = self.invalidated_selected_scene
        if scene_name is None:
            return
        scene = self._scene(scene_name)
        if scene is None:
            self.invalidated_selected_scene = None
            return
        track = self.selected_track
        if (
            self.active_bake is not None
            or _native_playback_active()
            or track is None
            or track.scene_name != scene_name
            or not track.prepared
            or track.cancel_requested
        ):
            return
        self.invalidated_selected_scene = None
        self.refresh_inference_settings(scene, rebuild_selected=True)
        self.request_selected_frame(scene)

    def _fail_stream(
        self,
        scene: bpy.types.Scene,
        operation_id: str,
        message: str,
        *,
        cancel_worker: bool,
    ) -> None:
        stream = self.active_stream
        owns_error = (
            stream is not None
            and stream.operation_id == operation_id
            and stream.stop_requested
            and scene.audio2face.status == "ERROR"
        )
        if (
            cancel_worker
            and stream is not None
            and stream.operation_id == operation_id
            and not stream.stop_requested
        ):
            try:
                self._request_stream_cancel(stream)
            except (OSError, RuntimeError, ValueError) as exc:
                self._reject_worker_contract(
                    f"{message}; failed to cancel the active stream: {exc}"
                )
                return
        get_live_stream_controller().stop(reset=False, notify=False)
        self._clear_pcm_ingress()
        if not cancel_worker:
            self._release_active_stream(operation_id)
        if not owns_error:
            self._set_status(scene, "ERROR", message)

    def stop(self, scene: bpy.types.Scene) -> None:
        self._require_editable_scene(scene)
        stream = self.active_stream
        get_live_stream_controller().stop(reset=False, notify=False)
        if self.client.state in {Lifecycle.STOPPED, Lifecycle.FAILED}:
            self._clear_model_state()
            self._release_active_stream()
            self._release_selected_track()
            self._release_active_bake()
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
        track = self.selected_track
        if track is not None:
            owner = self._scene(track.scene_name)
            if owner is not None and owner.is_editable:
                self._set_status(owner, "STOPPING", "Worker shutdown requested")
        bake = self.active_bake
        if bake is not None:
            owner = self._scene(bake.scene_name)
            if owner is not None and owner.is_editable:
                self._set_status(owner, "STOPPING", "Worker shutdown requested")
        self.expected_worker_exit = True
        request_id = self.client.begin_shutdown(timeout=SHUTDOWN_TIMEOUT_SECONDS)
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
        get_live_stream_controller().stop(reset=False, notify=False)
        self._release_active_stream()
        self._release_selected_track()
        self._release_active_bake()
        for scene in scenes:
            self._set_status(scene, "ERROR", message)
        with self.pending_lock:
            self.pending.clear()
            self.pcm_ingress = None
        self.expected_worker_exit = True
        self.client.begin_shutdown(timeout=SHUTDOWN_TIMEOUT_SECONDS)

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
        result = envelope["result"]
        if self.expected_worker_exit and pending.method != "shutdown":
            return
        scene = self._scene(pending.scene_name)
        if scene is None or not scene.is_editable:
            if pending.method == "hello":
                self.handshake_spec = None
                self.expected_worker_exit = True
                self.client.begin_shutdown(timeout=SHUTDOWN_TIMEOUT_SECONDS)
            if pending.method == "cancel":
                if result:
                    self._reject_worker_contract(
                        "worker returned an invalid operation-cancel response"
                    )
            return
        settings = scene.audio2face

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
            if settings.status == "LOADING_MODEL":
                self._set_status(
                    scene,
                    "MODEL_READY",
                    "Loaded Audio2Face 3.0 and Audio2Emotion 3.0 models",
                )
                try:
                    self._ensure_selected_track(scene)
                except (OSError, RuntimeError, ValueError) as exc:
                    self._set_status(scene, "ERROR", str(exc))
        elif pending.method == "track_start":
            track = self.selected_track
            if (
                track is None
                or track.operation_id != pending.operation_id
                or track.scene_name != scene.name
            ):
                return
            if result:
                self._reject_worker_contract(
                    "worker returned a noncanonical track-start response"
                )
                return
            try:
                self._send_next_track_chunk(scene, track)
            except (OSError, RuntimeError, ValueError) as exc:
                self._fail_selected_track(scene, track, str(exc))
        elif pending.method == "track_chunk":
            track = self.selected_track
            if track is None or track.operation_id != pending.operation_id:
                return
            if result:
                self._reject_worker_contract(
                    "worker returned a noncanonical track-chunk response"
                )
                return
            try:
                self._send_next_track_chunk(scene, track)
            except (OSError, RuntimeError, ValueError) as exc:
                self._fail_selected_track(scene, track, str(exc))
        elif pending.method == "track_prepare":
            track = self.selected_track
            if track is None or track.operation_id != pending.operation_id:
                return
            if result:
                self._reject_worker_contract(
                    "worker returned a noncanonical track-prepare response"
                )
                return
            track.prepared = True
            try:
                bake = self.active_bake
                if bake is None and _native_playback_active():
                    self.invalidate_selected_settings(scene)
                    return
                self._request_track_render(
                    scene,
                    track,
                    (
                        bake.settings_timeline
                        if bake is not None and bake.scene_name == scene.name
                        else self._evaluate_settings_timeline(scene, track)[0]
                    ),
                )
            except (OSError, RuntimeError, ValueError) as exc:
                if self.active_bake is not None:
                    self._fail_bake(scene, str(exc))
                else:
                    track.render_error = str(exc)
                    self._set_status(scene, "ERROR", track.render_error)
        elif pending.method == "track_render":
            track = self.selected_track
            if (
                track is None
                or track.operation_id != pending.operation_id
                or track.scene_name != scene.name
            ):
                return
            if track.cancel_requested:
                return
            if set(result) != {"revision", "frame_count", "superseded"}:
                self._reject_worker_contract(
                    "worker returned a noncanonical track-render response"
                )
                return
            revision = result["revision"]
            frame_count = result["frame_count"]
            superseded = result["superseded"]
            if (
                isinstance(revision, bool)
                or not isinstance(revision, int)
                or revision <= 0
                or isinstance(frame_count, bool)
                or not isinstance(frame_count, int)
                or frame_count < 0
                or type(superseded) is not bool
            ):
                self._reject_worker_contract(
                    "worker returned an invalid track-render response"
                )
                return
            if revision != track.render_revision:
                return
            stage = track.stage
            if stage is None or stage.revision != revision:
                self._reject_worker_contract(
                    "worker completed a track render without staged frames"
                )
                return
            if superseded:
                if frame_count != 0:
                    self._reject_worker_contract(
                        "superseded track render reported completed frames"
                    )
                    return
                track.stage = None
                return
            if (
                frame_count <= 0
                or stage.total_frames != frame_count
                or len(stage.timestamps) != frame_count
                or len(stage.weights) != frame_count
                or len(stage.effective_emotions) != frame_count
            ):
                self._reject_worker_contract(
                    "worker completed an incomplete track render"
                )
                return
            initial_render = track.published_timeline is None
            render_error = track.render_error
            track.timestamps = tuple(stage.timestamps)
            track.weights = tuple(stage.weights)
            track.effective_emotions = tuple(stage.effective_emotions)
            track.published_timeline = track.render_timeline
            track.stage = None
            track.render_error = None
            try:
                bake = self.active_bake
                if bake is not None and bake.scene_name == scene.name:
                    self._finish_bake(scene)
                else:
                    if initial_render and settings.status in {
                        "TRACK_UPLOADING",
                        "TRACK_PREPARING",
                    }:
                        self._set_status(scene, "MODEL_READY", "Selected WAV is ready")
                    elif (
                        render_error is not None
                        and settings.status == "ERROR"
                        and settings.status_message == render_error
                    ):
                        self._set_status(scene, "MODEL_READY", "Selected WAV is ready")
                    self.request_selected_frame(scene)
            except (OSError, RuntimeError, ValueError) as exc:
                if self.active_bake is not None:
                    self._fail_bake(scene, str(exc))
                else:
                    self._set_status(scene, "ERROR", str(exc))
        elif pending.method == "cancel":
            if result:
                self._reject_worker_contract(
                    "worker returned an invalid operation-cancel response",
                )
                return
            track = self.selected_track
            if track is not None and track.operation_id == pending.operation_id:
                return
            stream = self.active_stream
            if stream is None or stream.operation_id != pending.operation_id:
                return
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
            stream.prebuffer_samples = prebuffer_samples
            if (
                not stream.stop_requested
                and settings.status == "STREAM_STARTING"
            ):
                self._set_status(scene, "STREAMING", "PCM stream is ready")
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
        elif pending.method == "shutdown":
            if result:
                self._reject_worker_contract(
                    "worker returned an invalid shutdown response"
                )
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
        if self.expected_worker_exit and pending.method != "shutdown":
            return
        scene = self._scene(pending.scene_name)
        if scene is None or not scene.is_editable:
            track = self.selected_track
            if track is not None and track.operation_id == pending.operation_id:
                if (
                    pending.method == "cancel"
                    and error["code"] == "operation_not_found"
                ):
                    return
                self._release_active_bake()
                self._release_selected_track(track.operation_id)
                self._reject_worker_contract(message)
                return
            stream = self.active_stream
            if (
                pending.method == "cancel"
                and stream is not None
                and stream.operation_id == pending.operation_id
            ):
                if error["code"] == "operation_not_found":
                    return
                self._release_active_stream(stream.operation_id)
                self._reject_worker_contract(message)
            return
        if pending.operation_id is not None:
            track = self.selected_track
            if track is not None and track.operation_id == pending.operation_id:
                if pending.method == "track_render":
                    track.stage = None
                    track.render_error = message
                if track.cancel_requested and pending.method != "cancel":
                    return
                if (
                    track.cancel_requested
                    and error["code"] == "operation_not_found"
                    and pending.method == "cancel"
                ):
                    return
                if pending.method in {
                    "track_start",
                    "track_chunk",
                    "track_prepare",
                }:
                    self._fail_selected_track(scene, track, message)
                elif self.active_bake is not None:
                    self._fail_bake(scene, message)
                else:
                    self._set_status(scene, "ERROR", message)
                return
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

    def _handle_track_event(
        self,
        scene: bpy.types.Scene | None,
        track: SelectedTrack,
        event: str,
        data: dict[str, Any],
    ) -> None:
        if event == "track_preview":
            if track.cancel_requested:
                return
            if set(data) != {
                "revision",
                "timestamp_sample",
                "weights",
                "effective_emotions",
            }:
                self._reject_worker_contract(
                    "worker returned invalid track-preview data"
                )
                return
            revision = data["revision"]
            if (
                isinstance(revision, bool)
                or not isinstance(revision, int)
                or revision <= 0
            ):
                self._reject_worker_contract(
                    "worker returned invalid track-preview revision"
                )
                return
            if revision != track.render_revision:
                return
            model_schema = self.model_schema
            if model_schema is None:
                return
            try:
                weights, emotions = validate_stream_frame(
                    tuple(model_schema["channels"]),
                    _model_emotion_channels(model_schema),
                    data["timestamp_sample"],
                    data["weights"],
                    data["effective_emotions"],
                )
            except (LiveStreamError, TypeError, ValueError) as exc:
                self._reject_worker_contract(str(exc))
                return
            if scene is None:
                return
            try:
                if not self._settings_timeline_matches_current_frame(
                    scene,
                    track,
                    track.render_timeline,
                ):
                    return
                target_sample = self._track_sample(
                    scene,
                    track,
                    int(scene.frame_current),
                )
                if target_sample != data["timestamp_sample"]:
                    return
                apply_model_frame(
                    scene.audio2face,
                    tuple(model_schema["channels"]),
                    _model_emotion_channels(model_schema),
                    weights,
                    emotions,
                )
            except (LiveStreamError, RuntimeError, ValueError) as exc:
                self._fail_selected_track(scene, track, str(exc))
            return
        if event == "track_frame_batch":
            if track.cancel_requested:
                return
            if set(data) != {
                "revision",
                "offset",
                "total_frames",
                "timestamp_samples",
                "weights",
                "effective_emotions",
            }:
                self._reject_worker_contract(
                    "worker returned invalid track-frame-batch data"
                )
                return
            revision = data["revision"]
            offset = data["offset"]
            total_frames = data["total_frames"]
            if (
                isinstance(revision, bool)
                or not isinstance(revision, int)
                or revision <= 0
                or isinstance(offset, bool)
                or not isinstance(offset, int)
                or offset < 0
                or isinstance(total_frames, bool)
                or not isinstance(total_frames, int)
                or total_frames <= 0
            ):
                self._reject_worker_contract(
                    "worker returned invalid track-frame-batch bounds"
                )
                return
            if revision != track.render_revision:
                return
            stage = track.stage
            if stage is None or stage.revision != revision:
                self._reject_worker_contract(
                    "worker returned track frames without an active render"
                )
                return
            timestamps = data["timestamp_samples"]
            weights_rows = data["weights"]
            emotion_rows = data["effective_emotions"]
            if (
                type(timestamps) is not list
                or type(weights_rows) is not list
                or type(emotion_rows) is not list
                or not 1 <= len(timestamps) <= 64
                or len(weights_rows) != len(timestamps)
                or len(emotion_rows) != len(timestamps)
                or offset != len(stage.timestamps)
                or offset + len(timestamps) > total_frames
                or (
                    stage.total_frames is not None
                    and stage.total_frames != total_frames
                )
            ):
                self._reject_worker_contract(
                    "worker returned a noncanonical track-frame batch"
                )
                return
            model_schema = self.model_schema
            if model_schema is None:
                return
            validated_weights: list[tuple[float, ...]] = []
            validated_emotions: list[tuple[float, ...]] = []
            previous_timestamp = stage.timestamps[-1] if stage.timestamps else -1
            try:
                for timestamp, row, emotion_row in zip(
                    timestamps,
                    weights_rows,
                    emotion_rows,
                    strict=True,
                ):
                    row_weights, row_emotions = validate_stream_frame(
                        tuple(model_schema["channels"]),
                        _model_emotion_channels(model_schema),
                        timestamp,
                        row,
                        emotion_row,
                    )
                    if timestamp <= previous_timestamp:
                        raise LiveStreamError(
                            "track frame timestamps must be strictly increasing"
                        )
                    previous_timestamp = timestamp
                    validated_weights.append(row_weights)
                    validated_emotions.append(row_emotions)
            except (LiveStreamError, TypeError, ValueError) as exc:
                self._reject_worker_contract(str(exc))
                return
            stage.total_frames = total_frames
            stage.timestamps.extend(timestamps)
            stage.weights.extend(validated_weights)
            stage.effective_emotions.extend(validated_emotions)
            return
        if event == "track_ended":
            if data != {"reason": "canceled"}:
                self._reject_worker_contract(
                    "worker returned invalid track-ended data"
                )
                return
            try:
                self._complete_selected_track_cancel(scene, track)
            except (OSError, RuntimeError, ValueError) as exc:
                if scene is None:
                    self._reject_worker_contract(str(exc))
                    return
                self._set_status(scene, "ERROR", str(exc))
            return
        if event == "error":
            if set(data) != {"code", "message"}:
                self._reject_worker_contract(
                    "worker returned invalid operation error data"
                )
                return
            code = data["code"]
            worker_message = data["message"]
            if (
                not isinstance(code, str)
                or not code
                or not isinstance(worker_message, str)
                or not worker_message
            ):
                self._reject_worker_contract(
                    "worker returned invalid operation error data"
                )
                return
            self._release_active_bake()
            self._release_selected_track(track.operation_id)
            if scene is None:
                return
            self._set_status(scene, "ERROR", f"{code}: {worker_message}")
            return
        self._reject_worker_contract(f"worker returned unsupported track event {event!r}")

    def _handle_unavailable_stream_event(
        self,
        stream: ActiveStream,
        event: str,
    ) -> None:
        """Drain one stream after its owning scene can no longer be edited."""

        if event == "stream_ended":
            live = get_live_stream_controller()
            if live.active:
                live.stop(reset=False, notify=False)
            self._release_active_stream(stream.operation_id)
            return
        if event == "error":
            live = get_live_stream_controller()
            if live.active:
                live.stop(reset=False, notify=False)
            self._release_active_stream(stream.operation_id)
            return
        if event in {"stream_credit", "stream_frame"}:
            return
        self._reject_worker_contract(f"worker returned unsupported stream event {event!r}")

    def _handle_event(self, envelope: dict[str, Any]) -> None:
        event = envelope["event"]
        data = envelope["data"]
        operation_id = envelope["operation_id"]
        if self.expected_worker_exit:
            track = self.selected_track
            if (
                track is not None
                and track.operation_id == operation_id
                and event in {"track_ended", "error"}
            ):
                self._release_active_bake()
                self._release_selected_track(operation_id)
            stream = self.active_stream
            if (
                stream is not None
                and stream.operation_id == operation_id
                and event in {"stream_ended", "error"}
            ):
                get_live_stream_controller().stop(reset=False, notify=False)
                self._release_active_stream(operation_id)
            return
        track = self.selected_track
        if track is not None and track.operation_id == operation_id:
            scene = self._scene(track.scene_name)
            if scene is None or not scene.is_editable:
                scene = None
            self._handle_track_event(scene, track, event, data)
            return
        stream = self.active_stream
        if stream is None or stream.operation_id != operation_id:
            self._reject_worker_contract(
                "worker returned an event for an unknown operation ID"
            )
            return
        scene = self._scene(stream.scene_name)
        if scene is None or not scene.is_editable:
            self._handle_unavailable_stream_event(stream, event)
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
        elif event == "stream_frame":
            if set(data) != {
                "timestamp_sample",
                "weights",
                "effective_emotions",
            }:
                self._reject_worker_contract(
                    "worker returned invalid stream-frame data",
                )
                return
            if stream.stop_requested:
                return
            try:
                get_live_stream_controller().receive(
                    operation_id,
                    data["timestamp_sample"],
                    data["weights"],
                    data["effective_emotions"],
                )
            except (LiveStreamError, TypeError, ValueError) as exc:
                self._reject_worker_contract(str(exc))
                return
        elif event == "stream_ended":
            if data:
                self._reject_worker_contract(
                    "stream-ended event data must be empty",
                )
                return
            explicit_stop = stream.stop_requested
            if explicit_stop:
                get_live_stream_controller().stop(reset=False, notify=False)
                self._release_active_stream(operation_id)
                if (
                    not self.expected_worker_exit
                    and self.client.state == Lifecycle.RUNNING
                    and settings.status in _STREAM_STATUSES
                ):
                    self._set_status(
                        scene,
                        "MODEL_READY",
                        "PCM stream stopped; model remains ready",
                    )
                    try:
                        self._ensure_selected_track(scene)
                    except (OSError, RuntimeError, ValueError) as exc:
                        self._set_status(scene, "ERROR", str(exc))
            else:
                stream.worker_ended = True
                live = get_live_stream_controller()
                try:
                    live.mark_terminal(operation_id)
                except LiveStreamError as exc:
                    self._release_active_stream(operation_id)
                    self._set_status(scene, "ERROR", str(exc))
                    return
                if (
                    not self.expected_worker_exit
                    and self.client.state == Lifecycle.RUNNING
                    and live.active
                    and settings.status in _STREAM_STATUSES
                ):
                    self._set_status(
                        scene,
                        "STREAM_ENDING",
                        "Finishing buffered streamed audio and Audio2Face values",
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
        if self.optimization_failed:
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
        self._refresh_invalidated_selected_settings()
        self._cancel_orphaned_operation()
        self._poll_pcm_ingress()
        self.client.tick()
        for event in self.client.poll():
            if isinstance(event, ControlMessage):
                self._handle_control(event.envelope)
            elif isinstance(event, ClientDiagnostic):
                self.last_worker_diagnostic = event.message[-1000:]
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
                self._release_active_stream()
                self._release_selected_track()
                self._release_active_bake()
                for scene in self._editable_scenes():
                    settings = scene.audio2face
                    if self.rejected_reason:
                        self._set_status(scene, "ERROR", self.rejected_reason)
                    elif expected_exit and event.returncode == 0:
                        if settings.status != "ERROR":
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

        if (
            self.handshake_deadline is not None
            and not self.negotiated
            and not self.expected_worker_exit
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
        self._release_active_bake()
        self._release_selected_track()
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
    stream_active = False
    try:
        controller.poll()
        stream_active = get_live_stream_controller().tick()
        if stream_active:
            controller._tag_runtime_setup_redraw()
    except Exception as exc:  # Keep timer alive, but surface the main-thread failure.
        stream_active = False
        controller._release_active_bake()
        if controller.selected_track is not None:
            controller.selected_track.render_error = None
        for scene in controller._editable_scenes():
            controller._set_status(scene, "ERROR", f"Blender runtime failure: {exc}")
    track = controller.selected_track
    track_busy = track is not None and (
        not track.prepared or track.stage is not None or track.cancel_requested
    )
    return (
        PRESENTATION_INTERVAL_SECONDS
        if (
            stream_active
            or controller.active_stream is not None
            or track_busy
        )
        else POLL_INTERVAL_SECONDS
    )


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
def _frame_change_post_handler(
    scene: bpy.types.Scene,
    _depsgraph: bpy.types.Depsgraph | None = None,
) -> None:
    """Apply Selected Audio values after Blender evaluates the current frame."""

    controller = _CONTROLLER
    if controller is None or controller.evaluating_settings_timeline:
        return
    controller.refresh_frame_inference_settings(scene)
    controller.request_selected_frame(scene)


@bpy.app.handlers.persistent
def _depsgraph_update_post_handler(
    scene: bpy.types.Scene,
    depsgraph: bpy.types.Depsgraph,
) -> None:
    """Invalidate Selected settings when an Action used by the scene changes."""

    controller = _CONTROLLER
    if controller is None or controller.evaluating_settings_timeline:
        return
    animation_data = scene.animation_data
    if animation_data is None:
        return
    actions = [animation_data.action]
    actions.extend(
        strip.action
        for nla_track in animation_data.nla_tracks
        for strip in nla_track.strips
    )
    if any(
        update.id == action
        for update in depsgraph.updates
        for action in actions
        if action is not None
    ):
        controller.invalidate_selected_settings(scene)


@bpy.app.handlers.persistent
def _load_pre_handler(_unused: object) -> None:
    """Drop process, thread, and presentation state before Blender replaces its data."""

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
    if _depsgraph_update_post_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_depsgraph_update_post_handler)
    if _frame_change_post_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_frame_change_post_handler)
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
    if _frame_change_post_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_frame_change_post_handler)
    if _depsgraph_update_post_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_depsgraph_update_post_handler)
    _dispose_runtime_state()
