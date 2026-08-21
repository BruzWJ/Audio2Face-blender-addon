"""Blender main-thread controller for the queue-only sidecar client."""

from __future__ import annotations

import base64
import copy
import math
import queue
import struct
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bpy

from .live_stream import (
    LiveStreamError,
    get_live_stream_controller,
    unregister_live_stream,
)
from .preferences import get_preferences
from .preview import get_preview_controller, unregister_preview
from .properties import apply_model_defaults, tuning_parameters
from .runtime_bundle import (
    BundleError,
    BundleLaunchSpec,
    current_platform_id,
    resolve_runtime_bundle,
)
from .runtime_catalog import RuntimeCatalogError, load_runtime_catalog
from .runtime_install import (
    InstallProgress,
    RuntimeInstallCancelled,
    RuntimeInstallError,
    install_managed_runtime,
    validate_install_receipt,
)
from .sidecar import (
    ClientDiagnostic,
    ControlMessage,
    Lifecycle,
    ProcessExited,
    SidecarClient,
    SidecarError,
)
from .wav_stream import WavStreamError, WavStreamSource


@dataclass(slots=True)
class PendingRequest:
    method: str
    scene_name: str
    then_generate: bool = False
    then_stream_wav: bool = False
    model_signature: tuple[str, str, int] | None = None
    stream_id: str | None = None


WORKER_PROFILE = "nvidia-a2f3-a2e3-gpu-arkit52/1"
POLL_INTERVAL_SECONDS = 0.10
PREVIEW_INTERVAL_SECONDS = 1.0 / 60.0
SHUTDOWN_TIMEOUT_SECONDS = 2.0
HANDSHAKE_TIMEOUT_SECONDS = 15.0
STREAM_PREBUFFER_SECONDS = 1.25
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
        # One sidecar owns exactly one managed model pair, so its signature is global rather
        # than attached to a Blender scene.
        self.loaded_signature: tuple[str, str, int] | None = None
        self.model_sample_rate: int | None = None
        self.model_parameter_defaults: dict[str, Any] | None = None
        self.model_emotion_names: tuple[str, ...] | None = None
        self.scene_model_signatures: dict[int, tuple[str, str, int]] = {}
        self.rejected_reason: str | None = None
        self.install_thread: threading.Thread | None = None
        self.install_cancel: threading.Event | None = None
        self.install_activation_lock = threading.Lock()
        self.install_events: queue.Queue[tuple[str, str | None]] = queue.Queue()
        # Progress can be emitted once per archive member.  Keep only the most
        # recent snapshot so a large runtime cannot flood Blender's main-thread
        # timer with thousands of redundant RNA writes.
        self.install_progress_lock = threading.Lock()
        self.install_latest_progress: InstallProgress | None = None
        self.install_scene: str | None = None
        self.install_message = ""
        self.handshake_deadline: float | None = None
        self.last_worker_diagnostic = ""
        self.reset_scene_state_on_poll = True
        self.expected_worker_exit = False
        self.stream_source_thread: threading.Thread | None = None
        self.stream_source_cancel: threading.Event | None = None
        self.stream_playback_started: threading.Event | None = None
        self.stream_source_events: queue.Queue[tuple[str, str, str | None]] = queue.Queue()
        self.stream_audio_paths: dict[str, Path] = {}
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

    def _queue_install_progress(self, event: InstallProgress) -> None:
        """Publish the latest installer snapshot without touching :mod:`bpy`."""

        with self.install_progress_lock:
            self.install_latest_progress = event

    def _take_install_progress(self) -> InstallProgress | None:
        with self.install_progress_lock:
            event = self.install_latest_progress
            self.install_latest_progress = None
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
    def _absolute_blender_path(value: str) -> Path:
        return Path(bpy.path.abspath(value)).expanduser().resolve(strict=False)

    @staticmethod
    def result_directory() -> Path:
        path = bpy.utils.extension_path_user(__package__, path="results", create=True)
        return Path(path).resolve(strict=False)

    @staticmethod
    def data_root(*, create: bool) -> Path:
        """Return Blender's writable, upgrade-stable directory for this extension."""

        try:
            path = bpy.utils.extension_path_user(__package__, path="", create=create)
        except (OSError, RuntimeError, ValueError) as exc:
            raise SidecarError(f"cannot access Audio2Face extension storage: {exc}") from exc
        return Path(path).expanduser().resolve(strict=False)

    def runtime_spec(self) -> BundleLaunchSpec:
        try:
            artifact = load_runtime_catalog().artifact_for(current_platform_id())
            spec = resolve_runtime_bundle(
                self.data_root(create=False),
                require_engine=True,
            )
            validate_install_receipt(spec, artifact)
            return spec
        except (BundleError, RuntimeCatalogError, RuntimeInstallError) as exc:
            raise SidecarError(str(exc)) from exc

    def runtime_availability(self) -> tuple[bool, str]:
        try:
            spec = self.runtime_spec()
        except SidecarError as exc:
            return False, str(exc)
        return True, f"Managed {spec.platform} GPU runtime is ready"

    def install_availability(self) -> tuple[bool, str]:
        """Return whether this add-on release publishes a platform artifact."""

        try:
            platform_id = current_platform_id()
            catalog = load_runtime_catalog()
            catalog.artifact_for(platform_id)
        except (BundleError, RuntimeCatalogError) as exc:
            return False, str(exc)
        return True, f"Managed runtime {catalog.release} is available for download"

    @property
    def install_in_progress(self) -> bool:
        # Keep the operation reserved until its terminal queue event has been
        # consumed on Blender's main thread.
        return self.install_thread is not None

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

    def install_runtime(self, scene: bpy.types.Scene) -> None:
        """Start the verified runtime/model install without blocking Blender's UI."""

        self._require_editable_scene(scene)
        if self.install_in_progress:
            raise SidecarError("managed-runtime installation is already running")
        if self.client.state not in {Lifecycle.STOPPED, Lifecycle.FAILED}:
            raise SidecarError("stop the Audio2Face worker before installing its runtime")
        if not bpy.app.online_access:
            raise SidecarError(
                "Blender Online Access is disabled; enable it in Preferences, then install again"
            )
        preferences = get_preferences()
        if preferences is None or not preferences.runtime_license_accepted:
            raise SidecarError(
                "accept the NVIDIA runtime and both model license terms first"
            )
        try:
            platform_id = current_platform_id()
            artifact = load_runtime_catalog().artifact_for(platform_id)
        except (BundleError, RuntimeCatalogError) as exc:
            raise SidecarError(str(exc)) from exc

        data_root = self.data_root(create=True)
        canceled = threading.Event()
        with self.install_progress_lock:
            self.install_latest_progress = None
        self.install_cancel = canceled
        self.install_scene = scene.name
        message = "Preparing managed-runtime download"
        for candidate in self._editable_scenes():
            candidate_settings = candidate.audio2face
            candidate_settings.runtime_install_progress = 0.0
        self.install_message = message
        self._set_status(scene, "INSTALLING_RUNTIME", message)

        def progress(event: InstallProgress) -> None:
            self._queue_install_progress(event)

        def run_install() -> None:
            try:
                install_managed_runtime(
                    artifact,
                    data_root,
                    progress=progress,
                    canceled=canceled,
                    activation_lock=self.install_activation_lock,
                )
            except RuntimeInstallCancelled:
                self.install_events.put(("canceled", None))
            except (RuntimeInstallError, OSError, ValueError) as exc:
                self.install_events.put(("error", str(exc)))
            except Exception as exc:  # Never let a background exception disappear.
                self.install_events.put(("error", f"managed-runtime installation failed: {exc}"))
            else:
                self.install_events.put(("complete", None))

        self.install_thread = threading.Thread(
            name="a2f-managed-runtime-install",
            target=run_install,
            daemon=True,
        )
        try:
            self.install_thread.start()
        except RuntimeError as exc:
            self.install_thread = None
            self.install_cancel = None
            self.install_scene = None
            raise SidecarError(f"could not start managed-runtime installer: {exc}") from exc

    def cancel_runtime_install(self, scene: bpy.types.Scene) -> None:
        self._require_editable_scene(scene)
        if not self.install_in_progress or self.install_cancel is None:
            raise SidecarError("managed-runtime installation is not running")
        with self.install_activation_lock:
            self.install_cancel.set()
        owner_scene = self._scene(self.install_scene) or scene
        message = "Canceling managed-runtime installation"
        owner_scene.audio2face.status_message = message
        self.install_message = message

    def start(self, scene: bpy.types.Scene) -> None:
        self._require_editable_scene(scene)
        if self.install_in_progress:
            raise SidecarError("wait for managed-runtime installation to finish")
        if self.client.state == Lifecycle.RUNNING:
            raise SidecarError("worker is already running")
        if self.client.state == Lifecycle.STOPPING:
            raise SidecarError("worker is still shutting down")
        spec = self.runtime_spec()
        self.last_worker_diagnostic = ""
        self.client.start(
            spec.executable,
            cwd=spec.root,
            env=spec.env,
        )
        with self.pending_lock:
            self.pending.clear()
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
        then_generate: bool = False,
        then_stream_wav: bool = False,
        model_signature: tuple[str, str, int] | None = None,
        stream_id: str | None = None,
    ) -> str:
        with self.pending_lock:
            # Install correlation state before the main-thread poller can consume
            # a fast worker response.  This also covers calls made by an audio
            # source thread.
            request_id = self.client.request(method, params)
            self.pending[request_id] = PendingRequest(
                method=method,
                scene_name=scene.name,
                then_generate=then_generate,
                then_stream_wav=then_stream_wav,
                model_signature=model_signature,
                stream_id=stream_id,
            )
        return request_id

    def _send_hello(self, scene: bpy.types.Scene) -> None:
        self._request(
            scene,
            "hello",
            {},
        )
        self._set_status(scene, "STARTING", "Starting managed Audio2Face GPU worker")

    @staticmethod
    def _model_signature(
        settings: object,
        spec: BundleLaunchSpec,
    ) -> tuple[str, str, int]:
        return (
            str(spec.audio2face_model),
            str(spec.audio2emotion_model),
            int(settings.identity_index),
        )

    @staticmethod
    def _scene_key(scene: bpy.types.Scene) -> int:
        """Return a stable identity for one live Blender scene datablock."""

        try:
            return int(scene.as_pointer())
        except AttributeError:  # Ordinary-Python test doubles have no RNA pointer.
            return id(scene)

    def _clear_model_state(self) -> None:
        self.loaded_signature = None
        self.model_sample_rate = None
        self.model_parameter_defaults = None
        self.model_emotion_names = None
        self.scene_model_signatures.clear()

    def _cache_model_schema(
        self,
        scene: bpy.types.Scene,
        signature: tuple[str, str, int],
        defaults: dict[str, Any],
        emotion_names: list[str],
    ) -> None:
        self.loaded_signature = signature
        self.model_parameter_defaults = copy.deepcopy(defaults)
        self.model_emotion_names = tuple(emotion_names)
        self.scene_model_signatures[self._scene_key(scene)] = signature

    def _ensure_scene_model_schema(self, scene: bpy.types.Scene) -> None:
        """Populate model-derived controls for any scene using the loaded worker."""

        signature = self.loaded_signature
        defaults = self.model_parameter_defaults
        emotion_names = self.model_emotion_names
        if signature is None or defaults is None or emotion_names is None:
            raise SidecarError("loaded worker model metadata is unavailable")
        scene_key = self._scene_key(scene)
        if self.scene_model_signatures.get(scene_key) == signature:
            return
        try:
            apply_model_defaults(
                scene.audio2face,
                copy.deepcopy(defaults),
                list(emotion_names),
            )
        except ValueError as exc:
            raise SidecarError(str(exc)) from exc
        self.scene_model_signatures[scene_key] = signature

    def _submit_model_load(
        self,
        scene: bpy.types.Scene,
        spec: BundleLaunchSpec,
        *,
        then_generate: bool,
        then_stream_wav: bool = False,
    ) -> None:
        settings = scene.audio2face
        signature = self._model_signature(settings, spec)
        self._clear_model_state()
        self._request(
            scene,
            "load_model",
            {
                "audio2face_model_path": str(spec.audio2face_model),
                "audio2emotion_model_path": str(spec.audio2emotion_model),
                "identity_index": int(settings.identity_index),
            },
            then_generate=then_generate,
            then_stream_wav=then_stream_wav,
            model_signature=signature,
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
        spec = self.runtime_spec()
        if self.loaded_signature != self._model_signature(settings, spec):
            self._submit_model_load(scene, spec, then_generate=True)
            return

        self._ensure_scene_model_schema(scene)

        audio_path = self._absolute_blender_path(settings.audio_path)
        if not audio_path.is_file():
            raise SidecarError(f"audio file does not exist: {audio_path}")
        output_directory = self.result_directory()
        job_id = uuid.uuid4().hex
        result_path = output_directory / f"{job_id}.a2f.json"
        self._request(
            scene,
            "generate",
            {
                "job_id": job_id,
                "audio_path": str(audio_path),
                "result_path": str(result_path),
                "settings": tuning_parameters(settings),
            },
        )
        settings.current_job_id = job_id
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
        self._ensure_scene_model_schema(scene)
        stream_id = uuid.uuid4().hex
        playback_started = threading.Event() if audio_path is not None else None
        get_preview_controller().stop()
        try:
            get_live_stream_controller().prepare(
                scene,
                stream_id,
                sample_rate,
                audio_path=audio_path,
                playback_started=(
                    playback_started.set if playback_started is not None else None
                ),
                playback_stopped=lambda: self._finish_stream_presentation(
                    scene.name,
                    stream_id,
                ),
            )
        except LiveStreamError as exc:
            raise SidecarError(str(exc)) from exc
        try:
            self._request(
                scene,
                "stream_start",
                {
                    "stream_id": stream_id,
                    "sample_rate": sample_rate,
                    "settings": tuning_parameters(scene.audio2face),
                },
                stream_id=stream_id,
            )
        except Exception:
            get_live_stream_controller().stop(reset=False)
            raise

        settings = scene.audio2face
        settings.stream_id = stream_id
        settings.stream_sample_rate = sample_rate
        settings.stream_prebuffer_samples = 0
        settings.stream_time = 0.0
        self.stream_scene_names[stream_id] = scene.name
        self.stream_source_cancel = threading.Event()
        self.stream_playback_started = playback_started
        if audio_path is not None:
            self.stream_audio_paths[stream_id] = audio_path
        self._set_status(scene, "STREAM_STARTING", "Preparing incremental PCM stream")
        return stream_id

    def start_wav_stream(self, scene: bpy.types.Scene) -> str | None:
        """Start the built-in selected-WAV source on the incremental model path."""

        self._require_editable_scene(scene)
        self._require_operation_idle()
        self._require_worker_ready()
        settings = scene.audio2face
        audio_path = self._absolute_blender_path(settings.audio_path)
        if not audio_path.is_file():
            raise SidecarError(f"audio file does not exist: {audio_path}")
        spec = self.runtime_spec()
        if self.loaded_signature != self._model_signature(settings, spec):
            self._submit_model_load(
                scene,
                spec,
                then_generate=False,
                then_stream_wav=True,
            )
            return None
        return self._submit_stream_start(scene, audio_path=audio_path)

    def start_pcm_stream(self, scene: bpy.types.Scene) -> str:
        """Begin source-agnostic mono-f32 PCM ingress for Blender integrations."""

        self._require_editable_scene(scene)
        self._require_operation_idle()
        self._require_worker_ready()
        settings = scene.audio2face
        spec = self.runtime_spec()
        if self.loaded_signature != self._model_signature(settings, spec):
            raise SidecarError(
                "model settings changed; restart the worker before opening a PCM stream"
            )
        return self._submit_stream_start(scene, audio_path=None)

    def _stream_scene(self, stream_id: str) -> bpy.types.Scene | None:
        return next(
            (
                scene
                for scene in bpy.data.scenes
                if scene.is_editable and scene.audio2face.stream_id == stream_id
            ),
            None,
        )

    @staticmethod
    def _validate_f32le_chunk(audio_f32le: bytes | bytearray | memoryview) -> bytes:
        try:
            payload = bytes(audio_f32le)
        except (TypeError, ValueError) as exc:
            raise SidecarError("stream audio must be a bytes-like mono f32le payload") from exc
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
        audio_f32le: bytes | bytearray | memoryview,
        *,
        stream_id: str | None = None,
    ) -> str:
        """Queue one mono-f32le chunk; safe to call from an audio-source thread."""

        active_id = stream_id or get_live_stream_controller().stream_id
        if not active_id:
            raise SidecarError("no PCM stream is active")
        scene_name = self.stream_scene_names.get(active_id)
        if scene_name is None:
            raise SidecarError("the requested PCM stream is not active")
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
                    "stream_id": active_id,
                    "audio_f32le_base64": base64.b64encode(payload).decode("ascii"),
                },
            )
            self.pending[request_id] = PendingRequest(
                "stream_chunk",
                scene_name,
                stream_id=active_id,
            )
        return request_id

    def _queue_stream_request(
        self,
        stream_id: str,
        method: str,
        params: dict[str, Any],
    ) -> str:
        scene_name = self.stream_scene_names.get(stream_id)
        if scene_name is None:
            raise SidecarError("the requested PCM stream is not active")
        with self.pending_lock:
            request_id = self.client.request(method, params)
            self.pending[request_id] = PendingRequest(
                method,
                scene_name,
                stream_id=stream_id,
            )
        return request_id

    def _start_wav_stream_source(
        self,
        scene: bpy.types.Scene,
        stream_id: str,
        prebuffer_samples: int,
    ) -> None:
        if self.stream_source_thread is not None and self.stream_source_thread.is_alive():
            raise SidecarError("a selected-WAV stream source is already running")
        audio_path = self.stream_audio_paths.get(stream_id)
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
                    required_prebuffer = max(
                        int(STREAM_PREBUFFER_SECONDS * sample_rate),
                        prebuffer_samples,
                    )
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
                        self.push_stream_audio(chunk, stream_id=stream_id)
                        samples_sent += len(chunk) // 4
                if canceled.is_set():
                    return
                self._queue_stream_request(
                    stream_id,
                    "stream_end",
                    {"stream_id": stream_id},
                )
                self.stream_source_events.put(("ending", stream_id, None))
            except (OSError, SidecarError, WavStreamError, ValueError) as exc:
                if not canceled.is_set():
                    self.stream_source_events.put(("error", stream_id, str(exc)))
            except Exception as exc:
                if not canceled.is_set():
                    self.stream_source_events.put(
                        ("error", stream_id, f"selected-WAV stream failed: {exc}")
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
                kind, stream_id, message = self.stream_source_events.get_nowait()
            except queue.Empty:
                return
            scene = self._stream_scene(stream_id)
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
                    {"job_id": stream_id},
                    stream_id=stream_id,
                )
            except SidecarError:
                pass
            get_live_stream_controller().stop(reset=False)
            self._clear_stream_state(scene)
            self._set_status(scene, "ERROR", message)

    def end_stream(self, scene: bpy.types.Scene) -> None:
        """Close PCM input and let the worker drain the model's padded tail."""

        self._require_editable_scene(scene)
        stream_id = scene.audio2face.stream_id
        if not stream_id:
            raise SidecarError("there is no active PCM stream")
        if self.stream_source_cancel is not None:
            self.stream_source_cancel.set()
        if stream_id not in self.stream_scene_names:
            raise SidecarError("stream input has already ended; stop its buffered playback instead")
        self._request(
            scene,
            "stream_end",
            {"stream_id": stream_id},
            stream_id=stream_id,
        )
        self._set_status(scene, "STREAM_ENDING", "Draining final streamed frames")

    def stop_stream(self, scene: bpy.types.Scene) -> None:
        """Cancel one stream without stopping or unloading the GPU worker."""

        self._require_editable_scene(scene)
        stream_id = scene.audio2face.stream_id
        if not stream_id:
            raise SidecarError("there is no active PCM stream")
        if stream_id not in self.stream_scene_names:
            get_live_stream_controller().stop()
            self._clear_stream_state(scene, stream_id=stream_id)
            self._set_status(
                scene,
                "MODEL_READY",
                "Buffered PCM stream stopped; model remains ready",
            )
            return
        if self.stream_source_cancel is not None:
            self.stream_source_cancel.set()
        self.stream_stop_requests.add(stream_id)
        try:
            self._request(
                scene,
                "cancel",
                {"job_id": stream_id},
                stream_id=stream_id,
            )
        except Exception:
            self.stream_stop_requests.discard(stream_id)
            raise
        self._set_status(scene, "STREAM_ENDING", "Stopping PCM stream")

    def _release_worker_stream_state(self, stream_id: str) -> None:
        if stream_id:
            self.stream_audio_paths.pop(stream_id, None)
            self.stream_scene_names.pop(stream_id, None)
            self.stream_stop_requests.discard(stream_id)
        self.stream_source_thread = None
        self.stream_source_cancel = None
        self.stream_playback_started = None

    def _clear_stream_state(
        self,
        scene: bpy.types.Scene,
        *,
        stream_id: str | None = None,
    ) -> None:
        resolved_id = stream_id or scene.audio2face.stream_id
        if resolved_id:
            self._release_worker_stream_state(resolved_id)
        if stream_id is None or scene.audio2face.stream_id == stream_id:
            scene.audio2face.stream_id = ""
            scene.audio2face.stream_sample_rate = 0
            scene.audio2face.stream_prebuffer_samples = 0

    def pcm_stream_requirements(
        self,
        scene: bpy.types.Scene,
    ) -> tuple[int, int] | None:
        """Return model-rate/prebuffer metadata once an external PCM stream is ready."""

        self._require_editable_scene(scene)
        settings = scene.audio2face
        if not settings.stream_id:
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

    def _finish_stream_presentation(self, scene_name: str, stream_id: str) -> None:
        scene = self._scene(scene_name)
        if scene is None or not scene.is_editable:
            self._release_worker_stream_state(stream_id)
            return
        if scene.audio2face.stream_id != stream_id:
            self._release_worker_stream_state(stream_id)
            return
        self._clear_stream_state(scene, stream_id=stream_id)
        if scene.audio2face.status not in {"ERROR", "IDLE", "STOPPING"}:
            self._set_status(scene, "MODEL_READY", "PCM stream ended; model remains ready")

    def _fail_stream(
        self,
        scene: bpy.types.Scene,
        stream_id: str,
        message: str,
        *,
        cancel_worker: bool,
    ) -> None:
        if self.stream_source_cancel is not None:
            self.stream_source_cancel.set()
        if cancel_worker and scene.audio2face.stream_id == stream_id:
            try:
                self._request(
                    scene,
                    "cancel",
                    {"job_id": stream_id},
                    stream_id=stream_id,
                )
            except SidecarError:
                pass
        get_live_stream_controller().stop(reset=False)
        self._clear_stream_state(scene, stream_id=stream_id)
        self._set_status(scene, "ERROR", message)

    def cancel(self, scene: bpy.types.Scene) -> None:
        self._require_editable_scene(scene)
        settings = scene.audio2face
        if not settings.current_job_id:
            raise SidecarError("there is no active generation job")
        self._request(
            scene,
            "cancel",
            {"job_id": settings.current_job_id},
        )
        self._set_status(scene, "CANCELLING", "Cancellation requested")

    def stop(self, scene: bpy.types.Scene) -> None:
        self._require_editable_scene(scene)
        if self.stream_source_cancel is not None:
            self.stream_source_cancel.set()
        get_live_stream_controller().stop()
        if self.client.state in {Lifecycle.STOPPED, Lifecycle.FAILED}:
            self._clear_model_state()
            self._clear_stream_state(scene)
            self._set_status(scene, "IDLE", "Worker is already stopped")
            return
        if self.client.state == Lifecycle.STOPPING:
            self._set_status(scene, "STOPPING", "Worker shutdown is already in progress")
            return
        request_id = self.client.begin_shutdown(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        self.expected_worker_exit = True
        if request_id:
            with self.pending_lock:
                self.pending[request_id] = PendingRequest("shutdown", scene.name)
        self._set_status(scene, "STOPPING", "Worker shutdown requested")

    def _handle_response(self, envelope: dict[str, Any]) -> None:
        with self.pending_lock:
            pending = self.pending.pop(envelope["id"], None)
        if pending is None:
            message = "worker returned a response for an unknown request ID"
            self.rejected_reason = message
            scene = self._scene(self.startup_scene)
            if scene is not None:
                self._set_status(scene, "ERROR", message)
            self.client.begin_shutdown(timeout=SHUTDOWN_TIMEOUT_SECONDS)
            return
        scene = self._scene(pending.scene_name)
        if scene is None or not scene.is_editable:
            return
        settings = scene.audio2face
        result = envelope["result"]

        # Once shutdown starts, late model/job responses must not revive the UI.
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
                self.rejected_reason = (
                    f"worker does not implement the exact {WORKER_PROFILE} contract"
                )
                self._set_status(scene, "ERROR", self.rejected_reason)
                self.client.begin_shutdown(timeout=SHUTDOWN_TIMEOUT_SECONDS)
                return

            self.negotiated = True
            self.handshake_deadline = None
            self._submit_model_load(
                scene,
                self.runtime_spec(),
                then_generate=False,
            )
            return

        if pending.method == "load_model":
            expected_fields = {"parameter_defaults", "emotion_names", "sample_rate"}
            if set(result) != expected_fields:
                message = "worker returned a noncanonical model response"
                self._clear_model_state()
                self._set_status(scene, "ERROR", message)
                return
            defaults = result["parameter_defaults"]
            emotion_names = result["emotion_names"]
            sample_rate = result["sample_rate"]
            if (
                isinstance(sample_rate, bool)
                or not isinstance(sample_rate, int)
                or sample_rate <= 0
            ):
                self._clear_model_state()
                self._set_status(scene, "ERROR", "worker returned an invalid model sample rate")
                return
            try:
                apply_model_defaults(settings, defaults, emotion_names)
            except ValueError as exc:
                self._clear_model_state()
                self._set_status(scene, "ERROR", str(exc))
                return
            if pending.model_signature is None:
                self._clear_model_state()
                self._set_status(scene, "ERROR", "model response lost its request identity")
                return
            self._cache_model_schema(
                scene,
                pending.model_signature,
                defaults,
                emotion_names,
            )
            self.model_sample_rate = sample_rate
            self._set_status(
                scene,
                "MODEL_READY",
                "Loaded Audio2Face 3.0 and Audio2Emotion 3.0 models",
            )
            if pending.then_generate:
                self.generate(scene)
            elif pending.then_stream_wav:
                self.start_wav_stream(scene)
        elif pending.method == "generate":
            if result:
                self._set_status(scene, "ERROR", "worker returned a noncanonical generate response")
            else:
                self._set_status(scene, "GENERATING", "Worker accepted generation job")
        elif pending.method == "cancel":
            if pending.stream_id is not None:
                if settings.stream_id != pending.stream_id:
                    return
                if result:
                    self._fail_stream(
                        scene,
                        pending.stream_id,
                        "worker returned an invalid stream-cancel response",
                        cancel_worker=False,
                    )
                else:
                    self._set_status(
                        scene,
                        "STREAM_ENDING",
                        "Worker accepted stream stop",
                    )
                return
            if settings.status in {"COMPLETED", "ERROR"} or (
                not settings.current_job_id and not settings.stream_id
            ):
                return
            if not result:
                self._set_status(scene, "CANCELLING", "Worker accepted cancellation")
            else:
                self._set_status(scene, "ERROR", "worker returned an invalid cancel response")
        elif pending.method == "stream_start":
            response_rate = result.get("sample_rate")
            prebuffer_samples = result.get("prebuffer_samples")
            if (
                set(result) != {"sample_rate", "prebuffer_samples"}
                or isinstance(response_rate, bool)
                or not isinstance(response_rate, int)
                or response_rate <= 0
                or response_rate != self.model_sample_rate
                or isinstance(prebuffer_samples, bool)
                or not isinstance(prebuffer_samples, int)
                or prebuffer_samples < 0
                or prebuffer_samples > response_rate * 60
            ):
                self._fail_stream(
                    scene,
                    settings.stream_id,
                    "worker returned a noncanonical stream response",
                    cancel_worker=True,
                )
                return
            self._set_status(scene, "STREAMING", "PCM stream is ready")
            settings.stream_prebuffer_samples = prebuffer_samples
            if settings.stream_id in self.stream_audio_paths:
                self._start_wav_stream_source(
                    scene,
                    settings.stream_id,
                    prebuffer_samples,
                )
        elif pending.method == "stream_chunk":
            if result:
                self._fail_stream(
                    scene,
                    settings.stream_id,
                    "worker returned an invalid stream-chunk response",
                    cancel_worker=True,
                )
        elif pending.method == "stream_end":
            if result:
                self._fail_stream(
                    scene,
                    settings.stream_id,
                    "worker returned an invalid stream-end response",
                    cancel_worker=True,
                )
            else:
                self._set_status(scene, "STREAM_ENDING", "Worker is draining final frames")
        elif pending.method == "shutdown":
            if result:
                self._set_status(scene, "ERROR", "worker returned an invalid shutdown response")
            else:
                self._set_status(scene, "STOPPING", "Worker is exiting")

    def _handle_error(self, envelope: dict[str, Any]) -> None:
        with self.pending_lock:
            pending = self.pending.pop(envelope.get("id"), None)
        if pending is not None and pending.method == "load_model":
            self._clear_model_state()
        scene = self._scene(pending.scene_name if pending else self.startup_scene)
        if scene is None or not scene.is_editable:
            return
        error = envelope["error"]
        message = f"{error['code']}: {error['message']}"
        if pending is not None and pending.stream_id is not None:
            if scene.audio2face.stream_id != pending.stream_id:
                return
            self._fail_stream(
                scene,
                pending.stream_id,
                message,
                cancel_worker=pending.method != "cancel",
            )
            return
        if pending is not None and pending.method == "cancel" and (
            scene.audio2face.status in {"COMPLETED", "ERROR"}
            or (
                not scene.audio2face.current_job_id
                and not scene.audio2face.stream_id
            )
        ):
            return
        if (
            pending is not None
            and pending.method == "cancel"
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
        job_id = envelope["job_id"]
        scene = next(
            (
                candidate
                for candidate in bpy.data.scenes
                if candidate.is_editable
                and (
                    candidate.audio2face.current_job_id == job_id
                    or candidate.audio2face.stream_id == job_id
                )
            ),
            None,
        )
        if scene is None:
            return  # Ignore stale events from an earlier file or stream.
        settings = scene.audio2face
        is_stream = settings.stream_id == job_id
        is_generation = settings.current_job_id == job_id
        if event in {"stream_frame", "stream_ended"} and not is_stream:
            self._set_status(
                scene,
                "ERROR",
                "worker routed a stream event to a generation job",
            )
            return
        if event in {"progress", "result", "canceled"} and not is_generation:
            self._fail_stream(
                scene,
                job_id,
                "worker routed a generation event to a PCM stream",
                cancel_worker=True,
            )
            return

        if event == "stream_frame":
            if set(data) != {"timestamp_sample", "weights"}:
                self._fail_stream(
                    scene,
                    job_id,
                    "worker returned invalid stream-frame data",
                    cancel_worker=True,
                )
                return
            try:
                get_live_stream_controller().receive(
                    job_id,
                    data["timestamp_sample"],
                    data["weights"],
                )
            except (LiveStreamError, TypeError, ValueError) as exc:
                self._fail_stream(
                    scene,
                    job_id,
                    str(exc),
                    cancel_worker=True,
                )
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
                self._fail_stream(
                    scene,
                    job_id,
                    "stream-ended event data must be empty",
                    cancel_worker=False,
                )
                return
            explicit_stop = job_id in self.stream_stop_requests
            if explicit_stop:
                get_live_stream_controller().stop()
                self._clear_stream_state(scene, stream_id=job_id)
                self._set_status(
                    scene,
                    "MODEL_READY",
                    "PCM stream stopped; model remains ready",
                )
            else:
                self._release_worker_stream_state(job_id)
                get_live_stream_controller().mark_terminal(job_id)
                if get_live_stream_controller().active:
                    self._set_status(
                        scene,
                        "STREAMING",
                        "Finishing buffered streamed audio and ARKit-52 values",
                    )
                else:
                    self._clear_stream_state(scene, stream_id=job_id)
                    self._set_status(
                        scene,
                        "MODEL_READY",
                        "PCM stream ended; model remains ready",
                    )
        elif event == "progress":
            if set(data) != {"progress", "stage"}:
                self._set_status(scene, "ERROR", "worker returned invalid progress data")
                return
            progress = data["progress"]
            stage = data["stage"]
            if (
                isinstance(progress, bool)
                or not isinstance(progress, (int, float))
                or not math.isfinite(progress)
                or progress < 0.0
                or progress > 1.0
                or not isinstance(stage, str)
                or not stage
            ):
                self._set_status(scene, "ERROR", "worker returned invalid progress data")
                return
            settings.progress = float(progress)
            self._set_status(scene, "GENERATING", stage)
        elif event == "result":
            if data:
                self._set_status(scene, "ERROR", "result event data must be empty")
                return
            path = self.result_directory() / f"{job_id}.a2f.json"
            if not path.is_file():
                self._set_status(scene, "ERROR", f"worker result file is missing: {path}")
                return
            settings.result_path = str(path)
            settings.progress = 1.0
            self._set_status(scene, "COMPLETED", f"Result ready: {path.name}")
        elif event == "canceled":
            if data:
                self._set_status(scene, "ERROR", "worker returned invalid cancellation data")
                return
            settings.progress = 0.0
            self._set_status(scene, "MODEL_READY", "Generation canceled")
            settings.current_job_id = ""
        elif event == "error":
            if set(data) != {"code", "message"}:
                message = "worker returned invalid operation error data"
                if is_stream:
                    self._fail_stream(
                        scene,
                        job_id,
                        message,
                        cancel_worker=False,
                    )
                else:
                    self._set_status(scene, "ERROR", message)
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
                if is_stream:
                    self._fail_stream(
                        scene,
                        job_id,
                        message,
                        cancel_worker=False,
                    )
                else:
                    self._set_status(scene, "ERROR", message)
                return
            message = f"{code}: {worker_message}"
            if is_stream:
                self._fail_stream(
                    scene,
                    job_id,
                    message,
                    cancel_worker=False,
                )
            else:
                self._set_status(scene, "ERROR", message)

    def _handle_control(self, envelope: dict[str, Any]) -> None:
        if envelope["type"] == "response":
            self._handle_response(envelope)
        elif envelope["type"] == "error":
            self._handle_error(envelope)
        elif envelope["type"] == "event":
            self._handle_event(envelope)

    def _apply_install_progress(self, payload: InstallProgress) -> None:
        self.install_message = payload.message
        for candidate in self._editable_scenes():
            candidate_settings = candidate.audio2face
            candidate_settings.runtime_install_progress = payload.progress
        scene = self._scene(self.install_scene)
        if scene is not None and scene.is_editable:
            scene.audio2face.status = "INSTALLING_RUNTIME"
            scene.audio2face.status_message = payload.message

    def _finish_install(self, kind: str, payload: str | None) -> None:
        scene = self._scene(self.install_scene)
        settings = (
            scene.audio2face
            if scene is not None and scene.is_editable
            else None
        )
        if kind == "complete":
            self.install_message = "Managed Audio2Face/Audio2Emotion runtime is ready"
        elif kind == "canceled":
            self.install_message = "Runtime installation canceled"
        else:
            if payload is None:
                raise RuntimeError("runtime installer error event has no message")
            self.install_message = payload

        if settings is not None:
            if kind == "complete":
                settings.runtime_install_progress = 1.0
                self._set_status(
                    scene,
                    "IDLE",
                    "Managed runtime and GPU-optimized models are ready; start the worker",
                )
            elif kind == "canceled":
                settings.runtime_install_progress = 0.0
                self._set_status(
                    scene,
                    "IDLE",
                    "Runtime installation canceled; any previous runtime is unchanged",
                )
            else:
                self._set_status(scene, "ERROR", self.install_message)

        self.install_thread = None
        self.install_cancel = None
        self.install_scene = None
        with self.install_progress_lock:
            self.install_latest_progress = None

    def _poll_install_events(self) -> None:
        latest_progress = self._take_install_progress()
        try:
            terminal = self.install_events.get_nowait()
        except queue.Empty:
            terminal = None

        if latest_progress is not None:
            self._apply_install_progress(latest_progress)
        if terminal is not None:
            self._finish_install(*terminal)

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
                settings.stream_id = ""
                settings.stream_sample_rate = 0
                settings.stream_prebuffer_samples = 0
                settings.stream_time = 0.0
                settings.runtime_install_progress = 0.0
            self.reset_scene_state_on_poll = False
        self._poll_install_events()
        self._poll_stream_source_events()
        self.client.tick()
        for event in self.client.poll():
            if isinstance(event, ControlMessage):
                self._handle_control(event.envelope)
            elif isinstance(event, ClientDiagnostic):
                self.last_worker_diagnostic = event.message[-1000:]
                scene = self._scene(self.startup_scene)
                if scene is not None and scene.is_editable:
                    scene.audio2face.status_message = event.message
            elif isinstance(event, ProcessExited):
                expected_exit = self.expected_worker_exit
                self.negotiated = False
                self.handshake_deadline = None
                with self.pending_lock:
                    self.pending.clear()
                if self.stream_source_cancel is not None:
                    self.stream_source_cancel.set()
                get_live_stream_controller().stop()
                for scene in self._editable_scenes():
                    settings = scene.audio2face
                    settings.stream_id = ""
                    settings.stream_sample_rate = 0
                    settings.stream_prebuffer_samples = 0
                    if self.rejected_reason and scene.name == self.startup_scene:
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
            detail = (
                f": {self.last_worker_diagnostic}" if self.last_worker_diagnostic else ""
            )
            message = f"Audio2Face worker handshake timed out{detail}"
            self.rejected_reason = message
            scene = self._scene(self.startup_scene)
            if scene is not None:
                self._set_status(scene, "ERROR", message)
            self.client.begin_shutdown(timeout=SHUTDOWN_TIMEOUT_SECONDS)

    def close(self) -> None:
        if self.install_cancel is not None:
            with self.install_activation_lock:
                self.install_cancel.set()
        if self.install_thread is not None and self.install_thread.is_alive():
            self.install_thread.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        if self.stream_source_cancel is not None:
            self.stream_source_cancel.set()
        if self.stream_source_thread is not None and self.stream_source_thread.is_alive():
            self.stream_source_thread.join(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        get_live_stream_controller().close()
        self.client.close(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        with self.pending_lock:
            self.pending.clear()
        self.negotiated = False
        self.handshake_deadline = None
        self._clear_model_state()
        self.install_thread = None
        self.install_cancel = None
        self.install_scene = None
        self.stream_source_thread = None
        self.stream_source_cancel = None
        self.stream_playback_started = None
        self.stream_audio_paths.clear()
        self.stream_scene_names.clear()
        self.stream_stop_requests.clear()
        with self.install_progress_lock:
            self.install_latest_progress = None


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
    finally:
        try:
            unregister_preview()
        finally:
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


__all__ = [
    "RuntimeController",
    "get_controller",
    "register_runtime",
    "unregister_runtime",
]
