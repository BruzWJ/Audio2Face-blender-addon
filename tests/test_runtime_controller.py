from __future__ import annotations

import base64
import importlib.util
import struct
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


MODEL_CHANNELS = tuple(f"modelChannel{index}" for index in range(52))
MODEL_EMOTIONS = [0.75]
AUDIO2FACE_DEFAULTS: dict[str, float | int] = {
    "input_strength": 1.0,
    "lower_face_smoothing": 0.006,
    "upper_face_smoothing": 0.001,
    "lower_face_strength": 1.0,
    "upper_face_strength": 1.0,
    "face_mask_level": 0.6,
    "face_mask_softness": 0.0085,
    "skin_strength": 1.0,
    "blink_strength": 1.0,
    "eyelid_open_offset": 0.0,
    "lip_open_offset": 0.0,
    "eyeballs_strength": 1.0,
    "saccade_strength": 0.6,
    "right_eye_rot_x_offset": 0.0,
    "right_eye_rot_y_offset": 0.0,
    "left_eye_rot_x_offset": 0.0,
    "left_eye_rot_y_offset": 0.0,
    "eye_saccade_seed": 0,
}


def _inference_settings_payload() -> dict[str, object]:
    return {
        "audio2face": AUDIO2FACE_DEFAULTS.copy(),
        "emotion_driver": {
            "emotion_strength": 0.6,
            "generated": {
                "emotion_contrast": 1.0, "max_emotions": 6,
                "live_blend_coef": 0.7, "transition_smoothing": 0.5,
            },
            "preferred": {
                "values": {"Joy": 0.75},
                "strength": 0.35,
            },
        },
    }


def _model_schema() -> dict[str, object]:
    return {
        "channels": list(MODEL_CHANNELS),
        "emotion_channels": [{"name": "Joy", "default": 0.0}],
        "audio2face_defaults": AUDIO2FACE_DEFAULTS.copy(),
    }


class _Settings:
    def __init__(self) -> None:
        self.status = "IDLE"
        self.status_message = "before"
        self.input_mode = "SELECTED"
        self.prediction_delay = 0.0
        self.audio_path = ""
        self.audio_first_frame = 1
        self.stream_time = 0.0
        self.target_objects: list[object] = []
        for name, value in AUDIO2FACE_DEFAULTS.items():
            setattr(self, name, value)


class _ReadOnlySettings:
    def __setattr__(self, name: str, value: object) -> None:
        raise AssertionError(f"linked scene RNA was written: {name}={value!r}")


class _Scene:
    def __init__(self, name: str, *, editable: bool, settings: object) -> None:
        self.name = name
        self.is_editable = editable
        self.audio2face = settings
        self.frame_start = 1
        self.frame_end = 250
        self.frame_current = 1
        self.render = SimpleNamespace(fps=24, fps_base=1.0)
        self.sequence_editor = SimpleNamespace(strips=[])
        self.frame_set_calls: list[int] = []

    def as_pointer(self) -> int:
        return id(self)

    def frame_set(self, frame: int) -> None:
        self.frame_current = frame
        self.frame_set_calls.append(frame)


class _SoundStrip(dict[str, str]):
    def __init__(self, frame_start: int, frame_end: int) -> None:
        super().__init__(
            audio2face_selected_audio_owner="audio2face/selected-wav/1"
        )
        self.content_start = frame_start
        self.content_end = frame_end + 1


def _install_selected_audio_span(
    scene: _Scene,
    frame_start: int = 1,
    frame_end: int = 250,
) -> None:
    scene.sequence_editor.strips[:] = [_SoundStrip(frame_start, frame_end)]


class _Scenes(list[_Scene]):
    def get(self, name: str | None) -> _Scene | None:
        return next((scene for scene in self if scene.name == name), None)


class _LiveController:
    def __init__(self) -> None:
        self.is_active = False
        self.receive_calls: list[tuple[object, ...]] = []
        self.terminal_calls: list[str] = []
        self.stop_calls: list[dict[str, object]] = []
        self.prepare_external_calls: list[tuple[object, ...]] = []

    def tick(self) -> bool:
        return False

    @property
    def active(self) -> bool:
        return self.is_active

    def stop(self, **kwargs: object) -> None:
        self.stop_calls.append(dict(kwargs))
        self.is_active = False

    def receive(self, *args: object) -> None:
        self.receive_calls.append(args)

    def prepare_external(self, *args: object) -> None:
        self.prepare_external_calls.append(args)
        self.is_active = True

    def mark_terminal(self, operation_id: str) -> None:
        self.terminal_calls.append(operation_id)


class _Timers:
    def __init__(self) -> None:
        self.registrations: dict[object, dict[str, object]] = {}

    def is_registered(self, callback: object) -> bool:
        return callback in self.registrations

    def register(self, callback: object, **kwargs: object) -> None:
        self.registrations[callback] = dict(kwargs)

    def unregister(self, callback: object) -> None:
        self.registrations.pop(callback)


def _plain_pending(runtime: ModuleType, method: str, scene_name: str) -> object:
    return runtime.PendingRequest(
        method,
        scene_name,
        model_signature=None,
        operation_id=None,
    )


def _model_pending(
    runtime: ModuleType,
    scene_name: str,
    signature: tuple[str, str],
) -> object:
    return runtime.PendingRequest(
        "load_model",
        scene_name,
        model_signature=signature,
        operation_id=None,
    )


def _stream_pending(
    runtime: ModuleType,
    method: str,
    scene_name: str,
    operation_id: str,
) -> object:
    return runtime.PendingRequest(
        method,
        scene_name,
        model_signature=None,
        operation_id=operation_id,
    )


@pytest.fixture
def runtime_module(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    bpy = ModuleType("bpy")
    bpy.types = SimpleNamespace(Scene=object)  # type: ignore[attr-defined]
    bpy.data = SimpleNamespace(  # type: ignore[attr-defined]
        scenes=_Scenes(),
        actions=object(),
        window_managers=[],
    )
    bpy.context = SimpleNamespace(scene=None, window=None)  # type: ignore[attr-defined]
    bpy.path = SimpleNamespace(abspath=lambda value: value)  # type: ignore[attr-defined]
    bpy.utils = SimpleNamespace(  # type: ignore[attr-defined]
        extension_path_user=lambda *_args, **_kwargs: "/tmp/a2f-runtime-test"
    )
    bpy.app = SimpleNamespace(  # type: ignore[attr-defined]
        online_access=True,
        timers=_Timers(),
        handlers=SimpleNamespace(
            persistent=lambda callback: callback,
            animation_playback_pre=[],
            animation_playback_post=[],
            frame_change_post=[],
            load_pre=[],
            load_post=[],
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    preferences = ModuleType("audio2face.preferences")
    preferences.get_preferences = lambda *_args, **_kwargs: SimpleNamespace(  # type: ignore[attr-defined]
        nvidia_terms_accepted=True,
        audio2face_model_directory="",
        audio2emotion_model_directory="",
    )
    monkeypatch.setitem(sys.modules, preferences.__name__, preferences)

    properties = ModuleType("audio2face.properties")
    properties.apply_model_schema = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    properties.inference_settings = lambda *_args, **_kwargs: _inference_settings_payload()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, properties.__name__, properties)

    live_controller = _LiveController()
    live_stream = ModuleType("audio2face.live_stream")
    live_stream.LiveStreamError = ValueError  # type: ignore[attr-defined]
    live_stream.get_live_stream_controller = lambda: live_controller  # type: ignore[attr-defined]
    live_stream.unregister_live_stream = lambda: None  # type: ignore[attr-defined]
    live_stream.model_frame_calls = []  # type: ignore[attr-defined]
    live_stream.apply_model_frame = (  # type: ignore[attr-defined]
        lambda settings, channels, emotions, weights, effective: (
            live_stream.model_frame_calls.append(  # type: ignore[attr-defined]
                (
                    settings,
                    tuple(channels),
                    tuple(emotions),
                    tuple(weights),
                    tuple(effective),
                )
            )
        )
    )
    live_stream.validate_stream_frame = (  # type: ignore[attr-defined]
        lambda _channels, _emotion_channels, _timestamp, weights, emotions: (
            tuple(weights),
            tuple(emotions),
        )
    )
    monkeypatch.setitem(sys.modules, live_stream.__name__, live_stream)

    module_name = "audio2face._runtime_controller_test"
    source = Path(__file__).resolve().parents[1] / "audio2face" / "runtime.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    module._test_live_controller = live_controller  # type: ignore[attr-defined]
    module._test_live_stream = live_stream  # type: ignore[attr-defined]
    return module, bpy


def _local_scene(bpy: ModuleType, name: str = "Scene") -> tuple[_Scene, _Settings]:
    settings = _Settings()
    scene = _Scene(name, editable=True, settings=settings)
    bpy.data.scenes = _Scenes([scene])  # type: ignore[attr-defined]
    bpy.context.scene = scene  # type: ignore[attr-defined]
    return scene, settings


def _activate_stream(
    runtime: ModuleType,
    controller: object,
    scene: _Scene,
    *,
    operation_id: str = "stream-1",
    prebuffer_samples: int | None = None,
    stop_requested: bool = False,
    worker_ended: bool = False,
) -> object:
    stream = runtime.ActiveStream(
        operation_id=operation_id,
        scene_name=scene.name,
        prebuffer_samples=prebuffer_samples,
        stop_requested=stop_requested,
        worker_ended=worker_ended,
    )
    stream.chunk_credit.set()
    controller.active_stream = stream
    runtime._test_live_controller.is_active = True
    return stream


def _activate_bake(
    runtime: ModuleType,
    controller: object,
    scene: _Scene,
    *,
    frame_start: int = 1,
    frame_end: int = 2,
) -> object:
    bake = runtime.ActiveBake(
        scene_name=scene.name,
        frame_start=frame_start,
        frame_end=frame_end,
        targets=tuple(
            item.object
            for item in scene.audio2face.target_objects
            if item.object is not None
        ),
        settings=_inference_settings_payload(),
        prediction_delay=float(scene.audio2face.prediction_delay),
    )
    controller.active_bake = bake
    return bake


def _activate_track(
    runtime: ModuleType,
    controller: object,
    scene: _Scene,
    *,
    operation_id: str = "track-1",
    audio_samples: int = 48_000,
    prepared: bool = True,
) -> object:
    _install_selected_audio_span(scene)
    wav_source = SimpleNamespace(
        close_calls=0,
        metadata=SimpleNamespace(output_frames=audio_samples),
    )

    def close() -> None:
        wav_source.close_calls += 1

    wav_source.close = close
    track = runtime.SelectedTrack(
        operation_id=operation_id,
        scene_name=scene.name,
        path=Path(scene.audio2face.audio_path or "/audio/selected.wav"),
        wav_source=wav_source,
        chunks=iter(()),
        uploaded_samples=audio_samples,
        prepared=prepared,
    )
    controller.selected_track = track
    return track


_SELECTED_AUDIO_CHUNKS = (struct.pack("<ff", 0.1, 0.2), struct.pack("<f", 0.3))


class _FakeSelectedWavSource:
    def __init__(self, path: Path, **kwargs: object) -> None:
        self.path = path
        self.kwargs = kwargs
        self.advances = 0
        self.close_calls = 0
        self.closed = False
        self.metadata = SimpleNamespace(output_frames=16_000)

    def __iter__(self):  # type: ignore[no-untyped-def]
        try:
            for chunk in _SELECTED_AUDIO_CHUNKS:
                self.advances += 1
                yield chunk
        finally:
            self.close()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.close_calls += 1


def _start_selected_track(runtime, bpy, monkeypatch, tmp_path, *, frame=1):  # type: ignore[no-untyped-def]
    scene, settings = _local_scene(bpy)
    scene.frame_current = frame
    audio_path = tmp_path / "selected.wav"
    audio_path.write_bytes(b"fixture")
    settings.audio_path = str(audio_path)
    _install_selected_audio_span(scene)
    controller = runtime.RuntimeController()
    controller._require_worker_ready = lambda: None
    controller.model_sample_rate = 16_000
    controller.model_schema = _model_schema()
    controller.loaded_signature = ("face/model.json", "emotion/model.json")
    controller.client._state = runtime.Lifecycle.RUNNING
    controller.negotiated = True
    monkeypatch.setattr(runtime, "WavStreamSource", _FakeSelectedWavSource)
    requests: list[tuple[str, dict[str, object]]] = []
    controller.client.request = lambda method, params: (
        requests.append((method, params)) or f"request-{len(requests)}"
    )
    controller._ensure_selected_track(scene)
    track = controller.selected_track
    assert track is not None
    return controller, track, requests


def test_log_directory_rejects_a_filesystem_alias(
    runtime_module: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    runtime, bpy = runtime_module
    directory = tmp_path / "extension-data"
    directory.mkdir()
    alias = tmp_path / "extension-data-alias"
    try:
        alias.symlink_to(directory, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    bpy.utils.extension_path_user = lambda *_args, **_kwargs: str(alias)

    with pytest.raises(runtime.SidecarError, match="filesystem alias"):
        runtime.RuntimeController.log_directory()


def test_selected_paths_reject_blender_relative_spelling(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime = runtime_module[0]

    with pytest.raises(runtime.SidecarError, match="canonical absolute path"):
        runtime.RuntimeController._selected_path("//models/Audio2Face", "model")


def test_selected_directory_paths_accept_only_blender_terminal_separator(
    runtime_module: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    runtime, _bpy = runtime_module
    selected = tmp_path / "Audio2Face"
    selected.mkdir()

    assert (
        runtime.RuntimeController._selected_directory_path(f"{selected}/", "model")
        == selected
    )
    with pytest.raises(runtime.SidecarError, match="canonical absolute path"):
        runtime.RuntimeController._selected_directory_path(
            f"{tmp_path}/./Audio2Face/",
            "model",
        )


def test_optimization_progress_keeps_only_latest_snapshot(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    bpy.data.scenes = _Scenes(  # type: ignore[attr-defined]
        [_Scene("Linked", editable=False, settings=_ReadOnlySettings())]
    )
    controller = runtime.RuntimeController()

    for index in range(10_000):
        controller._queue_optimization_progress(
            runtime.OptimizationProgress(index / 10_000, f"step {index}")
        )
    controller._poll_optimization_events()

    assert controller.optimization_message == "step 9999"
    assert controller.optimization_progress == pytest.approx(0.9999)
    assert controller.optimization_failed is False


def test_optimization_error_is_retained_for_preferences(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, _bpy = runtime_module
    controller = runtime.RuntimeController()

    controller._finish_optimization("error", "readable TensorRT failure")

    assert controller.optimization_failed is True
    assert controller.optimization_message == "readable TensorRT failure"


def test_optimization_eligibility_reports_the_bundled_runtime_blocker(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, _bpy = runtime_module
    controller = runtime.RuntimeController()
    setup = runtime.RuntimeSetupSnapshot(
        runtime_status=runtime.SetupStatus(
            False,
            "bundled linux-x64 runtime is missing",
        ),
        model_status=runtime.SetupStatus(False, "models unavailable"),
        engine_status=runtime.SetupStatus(False, "engines unavailable"),
        model_spec=None,
    )

    assert controller.optimization_eligibility(setup) == (
        False,
        "bundled linux-x64 runtime is missing",
    )


def test_setup_snapshot_validates_the_saved_model_pair_once(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, _bpy = runtime_module
    selected: list[Path] = []
    for name in ("face", "emotion"):
        directory = tmp_path / name
        directory.mkdir()
        selected.append(directory)
    monkeypatch.setattr(
        runtime,
        "get_preferences",
        lambda: SimpleNamespace(
            audio2face_model_directory=str(selected[0]),
            audio2emotion_model_directory=str(selected[1]),
        ),
    )

    def validate_pair(
        face: Path,
        emotion: Path,
    ) -> tuple[Path, Path]:
        assert (face, emotion) == tuple(selected)
        return face / "model.json", emotion / "model.json"

    monkeypatch.setattr(runtime, "validate_model_pair", validate_pair)
    engine_checks: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        runtime,
        "validate_model_engines",
        lambda face, emotion: engine_checks.append((face, emotion)),
    )
    monkeypatch.setattr(
        runtime,
        "resolve_runtime_bundle",
        lambda: SimpleNamespace(platform="linux-x64"),
    )

    setup = runtime.RuntimeController().setup_snapshot()

    assert setup.runtime_status.ready is True
    assert setup.model_status.ready is True
    assert setup.engine_status.ready is True
    assert setup.model_spec.audio2face_model == selected[0] / "model.json"
    assert setup.model_spec.audio2emotion_model == selected[1] / "model.json"
    assert engine_checks == (
        [(selected[0] / "model.json", selected[1] / "model.json")]
    )
    monkeypatch.setattr(
        runtime,
        "validate_model_engines",
        lambda _face, _emotion: (_ for _ in ()).throw(
            runtime.ModelInputError("network.trt is missing")
        ),
    )
    setup = runtime.RuntimeController().setup_snapshot()
    assert setup.model_status.ready is True
    assert setup.engine_status == runtime.SetupStatus(
        False,
        "Click Optimize Models to generate the GPU-specific TensorRT engines "
        "from the downloaded ONNX models",
    )
    assert setup.model_spec is not None


def test_setup_snapshot_explains_a_missing_selection(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _bpy = runtime_module
    monkeypatch.setattr(
        runtime,
        "get_preferences",
        lambda: SimpleNamespace(
            audio2face_model_directory="",
            audio2emotion_model_directory="",
        ),
    )

    setup = runtime.RuntimeController().setup_snapshot()

    assert setup.model_status == runtime.SetupStatus(
        False,
        "select the complete downloaded Audio2Face model folder in Add-on Preferences",
    )
    assert setup.model_spec is None


def test_setup_snapshot_reports_missing_bundled_runtime(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, _bpy = runtime_module

    setup = runtime.RuntimeController().setup_snapshot()

    assert setup.runtime_status.ready is False
    assert "bundled" in setup.runtime_status.message
    assert "runtime is missing" in setup.runtime_status.message


def test_optimization_eligibility_requires_nvidia_terms_acceptance(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = runtime_module[0]
    controller = runtime.RuntimeController()
    spec = SimpleNamespace()
    setup = runtime.RuntimeSetupSnapshot(
        runtime_status=runtime.SetupStatus(True, "bundled GPU runtime is valid"),
        model_status=runtime.SetupStatus(True, "selected models are valid"),
        engine_status=runtime.SetupStatus(False, "engines are not required"),
        model_spec=spec,
    )

    monkeypatch.setattr(
        runtime,
        "get_preferences",
        lambda: SimpleNamespace(nvidia_terms_accepted=False),
    )
    allowed, reason = controller.optimization_eligibility(setup)
    assert allowed is False
    assert "accept the NVIDIA terms" in reason

    monkeypatch.setattr(
        runtime,
        "get_preferences",
        lambda: SimpleNamespace(nvidia_terms_accepted=True),
    )
    assert controller.optimization_eligibility(setup) == (
        True,
        "The bundled GPU runtime and selected model inputs are ready",
    )


def test_exact_hello_contract_automatically_loads_bundled_model(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.pending["hello"] = _plain_pending(runtime, "hello", scene.name)
    loaded: list[object] = []
    controller.handshake_spec = "bundled-spec"
    controller.setup_snapshot = lambda: pytest.fail(
        "hello re-resolved preferences instead of using its retained specification"
    )

    def submit_model_load(
        target: object,
        spec: object,
    ) -> None:
        loaded.append((target, spec))

    controller._submit_model_load = submit_model_load

    controller._handle_response(
        {
            "id": "hello",
            "result": {
                "worker_profile": runtime.WORKER_PROFILE,
                "worker_version": "0.1.0",
            },
        }
    )

    assert controller.negotiated is True
    assert controller.handshake_deadline is None
    assert controller.handshake_spec is None
    assert loaded == [(scene, "bundled-spec")]
    assert settings.status != "ERROR"


def test_noncanonical_hello_is_rejected_and_stopped(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.pending["hello"] = _plain_pending(runtime, "hello", scene.name)
    controller.handshake_spec = "bundled-spec"
    stopped: list[float] = []
    controller.client.begin_shutdown = lambda *, timeout: stopped.append(timeout)

    controller._handle_response(
        {
            "id": "hello",
            "result": {
                "worker_profile": "another-worker/1",
                "worker_version": "0.1.0",
            },
        }
    )

    assert controller.negotiated is False
    assert controller.handshake_spec is None
    assert settings.status == "ERROR"
    assert stopped == [runtime.SHUTDOWN_TIMEOUT_SECONDS]


def test_hello_without_its_startup_spec_is_rejected(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.pending["hello"] = _plain_pending(runtime, "hello", scene.name)
    stopped: list[float] = []
    controller.client.begin_shutdown = lambda *, timeout: stopped.append(timeout)

    controller._handle_response(
        {
            "id": "hello",
            "result": {
                "worker_profile": runtime.WORKER_PROFILE,
                "worker_version": "0.1.0",
            },
        }
    )

    assert controller.negotiated is False
    assert settings.status == "ERROR"
    assert settings.status_message == "worker startup specification is unavailable"
    assert stopped == [runtime.SHUTDOWN_TIMEOUT_SECONDS]


def test_exact_model_response_marks_model_ready(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    signature = ("audio2face/model.json", "audio2emotion/model.json")
    controller.pending["load"] = _model_pending(runtime, scene.name, signature)

    controller._handle_response(
        {
            "id": "load",
            "result": {
                "model_schema": _model_schema(),
                "sample_rate": 16_000,
            },
        }
    )

    assert settings.status == "MODEL_READY"
    assert controller.loaded_signature == signature
    assert controller.model_sample_rate == 16_000
    assert controller.model_schema == _model_schema()


def test_loaded_model_schema_is_applied_when_each_scene_starts_a_stream(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, bpy = runtime_module
    first_scene, first_settings = _local_scene(bpy, "First")
    second_settings = _Settings()
    second_scene = _Scene("Second", editable=True, settings=second_settings)
    bpy.data.scenes.append(second_scene)  # type: ignore[attr-defined]
    controller = runtime.RuntimeController()
    signature = ("audio2face/model.json", "audio2emotion/model.json")
    model_schema = _model_schema()
    applications: list[object] = []

    def apply(
        settings: object,
        payload: object,
        applied_signature: tuple[str, str],
    ) -> None:
        assert applied_signature == signature
        assert payload == model_schema
        applications.append(settings)

    monkeypatch.setattr(runtime, "apply_model_schema", apply)
    controller.pending["load"] = _model_pending(
        runtime,
        first_scene.name,
        signature,
    )
    controller._handle_response(
        {
            "id": "load",
            "result": {
                "model_schema": model_schema,
                "sample_rate": 16_000,
            },
        }
    )

    controller._ensure_scene_model_schema(second_scene)
    controller._ensure_scene_model_schema(second_scene)

    assert applications == [first_settings, second_settings, second_settings]
    assert controller.model_schema is model_schema


def test_model_load_submits_both_bundled_models(
    runtime_module: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    spec = SimpleNamespace(
        audio2face_model=tmp_path / "audio2face" / "model.json",
        audio2emotion_model=tmp_path / "audio2emotion" / "model.json",
    )
    submitted: list[tuple[str, dict[str, object], dict[str, object]]] = []

    def request(
        _scene: object,
        method: str,
        params: dict[str, object],
        **kwargs: object,
    ) -> None:
        submitted.append((method, params, kwargs))

    controller._request = request
    controller._submit_model_load(
        scene,
        spec,
    )

    assert submitted == [
        (
            "load_model",
            {
                "audio2face_model_path": str(spec.audio2face_model),
                "audio2emotion_model_path": str(spec.audio2emotion_model),
            },
            {
                "model_signature": (
                    str(spec.audio2face_model),
                    str(spec.audio2emotion_model),
                ),
                "operation_id": None,
            },
        )
    ]


def test_model_response_rejects_an_unknown_field(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.pending["load"] = _model_pending(
        runtime,
        scene.name,
        ("audio2face/model.json", "audio2emotion/model.json"),
    )
    shutdown_timeouts: list[float] = []
    controller.client.begin_shutdown = lambda *, timeout: shutdown_timeouts.append(
        timeout
    )

    controller._handle_response(
        {
            "id": "load",
            "result": {
                "model_schema": _model_schema(),
                "sample_rate": 16_000,
                "unexpected": True,
            },
        }
    )

    assert settings.status == "ERROR"
    assert controller.rejected_reason == "worker returned a noncanonical model response"
    assert shutdown_timeouts == [runtime.SHUTDOWN_TIMEOUT_SECONDS]


def test_error_for_unknown_request_id_is_a_terminal_contract_violation(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    _scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    shutdown_timeouts: list[float] = []
    controller.client.begin_shutdown = lambda *, timeout: shutdown_timeouts.append(
        timeout
    )

    controller._handle_error(
        {
            "id": "unknown",
            "error": {"code": "invalid_params", "message": "bad request"},
        }
    )

    assert settings.status == "ERROR"
    assert controller.rejected_reason == (
        "worker returned an error for an unknown request ID"
    )
    assert shutdown_timeouts == [runtime.SHUTDOWN_TIMEOUT_SECONDS]


def test_idless_worker_error_is_a_terminal_diagnostic(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    _scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    shutdown_timeouts: list[float] = []
    controller.client.begin_shutdown = lambda *, timeout: shutdown_timeouts.append(
        timeout
    )

    controller._handle_error(
        {"error": {"code": "invalid_json", "message": "could not parse input"}}
    )

    assert settings.status == "ERROR"
    assert settings.status_message == "invalid_json: could not parse input"
    assert controller.rejected_reason == settings.status_message
    assert shutdown_timeouts == [runtime.SHUTDOWN_TIMEOUT_SECONDS]


def test_event_routing_rejects_an_unknown_operation_id(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.startup_scene = scene.name
    shutdown_timeouts: list[float] = []
    controller.client.begin_shutdown = lambda *, timeout: shutdown_timeouts.append(
        timeout
    )

    controller._handle_event(
        {
            "event": "stream_frame",
            "operation_id": "operation-1",
            "data": {
                "timestamp_sample": 0,
                "weights": [0.0] * len(MODEL_CHANNELS),
                "effective_emotions": MODEL_EMOTIONS.copy(),
            },
        }
    )

    assert settings.status == "ERROR"
    assert settings.status_message == (
        "worker returned an event for an unknown operation ID"
    )
    assert controller.rejected_reason == settings.status_message
    assert shutdown_timeouts == [runtime.SHUTDOWN_TIMEOUT_SECONDS]


def test_terminal_contract_rejection_clears_the_active_stream_from_all_scenes(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    first_settings = _local_scene(bpy, "First")[1]
    second_settings = _Settings()
    second = _Scene("Second", editable=True, settings=second_settings)
    bpy.data.scenes.append(second)  # type: ignore[attr-defined]
    controller = runtime.RuntimeController()
    stream = _activate_stream(
        runtime,
        controller,
        second,
        prebuffer_samples=60_000,
    )
    shutdown_timeouts: list[float] = []
    controller.client.begin_shutdown = lambda *, timeout: shutdown_timeouts.append(
        timeout
    )

    controller._reject_worker_contract("worker contract failed")

    assert first_settings.status == "ERROR"
    assert second_settings.status == "ERROR"
    assert controller.active_stream is None
    assert runtime._test_live_controller.stop_calls == [
        {"reset": False, "notify": False}
    ]
    assert shutdown_timeouts == [runtime.SHUTDOWN_TIMEOUT_SECONDS]


def test_rejected_worker_controls_cannot_revive_scene_state(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    settings = _local_scene(bpy)[1]
    controller = runtime.RuntimeController()
    controller.client.begin_shutdown = lambda *, timeout: None
    controller._reject_worker_contract("terminal contract failure")

    controller._handle_control(
        {
            "type": "event",
            "event": "stream_frame",
            "operation_id": "operation-1",
            "data": {
                "timestamp_sample": 0,
                "weights": [0.0] * len(MODEL_CHANNELS),
                "effective_emotions": MODEL_EMOTIONS.copy(),
            },
        }
    )

    assert settings.status == "ERROR"
    assert settings.status_message == "terminal contract failure"


def test_rejected_worker_exit_preserves_error_on_every_editable_scene(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    first, first_settings = _local_scene(bpy, "First")
    second_settings = _Settings()
    second = _Scene("Second", editable=True, settings=second_settings)
    bpy.data.scenes.append(second)  # type: ignore[attr-defined]
    controller = runtime.RuntimeController()
    controller.startup_scene = first.name
    controller.client.begin_shutdown = lambda *, timeout: None
    controller._reject_worker_contract("terminal contract failure")
    controller.client.tick = lambda: None
    controller.client.poll = lambda: [runtime.ProcessExited(0)]

    controller.poll()

    assert first_settings.status == "ERROR"
    assert second_settings.status == "ERROR"
    assert first_settings.status_message == "terminal contract failure"
    assert second_settings.status_message == "terminal contract failure"


def test_cancel_error_terminates_the_matching_stream(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = "STREAM_ENDING"
    controller = runtime.RuntimeController()
    _activate_stream(runtime, controller, scene)
    controller.pending["cancel"] = _stream_pending(
        runtime, "cancel", scene.name, "stream-1"
    )

    controller._handle_error(
        {"id": "cancel", "error": {"code": "busy", "message": "try again"}}
    )

    assert settings.status == "ERROR"
    assert settings.status_message == "busy: try again"
    assert controller.active_stream is None


def test_late_cancel_response_does_not_erase_terminal_error(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = "ERROR"
    settings.status_message = "terminal result"
    controller = runtime.RuntimeController()
    controller.pending["cancel"] = _stream_pending(
        runtime, "cancel", scene.name, "stream-1"
    )

    controller._handle_response(
        {"id": "cancel", "result": {}}
    )

    assert settings.status == "ERROR"
    assert settings.status_message == "terminal result"


def test_late_non_shutdown_response_does_not_replace_stopping_state(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = "STOPPING"
    controller = runtime.RuntimeController()
    controller.client._state = runtime.Lifecycle.STOPPING
    controller.pending["load"] = _model_pending(
        runtime,
        scene.name,
        ("audio2face/model.json", "audio2emotion/model.json"),
    )

    controller._handle_response(
        {
            "id": "load",
            "result": {
                "model_schema": _model_schema(),
                "sample_rate": 16_000,
            },
        }
    )

    assert settings.status == "STOPPING"


def test_public_runtime_guard_rejects_linked_scene_without_rna_write(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, _bpy = runtime_module
    linked = _Scene("Linked", editable=False, settings=_ReadOnlySettings())

    with pytest.raises(runtime.SidecarError, match="editable local"):
        runtime.RuntimeController._require_editable_scene(linked)


def test_deleted_stream_scene_is_canceled_and_drained_until_terminal_event(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.model_schema = _model_schema()
    stream = _activate_stream(
        runtime,
        controller,
        scene,
    )
    requests: list[tuple[str, dict[str, object]]] = []

    def request(method: str, params: dict[str, object]) -> str:
        assert controller.pending_lock.locked()
        requests.append((method, params))
        return "cancel-deleted-stream"

    controller.client.request = request
    bpy.data.scenes = _Scenes()  # type: ignore[attr-defined]
    bpy.context.scene = None  # type: ignore[attr-defined]

    controller._cancel_orphaned_operation()

    assert requests == [("cancel", {"operation_id": stream.operation_id})]
    assert stream.stop_requested is True
    assert controller.active_stream is stream
    assert controller.pending["cancel-deleted-stream"] == _stream_pending(
        runtime,
        "cancel",
        scene.name,
        stream.operation_id,
    )
    assert runtime._test_live_controller.stop_calls == [
        {"reset": False, "notify": False}
    ]

    controller._handle_response({"id": "cancel-deleted-stream", "result": {}})
    stream.chunk_credit.clear()
    controller._handle_event(
        {"event": "stream_credit", "operation_id": stream.operation_id, "data": {}}
    )
    controller._handle_event(
        {
            "event": "stream_frame",
            "operation_id": stream.operation_id,
            "data": {
                "timestamp_sample": 0,
                "weights": [0.0] * len(MODEL_CHANNELS),
                "effective_emotions": MODEL_EMOTIONS.copy(),
            },
        }
    )

    assert controller.active_stream is stream
    assert stream.chunk_credit.is_set() is False
    assert runtime._test_live_controller.receive_calls == []
    assert controller.rejected_reason is None

    controller._handle_event(
        {"event": "stream_ended", "operation_id": stream.operation_id, "data": {}}
    )

    assert controller.active_stream is None
    assert controller.pending == {}
    assert controller.rejected_reason is None
    assert runtime._test_live_controller.stop_calls == [
        {"reset": False, "notify": False}
    ]


def test_cancel_state_changes_only_after_the_request_is_submitted(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    track = _activate_track(runtime, controller, scene)
    stream = runtime.ActiveStream("stream-1", scene.name)

    def reject(_method: str, _params: dict[str, object]) -> str:
        raise OSError("transport failed")

    controller.client.request = reject
    with pytest.raises(OSError, match="transport failed"):
        controller._cancel_selected_track(track)
    with pytest.raises(OSError, match="transport failed"):
        controller._request_stream_cancel(stream)

    assert track.cancel_requested is False
    assert stream.stop_requested is False


def test_uneditable_stream_scene_releases_on_terminal_error(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    stream = _activate_stream(runtime, controller, scene)
    requests: list[tuple[str, dict[str, object]]] = []

    def request(method: str, params: dict[str, object]) -> str:
        assert controller.pending_lock.locked()
        requests.append((method, params))
        return "cancel-uneditable-stream"

    controller.client.request = request
    scene.is_editable = False

    controller._cancel_orphaned_operation()
    controller._handle_response({"id": "cancel-uneditable-stream", "result": {}})

    assert requests == [("cancel", {"operation_id": stream.operation_id})]
    assert controller.active_stream is stream
    controller._handle_event(
        {
            "event": "error",
            "operation_id": stream.operation_id,
            "data": {"code": "inference_failed", "message": "worker stopped"},
        }
    )

    assert controller.active_stream is None
    assert controller.pending == {}
    assert controller.rejected_reason is None


def test_deleted_stream_scene_cancel_not_found_waits_for_terminal_event(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    stream = _activate_stream(runtime, controller, scene)
    controller.client.request = lambda _method, _params: "cancel-missing-stream"
    bpy.data.scenes = _Scenes()  # type: ignore[attr-defined]
    bpy.context.scene = None  # type: ignore[attr-defined]

    controller._cancel_orphaned_operation()
    controller._handle_error(
        {
            "id": "cancel-missing-stream",
            "error": {
                "code": "operation_not_found",
                "message": "operation already ended",
            },
        }
    )

    assert controller.active_stream is stream
    assert controller.rejected_reason is None

    controller._handle_event(
        {"event": "stream_ended", "operation_id": stream.operation_id, "data": {}}
    )

    assert controller.active_stream is None
    assert controller.rejected_reason is None


def test_setting_edit_updates_an_active_stream_without_stalling_audio(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.input_mode = "STREAM"
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    stream = _activate_stream(
        runtime,
        controller,
        scene,
        prebuffer_samples=0,
    )
    requests: list[tuple[str, dict[str, object]]] = []
    controller.client.request = lambda method, params: (
        requests.append((method, params)) or f"request-{len(requests)}"
    )
    controller.refresh_inference_settings(scene)
    controller.queue_pcm_audio(struct.pack("<f", 0.5), scene_name=scene.name)
    controller._poll_pcm_ingress()

    assert requests[0] == (
        "stream_settings",
        {
            "operation_id": stream.operation_id,
            "settings": _inference_settings_payload(),
        },
    )
    assert [method for method, _params in requests] == [
        "stream_settings",
        "stream_chunk",
    ]
    controller._handle_response({"id": "request-1", "result": {}})

    assert controller.rejected_reason is None
    assert controller.active_stream is stream


def test_stream_start_submits_the_same_complete_inference_settings(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    controller.model_schema = _model_schema()
    controller.loaded_signature = ("face/model.json", "emotion/model.json")
    requests: list[tuple[str, dict[str, object], dict[str, object]]] = []

    def request(
        _scene: object,
        method: str,
        params: dict[str, object],
        **kwargs: object,
    ) -> None:
        requests.append((method, params, kwargs))

    controller._request = request
    controller._submit_stream_start(scene)

    assert len(requests) == 1
    assert controller.active_stream is not None
    operation_id = controller.active_stream.operation_id
    method, params, kwargs = requests[0]
    assert method == "stream_start"
    assert params == {
        "operation_id": operation_id,
        "sample_rate": 16_000,
        "settings": _inference_settings_payload(),
    }
    assert kwargs["operation_id"] == operation_id


def test_stream_to_selected_waits_for_stream_terminal_then_starts_track(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.input_mode = "STREAM"
    audio_path = tmp_path / "selected.wav"
    audio_path.write_bytes(b"fixture")
    settings.audio_path = str(audio_path)
    controller = runtime.RuntimeController()
    controller.client._state = runtime.Lifecycle.RUNNING
    controller.negotiated = True
    controller.model_sample_rate = 16_000
    controller.model_schema = _model_schema()
    controller.loaded_signature = ("face/model.json", "emotion/model.json")
    stream = _activate_stream(runtime, controller, scene)
    monkeypatch.setattr(runtime, "WavStreamSource", _FakeSelectedWavSource)
    requests: list[tuple[str, dict[str, object]]] = []
    controller.client.request = lambda method, params: (
        requests.append((method, params)) or f"request-{len(requests)}"
    )

    settings.input_mode = "SELECTED"
    controller.input_mode_changed(scene)

    assert requests == [("cancel", {"operation_id": stream.operation_id})]
    assert stream.stop_requested is True
    assert controller.selected_track is None
    assert runtime._test_live_controller.stop_calls == [
        {"reset": False, "notify": False}
    ]
    assert scene.frame_set_calls == []

    controller._handle_response({"id": "request-1", "result": {}})
    controller._handle_event(
        {"event": "stream_ended", "operation_id": stream.operation_id, "data": {}}
    )

    assert controller.active_stream is None
    assert controller.selected_track is not None
    assert controller.selected_track.path == audio_path
    assert [method for method, _params in requests] == ["cancel", "track_start"]
    assert all(
        call == {"reset": False, "notify": False}
        for call in runtime._test_live_controller.stop_calls
    )
    assert scene.frame_set_calls == []


def test_selected_to_stream_retains_first_pcm_until_track_terminal(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.audio_path = "/audio/selected.wav"
    controller = runtime.RuntimeController()
    controller.client._state = runtime.Lifecycle.RUNNING
    controller.negotiated = True
    controller.model_sample_rate = 16_000
    controller.model_schema = _model_schema()
    controller.loaded_signature = ("face/model.json", "emotion/model.json")
    track = _activate_track(runtime, controller, scene)
    requests: list[tuple[str, dict[str, object]]] = []
    controller.client.request = lambda method, params: (
        requests.append((method, params)) or f"request-{len(requests)}"
    )
    payload = struct.pack("<ff", -0.25, 0.75)

    settings.input_mode = "STREAM"
    controller.input_mode_changed(scene)
    controller.queue_pcm_audio(payload, scene_name=scene.name)
    controller._poll_pcm_ingress()

    assert requests == [("cancel", {"operation_id": track.operation_id})]
    assert controller.pcm_ingress is not None
    assert list(controller.pcm_ingress.chunks) == [payload]
    assert controller.active_stream is None

    controller._handle_response({"id": "request-1", "result": {}})
    controller._handle_event(
        {
            "event": "track_ended",
            "operation_id": track.operation_id,
            "data": {"reason": "canceled"},
        }
    )

    assert controller.selected_track is None
    assert controller.pcm_ingress is not None
    assert list(controller.pcm_ingress.chunks) == [payload]

    controller._poll_pcm_ingress()
    assert [method for method, _params in requests] == ["cancel", "stream_start"]
    controller._handle_response(
        {
            "id": "request-2",
            "result": {"sample_rate": 16_000, "prebuffer_samples": 0},
        }
    )
    controller._poll_pcm_ingress()

    assert [method for method, _params in requests] == [
        "cancel",
        "stream_start",
        "stream_chunk",
    ]
    assert base64.b64decode(
        requests[-1][1]["audio_f32le_base64"], validate=True
    ) == payload


def test_selected_track_uploads_once_then_starts_one_continuous_render(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, bpy = runtime_module
    controller, track, requests = _start_selected_track(
        runtime, bpy, monkeypatch, tmp_path, frame=25
    )
    source = track.wav_source

    assert (source.path, source.kwargs) == (
        tmp_path / "selected.wav",
        {"output_sample_rate": 16_000, "chunk_frames": 65_536},
    )
    assert source.advances == 0
    assert [method for method, _params in requests] == ["track_start"]
    assert bpy.data.scenes[0].frame_current == 25

    source.metadata.output_frames = 3
    controller._handle_response(
        {
            "id": "request-1",
            "result": {},
        }
    )
    controller._handle_response(
        {
            "id": "request-2",
            "result": {},
        }
    )
    controller._handle_response(
        {
            "id": "request-3",
            "result": {},
        }
    )

    assert [method for method, _params in requests] == [
        "track_start",
        "track_chunk",
        "track_chunk",
        "track_prepare",
    ]
    assert source.advances == 2
    assert [
        base64.b64decode(params["audio_f32le_base64"], validate=True)
        for method, params in requests
        if method == "track_chunk"
    ] == list(_SELECTED_AUDIO_CHUNKS)

    bpy.data.scenes[0].frame_current = 1
    controller._handle_response(
        {
            "id": "request-4",
            "result": {},
        }
    )

    assert track.prepared is True
    assert source.close_calls == 1
    assert requests[-1] == (
        "track_render",
        {
            "operation_id": track.operation_id,
            "revision": 1,
            "settings": _inference_settings_payload(),
            "preview_sample": 0,
        },
    )
    assert track.render_revision == 1
    assert track.stage == runtime.TrackRenderStage(1)


def test_setting_edits_supersede_render_revisions_and_ignore_stale_output(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.audio_path = "/audio/selected.wav"
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 48_000
    controller.model_schema = _model_schema()
    track = _activate_track(runtime, controller, scene)
    requests: list[tuple[str, dict[str, object]]] = []

    def evaluated_settings(value: object) -> dict[str, object]:
        payload = _inference_settings_payload()
        payload["audio2face"]["skin_strength"] = value.skin_strength
        return payload

    monkeypatch.setattr(runtime, "inference_settings", evaluated_settings)
    controller.client.request = lambda method, params: (
        requests.append((method, params)) or f"request-{len(requests)}"
    )

    controller.refresh_inference_settings(scene)
    settings.skin_strength = 0.25
    controller.refresh_inference_settings(scene)

    assert [method for method, _params in requests] == [
        "track_render",
        "track_render",
    ]
    assert [params["revision"] for _method, params in requests] == [1, 2]
    assert requests[1][1]["settings"]["audio2face"]["skin_strength"] == 0.25
    assert track.render_revision == 2
    assert track.stage == runtime.TrackRenderStage(2)

    controller._handle_event(
        {
            "event": "track_preview",
            "operation_id": track.operation_id,
            "data": {
                "revision": 1,
                "timestamp_sample": 0,
                "weights": [0.1] * len(MODEL_CHANNELS),
                "effective_emotions": MODEL_EMOTIONS.copy(),
            },
        }
    )
    controller._handle_event(
        {
            "event": "track_frame_batch",
            "operation_id": track.operation_id,
            "data": {
                "revision": 1,
                "offset": 0,
                "total_frames": 1,
                "timestamp_samples": [0],
                "weights": [[0.1] * len(MODEL_CHANNELS)],
                "effective_emotions": [MODEL_EMOTIONS.copy()],
            },
        }
    )
    controller._handle_response(
        {
            "id": "request-1",
            "result": {
                "revision": 1,
                "frame_count": 0,
                "superseded": True,
            },
        }
    )
    assert track.stage == runtime.TrackRenderStage(2)
    assert runtime._test_live_stream.model_frame_calls == []

    controller._handle_event(
        {
            "event": "track_preview",
            "operation_id": track.operation_id,
            "data": {
                "revision": 2,
                "timestamp_sample": 0,
                "weights": [0.2] * len(MODEL_CHANNELS),
                "effective_emotions": MODEL_EMOTIONS.copy(),
            },
        }
    )
    assert [call[3][0] for call in runtime._test_live_stream.model_frame_calls] == [
        0.2
    ]
    assert controller.selected_track is track
    assert runtime._test_live_controller.stop_calls == []


def test_reverting_to_published_settings_supersedes_inflight_render(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.audio_path = "/audio/selected.wav"
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 48_000
    controller.model_schema = _model_schema()
    track = _activate_track(runtime, controller, scene)

    def evaluated_settings(value: object) -> dict[str, object]:
        payload = _inference_settings_payload()
        payload["audio2face"]["skin_strength"] = value.skin_strength
        return payload

    monkeypatch.setattr(runtime, "inference_settings", evaluated_settings)
    track.published_settings = evaluated_settings(settings)
    requests: list[tuple[str, dict[str, object]]] = []
    controller.client.request = lambda method, params: (
        requests.append((method, params)) or f"request-{len(requests)}"
    )

    settings.skin_strength = 0.25
    controller.refresh_inference_settings(scene)
    settings.skin_strength = 1.0
    controller.refresh_inference_settings(scene)
    controller.refresh_inference_settings(scene)

    assert [params["revision"] for _method, params in requests] == [1, 2]
    assert requests[1][1]["settings"] == track.published_settings
    assert track.render_revision == 2
    assert track.render_settings == track.published_settings
    assert track.stage == runtime.TrackRenderStage(2)


def test_native_frame_changes_only_sample_the_published_cache(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.audio_path = "/audio/selected.wav"
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 48_000
    controller.model_schema = _model_schema()
    track = _activate_track(
        runtime,
        controller,
        scene,
        audio_samples=96_001,
    )
    track.published_settings = _inference_settings_payload()
    track.timestamps = (0, 48_000, 96_000)
    track.weights = tuple(
        (value,) * len(MODEL_CHANNELS) for value in (0.0, 1.0, 2.0)
    )
    track.effective_emotions = ((0.0,), (0.5,), (1.0,))
    requests: list[tuple[str, dict[str, object]]] = []
    controller.client.request = lambda method, params: (
        requests.append((method, params)) or f"request-{len(requests)}"
    )
    runtime._CONTROLLER = controller
    _activate_bake(runtime, controller, scene)

    scene.frame_current = 13
    runtime._frame_change_post_handler(scene)
    scene.frame_current = 37
    runtime._frame_change_post_handler(scene)

    assert requests == []
    assert controller.active_bake is not None
    assert [call[3][0] for call in runtime._test_live_stream.model_frame_calls] == [
        0.5,
        1.5,
    ]
    assert [call[4][0] for call in runtime._test_live_stream.model_frame_calls] == [
        0.25,
        0.75,
    ]


def test_bounded_track_frame_batches_publish_atomically_on_render_completion(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.audio_path = "/audio/selected.wav"
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 48_000
    controller.model_schema = _model_schema()
    track = _activate_track(runtime, controller, scene, audio_samples=96_001)
    old_weights = ((0.75,) * len(MODEL_CHANNELS),)
    track.timestamps = (0,)
    track.weights = old_weights
    track.effective_emotions = ((0.75,),)
    track.published_settings = {"old": True}
    requests: list[tuple[str, dict[str, object]]] = []
    controller.client.request = lambda method, params: (
        requests.append((method, params)) or f"request-{len(requests)}"
    )

    controller._request_track_render(scene, track, _inference_settings_payload())
    timestamps = [index * 1_000 for index in range(65)]
    rows = [
        [index / 100.0] * len(MODEL_CHANNELS) for index in range(65)
    ]
    emotions = [[index / 100.0] for index in range(65)]
    for offset, stop in ((0, 64), (64, 65)):
        controller._handle_event(
            {
                "event": "track_frame_batch",
                "operation_id": track.operation_id,
                "data": {
                    "revision": 1,
                    "offset": offset,
                    "total_frames": 65,
                    "timestamp_samples": timestamps[offset:stop],
                    "weights": rows[offset:stop],
                    "effective_emotions": emotions[offset:stop],
                },
            }
        )

    assert requests[0][0] == "track_render"
    assert track.timestamps == (0,)
    assert track.weights == old_weights
    assert track.stage is not None
    assert len(track.stage.timestamps) == 65

    controller._handle_response(
        {
            "id": "request-1",
            "result": {
                "revision": 1,
                "frame_count": 65,
                "superseded": False,
            },
        }
    )

    assert track.timestamps == tuple(timestamps)
    assert track.weights == tuple(tuple(row) for row in rows)
    assert track.effective_emotions == tuple(tuple(row) for row in emotions)
    assert track.published_settings == _inference_settings_payload()
    assert track.stage is None
    assert controller.rejected_reason is None
    assert settings.status == "MODEL_READY"


def test_native_frame_change_does_not_create_a_worker_track(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.audio_path = "/audio/selected.wav"
    controller = runtime.RuntimeController()

    def fail_if_called(_scene: object) -> None:
        raise AssertionError("native transport created a worker track")

    controller._ensure_selected_track = fail_if_called
    runtime._CONTROLLER = controller

    runtime._frame_change_post_handler(scene)

    assert controller.selected_track is None
    assert controller.pending == {}


def test_stream_audio_request_is_exact_base64_f32le(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    _activate_stream(runtime, controller, scene)
    requests: list[tuple[str, dict[str, object]]] = []

    def request(method: str, params: dict[str, object]) -> str:
        assert controller.pending_lock.locked()
        requests.append((method, params))
        return "chunk-request"

    controller.client.request = request
    payload = struct.pack("<fff", -0.5, 0.0, 0.75)

    controller._send_stream_audio(
        payload,
        operation_id="stream-1",
    )

    assert requests == [
        (
            "stream_chunk",
            {
                "operation_id": "stream-1",
                "audio_f32le_base64": base64.b64encode(payload).decode("ascii"),
            },
        )
    ]
    assert base64.b64decode(requests[0][1]["audio_f32le_base64"], validate=True) == payload
    assert controller.pending["chunk-request"] == _stream_pending(
        runtime, "stream_chunk", scene.name, "stream-1"
    )


def test_releasing_stream_removes_every_pending_operation_request(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    _activate_stream(runtime, controller, scene)
    controller.pending.update(
        {
            "chunk": _stream_pending(
                runtime, "stream_chunk", scene.name, "stream-1"
            ),
            "end": _stream_pending(
                runtime, "stream_end", scene.name, "stream-1"
            ),
            "cancel": _stream_pending(
                runtime, "cancel", scene.name, "stream-1"
            ),
            "other": _stream_pending(
                runtime, "stream_chunk", scene.name, "stream-2"
            ),
        }
    )

    controller._release_active_stream("stream-1")

    assert controller.active_stream is None
    assert controller.pending == {
        "other": _stream_pending(
            runtime, "stream_chunk", scene.name, "stream-2"
        )
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "must not be empty"),
        (b"\x00\x00\x00", "divisible by four"),
        (struct.pack("<f", float("nan")), "must be finite"),
        (struct.pack("<f", float("inf")), "must be finite"),
        (b"\x00" * (16_000 * 4 + 4), "model-rate second"),
        (b"\x00" * (256 * 1024 + 4), "exceeds"),
        (bytearray(b"\x00\x00\x00\x00"), "exact bytes"),
        (memoryview(b"\x00\x00\x00\x00"), "exact bytes"),
        ("not bytes", "exact bytes"),
    ],
)
def test_pcm_ingress_rejects_noncanonical_f32le(
    runtime_module: tuple[ModuleType, ModuleType],
    payload: object,
    message: str,
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000

    with pytest.raises(runtime.SidecarError, match=message):
        controller.queue_pcm_audio(payload, scene_name=scene.name)


def test_first_live_pcm_chunk_auto_starts_and_flushes_from_main_thread(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.input_mode = "STREAM"
    controller = runtime.RuntimeController()
    controller._require_worker_ready = lambda: None
    controller.model_sample_rate = 16_000
    controller.model_schema = _model_schema()
    controller.loaded_signature = ("face", "emotion")
    calls: list[tuple[str, dict[str, object], threading.Thread]] = []
    failures: list[BaseException] = []

    def request(method: str, params: dict[str, object]) -> str:
        calls.append((method, params, threading.current_thread()))
        return f"request-{len(calls)}"

    controller.client.request = request

    def push() -> None:
        try:
            controller.queue_pcm_audio(struct.pack("<f", 0.25), scene_name=scene.name)
        except BaseException as exc:  # pragma: no cover - asserted on main thread
            failures.append(exc)

    source_thread = threading.Thread(target=push, name="test-audio-source")
    source_thread.start()
    source_thread.join(timeout=2.0)

    assert not source_thread.is_alive()
    assert failures == []
    assert calls == []
    assert controller.pcm_ingress is not None
    assert list(controller.pcm_ingress.chunks) == [struct.pack("<f", 0.25)]

    controller._poll_pcm_ingress()

    assert len(calls) == 1
    assert calls[0][0] == "stream_start"
    assert calls[0][2] is threading.current_thread()
    operation_id = calls[0][1]["operation_id"]
    assert isinstance(operation_id, str)
    assert controller.active_stream is not None
    assert controller.active_stream.operation_id == operation_id

    controller._handle_response(
        {
            "id": "request-1",
            "result": {"sample_rate": 16_000, "prebuffer_samples": 32_000},
        }
    )
    controller._poll_pcm_ingress()

    assert [method for method, _params, _thread in calls] == [
        "stream_start",
        "stream_chunk",
    ]
    assert base64.b64decode(
        calls[1][1]["audio_f32le_base64"], validate=True
    ) == struct.pack("<f", 0.25)
    assert controller.pending["request-2"] == _stream_pending(
        runtime, "stream_chunk", scene.name, operation_id
    )


def test_stream_audio_ingress_has_bounded_pending_backpressure(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    _activate_stream(runtime, controller, scene)
    controller.pending.update(
        {
            f"chunk-{index}": _stream_pending(
                runtime, "stream_chunk", scene.name, "stream-1"
            )
            for index in range(runtime.MAX_PENDING_STREAM_CHUNKS)
        }
    )
    controller.client.request = lambda *_args, **_kwargs: pytest.fail(
        "a full source queue reached the sidecar"
    )

    with pytest.raises(runtime.SidecarError, match="queue is full"):
        controller.queue_pcm_audio(struct.pack("<f", 0.0), scene_name=scene.name)


def test_exact_stream_start_response_marks_stream_ready(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = "STREAM_STARTING"
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    stream = _activate_stream(runtime, controller, scene)
    controller.pending["start"] = _stream_pending(
        runtime,
        "stream_start",
        scene.name,
        "stream-1",
    )

    assert controller.pcm_stream_requirements(scene) == (16_000, None)

    controller._handle_response(
        {
            "id": "start",
            "result": {"sample_rate": 16_000, "prebuffer_samples": 0},
        }
    )

    assert settings.status == "STREAMING"
    assert settings.status_message == "PCM stream is ready"
    assert stream.prebuffer_samples == 0
    assert controller.pcm_stream_requirements(scene) == (16_000, 0)


def test_pcm_stream_requirements_need_a_loaded_worker_model(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)

    with pytest.raises(runtime.SidecarError, match="start the Audio2Face worker"):
        runtime.RuntimeController().pcm_stream_requirements(scene)


def test_live_pcm_waits_for_the_previous_worker_chunk_credit(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.input_mode = "STREAM"
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    _activate_stream(runtime, controller, scene, prebuffer_samples=0)
    first = struct.pack("<f", 0.1)
    second = struct.pack("<f", 0.2)
    controller.queue_pcm_audio(first, scene_name=scene.name)
    controller.queue_pcm_audio(second, scene_name=scene.name)
    requests: list[tuple[str, dict[str, object]]] = []

    def request(method: str, params: dict[str, object]) -> str:
        requests.append((method, params))
        return f"request-{len(requests)}"

    controller.client.request = request
    controller._poll_pcm_ingress()
    controller._poll_pcm_ingress()

    assert [method for method, _params in requests] == ["stream_chunk"]
    assert controller.pcm_ingress is not None
    assert list(controller.pcm_ingress.chunks) == [second]

    controller._handle_response({"id": "request-1", "result": {}})
    controller._handle_event(
        {"event": "stream_credit", "operation_id": "stream-1", "data": {}}
    )
    controller._poll_pcm_ingress()

    assert [method for method, _params in requests] == [
        "stream_chunk",
        "stream_chunk",
    ]
    assert controller.pcm_ingress is not None
    assert list(controller.pcm_ingress.chunks) == []


def test_stream_start_response_rejects_noninteger_sample_rate(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    _activate_stream(runtime, controller, scene)
    controller.pending["start"] = _stream_pending(
        runtime,
        "stream_start",
        scene.name,
        "stream-1",
    )

    controller._handle_response(
        {
            "id": "start",
            "result": {"sample_rate": 16_000.0, "prebuffer_samples": 0},
        }
    )

    assert settings.status == "ERROR"
    assert controller.active_stream is None
    assert runtime._test_live_controller.stop_calls == [
        {"reset": False, "notify": False}
    ]


def test_exact_stream_frame_routes_negative_timestamp_and_arkit52(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    _activate_stream(runtime, controller, scene)
    weights = [0.25] * len(MODEL_CHANNELS)

    controller._handle_event(
        {
            "event": "stream_frame",
            "operation_id": "stream-1",
            "data": {
                "timestamp_sample": -320,
                "weights": weights,
                "effective_emotions": MODEL_EMOTIONS.copy(),
            },
        }
    )

    assert runtime._test_live_controller.receive_calls == [
        ("stream-1", -320, weights, MODEL_EMOTIONS)
    ]
    assert settings.status == "STREAMING"


def test_late_frame_from_a_canceling_stream_is_drained_without_delivery(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = "STREAM_ENDING"
    controller = runtime.RuntimeController()
    controller.model_schema = _model_schema()
    stream = _activate_stream(
        runtime,
        controller,
        scene,
        stop_requested=True,
    )

    controller._handle_event(
        {
            "event": "stream_frame",
            "operation_id": "stream-1",
            "data": {
                "timestamp_sample": 0,
                "weights": [0.0] * len(MODEL_CHANNELS),
                "effective_emotions": MODEL_EMOTIONS.copy(),
            },
        }
    )

    assert runtime._test_live_controller.receive_calls == []
    assert controller.rejected_reason is None
    assert controller.active_stream is stream


def test_inflight_chunk_error_during_cancel_waits_for_terminal_event(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = "STREAM_ENDING"
    controller = runtime.RuntimeController()
    stream = _activate_stream(
        runtime,
        controller,
        scene,
        stop_requested=True,
    )
    controller.pending["chunk"] = _stream_pending(
        runtime,
        "stream_chunk",
        scene.name,
        "stream-1",
    )

    controller._handle_error(
        {
            "id": "chunk",
            "error": {"code": "operation_not_found", "message": "stream canceled"},
        }
    )

    assert controller.rejected_reason is None
    assert controller.active_stream is stream
    assert settings.status == "STREAM_ENDING"


def test_cancel_racing_natural_completion_drains_terminal(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = "STREAM_ENDING"
    controller = runtime.RuntimeController()
    stream = _activate_stream(
        runtime,
        controller,
        scene,
        stop_requested=True,
    )
    controller.pending["cancel"] = _stream_pending(
        runtime,
        "cancel",
        scene.name,
        "stream-1",
    )

    controller._handle_error(
        {
            "id": "cancel",
            "error": {
                "code": "operation_not_found",
                "message": "operation already ended",
            },
        }
    )
    assert controller.active_stream is stream

    controller._handle_event(
        {"event": "stream_ended", "operation_id": "stream-1", "data": {}}
    )

    assert controller.rejected_reason is None
    assert controller.active_stream is None


def test_malformed_stream_frame_terminates_the_active_stream(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    stream = _activate_stream(
        runtime,
        controller,
        scene,
    )
    shutdown_timeouts: list[float] = []
    controller.client.begin_shutdown = lambda *, timeout: shutdown_timeouts.append(
        timeout
    )

    controller._handle_event(
        {
            "event": "stream_frame",
            "operation_id": "stream-1",
            "data": {
                "timestamp_sample": 0,
                "weights": [0.0] * len(MODEL_CHANNELS),
                "effective_emotions": MODEL_EMOTIONS.copy(),
                "unexpected": True,
            },
        }
    )

    assert settings.status == "ERROR"
    assert controller.active_stream is None
    assert runtime._test_live_controller.stop_calls == [
        {"reset": False, "notify": False}
    ]
    assert shutdown_timeouts == [runtime.SHUTDOWN_TIMEOUT_SECONDS]


def test_stream_tail_keeps_ui_state_until_presentation_stops(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    stream = _activate_stream(
        runtime,
        controller,
        scene,
    )

    controller._handle_event(
        {"event": "stream_ended", "operation_id": "stream-1", "data": {}}
    )

    assert controller.active_stream is stream
    assert stream.worker_ended is True
    assert settings.status == "STREAMING"
    assert "Finishing buffered" in settings.status_message

    runtime._test_live_controller.is_active = False
    controller._finish_stream_presentation(
        scene.name,
        "stream-1",
    )

    assert controller.active_stream is None
    assert settings.status == "MODEL_READY"


def test_stream_request_error_preserves_failure_through_terminal_event(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = "STREAMING"
    controller = runtime.RuntimeController()
    _activate_stream(runtime, controller, scene)
    controller.pending["chunk"] = _stream_pending(
        runtime,
        "stream_chunk",
        scene.name,
        "stream-1",
    )
    requests: list[tuple[str, dict[str, object]]] = []

    def request(method: str, params: dict[str, object]) -> str:
        requests.append((method, params))
        return "cancel-stream"

    controller.client.request = request
    controller._handle_error(
        {
            "id": "chunk",
            "error": {
                "code": "backpressure",
                "message": "queue full",
            },
        }
    )

    assert requests == [("cancel", {"operation_id": "stream-1"})]
    assert controller.pending["cancel-stream"].operation_id == "stream-1"
    assert settings.status == "ERROR"
    assert settings.status_message == "backpressure: queue full"

    controller._handle_response({"id": "cancel-stream", "result": {}})
    controller._handle_event(
        {"event": "stream_ended", "operation_id": "stream-1", "data": {}}
    )

    assert controller.rejected_reason is None
    assert controller.active_stream is None
    assert settings.status == "ERROR"
    assert settings.status_message == "backpressure: queue full"
    assert runtime._test_live_controller.terminal_calls == []


def test_stop_worker_drains_active_stream_terminal_without_presentation_completion(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = "STREAMING"
    controller = runtime.RuntimeController()
    _activate_stream(
        runtime,
        controller,
        scene,
    )
    controller.client._state = runtime.Lifecycle.RUNNING
    controller.client.begin_shutdown = lambda *, timeout: "shutdown"
    runtime._test_live_controller.is_active = True

    controller.stop(scene)

    assert settings.status == "STOPPING"
    controller._handle_event(
        {"event": "stream_ended", "operation_id": "stream-1", "data": {}}
    )

    assert controller.rejected_reason is None
    assert controller.active_stream is None
    assert settings.status == "STOPPING"
    assert runtime._test_live_controller.terminal_calls == []


@pytest.mark.parametrize("response_kind", ["error", "noncanonical"])
def test_rejected_stream_cancel_cleans_local_state(
    runtime_module: tuple[ModuleType, ModuleType],
    response_kind: str,
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = "STREAM_ENDING"
    controller = runtime.RuntimeController()
    _activate_stream(runtime, controller, scene)
    controller.pending["cancel"] = _stream_pending(
        runtime,
        "cancel",
        scene.name,
        "stream-1",
    )

    if response_kind == "error":
        controller._handle_error(
            {
                "id": "cancel",
                "error": {"code": "operation_not_found", "message": "already ended"},
            }
        )
    else:
        controller._handle_response(
            {"id": "cancel", "result": {"unexpected": True}}
        )

    assert controller.active_stream is None
    assert settings.status == "ERROR"


def test_canceled_stream_ends_local_audio_instead_of_finishing_playback(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    _activate_stream(
        runtime,
        controller,
        scene,
        stop_requested=True,
    )

    controller._handle_event(
        {"event": "stream_ended", "operation_id": "stream-1", "data": {}}
    )

    assert runtime._test_live_controller.stop_calls == [
        {"reset": False, "notify": False}
    ]
    assert runtime._test_live_controller.terminal_calls == []
    assert controller.active_stream is None
    assert settings.status == "MODEL_READY"


def test_malformed_stream_ended_still_cleans_the_terminal_stream(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    stream = _activate_stream(
        runtime,
        controller,
        scene,
    )

    controller._handle_event(
        {
            "event": "stream_ended",
            "operation_id": "stream-1",
            "data": {"unexpected": True},
        }
    )

    assert settings.status == "ERROR"
    assert controller.active_stream is None
    assert runtime._test_live_controller.stop_calls == [
        {"reset": False, "notify": False}
    ]


def test_malformed_error_event_cleans_its_active_stream(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    stream = _activate_stream(
        runtime,
        controller,
        scene,
    )

    controller._handle_event(
        {
            "event": "error",
            "operation_id": "stream-1",
            "data": {"code": "inference_failed", "message": 42},
        }
    )

    assert settings.status == "ERROR"
    assert controller.active_stream is None
    assert runtime._test_live_controller.stop_calls == [
        {"reset": False, "notify": False}
    ]


def test_bake_waits_for_one_continuous_render_without_reuploading_audio(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    audio_path = tmp_path / "selected.wav"
    audio_path.write_bytes(b"fixture")
    settings.audio_path = str(audio_path)
    settings.target_objects = [SimpleNamespace(object=object())]
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 48_000
    controller.model_schema = _model_schema()
    controller.loaded_signature = ("face/model.json", "emotion/model.json")
    track = _activate_track(runtime, controller, scene)
    plan = object()
    monkeypatch.setattr(controller, "_require_worker_ready", lambda: None)
    monkeypatch.setattr(runtime, "plan_bake_targets", lambda *_args: (plan,))
    monkeypatch.setattr(
        runtime,
        "configure_selected_audio",
        lambda *_args, **_kwargs: (1, 2),
    )
    requests: list[tuple[str, dict[str, object]]] = []
    controller.client.request = lambda method, params: (
        requests.append((method, params)) or "request-1"
    )

    controller.bake_selected_audio(scene)

    assert controller.selected_track is track
    assert controller.active_bake is not None
    assert controller.active_bake.targets == (plan,)
    assert requests == [
        (
            "track_render",
            {
                "operation_id": track.operation_id,
                "revision": 1,
                "settings": _inference_settings_payload(),
                "preview_sample": 0,
            },
        )
    ]
    assert track.stage == runtime.TrackRenderStage(1)
    assert track.wav_source.close_calls == 0
    assert runtime._test_live_controller.stop_calls == []


def test_bake_samples_the_exact_published_cache_on_blender_frames(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    audio_path = tmp_path / "selected.wav"
    audio_path.write_bytes(b"fixture")
    settings.audio_path = str(audio_path)
    target = object()
    settings.target_objects = [SimpleNamespace(object=target)]
    scene.frame_current = 6
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 48_000
    controller.model_schema = _model_schema()
    controller.loaded_signature = ("face/model.json", "emotion/model.json")
    track = _activate_track(
        runtime,
        controller,
        scene,
        audio_samples=6_001,
    )
    _install_selected_audio_span(scene, 10, 12)
    track.timestamps = (0, 2_000, 4_000)
    track.weights = tuple(
        (value,) * len(MODEL_CHANNELS) for value in (0.1, 0.2, 0.3)
    )
    track.effective_emotions = ((0.1,), (0.2,), (0.3,))
    track.published_settings = _inference_settings_payload()
    plan = object()
    monkeypatch.setattr(controller, "_require_worker_ready", lambda: None)
    monkeypatch.setattr(runtime, "plan_bake_targets", lambda *_args: (plan,))
    monkeypatch.setattr(
        runtime,
        "configure_selected_audio",
        lambda *_args, **_kwargs: (10, 12),
    )
    requests: list[tuple[str, dict[str, object]]] = []
    controller.client.request = lambda method, params: (
        requests.append((method, params)) or f"request-{len(requests)}"
    )
    calls: list[tuple[object, ...]] = []

    def build_actions(*args: object) -> tuple[object, ...]:
        calls.append(args)
        return (object(),)

    monkeypatch.setattr(runtime, "bake_shape_key_actions", build_actions)

    controller.bake_selected_audio(scene)

    assert requests == []
    assert len(calls) == 1
    assert tuple(calls[0][0]) == (10, 11, 12)
    assert calls[0][1] == track.weights
    assert calls[0][2] == (plan,)
    assert calls[0][3] is bpy.data.actions
    assert controller.active_bake is None
    assert controller.selected_track is track
    assert scene.frame_current == 6
    assert scene.frame_set_calls == []
    assert track.wav_source.close_calls == 0
    assert settings.status == "MODEL_READY"


def test_runtime_registration_does_not_access_restricted_blend_data(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    unrestricted_data = bpy.data
    bpy.data = object()  # type: ignore[attr-defined]

    try:
        runtime.register_runtime()
        assert bpy.app.timers.is_registered(runtime._timer_callback)
        assert runtime._load_pre_handler in bpy.app.handlers.load_pre
        assert runtime._load_post_handler in bpy.app.handlers.load_post
        assert bpy.app.handlers.animation_playback_pre == []
        assert bpy.app.handlers.animation_playback_post == []
        assert runtime._frame_change_post_handler in bpy.app.handlers.frame_change_post
    finally:
        bpy.data = unrestricted_data  # type: ignore[attr-defined]
        runtime.unregister_runtime()
