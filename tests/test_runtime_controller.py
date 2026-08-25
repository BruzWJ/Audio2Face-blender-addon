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
            "mode": "automatic",
            "emotion_strength": 0.6,
            "emotion_contrast": 1.0,
            "max_emotions": 6,
            "live_blend_coef": 0.7,
            "transition_smoothing": 0.5,
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
        self.playback_loop = False
        self.playback_state = "PLAYING"
        self.prediction_delay = 0.0
        self.audio_path = ""
        self.stream_time = 0.0
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

    def as_pointer(self) -> int:
        return id(self)


class _Scenes(list[_Scene]):
    def get(self, name: str | None) -> _Scene | None:
        return next((scene for scene in self if scene.name == name), None)


class _LiveController:
    def __init__(self) -> None:
        self.operation_id: str | None = None
        self.is_active = False
        self.receive_calls: list[tuple[object, ...]] = []
        self.reset_calls: list[str] = []
        self.terminal_calls: list[str] = []
        self.stop_calls: list[dict[str, object]] = []
        self.seek_stop_calls: list[tuple[float, bool]] = []
        self.prepare_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def tick(self) -> bool:
        return False

    @property
    def active(self) -> bool:
        return self.is_active

    @property
    def can_seek(self) -> bool:
        return self.is_active

    def stop(self, **kwargs: object) -> None:
        self.stop_calls.append(dict(kwargs))
        self.is_active = False

    def stop_for_seek(self, position: float, *, paused: bool) -> None:
        self.seek_stop_calls.append((position, paused))
        self.is_active = False

    def receive(self, *args: object) -> None:
        self.receive_calls.append(args)

    def reset_frames(self, operation_id: str) -> None:
        self.reset_calls.append(operation_id)

    def prepare(self, *args: object, **kwargs: object) -> None:
        self.prepare_calls.append((args, dict(kwargs)))
        self.operation_id = str(args[1])
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
    bpy.data = SimpleNamespace(scenes=_Scenes())  # type: ignore[attr-defined]
    bpy.context = SimpleNamespace(scene=None)  # type: ignore[attr-defined]
    bpy.path = SimpleNamespace(abspath=lambda value: value)  # type: ignore[attr-defined]
    bpy.utils = SimpleNamespace(  # type: ignore[attr-defined]
        extension_path_user=lambda *_args, **_kwargs: "/tmp/a2f-runtime-test"
    )
    bpy.app = SimpleNamespace(  # type: ignore[attr-defined]
        online_access=True,
        timers=_Timers(),
        handlers=SimpleNamespace(
            persistent=lambda callback: callback,
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
    live_stream.clear_playback_position = lambda _settings: None  # type: ignore[attr-defined]
    live_stream.get_live_stream_controller = lambda: live_controller  # type: ignore[attr-defined]
    live_stream.unregister_live_stream = lambda: None  # type: ignore[attr-defined]
    live_stream.validate_stream_frame = lambda *_args: ()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, live_stream.__name__, live_stream)

    module_name = "audio2face._runtime_controller_test"
    source = Path(__file__).resolve().parents[1] / "audio2face" / "runtime.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    module._test_live_controller = live_controller  # type: ignore[attr-defined]
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
    audio_path: Path | None = None,
    start_position: float = 0.0,
    timestamp_offset: int | None = None,
    prebuffer_samples: int | None = None,
    stop_requested: bool = False,
    worker_ended: bool = False,
) -> object:
    wav_source = (
        runtime.SelectedWavSource(
            audio_path=audio_path,
            start_position=start_position,
            cancel=threading.Event(),
            playing=threading.Event(),
            playback_started=threading.Event(),
            timestamp_offset=timestamp_offset,
        )
        if audio_path is not None
        else None
    )
    stream = runtime.ActiveStream(
        operation_id=operation_id,
        scene_name=scene.name,
        wav_source=wav_source,
        prebuffer_samples=prebuffer_samples,
        stop_requested=stop_requested,
        worker_ended=worker_ended,
    )
    if wav_source is not None:
        wav_source.playing.set()
    stream.chunk_credit.set()
    controller.active_stream = stream
    runtime._test_live_controller.operation_id = operation_id
    runtime._test_live_controller.is_active = True
    return stream


def test_short_lived_status_notice_never_becomes_visible(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    now = [10.0]
    redraws: list[None] = []
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        controller,
        "_tag_runtime_setup_redraw",
        lambda: redraws.append(None),
    )

    controller._set_status(scene, "STREAM_STARTING", "Preparing audio inference")
    assert controller.status_notice(scene) is None

    now[0] += runtime._STATUS_NOTICE_DELAY_SECONDS / 2.0
    controller._set_status(scene, "STREAMING", "PCM stream is ready")
    redraws.clear()
    now[0] += runtime._STATUS_NOTICE_DELAY_SECONDS
    controller._poll_status_notices()

    assert controller.status_notice(scene) is None
    assert redraws == []


def test_status_notice_appears_once_after_persistence_delay(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    now = [20.0]
    redraws: list[None] = []
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        controller,
        "_tag_runtime_setup_redraw",
        lambda: redraws.append(None),
    )

    controller._set_status(scene, "LOADING_MODEL", "Loading models")
    redraws.clear()
    now[0] += runtime._STATUS_NOTICE_DELAY_SECONDS - 0.001
    controller._poll_status_notices()
    assert controller.status_notice(scene) is None
    assert redraws == []

    now[0] += 0.001
    controller._poll_status_notices()
    assert controller.status_notice(scene) == ("LOADING_MODEL", "Loading models")
    assert redraws == [None]

    controller._poll_status_notices()
    assert redraws == [None]


def test_status_message_change_preserves_notice_persistence(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    now = [30.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])

    controller._set_status(scene, "STARTING", "Starting worker")
    now[0] += runtime._STATUS_NOTICE_DELAY_SECONDS / 2.0
    controller._set_status(scene, "STARTING", "Loading playback data")
    now[0] += runtime._STATUS_NOTICE_DELAY_SECONDS / 2.0
    controller._poll_status_notices()

    assert controller.status_notice(scene) == (
        "STARTING",
        "Loading playback data",
    )

    controller._set_status(scene, "STARTING", "Loading audio")
    assert controller.status_notice(scene) == ("STARTING", "Loading audio")


def test_error_status_notice_is_immediate_without_tracking_state(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()

    settings.status = "ERROR"
    settings.status_message = "Audio device failed"

    assert controller.status_notice(scene) == ("ERROR", "Audio device failed")


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
        audio_path=Path("voice.wav"),
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
    assert stream.wav_source is not None and stream.wav_source.cancel.is_set()
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


def test_selected_wav_play_starts_streaming_inference(
    runtime_module: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    audio_path = tmp_path / "submitted.wav"
    audio_path.write_bytes(b"RIFF")
    settings.audio_path = str(audio_path)

    controller = runtime.RuntimeController()
    controller._require_operation_idle = lambda: None
    controller._require_worker_ready = lambda: None
    spec = SimpleNamespace(
        audio2face_model=tmp_path / "audio2face" / "model.json",
        audio2emotion_model=tmp_path / "audio2emotion" / "model.json",
    )
    controller.setup_snapshot = lambda: SimpleNamespace(
        require_inference_spec=lambda: spec
    )
    signature = (
        str(spec.audio2face_model),
        str(spec.audio2emotion_model),
    )
    controller.loaded_signature = signature
    controller.model_sample_rate = 16_000
    controller.model_schema = _model_schema()
    requests: list[tuple[str, dict[str, object]]] = []

    def request(
        _scene: object,
        method: str,
        params: dict[str, object],
        **_kwargs: object,
    ) -> None:
        requests.append((method, params))

    controller._request = request
    controller.start_selected_audio(scene)

    assert controller.active_stream is not None
    operation_id = controller.active_stream.operation_id
    assert requests[0][0] == "stream_start"
    assert set(requests[0][1]) == {"operation_id", "sample_rate", "settings"}
    assert requests[0][1]["operation_id"] == operation_id
    assert requests[0][1]["sample_rate"] == 16_000
    assert requests[0][1]["settings"] == _inference_settings_payload()
    assert controller.active_stream.operation_id == operation_id
    assert controller.active_stream.wav_source is not None
    assert controller.active_stream.wav_source.audio_path == audio_path.resolve()
    assert runtime._test_live_controller.prepare_calls[0][0][4] == ("Joy",)
    assert settings.status == "STREAM_STARTING"


def test_selected_restart_waits_for_pending_model_load(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    restart = (scene.name, 1.25, True)
    controller.selected_restart = restart
    controller.pending["load"] = _model_pending(
        runtime,
        scene.name,
        ("audio2face/model.json", "audio2emotion/model.json"),
    )
    starts: list[tuple[object, float, bool]] = []
    controller.start_selected_audio = (
        lambda target, *, position, paused: starts.append(
            (target, position, paused)
        )
    )

    controller._poll_selected_restart()

    assert controller.selected_restart == restart
    assert starts == []


def test_selected_audio_seek_preserves_ui_and_cancels_only_the_stream(
    runtime_module: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.playback_state = "PAUSED"
    controller = runtime.RuntimeController()
    stream = _activate_stream(
        runtime,
        controller,
        scene,
        audio_path=tmp_path / "voice.wav",
    )
    requests: list[tuple[str, dict[str, object]]] = []

    def request(
        _scene: object,
        method: str,
        params: dict[str, object],
        **_kwargs: object,
    ) -> None:
        requests.append((method, params))

    controller._request = request

    controller.seek_selected_audio(scene, 1.25, paused=True)

    assert runtime._test_live_controller.seek_stop_calls == [(1.25, True)]
    assert controller.selected_restart == (scene.name, 1.25, True)
    assert stream.stop_requested is True
    assert requests == [("cancel", {"operation_id": "stream-1"})]
    assert settings.status == "STREAM_ENDING"


def test_selected_audio_seek_failure_does_not_leave_a_restart_armed(
    runtime_module: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    stream = _activate_stream(
        runtime,
        controller,
        scene,
        audio_path=tmp_path / "voice.wav",
    )
    requests: list[tuple[str, dict[str, object]]] = []
    controller._request = lambda _scene, method, params, **_kwargs: requests.append(
        (method, params)
    )

    def reject_seek(_position: float, *, paused: bool) -> None:
        assert paused is True
        raise runtime.LiveStreamError("audio device rejected seek")

    runtime._test_live_controller.stop_for_seek = reject_seek

    with pytest.raises(runtime.SidecarError, match="audio device rejected seek"):
        controller.seek_selected_audio(scene, 1.25, paused=True)

    assert requests == [("cancel", {"operation_id": "stream-1"})]
    assert stream.stop_requested is True
    assert controller.selected_restart is None
    assert runtime._test_live_controller.stop_calls == []


def test_selected_to_stream_detaches_wav_then_uses_queued_pcm(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.input_mode = "STREAM"
    settings.status = "STREAMING"
    controller = runtime.RuntimeController()
    controller._require_worker_ready = lambda: None
    controller.client._state = runtime.Lifecycle.RUNNING
    spec = SimpleNamespace(
        audio2face_model=Path("/models/audio2face/model.json"),
        audio2emotion_model=Path("/models/audio2emotion/model.json"),
    )
    signature = controller._model_signature(spec)
    controller.loaded_signature = signature
    controller.model_sample_rate = 16_000
    controller.model_schema = _model_schema()
    controller.setup_snapshot = lambda: SimpleNamespace(
        require_inference_spec=lambda: spec
    )
    selected = _activate_stream(
        runtime,
        controller,
        scene,
        audio_path=Path("voice.wav"),
    )
    requests: list[tuple[str, dict[str, object]]] = []

    def request(method: str, params: dict[str, object]) -> str:
        requests.append((method, params))
        return f"request-{len(requests)}"

    controller.client.request = request
    controller._reconcile_input_media()

    assert requests == [("cancel", {"operation_id": "stream-1"})]
    assert selected.wav_source is not None and selected.wav_source.cancel.is_set()
    assert selected.stop_requested is True
    assert controller.active_stream is selected
    assert runtime._test_live_controller.stop_calls == [
        {"reset": False, "notify": False}
    ]
    assert controller.loaded_signature == signature
    assert controller.model_sample_rate == 16_000
    assert controller.model_schema == _model_schema()

    payload = struct.pack("<f", 0.25)
    controller.queue_pcm_audio(payload, scene_name=scene.name)
    controller._poll_pcm_ingress()
    assert list(controller.pcm_ingress.chunks) == [payload]
    assert len(requests) == 1

    controller._handle_response({"id": "request-1", "result": {}})
    controller._handle_event(
        {"event": "stream_ended", "operation_id": "stream-1", "data": {}}
    )
    controller._poll_pcm_ingress()

    assert [method for method, _params in requests] == ["cancel", "stream_start"]
    assert controller.active_stream is not None
    assert controller.active_stream.wav_source is None
    assert controller.pcm_ingress is not None
    assert list(controller.pcm_ingress.chunks) == [payload]
    assert controller.loaded_signature == signature


def test_stream_to_selected_detaches_pcm_without_stopping_worker(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.audio_path = "voice.wav"
    settings.status = "STREAMING"
    controller = runtime.RuntimeController()
    controller.client._state = runtime.Lifecycle.RUNNING
    controller.loaded_signature = ("face", "emotion")
    controller.model_sample_rate = 16_000
    controller.model_schema = _model_schema()
    stream = _activate_stream(runtime, controller, scene)
    controller.queue_pcm_audio(struct.pack("<f", 0.25), scene_name=scene.name)
    requests: list[tuple[str, dict[str, object]]] = []
    controller.client.request = lambda method, params: (
        requests.append((method, params)) or "cancel-request"
    )

    controller._reconcile_input_media()
    controller._poll_pcm_ingress()

    assert requests == [("cancel", {"operation_id": "stream-1"})]
    assert stream.stop_requested is True
    assert controller.active_stream is stream
    assert controller.pcm_ingress is None
    assert settings.audio_path == "voice.wav"
    assert controller.client.state == runtime.Lifecycle.RUNNING
    assert controller.loaded_signature == ("face", "emotion")
    assert controller.model_sample_rate == 16_000
    assert controller.model_schema == _model_schema()
    assert runtime._test_live_controller.stop_calls == [
        {"reset": False, "notify": False}
    ]


def test_replacing_selected_wav_detaches_the_previous_media(
    runtime_module: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    old_path = (tmp_path / "old.wav").resolve()
    settings.audio_path = str((tmp_path / "new.wav").resolve())
    settings.status = "STREAMING"
    controller = runtime.RuntimeController()
    controller.client._state = runtime.Lifecycle.RUNNING
    controller.loaded_signature = ("face", "emotion")
    stream = _activate_stream(
        runtime,
        controller,
        scene,
        audio_path=old_path,
    )
    requests: list[tuple[str, dict[str, object]]] = []
    controller.client.request = lambda method, params: (
        requests.append((method, params)) or "cancel-request"
    )

    controller._reconcile_input_media()

    assert requests == [("cancel", {"operation_id": "stream-1"})]
    assert stream.stop_requested is True
    assert controller.active_stream is stream
    assert controller.client.state == runtime.Lifecycle.RUNNING
    assert controller.loaded_signature == ("face", "emotion")
    assert runtime._test_live_controller.stop_calls == [
        {"reset": False, "notify": False}
    ]


def test_mode_switch_releases_an_already_terminal_stream_without_cancel(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.input_mode = "STREAM"
    settings.status = "STREAMING"
    controller = runtime.RuntimeController()
    controller.client._state = runtime.Lifecycle.RUNNING
    controller.loaded_signature = ("face", "emotion")
    controller.model_sample_rate = 16_000
    controller.model_schema = _model_schema()
    controller.selected_restart = (scene.name, 1.25, True)
    _activate_stream(
        runtime,
        controller,
        scene,
        audio_path=Path("voice.wav"),
        worker_ended=True,
    )
    requests: list[tuple[str, dict[str, object]]] = []
    controller.client.request = lambda method, params: (
        requests.append((method, params)) or "unexpected"
    )

    controller._reconcile_input_media()
    controller._poll_selected_restart()

    assert requests == []
    assert controller.active_stream is None
    assert controller.selected_restart is None
    assert controller.client.state == runtime.Lifecycle.RUNNING
    assert controller.loaded_signature == ("face", "emotion")
    assert controller.model_sample_rate == 16_000
    assert controller.model_schema == _model_schema()
    assert settings.status == "MODEL_READY"
    assert runtime._test_live_controller.stop_calls == [
        {"reset": False, "notify": False}
    ]


@pytest.mark.parametrize("input_mode", ["SELECTED", "STREAM"])
def test_inference_edit_updates_active_stream_without_touching_transport(
    runtime_module: tuple[ModuleType, ModuleType],
    input_mode: str,
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.input_mode = input_mode
    settings.playback_state = "PAUSED"
    controller = runtime.RuntimeController()
    stream = _activate_stream(runtime, controller, scene)
    requests: list[tuple[str, dict[str, object]]] = []
    controller.client.request = lambda method, params: (
        requests.append((method, params)) or "settings-request"
    )

    controller.refresh_inference_settings(scene)

    assert requests == []
    assert stream.refresh_deadline is not None
    assert controller.selected_restart is None
    assert stream.stop_requested is False
    assert runtime._test_live_controller.seek_stop_calls == []

    stream.refresh_deadline = 0.0
    controller._poll_inference_refresh()

    assert requests == [
        (
            "stream_settings",
            {
                "operation_id": "stream-1",
                "settings": _inference_settings_payload(),
            },
        )
    ]
    assert stream.refresh_deadline is None
    assert settings.playback_state == "PAUSED"


def test_repeated_inference_edits_coalesce_into_the_pending_selected_restart(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene = _local_scene(bpy)[0]
    controller = runtime.RuntimeController()
    _activate_stream(runtime, controller, scene, audio_path=Path("voice.wav"))
    controller.selected_restart = (scene.name, 1.0, False)
    requests: list[object] = []
    controller._request = lambda *_args, **_kwargs: requests.append(object())

    controller.refresh_inference_settings(scene)

    assert requests == []
    assert controller.selected_restart == (scene.name, 1.0, False)


def test_settings_refresh_and_selected_wav_eof_have_atomic_queue_order(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = "STREAMING"
    controller = runtime.RuntimeController()
    stream = _activate_stream(
        runtime,
        controller,
        scene,
        audio_path=Path("voice.wav"),
    )
    stream.refresh_deadline = 0.0
    requests: list[tuple[str, dict[str, object]]] = []

    def request(method: str, params: dict[str, object]) -> str:
        requests.append((method, params))
        return f"request-{len(requests)}"

    controller.client.request = request

    class InterleavingLock:
        """Run selected-WAV EOF exactly after the refresh lock is released."""

        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.interleave = True

        def __enter__(self) -> InterleavingLock:
            self.lock.acquire()
            return self

        def __exit__(self, *_args: object) -> None:
            self.lock.release()
            if self.interleave:
                self.interleave = False
                controller._queue_stream_end("stream-1")

        def locked(self) -> bool:
            return self.lock.locked()

    controller.pending_lock = InterleavingLock()

    controller._poll_inference_refresh()

    assert [method for method, _params in requests] == [
        "stream_settings",
        "stream_end",
    ]


def test_settings_refresh_waits_for_the_previous_request_acknowledgment(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = "STREAMING"
    controller = runtime.RuntimeController()
    stream = _activate_stream(runtime, controller, scene)
    requests: list[tuple[str, dict[str, object]]] = []

    def request(method: str, params: dict[str, object]) -> str:
        requests.append((method, params))
        return f"settings-{len(requests)}"

    controller.client.request = request
    controller.refresh_inference_settings(scene)
    stream.refresh_deadline = 0.0
    controller._poll_inference_refresh()
    assert [method for method, _params in requests] == ["stream_settings"]

    controller.refresh_inference_settings(scene)
    stream.refresh_deadline = 0.0
    controller._handle_response({"id": "settings-1", "result": {}})
    controller._poll_inference_refresh()

    assert [method for method, _params in requests] == [
        "stream_settings",
        "stream_settings",
    ]


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
    controller._submit_stream_start(scene, audio_path=None)

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
    spec = SimpleNamespace(
        audio2face_model=Path("/models/audio2face/model.json"),
        audio2emotion_model=Path("/models/audio2emotion/model.json"),
    )
    controller.loaded_signature = controller._model_signature(spec)
    controller.setup_snapshot = lambda: SimpleNamespace(
        require_inference_spec=lambda: spec
    )
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


def test_audio2emotion_stream_prebuffer_reaches_wav_source(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = "STREAM_STARTING"
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    _activate_stream(runtime, controller, scene, audio_path=Path("voice.wav"))
    controller.pending["start"] = _stream_pending(
        runtime,
        "stream_start",
        scene.name,
        "stream-1",
    )
    starts: list[tuple[object, str, object, int, int]] = []
    controller._start_wav_stream_source = (
        lambda source_scene, operation_id, wav_source, sample_rate, prebuffer_samples: starts.append(
            (source_scene, operation_id, wav_source, sample_rate, prebuffer_samples)
        )
    )

    controller._handle_response(
        {
            "id": "start",
            "result": {
                "sample_rate": 16_000,
                "prebuffer_samples": 60_000,
            },
        }
    )

    assert controller.active_stream is not None
    assert starts == [
        (scene, "stream-1", controller.active_stream.wav_source, 16_000, 60_000)
    ]
    assert settings.status == "STREAMING"
    assert controller.pcm_stream_requirements(scene) == (16_000, 60_000)


def test_selected_wav_source_honors_worker_prebuffer_without_local_override(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    stream = _activate_stream(
        runtime,
        controller,
        scene,
        audio_path=Path("voice.wav"),
    )

    class PlaybackGate:
        def __init__(self) -> None:
            self.wait_calls = 0

        def wait(self, _timeout: float) -> bool:
            self.wait_calls += 1
            return True

    class Source:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> Source:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def __iter__(self) -> object:
            return iter((struct.pack("<f", 0.0), struct.pack("<f", 0.0)))

    playback_gate = PlaybackGate()
    assert stream.wav_source is not None
    stream.wav_source.playback_started = playback_gate
    submitted: list[bytes] = []
    ended: list[str] = []

    def send(payload: bytes, *, operation_id: str) -> None:
        stream.chunk_credit.clear()
        submitted.append(payload)
        stream.chunk_credit.set()

    controller._send_stream_audio = send
    controller._queue_stream_end = lambda operation_id: ended.append(operation_id)
    monkeypatch.setattr(runtime, "WavStreamSource", Source)

    controller._start_wav_stream_source(
        scene,
        "stream-1",
        stream.wav_source,
        16_000,
        prebuffer_samples=1,
    )
    assert stream.wav_source.thread is not None
    stream.wav_source.thread.join(timeout=2.0)

    assert not stream.wav_source.thread.is_alive()
    assert playback_gate.wait_calls == 1
    assert submitted == [struct.pack("<f", 0.0), struct.pack("<f", 0.0)]
    assert ended == ["stream-1"]


def test_selected_wav_source_waits_for_each_worker_chunk_credit(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    stream = _activate_stream(
        runtime,
        controller,
        scene,
        audio_path=Path("voice.wav"),
    )
    assert stream.wav_source is not None
    stream.wav_source.playback_started.set()

    chunks = (struct.pack("<f", 0.1), struct.pack("<f", 0.2))

    class Source:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> Source:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def __iter__(self) -> object:
            return iter(chunks)

    submitted: list[bytes] = []
    first_sent = threading.Event()
    second_sent = threading.Event()
    ended: list[str] = []

    def send(payload: bytes, *, operation_id: str) -> None:
        assert operation_id == "stream-1"
        stream.chunk_credit.clear()
        submitted.append(payload)
        (first_sent if len(submitted) == 1 else second_sent).set()

    controller._send_stream_audio = send
    controller._queue_stream_end = lambda operation_id: ended.append(operation_id)
    monkeypatch.setattr(runtime, "WavStreamSource", Source)

    controller._start_wav_stream_source(
        scene,
        "stream-1",
        stream.wav_source,
        16_000,
        prebuffer_samples=0,
    )
    assert stream.wav_source.thread is not None
    assert first_sent.wait(1.0)
    assert submitted == [chunks[0]]
    assert not second_sent.is_set()

    stream.chunk_credit.set()
    assert second_sent.wait(1.0)
    assert submitted == list(chunks)
    assert ended == []

    stream.chunk_credit.set()
    stream.wav_source.thread.join(timeout=1.0)
    assert not stream.wav_source.thread.is_alive()
    assert ended == ["stream-1"]


def test_cancel_interrupts_a_selected_wav_waiting_for_chunk_credit(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    stream = _activate_stream(
        runtime,
        controller,
        scene,
        audio_path=Path("voice.wav"),
    )
    assert stream.wav_source is not None
    stream.wav_source.playback_started.set()

    class Source:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> Source:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def __iter__(self) -> object:
            return iter((struct.pack("<f", 0.1),))

    sent = threading.Event()
    ended: list[str] = []

    def send(_payload: bytes, *, operation_id: str) -> None:
        assert operation_id == "stream-1"
        stream.chunk_credit.clear()
        sent.set()

    controller._send_stream_audio = send
    controller._queue_stream_end = lambda operation_id: ended.append(operation_id)
    monkeypatch.setattr(runtime, "WavStreamSource", Source)

    controller._start_wav_stream_source(
        scene,
        "stream-1",
        stream.wav_source,
        16_000,
        prebuffer_samples=0,
    )
    assert stream.wav_source.thread is not None
    assert sent.wait(1.0)
    stream.wav_source.cancel.set()
    stream.wav_source.thread.join(timeout=1.0)

    assert not stream.wav_source.thread.is_alive()
    assert ended == []


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


def test_stream_reset_replaces_buffered_frames_without_stopping_transport(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.playback_state = "PAUSED"
    controller = runtime.RuntimeController()
    _activate_stream(runtime, controller, scene)

    controller._handle_event(
        {"event": "stream_reset", "operation_id": "stream-1", "data": {}}
    )

    assert runtime._test_live_controller.reset_calls == ["stream-1"]
    assert runtime._test_live_controller.stop_calls == []
    assert controller.active_stream is not None
    assert settings.playback_state == "PAUSED"


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


def test_cancel_racing_natural_completion_drains_terminal_before_restart(
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
    controller.selected_restart = (scene.name, 1.0, True)
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
    assert controller.selected_restart == (scene.name, 1.0, True)


def test_canceled_source_error_is_discarded_while_worker_drains(
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
    controller.stream_source_events.put(("error", "stream-1", "stale source error"))

    controller._poll_stream_source_events()

    assert controller.active_stream is stream
    assert settings.status == "STREAM_ENDING"


def test_seek_during_buffered_terminal_tail_restarts_without_worker_cancel(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.playback_state = "PLAYING"
    controller = runtime.RuntimeController()
    stream = _activate_stream(
        runtime,
        controller,
        scene,
        audio_path=Path("voice.wav"),
    )

    controller._handle_event(
        {"event": "stream_ended", "operation_id": "stream-1", "data": {}}
    )
    assert controller.active_stream is stream
    assert stream.worker_ended is True

    controller.seek_selected_audio(scene, 1.5, paused=False)

    assert runtime._test_live_controller.seek_stop_calls == [(1.5, False)]
    assert controller.selected_restart == (scene.name, 1.5, False)
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
        audio_path=Path("voice.wav"),
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
    assert stream.wav_source is not None and stream.wav_source.cancel.is_set()
    assert runtime._test_live_controller.stop_calls == [
        {"reset": False, "notify": False}
    ]
    assert shutdown_timeouts == [runtime.SHUTDOWN_TIMEOUT_SECONDS]


def test_natural_stream_tail_keeps_ui_state_until_presentation_stops(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    stream = _activate_stream(
        runtime,
        controller,
        scene,
        audio_path=Path("voice.wav"),
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
        natural=True,
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
                "code": "stream_backpressure",
                "message": "queue full",
            },
        }
    )

    assert requests == [("cancel", {"operation_id": "stream-1"})]
    assert controller.pending["cancel-stream"].operation_id == "stream-1"
    assert settings.status == "ERROR"
    assert settings.status_message == "stream_backpressure: queue full"

    controller._handle_response({"id": "cancel-stream", "result": {}})
    controller._handle_event(
        {"event": "stream_ended", "operation_id": "stream-1", "data": {}}
    )

    assert controller.rejected_reason is None
    assert controller.active_stream is None
    assert settings.status == "ERROR"
    assert settings.status_message == "stream_backpressure: queue full"
    assert runtime._test_live_controller.terminal_calls == []


def test_selected_wav_source_error_preserves_failure_through_terminal_event(
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
        audio_path=Path("voice.wav"),
    )
    requests: list[tuple[str, dict[str, object]]] = []

    def request(method: str, params: dict[str, object]) -> str:
        requests.append((method, params))
        return "cancel-source"

    controller.client.request = request
    controller.stream_source_events.put(
        ("error", "stream-1", "selected-WAV source failed")
    )

    controller._poll_stream_source_events()

    assert requests == [("cancel", {"operation_id": "stream-1"})]
    assert controller.pending["cancel-source"].operation_id == "stream-1"
    assert settings.status == "ERROR"

    controller._handle_response({"id": "cancel-source", "result": {}})
    controller._handle_event(
        {"event": "stream_ended", "operation_id": "stream-1", "data": {}}
    )

    assert controller.rejected_reason is None
    assert controller.active_stream is None
    assert settings.status == "ERROR"
    assert settings.status_message == "selected-WAV source failed"
    assert runtime._test_live_controller.terminal_calls == []


def test_stop_worker_drains_active_stream_terminal_without_playback_completion(
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
        audio_path=Path("voice.wav"),
    )
    controller.client._state = runtime.Lifecycle.RUNNING
    controller.client.begin_shutdown = lambda *, timeout: "shutdown"
    runtime._test_live_controller.operation_id = "stream-1"
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
        audio_path=Path("voice.wav"),
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
    assert stream.wav_source is not None and stream.wav_source.cancel.is_set()
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
        audio_path=Path("voice.wav"),
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
    assert stream.wav_source is not None and stream.wav_source.cancel.is_set()
    assert runtime._test_live_controller.stop_calls == [
        {"reset": False, "notify": False}
    ]


def test_runtime_survives_blend_file_replacement_with_fresh_controller(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, bpy = runtime_module
    cleanup_order: list[str] = []
    controllers: list[object] = []

    class StubController:
        def __init__(self) -> None:
            self.number = len(controllers) + 1
            controllers.append(self)

        def close(self) -> None:
            cleanup_order.append(f"close-{self.number}")

    monkeypatch.setattr(runtime, "RuntimeController", StubController)
    runtime.register_runtime()
    runtime.register_runtime()

    assert len(controllers) == 1
    assert bpy.app.handlers.load_pre == [runtime._load_pre_handler]
    assert bpy.app.handlers.load_post == [runtime._load_post_handler]
    assert bpy.app.timers.registrations[runtime._timer_callback] == {
        "first_interval": runtime.POLL_INTERVAL_SECONDS,
        "persistent": True,
    }

    runtime._load_pre_handler(None)

    assert runtime._CONTROLLER is None
    assert cleanup_order == ["close-1"]

    runtime._load_post_handler(None)

    assert len(controllers) == 2
    assert runtime._CONTROLLER is controllers[1]
    assert bpy.app.timers.is_registered(runtime._timer_callback)

    runtime.unregister_runtime()

    assert cleanup_order == [
        "close-1",
        "close-2",
    ]
    assert runtime._CONTROLLER is None
    assert bpy.app.handlers.load_pre == []
    assert bpy.app.handlers.load_post == []
    assert not bpy.app.timers.is_registered(runtime._timer_callback)


def test_runtime_timer_redraws_the_sidebar_while_frames_are_presented(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _bpy = runtime_module
    calls: list[str] = []
    runtime._CONTROLLER = SimpleNamespace(
        poll=lambda: calls.append("poll"),
        _tag_runtime_setup_redraw=lambda: calls.append("redraw"),
    )
    monkeypatch.setattr(
        runtime,
        "get_live_stream_controller",
        lambda: SimpleNamespace(tick=lambda: True),
    )

    assert runtime._timer_callback() == runtime.PLAYBACK_INTERVAL_SECONDS
    assert calls == ["poll", "redraw"]


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
    finally:
        bpy.data = unrestricted_data  # type: ignore[attr-defined]
        runtime.unregister_runtime()
