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


def _model_schema(*, parameter_default: float = 1.0) -> dict[str, object]:
    return {
        "identities": ["Model Identity"],
        "channels": list(MODEL_CHANNELS),
        "parameters": {"/input_strength": parameter_default},
        "emotion_channels": [{"name": "Joy", "default": 0.0}],
    }


class _Settings:
    def __init__(self) -> None:
        self.status = "IDLE"
        self.status_message = "before"
        self.progress = 0.5
        self.preview_state = "PLAYING"
        self.preview_time = 1.0
        self.preview_duration = 2.0
        self.preview_reset_on_stop = True
        self.result_operation_id = ""
        self.result_path = ""
        self.result_audio_path = ""
        self.audio_path = ""
        self.identity_index = 0
        self.stream_operation_id = ""
        self.stream_sample_rate = 0
        self.stream_prebuffer_samples = 0
        self.stream_time = 0.0
        self.stream_reset_on_stop = True

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
        self.terminal_calls: list[str] = []
        self.stop_calls: list[dict[str, object]] = []
        self.reset_on_stop = True

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
        continuation=None,
        model_signature=None,
        operation_id=None,
    )


def _model_pending(
    runtime: ModuleType,
    scene_name: str,
    signature: tuple[str, str, int],
) -> object:
    return runtime.PendingRequest(
        "load_model",
        scene_name,
        continuation="ready",
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
        continuation=None,
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
    properties.apply_model_schema = lambda *_args, **_kwargs: MODEL_CHANNELS  # type: ignore[attr-defined]
    properties.tuning_parameters = lambda *_args, **_kwargs: {}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, properties.__name__, properties)

    preview = ModuleType("audio2face.preview")
    preview.get_preview_controller = lambda: SimpleNamespace(tick=lambda: False)  # type: ignore[attr-defined]
    preview.unregister_preview = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, preview.__name__, preview)

    live_controller = _LiveController()
    live_stream = ModuleType("audio2face.live_stream")
    live_stream.LiveStreamError = ValueError  # type: ignore[attr-defined]
    live_stream.get_live_stream_controller = lambda: live_controller  # type: ignore[attr-defined]
    live_stream.unregister_live_stream = lambda: None  # type: ignore[attr-defined]
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


def test_extension_data_directory_rejects_a_filesystem_alias(
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
        runtime.RuntimeController.result_directory()


def test_selected_paths_reject_blender_relative_spelling(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, _bpy = runtime_module

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


def test_initial_poll_resets_only_editable_scenes(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    local_settings = _Settings()
    local = _Scene("Local", editable=True, settings=local_settings)
    linked = _Scene("Linked", editable=False, settings=_ReadOnlySettings())
    bpy.data.scenes = _Scenes([linked, local])  # type: ignore[attr-defined]

    controller = runtime.RuntimeController()
    controller.poll()

    assert controller.reset_scene_state_on_poll is False
    assert local_settings.status == "IDLE"
    assert local_settings.status_message == "Worker is stopped"
    assert local_settings.preview_state == "IDLE"


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
    runtime, bpy = runtime_module
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
    settings.identity_index = 99
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
        *,
        identity_index: int,
        continuation: str,
    ) -> None:
        loaded.append((target, spec, identity_index, continuation))

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
    assert loaded == [(scene, "bundled-spec", 0, "ready")]
    assert settings.identity_index == 0
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
    signature = ("audio2face/model.json", "audio2emotion/model.json", 0)
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
    assert controller.schema_scenes == {scene.as_pointer()}


def test_loaded_model_schema_is_cached_and_applied_once_per_scene(
    runtime_module: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, bpy = runtime_module
    first_scene, first_settings = _local_scene(bpy, "First")
    second_settings = _Settings()
    second_scene = _Scene("Second", editable=True, settings=second_settings)
    bpy.data.scenes.append(second_scene)  # type: ignore[attr-defined]
    controller = runtime.RuntimeController()
    signature = ("audio2face/model.json", "audio2emotion/model.json", 0)
    model_schema = _model_schema(parameter_default=1.0)
    applications: list[tuple[object, float]] = []

    def apply(
        settings: object,
        payload: object,
        applied_signature: tuple[str, str, int],
    ) -> tuple[str, ...]:
        assert applied_signature == signature
        assert isinstance(payload, dict)
        parameters = payload["parameters"]
        applications.append((settings, parameters["/input_strength"]))
        return MODEL_CHANNELS

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

    assert applications == [
        (first_settings, 1.0),
        (second_settings, 1.0),
    ]
    assert controller.model_schema is model_schema
    assert controller.schema_scenes == {
        first_scene.as_pointer(),
        second_scene.as_pointer(),
    }


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
    ) -> str:
        submitted.append((method, params, kwargs))
        return "load"

    controller._request = request
    controller._submit_model_load(
        scene,
        spec,
        identity_index=0,
        continuation="ready",
    )

    assert submitted == [
        (
            "load_model",
            {
                "audio2face_model_path": str(spec.audio2face_model),
                "audio2emotion_model_path": str(spec.audio2emotion_model),
                "identity_index": 0,
            },
            {
                "continuation": "ready",
                "model_signature": (
                    str(spec.audio2face_model),
                    str(spec.audio2emotion_model),
                    0,
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
        ("audio2face/model.json", "audio2emotion/model.json", 0),
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


def test_result_event_uses_only_the_controller_path_for_its_active_operation(
    runtime_module: tuple[ModuleType, ModuleType],
    tmp_path: Path,
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.result_operation_id = "operation-1"
    controller = runtime.RuntimeController()
    controller.generation_scene_names["operation-1"] = scene.name
    controller.result_directory = lambda: tmp_path
    (tmp_path / "operation-1.a2f.json").write_text("{}", encoding="utf-8")

    controller._handle_event(
        {
            "event": "result",
            "operation_id": "operation-1",
            "data": {},
        }
    )

    assert settings.status == "COMPLETED"
    assert settings.progress == 1.0
    assert settings.result_path.endswith("operation-1.a2f.json")
    assert settings.result_operation_id == "operation-1"
    assert controller.generation_scene_names == {}

    settings.result_operation_id = "operation-2"
    controller.generation_scene_names["operation-2"] = scene.name
    controller._handle_event(
        {
            "event": "result",
            "operation_id": "operation-2",
            "data": {"unexpected": True},
        }
    )
    assert settings.status == "ERROR"
    assert settings.result_operation_id == ""
    assert controller.generation_scene_names == {}


def test_event_routing_rejects_an_unknown_operation_id(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.result_operation_id = "operation-1"
    controller = runtime.RuntimeController()
    controller.startup_scene = scene.name
    shutdown_timeouts: list[float] = []
    controller.client.begin_shutdown = lambda *, timeout: shutdown_timeouts.append(
        timeout
    )

    controller._handle_event(
        {
            "event": "progress",
            "operation_id": "operation-1",
            "data": {"progress": 0.75, "stage": "unsolicited"},
        }
    )

    assert settings.status == "ERROR"
    assert settings.status_message == (
        "worker returned an event for an unknown operation ID"
    )
    assert settings.progress == 0.5
    assert controller.rejected_reason == settings.status_message
    assert shutdown_timeouts == [runtime.SHUTDOWN_TIMEOUT_SECONDS]


def test_event_routing_rejects_an_operation_in_both_maps(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.startup_scene = scene.name
    controller.generation_scene_names["operation-1"] = scene.name
    controller.stream_scene_names["operation-1"] = scene.name
    shutdown_timeouts: list[float] = []
    controller.client.begin_shutdown = lambda *, timeout: shutdown_timeouts.append(
        timeout
    )

    controller._handle_event(
        {
            "event": "progress",
            "operation_id": "operation-1",
            "data": {"progress": 0.75, "stage": "ambiguous"},
        }
    )

    assert settings.status == "ERROR"
    assert settings.status_message == (
        "worker event operation ID belongs to both operation maps"
    )
    assert controller.rejected_reason == settings.status_message
    assert shutdown_timeouts == [runtime.SHUTDOWN_TIMEOUT_SECONDS]


def test_terminal_contract_rejection_clears_all_active_scene_operations(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    first, first_settings = _local_scene(bpy, "First")
    second_settings = _Settings()
    second = _Scene("Second", editable=True, settings=second_settings)
    bpy.data.scenes.append(second)  # type: ignore[attr-defined]
    first_settings.result_operation_id = "generation-1"
    second_settings.stream_operation_id = "stream-1"
    second_settings.stream_sample_rate = 16_000
    second_settings.stream_prebuffer_samples = 60_000
    controller = runtime.RuntimeController()
    controller.generation_scene_names["generation-1"] = first.name
    controller.stream_scene_names["stream-1"] = second.name
    controller.stream_audio_paths["stream-1"] = Path("voice.wav")
    controller.stream_source_cancel = threading.Event()
    shutdown_timeouts: list[float] = []
    controller.client.begin_shutdown = lambda *, timeout: shutdown_timeouts.append(
        timeout
    )

    controller._reject_worker_contract("worker contract failed")

    assert first_settings.status == "ERROR"
    assert second_settings.status == "ERROR"
    assert first_settings.result_operation_id == ""
    assert second_settings.stream_operation_id == ""
    assert second_settings.stream_sample_rate == 0
    assert second_settings.stream_prebuffer_samples == 0
    assert controller.generation_scene_names == {}
    assert controller.stream_scene_names == {}
    assert controller.stream_audio_paths == {}
    assert controller.stream_source_cancel is None
    assert runtime._test_live_controller.stop_calls == [{"reset": False}]
    assert shutdown_timeouts == [runtime.SHUTDOWN_TIMEOUT_SECONDS]


def test_rejected_worker_controls_cannot_revive_scene_state(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.client.begin_shutdown = lambda *, timeout: None
    controller._reject_worker_contract("terminal contract failure")
    controller.generation_scene_names["operation-1"] = scene.name
    settings.result_operation_id = "operation-1"

    controller._handle_control(
        {
            "type": "event",
            "event": "progress",
            "operation_id": "operation-1",
            "data": {"progress": 0.5, "stage": "must be ignored"},
        }
    )

    assert settings.status == "ERROR"
    assert settings.status_message == "terminal contract failure"
    assert settings.progress == 0.5


def test_rejected_worker_exit_preserves_error_on_every_editable_scene(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    first, first_settings = _local_scene(bpy, "First")
    second_settings = _Settings()
    second = _Scene("Second", editable=True, settings=second_settings)
    bpy.data.scenes.append(second)  # type: ignore[attr-defined]
    controller = runtime.RuntimeController()
    controller.reset_scene_state_on_poll = False
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


def test_cancel_error_preserves_active_status(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = "CANCELLING"
    settings.result_operation_id = "operation"
    controller = runtime.RuntimeController()
    controller.pending["cancel"] = _plain_pending(runtime, "cancel", scene.name)

    controller._handle_error(
        {"id": "cancel", "error": {"code": "busy", "message": "try again"}}
    )

    assert settings.status == "CANCELLING"
    assert settings.status_message == "busy: try again"


@pytest.mark.parametrize("terminal_status", ["COMPLETED", "ERROR"])
def test_late_cancel_response_does_not_erase_terminal_status(
    runtime_module: tuple[ModuleType, ModuleType], terminal_status: str
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = terminal_status
    settings.status_message = "terminal result"
    controller = runtime.RuntimeController()
    controller.pending["cancel"] = _plain_pending(runtime, "cancel", scene.name)

    controller._handle_response(
        {"id": "cancel", "result": {}}
    )

    assert settings.status == terminal_status
    assert settings.status_message == "terminal result"


def test_late_non_shutdown_response_does_not_replace_stopping_state(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.status = "STOPPING"
    controller = runtime.RuntimeController()
    controller.pending["load"] = _model_pending(
        runtime,
        scene.name,
        ("audio2face/model.json", "audio2emotion/model.json", 0),
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


def test_generation_binds_result_to_exact_submitted_audio(
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
        0,
    )
    controller.loaded_signature = signature
    controller.model_schema = _model_schema()
    controller.schema_scenes.add(scene.as_pointer())
    controller.result_directory = lambda: tmp_path
    requests: list[tuple[str, dict[str, object]]] = []

    def request(
        _scene: object,
        method: str,
        params: dict[str, object],
        **_kwargs: object,
    ) -> str:
        requests.append((method, params))
        return "generate-request"

    controller._request = request
    controller.generate(scene)

    assert requests[0][0] == "generate"
    assert set(requests[0][1]) == {
        "operation_id",
        "audio_path",
        "result_path",
        "settings",
    }
    assert requests[0][1]["operation_id"] == settings.result_operation_id
    assert requests[0][1]["audio_path"] == str(audio_path.resolve())
    assert settings.result_audio_path == str(audio_path.resolve())
    settings.audio_path = str(tmp_path / "different.wav")
    assert settings.result_audio_path == str(audio_path.resolve())


def test_stream_audio_request_is_exact_base64_f32le(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    controller.stream_scene_names["stream-1"] = scene.name
    requests: list[tuple[str, dict[str, object]]] = []

    def request(method: str, params: dict[str, object]) -> str:
        assert controller.pending_lock.locked()
        requests.append((method, params))
        return "chunk-request"

    controller.client.request = request
    payload = struct.pack("<fff", -0.5, 0.0, 0.75)

    request_id = controller.push_stream_audio(
        payload,
        operation_id="stream-1",
    )

    assert request_id == "chunk-request"
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
    assert controller.pending[request_id] == _stream_pending(
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
def test_stream_audio_rejects_noncanonical_f32le(
    runtime_module: tuple[ModuleType, ModuleType],
    payload: object,
    message: str,
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    controller.stream_scene_names["stream-1"] = scene.name
    controller.client.request = lambda *_args, **_kwargs: pytest.fail(
        "invalid audio reached the sidecar"
    )

    with pytest.raises(runtime.SidecarError, match=message):
        controller.push_stream_audio(payload, operation_id="stream-1")


def test_stream_audio_ingress_is_safe_from_a_source_thread(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    controller.stream_scene_names["stream-1"] = scene.name
    runtime._test_live_controller.operation_id = "stream-1"
    calls: list[tuple[str, dict[str, object], threading.Thread]] = []
    failures: list[BaseException] = []

    def request(method: str, params: dict[str, object]) -> str:
        calls.append((method, params, threading.current_thread()))
        return "thread-chunk"

    controller.client.request = request

    def push() -> None:
        try:
            controller.push_stream_audio(
                struct.pack("<f", 0.25),
                operation_id="stream-1",
            )
        except BaseException as exc:  # pragma: no cover - asserted on main thread
            failures.append(exc)

    source_thread = threading.Thread(target=push, name="test-audio-source")
    source_thread.start()
    source_thread.join(timeout=2.0)

    assert not source_thread.is_alive()
    assert failures == []
    assert len(calls) == 1
    assert calls[0][0] == "stream_chunk"
    assert calls[0][2] is source_thread
    assert controller.pending["thread-chunk"] == _stream_pending(
        runtime, "stream_chunk", scene.name, "stream-1"
    )


def test_stream_audio_ingress_has_bounded_pending_backpressure(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    controller.stream_scene_names["stream-1"] = scene.name
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
        controller.push_stream_audio(
            struct.pack("<f", 0.0),
            operation_id="stream-1",
        )


def test_exact_stream_start_response_marks_stream_ready(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.stream_operation_id = "stream-1"
    settings.stream_sample_rate = 16_000
    settings.status = "STREAM_STARTING"
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    controller.stream_scene_names["stream-1"] = scene.name
    controller.pending["start"] = _stream_pending(
        runtime,
        "stream_start",
        scene.name,
        "stream-1",
    )

    assert controller.pcm_stream_requirements(scene) is None

    controller._handle_response(
        {
            "id": "start",
            "result": {"sample_rate": 16_000, "prebuffer_samples": 0},
        }
    )

    assert settings.status == "STREAMING"
    assert settings.status_message == "PCM stream is ready"
    assert settings.stream_operation_id == "stream-1"
    assert controller.pcm_stream_requirements(scene) == (16_000, 0)


def test_pcm_stream_requirements_rejects_inactive_scene(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, _settings = _local_scene(bpy)

    with pytest.raises(runtime.SidecarError, match="no active PCM stream"):
        runtime.RuntimeController().pcm_stream_requirements(scene)


def test_audio2emotion_stream_prebuffer_reaches_wav_source(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.stream_operation_id = "stream-1"
    settings.stream_sample_rate = 16_000
    settings.status = "STREAM_STARTING"
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    controller.stream_scene_names["stream-1"] = scene.name
    controller.stream_audio_paths["stream-1"] = Path("voice.wav")
    controller.pending["start"] = _stream_pending(
        runtime,
        "stream_start",
        scene.name,
        "stream-1",
    )
    starts: list[tuple[object, str, int]] = []
    controller._start_wav_stream_source = (
        lambda source_scene, operation_id, prebuffer_samples: starts.append(
            (source_scene, operation_id, prebuffer_samples)
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

    assert starts == [(scene, "stream-1", 60_000)]
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
    controller.stream_audio_paths["stream-1"] = Path("voice.wav")
    controller.stream_source_cancel = threading.Event()

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
    controller.stream_playback_started = playback_gate
    submitted: list[bytes] = []
    ended: list[str] = []
    controller.push_stream_audio = (
        lambda payload, *, operation_id: submitted.append(payload) or "chunk"
    )
    controller._queue_stream_end = lambda operation_id: ended.append(operation_id) or "end"
    monkeypatch.setattr(runtime, "WavStreamSource", Source)

    controller._start_wav_stream_source(scene, "stream-1", prebuffer_samples=1)
    assert controller.stream_source_thread is not None
    controller.stream_source_thread.join(timeout=2.0)

    assert not controller.stream_source_thread.is_alive()
    assert playback_gate.wait_calls == 1
    assert submitted == [struct.pack("<f", 0.0), struct.pack("<f", 0.0)]
    assert ended == ["stream-1"]


def test_stream_start_response_rejects_noninteger_sample_rate(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.stream_operation_id = "stream-1"
    settings.stream_sample_rate = 16_000
    controller = runtime.RuntimeController()
    controller.model_sample_rate = 16_000
    controller.stream_scene_names["stream-1"] = scene.name
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
    assert settings.stream_operation_id == ""
    assert runtime._test_live_controller.stop_calls == [{"reset": False}]


def test_exact_stream_frame_routes_negative_timestamp_and_arkit52(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.stream_operation_id = "stream-1"
    controller = runtime.RuntimeController()
    controller.stream_scene_names["stream-1"] = scene.name
    weights = [0.25] * len(MODEL_CHANNELS)

    controller._handle_event(
        {
            "event": "stream_frame",
            "operation_id": "stream-1",
            "data": {"timestamp_sample": -320, "weights": weights},
        }
    )

    assert runtime._test_live_controller.receive_calls == [
        ("stream-1", -320, weights)
    ]
    assert settings.status == "STREAMING"


def test_malformed_stream_frame_terminates_the_active_stream(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.stream_operation_id = "stream-1"
    settings.stream_sample_rate = 16_000
    controller = runtime.RuntimeController()
    controller.stream_scene_names["stream-1"] = scene.name
    controller.stream_audio_paths["stream-1"] = Path("voice.wav")
    controller.stream_source_cancel = threading.Event()
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
                "unexpected": True,
            },
        }
    )

    assert settings.status == "ERROR"
    assert settings.stream_operation_id == ""
    assert controller.stream_source_cancel is None
    assert "stream-1" not in controller.stream_scene_names
    assert "stream-1" not in controller.stream_audio_paths
    assert runtime._test_live_controller.stop_calls == [{"reset": False}]
    assert shutdown_timeouts == [runtime.SHUTDOWN_TIMEOUT_SECONDS]


def test_exact_stream_ended_cleans_stream_but_keeps_model_ready(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.stream_operation_id = "stream-1"
    settings.stream_sample_rate = 16_000
    controller = runtime.RuntimeController()
    controller.stream_scene_names["stream-1"] = scene.name
    controller.stream_audio_paths["stream-1"] = Path("voice.wav")
    controller.stream_source_cancel = threading.Event()

    controller._handle_event(
        {"event": "stream_ended", "operation_id": "stream-1", "data": {}}
    )

    assert runtime._test_live_controller.terminal_calls == ["stream-1"]
    assert settings.stream_operation_id == ""
    assert settings.stream_sample_rate == 0
    assert settings.status == "MODEL_READY"
    assert controller.stream_source_cancel is None
    assert controller.stream_scene_names == {}
    assert controller.stream_audio_paths == {}


def test_natural_stream_tail_keeps_ui_state_until_presentation_stops(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.stream_operation_id = "stream-1"
    settings.stream_sample_rate = 16_000
    controller = runtime.RuntimeController()
    controller.stream_scene_names["stream-1"] = scene.name
    controller.stream_audio_paths["stream-1"] = Path("voice.wav")
    runtime._test_live_controller.is_active = True

    controller._handle_event(
        {"event": "stream_ended", "operation_id": "stream-1", "data": {}}
    )

    assert settings.stream_operation_id == "stream-1"
    assert settings.stream_sample_rate == 16_000
    assert settings.status == "STREAMING"
    assert "Finishing buffered" in settings.status_message
    assert controller.stream_scene_names == {}
    assert controller.stream_audio_paths == {}

    runtime._test_live_controller.is_active = False
    controller._finish_stream_presentation(scene.name, "stream-1")

    assert settings.stream_operation_id == ""
    assert settings.stream_sample_rate == 0
    assert settings.status == "MODEL_READY"


def test_stream_request_error_cancels_worker_before_clearing_local_state(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.stream_operation_id = "stream-1"
    settings.stream_sample_rate = 16_000
    settings.status = "STREAMING"
    controller = runtime.RuntimeController()
    controller.stream_scene_names["stream-1"] = scene.name
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
    assert settings.stream_operation_id == ""
    assert settings.status == "ERROR"

    controller._handle_response({"id": "cancel-stream", "result": {}})
    assert settings.status == "ERROR"


@pytest.mark.parametrize("response_kind", ["error", "noncanonical"])
def test_rejected_stream_cancel_cleans_local_state(
    runtime_module: tuple[ModuleType, ModuleType],
    response_kind: str,
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.stream_operation_id = "stream-1"
    settings.stream_sample_rate = 16_000
    settings.status = "STREAM_ENDING"
    controller = runtime.RuntimeController()
    controller.stream_scene_names["stream-1"] = scene.name
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

    assert settings.stream_operation_id == ""
    assert settings.stream_sample_rate == 0
    assert settings.status == "ERROR"
    assert controller.stream_scene_names == {}


def test_explicit_stream_stop_ends_local_audio_instead_of_finishing_playback(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.stream_operation_id = "stream-1"
    settings.stream_sample_rate = 16_000
    controller = runtime.RuntimeController()
    controller.stream_scene_names["stream-1"] = scene.name
    controller.stream_stop_requests.add("stream-1")

    controller._handle_event(
        {"event": "stream_ended", "operation_id": "stream-1", "data": {}}
    )

    assert runtime._test_live_controller.stop_calls == [{"reset": True}]
    assert runtime._test_live_controller.terminal_calls == []
    assert controller.stream_stop_requests == set()
    assert settings.stream_operation_id == ""
    assert settings.status == "MODEL_READY"


def test_malformed_stream_ended_still_cleans_the_terminal_stream(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.stream_operation_id = "stream-1"
    settings.stream_sample_rate = 16_000
    controller = runtime.RuntimeController()
    controller.stream_scene_names["stream-1"] = scene.name
    controller.stream_source_cancel = threading.Event()

    controller._handle_event(
        {
            "event": "stream_ended",
            "operation_id": "stream-1",
            "data": {"unexpected": True},
        }
    )

    assert settings.status == "ERROR"
    assert settings.stream_operation_id == ""
    assert controller.stream_source_cancel is None
    assert controller.stream_scene_names == {}
    assert runtime._test_live_controller.stop_calls == [{"reset": False}]


def test_malformed_error_event_cleans_its_active_stream(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.stream_operation_id = "stream-1"
    settings.stream_sample_rate = 16_000
    controller = runtime.RuntimeController()
    controller.stream_scene_names["stream-1"] = scene.name
    controller.stream_source_cancel = threading.Event()

    controller._handle_event(
        {
            "event": "error",
            "operation_id": "stream-1",
            "data": {"code": "inference_failed", "message": 42},
        }
    )

    assert settings.status == "ERROR"
    assert settings.stream_operation_id == ""
    assert controller.stream_source_cancel is None
    assert controller.stream_scene_names == {}
    assert runtime._test_live_controller.stop_calls == [{"reset": False}]


def test_generation_event_cannot_be_routed_to_a_stream(
    runtime_module: tuple[ModuleType, ModuleType],
) -> None:
    runtime, bpy = runtime_module
    scene, settings = _local_scene(bpy)
    settings.stream_operation_id = "stream-1"
    settings.stream_sample_rate = 16_000
    controller = runtime.RuntimeController()
    controller.stream_scene_names["stream-1"] = scene.name

    controller._handle_event(
        {
            "event": "progress",
            "operation_id": "stream-1",
            "data": {"progress": 0.5, "stage": "wrong operation"},
        }
    )

    assert settings.status == "ERROR"
    assert settings.stream_operation_id == ""
    assert controller.stream_scene_names == {}
    assert runtime._test_live_controller.stop_calls == [{"reset": False}]


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
    monkeypatch.setattr(
        runtime,
        "unregister_preview",
        lambda: cleanup_order.append("preview"),
    )
    monkeypatch.setattr(
        runtime,
        "unregister_live_stream",
        lambda: cleanup_order.append("live"),
    )

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
    assert cleanup_order == ["close-1", "preview"]

    runtime._load_post_handler(None)

    assert len(controllers) == 2
    assert runtime._CONTROLLER is controllers[1]
    assert bpy.app.timers.is_registered(runtime._timer_callback)

    runtime.unregister_runtime()

    assert cleanup_order == [
        "close-1",
        "preview",
        "close-2",
        "preview",
    ]
    assert runtime._CONTROLLER is None
    assert bpy.app.handlers.load_pre == []
    assert bpy.app.handlers.load_post == []
    assert not bpy.app.timers.is_registered(runtime._timer_callback)
