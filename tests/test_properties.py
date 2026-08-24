from __future__ import annotations

import copy
import importlib.util
import re
import struct
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


CHANNELS = tuple(f"modelChannel{index}" for index in range(52))
MODEL_SIGNATURE = ("/models/audio2face/model.json", "/models/audio2emotion/model.json")
UI_SOURCE = (
    Path(__file__).resolve().parents[1] / "audio2face" / "ui.py"
).read_text(encoding="utf-8")

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
AUDIO2FACE_TUNING: dict[str, float | int] = {
    "input_strength": 2.0,
    "lower_face_smoothing": 0.04,
    "upper_face_smoothing": 0.05,
    "lower_face_strength": 1.5,
    "upper_face_strength": 0.75,
    "face_mask_level": 0.4,
    "face_mask_softness": 0.1,
    "skin_strength": 0.8,
    "blink_strength": 1.5,
    "eyelid_open_offset": -0.25,
    "lip_open_offset": 0.1,
    "eyeballs_strength": 0.5,
    "saccade_strength": 1.4,
    "right_eye_rot_x_offset": -4.0,
    "right_eye_rot_y_offset": -4.5,
    "left_eye_rot_x_offset": 4.0,
    "left_eye_rot_y_offset": 4.5,
    "eye_saccade_seed": 42,
}


class _Collection(list[SimpleNamespace]):
    def __init__(self, factory: Callable[[], SimpleNamespace]) -> None:
        super().__init__()
        self.factory = factory

    def add(self) -> SimpleNamespace:
        item = self.factory()
        self.append(item)
        return item


@pytest.fixture
def properties_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    bpy = ModuleType("bpy")
    bpy.types = SimpleNamespace(PropertyGroup=object, Object=object)  # type: ignore[attr-defined]
    props = ModuleType("bpy.props")
    for name in (
        "BoolProperty",
        "CollectionProperty",
        "EnumProperty",
        "FloatProperty",
        "IntProperty",
        "PointerProperty",
        "StringProperty",
    ):
        setattr(props, name, lambda **kwargs: kwargs)
    bpy.props = props  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", props)

    name = "audio2face._properties_test"
    source = Path(__file__).resolve().parents[1] / "audio2face" / "properties.py"
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        model_schema_signature="",
        **AUDIO2FACE_DEFAULTS,
        auto_audio2emotion=False,
        manual_emotions=_Collection(lambda: SimpleNamespace(name="", value=0.0)),
        preferred_emotions=_Collection(
            lambda: SimpleNamespace(name="", value=0.0)
        ),
        a2e_emotion_strength=0.6,
        a2e_emotion_contrast=1.0,
        a2e_max_emotions=6,
        a2e_live_blend_coef=0.7,
        a2e_transition_smoothing=0.5,
        a2e_preferred_emotion_strength=0.5,
    )


def _schema() -> dict[str, object]:
    return {
        "channels": list(CHANNELS),
        "audio2face_defaults": AUDIO2FACE_DEFAULTS.copy(),
        "emotion_channels": [
            {"name": "Neutral", "default": 0.5},
            {"name": "Joy", "default": 0.2},
        ],
    }


def test_emotion_configuration_is_not_hidden_by_auto_mode() -> None:
    assert "if settings.auto_audio2emotion:" not in UI_SOURCE
    assert 'text="Manual Emotion"' not in UI_SOURCE
    for name in (
        "a2e_preferred_emotion_strength",
        "a2e_emotion_strength",
        "a2e_emotion_contrast",
        "a2e_max_emotions",
        "a2e_live_blend_coef",
        "a2e_transition_smoothing",
    ):
        assert re.search(
            rf'auto_controls\.prop\(\s*settings,\s*"{name}",\s*slider=True,?\s*\)',
            UI_SOURCE,
        )
    assert '"a2f.load_preferred_emotion"' in UI_SOURCE
    assert '"a2f.clear_preferred_emotion"' in UI_SOURCE


def test_prediction_delay_uses_a_range_slider() -> None:
    assert (
        'playback_box.prop(settings, "prediction_delay", slider=True)'
        in UI_SOURCE
    )


def test_model_tuning_ui_exposes_only_the_fixed_audio2face_contract() -> None:
    assert 'text="Model Tuning"' in UI_SOURCE
    assert "AUDIO2FACE_SETTING_GROUPS" in UI_SOURCE
    assert "emotion_controls.enabled" not in UI_SOURCE


def test_runtime_status_box_uses_the_persistence_gate() -> None:
    assert "controller.status_notice(context.scene)" in UI_SOURCE
    assert "visible_statuses" not in UI_SOURCE


def test_playback_ui_uses_an_editable_absolute_time_slider() -> None:
    assert re.search(
        r'playback_box\.prop\(\s*settings,\s*PLAYBACK_POSITION_PATH,'
        r'\s*text="",\s*slider=True,?\s*\)',
        UI_SOURCE,
    )
    assert "playback_progress" not in UI_SOURCE
    assert "playback_duration" not in UI_SOURCE
    assert "playback_position(settings)" not in UI_SOURCE
    assert "seek_row.enabled" not in UI_SOURCE
    assert ".progress(" not in UI_SOURCE


def test_input_mode_switch_is_never_disabled() -> None:
    assert "input_mode_row = input_box.row(align=True)" in UI_SOURCE
    assert 'input_mode_row.prop(settings, "input_mode", expand=True)' in UI_SOURCE
    assert "input_mode_row.enabled" not in UI_SOURCE


def test_audio_path_update_replaces_the_selected_media_slider(
    properties_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    live_stream = ModuleType("audio2face.live_stream")
    live_stream.clear_playback_position = (  # type: ignore[attr-defined]
        lambda settings: calls.append(("clear", settings))
    )
    live_stream.configure_playback_position = (  # type: ignore[attr-defined]
        lambda settings, position, duration: calls.append(
            ("configure", settings, position, duration)
        )
    )
    wav_stream = ModuleType("audio2face.wav_stream")
    wav_stream.WavStreamError = ValueError  # type: ignore[attr-defined]
    wav_stream.wav_duration_seconds = (  # type: ignore[attr-defined]
        lambda path: {"first.wav": 2.0, "second.wav": 3.5}[path]
    )
    monkeypatch.setitem(sys.modules, live_stream.__name__, live_stream)
    monkeypatch.setitem(sys.modules, wav_stream.__name__, wav_stream)
    settings = SimpleNamespace(audio_path="first.wav")

    properties_module._audio_path_updated(settings, SimpleNamespace())
    settings.audio_path = "second.wav"
    properties_module._audio_path_updated(settings, SimpleNamespace())
    settings.audio_path = ""
    properties_module._audio_path_updated(settings, SimpleNamespace())

    assert calls == [
        ("clear", settings),
        ("configure", settings, 0.0, 2.0),
        ("clear", settings),
        ("configure", settings, 0.0, 3.5),
        ("clear", settings),
    ]
    assert "update=_audio_path_updated" in (
        properties_module.A2FSceneSettings.__annotations__["audio_path"]
    )


def test_shared_update_callback_refreshes_the_context_scene(
    properties_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    controller = SimpleNamespace(
        refresh_inference_settings=lambda scene: calls.append(scene)
    )
    runtime = ModuleType("audio2face.runtime")
    runtime.get_controller = lambda: controller  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "audio2face.runtime", runtime)
    scene = SimpleNamespace(audio2face=object())

    properties_module._inference_setting_updated(None, SimpleNamespace(scene=scene))

    assert calls == [scene]


def test_effective_emotions_update_visible_values_without_mutating_preferred(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    properties_module.load_preferred_emotion(settings)

    properties_module.apply_effective_emotions(
        settings,
        ("Neutral", "Joy"),
        (0.1, 0.9),
    )

    assert [(item.name, item.value) for item in settings.manual_emotions] == [
        ("Neutral", 0.1),
        ("Joy", 0.9),
    ]
    assert [(item.name, item.value) for item in settings.preferred_emotions] == [
        ("Neutral", 0.5),
        ("Joy", 0.2),
    ]


def test_effective_emotion_writes_suppress_only_the_worker_refresh(
    properties_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    controller = SimpleNamespace(
        refresh_inference_settings=lambda scene: calls.append(scene)
    )
    runtime = ModuleType("audio2face.runtime")
    runtime.get_controller = lambda: controller  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "audio2face.runtime", runtime)
    scene = SimpleNamespace(audio2face=object())
    context = SimpleNamespace(scene=scene)

    class CallbackItem:
        def __init__(self, name: str) -> None:
            self.name = name
            self._value = 0.0

        @property
        def value(self) -> float:
            return self._value

        @value.setter
        def value(self, value: float) -> None:
            self._value = value
            properties_module._inference_setting_updated(None, context)

    settings = SimpleNamespace(
        manual_emotions=[CallbackItem("Neutral"), CallbackItem("Joy")]
    )

    properties_module.apply_effective_emotions(
        settings,
        ("Neutral", "Joy"),
        (0.25, 0.75),
    )
    assert calls == []

    settings.manual_emotions[1].value = 0.5
    assert calls == [scene]


@pytest.mark.parametrize(
    ("channels", "values"),
    (
        (("Joy", "Neutral"), (0.1, 0.9)),
        (("Neutral", "Joy"), (0.1,)),
        (("Neutral", "Joy"), (0.1, float("nan"))),
    ),
)
def test_effective_emotions_require_the_exact_loaded_schema(
    properties_module: ModuleType,
    channels: tuple[str, ...],
    values: tuple[float, ...],
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)

    with pytest.raises(ValueError):
        properties_module.apply_effective_emotions(settings, channels, values)


def test_effective_emotion_display_clamps_without_mutating_preferred(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    properties_module.load_preferred_emotion(settings)

    properties_module.apply_effective_emotions(
        settings,
        ("Neutral", "Joy"),
        (-0.5, 1.5),
    )

    assert [item.value for item in settings.manual_emotions] == [0.0, 1.0]
    assert [item.value for item in settings.preferred_emotions] == [0.5, 0.2]


def test_prediction_delay_does_not_reset_inference(
    properties_module: ModuleType,
) -> None:
    annotation = properties_module.A2FSceneSettings.__annotations__[
        "prediction_delay"
    ]
    assert "update" not in annotation


def test_schema_materializes_dynamic_emotions(
    properties_module: ModuleType,
) -> None:
    assert set(properties_module.AUDIO2FACE_SETTING_FIELDS) == set(
        AUDIO2FACE_DEFAULTS
    )
    settings = _settings()

    assert (
        properties_module.apply_model_schema(
            settings,
            _schema(),
            MODEL_SIGNATURE,
        )
        == CHANNELS
    )
    assert [(item.name, item.value) for item in settings.manual_emotions] == [
        ("Neutral", 0.5),
        ("Joy", 0.2),
    ]


def test_reload_preserves_values_only_for_the_exact_same_schema(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    settings.manual_emotions[1].value = 0.82
    for name, value in AUDIO2FACE_TUNING.items():
        setattr(settings, name, value)
    properties_module.load_preferred_emotion(settings)

    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    assert [(item.name, item.value) for item in settings.manual_emotions] == [
        ("Neutral", 0.5),
        ("Joy", 0.82),
    ]
    assert [(item.name, item.value) for item in settings.preferred_emotions] == [
        ("Neutral", 0.5),
        ("Joy", 0.82),
    ]
    assert {
        name: getattr(settings, name)
        for name in AUDIO2FACE_DEFAULTS
    } == AUDIO2FACE_TUNING


def test_changed_schema_resets_every_control_to_advertised_defaults(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    settings.manual_emotions[1].value = 0.82
    for name, value in AUDIO2FACE_TUNING.items():
        setattr(settings, name, value)
    properties_module.load_preferred_emotion(settings)

    schema = _schema()
    changed_defaults = AUDIO2FACE_DEFAULTS.copy()
    changed_defaults["input_strength"] = 1.5
    schema["audio2face_defaults"] = changed_defaults
    properties_module.apply_model_schema(settings, schema, MODEL_SIGNATURE)
    settings.auto_audio2emotion = True

    assert not settings.preferred_emotions
    assert properties_module.inference_settings(settings) == {
        "audio2face": changed_defaults,
        "auto_audio2emotion": True,
        "manual_emotions": {"Neutral": 0.5, "Joy": 0.2},
        "audio2emotion": {
            "emotion_strength": 0.6,
            "emotion_contrast": 1.0,
            "max_emotions": 6,
            "live_blend_coef": 0.7,
            "transition_smoothing": 0.5,
            "preferred_emotion": None,
            "preferred_emotion_strength": 0.5,
        },
    }


def test_inference_settings_freezes_face_manual_and_automatic_controls(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    settings.manual_emotions[0].value = 0.25
    settings.manual_emotions[1].value = 0.75
    settings.auto_audio2emotion = True
    settings.a2e_emotion_strength = 0.8
    settings.a2e_emotion_contrast = 1.4
    settings.a2e_max_emotions = 3
    settings.a2e_live_blend_coef = 0.4
    settings.a2e_transition_smoothing = 0.9
    settings.a2e_preferred_emotion_strength = 0.35
    settings.input_strength = 2.0
    settings.blink_strength = 1.5
    settings.right_eye_rot_y_offset = -4.5
    settings.eye_saccade_seed = 4999
    properties_module.load_preferred_emotion(settings)
    settings.manual_emotions[0].value = 0.1
    settings.manual_emotions[1].value = 0.9

    expected_audio2face = AUDIO2FACE_DEFAULTS.copy()
    expected_audio2face.update(
        input_strength=2.0,
        blink_strength=1.5,
        right_eye_rot_y_offset=-4.5,
        eye_saccade_seed=4999,
    )
    assert properties_module.inference_settings(settings) == {
        "audio2face": expected_audio2face,
        "auto_audio2emotion": True,
        "manual_emotions": {"Neutral": 0.1, "Joy": 0.9},
        "audio2emotion": {
            "emotion_strength": 0.8,
            "emotion_contrast": 1.4,
            "max_emotions": 3,
            "live_blend_coef": 0.4,
            "transition_smoothing": 0.9,
            "preferred_emotion": {"Neutral": 0.25, "Joy": 0.75},
            "preferred_emotion_strength": 0.35,
        },
    }

    properties_module.clear_preferred_emotion(settings)
    assert (
        properties_module.inference_settings(settings)["audio2emotion"][
            "preferred_emotion"
        ]
        is None
    )


@pytest.mark.parametrize(
    "corrupt",
    (
        lambda settings: settings.manual_emotions.pop(),
        lambda settings: setattr(settings.manual_emotions[0], "name", "Renamed"),
        lambda settings: settings.preferred_emotions.pop(),
        lambda settings: setattr(settings.preferred_emotions[0], "name", "Renamed"),
    ),
)
def test_exact_schema_signature_never_masks_corrupt_saved_collections(
    properties_module: ModuleType,
    corrupt: Callable[[SimpleNamespace], object],
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    properties_module.load_preferred_emotion(settings)
    corrupt(settings)

    with pytest.raises(ValueError, match="saved|unsupported"):
        properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda schema: schema.update(extra=True),
        lambda schema: schema.update(channels=[]),
        lambda schema: schema.update(parameters={}),
        lambda schema: schema.update(
            emotion_channels=[{"name": "Joy", "default": 0}]
        ),
        lambda schema: schema["emotion_channels"].append(
            {"name": "Joy", "default": 0.1}
        ),
        lambda schema: schema["audio2face_defaults"].pop("skin_strength"),
        lambda schema: schema["audio2face_defaults"].update(extra=1.0),
        lambda schema: schema["audio2face_defaults"].update(
            input_strength=3.1
        ),
        lambda schema: schema["audio2face_defaults"].update(
            eye_saccade_seed=0.0
        ),
    ),
)
def test_invalid_schema_does_not_mutate_controls(
    properties_module: ModuleType,
    mutate: Callable[[dict[str, object]], object],
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    before_emotions = copy.deepcopy(
        [(item.name, item.value) for item in settings.manual_emotions]
    )
    before_audio2face = {
        name: getattr(settings, name)
        for name in AUDIO2FACE_DEFAULTS
    }
    schema = _schema()
    mutate(schema)

    with pytest.raises(ValueError):
        properties_module.apply_model_schema(settings, schema, MODEL_SIGNATURE)

    assert (
        [(item.name, item.value) for item in settings.manual_emotions]
        == before_emotions
    )
    assert {
        name: getattr(settings, name)
        for name in AUDIO2FACE_DEFAULTS
    } == before_audio2face


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        (
            "input_strength",
            1,
            r"model_schema\.audio2face_defaults\.input_strength "
            r"must be a finite float",
        ),
        (
            "input_strength",
            True,
            r"model_schema\.audio2face_defaults\.input_strength "
            r"must be a finite float",
        ),
        (
            "input_strength",
            float("nan"),
            r"model_schema\.audio2face_defaults\.input_strength "
            r"must be a finite float",
        ),
        (
            "face_mask_softness",
            0.0,
            r"model_schema\.audio2face_defaults\.face_mask_softness "
            r"must be in \[0\.001, 0\.5\]",
        ),
        (
            "right_eye_rot_x_offset",
            10.1,
            r"model_schema\.audio2face_defaults\.right_eye_rot_x_offset "
            r"must be in \[-10, 10\]",
        ),
        (
            "eye_saccade_seed",
            0.0,
            r"model_schema\.audio2face_defaults\.eye_saccade_seed "
            r"must be an integer in \[0, 4999\]",
        ),
        (
            "eye_saccade_seed",
            True,
            r"model_schema\.audio2face_defaults\.eye_saccade_seed "
            r"must be an integer in \[0, 4999\]",
        ),
        (
            "eye_saccade_seed",
            5000,
            r"model_schema\.audio2face_defaults\.eye_saccade_seed "
            r"must be an integer in \[0, 4999\]",
        ),
    ),
)
def test_model_schema_rejects_exact_malformed_audio2face_defaults(
    properties_module: ModuleType,
    name: str,
    value: object,
    message: str,
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    before_signature = settings.model_schema_signature
    before_values = {
        field: getattr(settings, field)
        for field in AUDIO2FACE_DEFAULTS
    }
    schema = _schema()
    schema["audio2face_defaults"][name] = value

    with pytest.raises(ValueError, match=message):
        properties_module.apply_model_schema(settings, schema, MODEL_SIGNATURE)

    assert settings.model_schema_signature == before_signature
    assert {
        field: getattr(settings, field)
        for field in AUDIO2FACE_DEFAULTS
    } == before_values


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        (
            "input_strength",
            1,
            r"Audio2Face settings\.input_strength must be a finite float",
        ),
        (
            "saccade_strength",
            float("inf"),
            r"Audio2Face settings\.saccade_strength must be a finite float",
        ),
        (
            "eye_saccade_seed",
            1.0,
            r"Audio2Face settings\.eye_saccade_seed "
            r"must be an integer in \[0, 4999\]",
        ),
    ),
)
def test_inference_settings_rejects_exact_malformed_audio2face_controls(
    properties_module: ModuleType,
    name: str,
    value: object,
    message: str,
) -> None:
    settings = _settings()
    setattr(settings, name, value)

    with pytest.raises(ValueError, match=message):
        properties_module.inference_settings(settings)


def test_inference_settings_accepts_declared_and_blender_float_endpoints(
    properties_module: ModuleType,
) -> None:
    for name, bounds in properties_module._AUDIO2FACE_FLOAT_RANGES.items():
        for endpoint in bounds:
            stored = struct.unpack("=f", struct.pack("=f", endpoint))[0]
            for value in (endpoint, stored):
                settings = _settings()
                setattr(settings, name, value)

                payload = properties_module.inference_settings(settings)

                assert payload["audio2face"][name] == endpoint


def test_inference_settings_rejects_next_float_above_endpoint(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    endpoint_bits = struct.unpack("=I", struct.pack("=f", 0.2))[0]
    settings.lip_open_offset = struct.unpack(
        "=f", struct.pack("=I", endpoint_bits + 1)
    )[0]

    with pytest.raises(ValueError, match=r"lip_open_offset must be in"):
        properties_module.inference_settings(settings)
