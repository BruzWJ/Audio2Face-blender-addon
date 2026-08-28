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
OPERATORS_SOURCE = (
    Path(__file__).resolve().parents[1] / "audio2face" / "operators.py"
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
    bpy.path = SimpleNamespace(abspath=lambda value: value)  # type: ignore[attr-defined]
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
        preferred_emotions=_Collection(
            lambda: SimpleNamespace(name="", value=0.0)
        ),
        mixed_emotions=_Collection(
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


def test_emotion_configuration_uses_parallel_collapsible_sections() -> None:
    for name in (
        "a2e_emotion_strength",
        "a2e_emotion_contrast",
        "a2e_max_emotions",
        "a2e_live_blend_coef",
        "a2e_transition_smoothing",
    ):
        assert re.search(
            rf'emotion_controls\.prop\(\s*settings,\s*"{name}",\s*slider=True,?\s*\)',
            UI_SOURCE,
        )
    assert re.search(
        r'emotion_header, emotion_body = layout\.panel\(\s*'
        r'"audio2face_emotion_tuning",\s*default_closed=True,?\s*\)',
        UI_SOURCE,
    )
    assert re.search(
        r'preferred_header, preferred_body = layout\.panel\(\s*'
        r'"audio2face_preferred_emotion",\s*default_closed=True,?\s*\)',
        UI_SOURCE,
    )
    assert UI_SOURCE.index("audio2face_preferred_emotion") < UI_SOURCE.index(
        "audio2face_emotion_tuning"
    )
    assert (
        'preferred_header.label(text="Preferred Emotion", icon="BOOKMARKS")'
        in UI_SOURCE
    )
    assert (
        'emotion_header.label(text="Emotion Tuning", icon="PREFERENCES")'
        in UI_SOURCE
    )
    assert re.search(
        r'preferred_body\.prop\(\s*settings,\s*'
        r'"a2e_preferred_emotion_strength",\s*slider=True,?\s*\)',
        UI_SOURCE,
    )
    assert "for emotion in settings.preferred_emotions:" in UI_SOURCE
    assert "for emotion in settings.mixed_emotions:" in UI_SOURCE
    assert "mixed_box = emotion_body.box()" in UI_SOURCE
    assert "mixed_controls.enabled = False" in UI_SOURCE
    assert UI_SOURCE.count('"a2f.reset_emotion_settings"') == 1
    assert "preferred_body.use_property_split = True" in UI_SOURCE
    assert "preferred_body.use_property_decorate = True" in UI_SOURCE
    assert "emotion_body.use_property_split = True" in UI_SOURCE
    assert "emotion_body.use_property_decorate = True" in UI_SOURCE


def test_preferred_emotions_are_saved_and_mixed_emotions_are_transient(
    properties_module: ModuleType,
) -> None:
    annotations = properties_module.A2FSceneSettings.__annotations__

    assert "SKIP_SAVE" not in annotations["preferred_emotions"]
    assert "SKIP_SAVE" in annotations["mixed_emotions"]
    assert "update=_inference_setting_updated" in annotations["auto_audio2emotion"]
    assert "update=_preferred_emotion_updated" in (
        properties_module.A2FPreferredEmotionItem.__annotations__["value"]
    )
    assert "update" not in (
        properties_module.A2FMixedEmotionItem.__annotations__["value"]
    )
    mixed_value = properties_module.A2FMixedEmotionItem.__annotations__["value"]
    assert "soft_min=0.0" in mixed_value
    assert "soft_max=2.0" in mixed_value
    assert "min=" not in mixed_value.replace("soft_min=", "")
    assert "max=" not in mixed_value.replace("soft_max=", "")


def test_automatic_emotion_strength_is_an_independent_two_x_multiplier(
    properties_module: ModuleType,
) -> None:
    annotation = properties_module.A2FSceneSettings.__annotations__[
        "a2e_emotion_strength"
    ]
    assert "min=0.0" in annotation
    assert "max=2.0" in annotation
    assert "subtype" not in annotation


def test_prediction_delay_uses_a_range_slider() -> None:
    assert (
        'playback_box.prop(settings, "prediction_delay", slider=True)'
        in UI_SOURCE
    )


def test_model_tuning_ui_exposes_only_the_fixed_audio2face_contract() -> None:
    assert 'text="Model Tuning"' in UI_SOURCE
    assert "AUDIO2FACE_SETTING_GROUPS" in UI_SOURCE
    assert "tuning_body.use_property_decorate = True" in UI_SOURCE
    assert re.search(
        r'tuning_header, tuning_body = layout\.panel\(\s*'
        r'"audio2face_model_tuning",\s*default_closed=True,?\s*\)',
        UI_SOURCE,
    )
    assert UI_SOURCE.count('"a2f.reset_model_tuning"') == 1
    assert "emotion_controls.enabled" not in UI_SOURCE


def test_runtime_status_box_uses_the_persistence_gate() -> None:
    assert "controller.status_notice(context.scene)" in UI_SOURCE


def test_playback_ui_uses_blender_native_transport_without_seconds_slider() -> None:
    assert "screen.is_animation_playing" in UI_SOURCE
    assert '"a2f.play_pause", text="Pause", icon="PAUSE"' in UI_SOURCE
    assert '"a2f.play_pause", text="Play", icon="PLAY"' in UI_SOURCE
    assert '"a2f.bake_animation"' in UI_SOURCE
    assert '"a2f.cancel_bake"' in UI_SOURCE


def test_selected_play_operator_calls_blender_native_timeline_transport() -> None:
    assert "screen.is_animation_playing" in OPERATORS_SOURCE
    assert "bpy.ops.screen.animation_pause()" in OPERATORS_SOURCE
    assert "bpy.ops.screen.animation_play()" in OPERATORS_SOURCE
    assert "configure_selected_audio" in OPERATORS_SOURCE
    assert "controller.start_selected_audio(" in OPERATORS_SOURCE
    assert "controller.cancel_selected_audio(scene)" in OPERATORS_SOURCE
    assert OPERATORS_SOURCE.index("controller.start_selected_audio(") < (
        OPERATORS_SOURCE.index("bpy.ops.screen.animation_play()")
    )


def test_input_mode_switch_is_never_disabled() -> None:
    assert "input_mode_row = input_box.row(align=True)" in UI_SOURCE
    assert 'input_mode_row.prop(settings, "input_mode", expand=True)' in UI_SOURCE
    assert "input_mode_row.enabled" not in UI_SOURCE


def test_target_ui_uses_the_shape_key_object_contract() -> None:
    assert 'text="Target Objects"' in UI_SOURCE
    assert 'text="Add Selected Objects"' in UI_SOURCE
    assert '"target_objects"' in UI_SOURCE


def test_audio_path_update_replaces_the_selected_native_sound_strip(
    properties_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    timeline = ModuleType("audio2face.selected_audio_timeline")
    timeline.configure_selected_audio = (  # type: ignore[attr-defined]
        lambda scene, path: calls.append(("configure", scene, path))
    )
    timeline.remove_selected_audio_strips = (  # type: ignore[attr-defined]
        lambda scene: calls.append(("remove", scene))
    )
    monkeypatch.setitem(sys.modules, timeline.__name__, timeline)
    settings = SimpleNamespace(audio_path="first.wav")
    scene = SimpleNamespace(audio2face=settings)
    context = SimpleNamespace(scene=scene)

    properties_module._audio_path_updated(settings, context)
    settings.audio_path = "second.wav"
    properties_module._audio_path_updated(settings, context)
    settings.audio_path = ""
    properties_module._audio_path_updated(settings, context)

    assert calls == [
        ("configure", scene, "first.wav"),
        ("configure", scene, "second.wav"),
        ("remove", scene),
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


def test_mixed_output_and_preferred_input_have_strict_ownership(
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

    class PreferredCallbackItem:
        def __init__(self) -> None:
            self.name = ""
            self._value = 0.0

        @property
        def value(self) -> float:
            return self._value

        @value.setter
        def value(self, value: float) -> None:
            self._value = value
            properties_module._preferred_emotion_updated(self, context)

    settings = _settings()
    settings.preferred_emotions = _Collection(PreferredCallbackItem)
    settings.mixed_emotions = _Collection(
        lambda: SimpleNamespace(name="", value=0.0)
    )
    scene = SimpleNamespace(audio2face=settings)
    context = SimpleNamespace(scene=scene)
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)

    properties_module.apply_mixed_emotions(
        settings,
        ("Neutral", "Joy"),
        (0.25, 0.75),
    )
    assert calls == []
    assert [item.value for item in settings.mixed_emotions] == [0.25, 0.75]
    assert [item.value for item in settings.preferred_emotions] == [0.5, 0.2]

    settings.preferred_emotions[1].value = 0.35
    assert calls == [scene]

    properties_module.apply_mixed_emotions(
        settings,
        ("Neutral", "Joy"),
        (0.6, 0.4),
    )
    assert [item.value for item in settings.mixed_emotions] == [0.6, 0.4]
    assert [item.value for item in settings.preferred_emotions] == [0.5, 0.35]
    assert calls == [scene]


def test_preferred_source_is_value_driven_and_all_zero_means_clear(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    settings.a2e_preferred_emotion_strength = 0.3

    preferred = properties_module.inference_settings(settings)["emotion_driver"][
        "preferred"
    ]
    assert preferred == {
        "values": {"Neutral": 0.5, "Joy": 0.2},
        "strength": 0.3,
    }

    for item in settings.preferred_emotions:
        item.value = 0.0
    assert properties_module.inference_settings(settings)["emotion_driver"][
        "preferred"
    ] is None

    settings.preferred_emotions[1].value = 0.4
    assert properties_module.inference_settings(settings)["emotion_driver"][
        "preferred"
    ] == {
        "values": {"Neutral": 0.0, "Joy": 0.4},
        "strength": 0.3,
    }


@pytest.mark.parametrize(
    ("channels", "values"),
    (
        (("Joy", "Neutral"), (0.1, 0.9)),
        (("Neutral", "Joy"), (0.1,)),
        (("Neutral", "Joy"), (0.1, float("nan"))),
    ),
)
def test_mixed_emotions_require_the_exact_loaded_schema(
    properties_module: ModuleType,
    channels: tuple[str, ...],
    values: tuple[float, ...],
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)

    with pytest.raises(ValueError):
        properties_module.apply_mixed_emotions(settings, channels, values)


def test_mixed_emotion_display_preserves_worker_values_without_mutating_preferred(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)

    properties_module.apply_mixed_emotions(
        settings,
        ("Neutral", "Joy"),
        (-0.5, 1.5),
    )

    assert [item.value for item in settings.mixed_emotions] == [-0.5, 1.5]
    assert [item.value for item in settings.preferred_emotions] == [0.5, 0.2]


def test_prediction_delay_is_not_an_inference_refresh_trigger(
    properties_module: ModuleType,
) -> None:
    annotation = properties_module.A2FSceneSettings.__annotations__[
        "prediction_delay"
    ]
    assert "update" not in annotation


def test_reset_helpers_only_unset_their_owned_rna_properties(
    properties_module: ModuleType,
) -> None:
    unset: list[str] = []
    settings = SimpleNamespace(property_unset=unset.append)

    properties_module.reset_model_tuning(settings)
    assert unset == list(properties_module.AUDIO2FACE_SETTING_FIELDS)
    unset.clear()
    properties_module.reset_emotion_settings(settings)
    assert unset == list(properties_module.EMOTION_SETTING_FIELDS)
    assert "a2e_preferred_emotion_strength" not in unset


def test_schema_materializes_preferred_defaults_and_empty_mixed_output(
    properties_module: ModuleType,
) -> None:
    assert set(properties_module.AUDIO2FACE_SETTING_FIELDS) == set(
        AUDIO2FACE_DEFAULTS
    )
    settings = _settings()

    properties_module.apply_model_schema(
        settings,
        _schema(),
        MODEL_SIGNATURE,
    )
    assert [(item.name, item.value) for item in settings.preferred_emotions] == [
        ("Neutral", 0.5),
        ("Joy", 0.2),
    ]
    assert [(item.name, item.value) for item in settings.mixed_emotions] == [
        ("Neutral", 0.0),
        ("Joy", 0.0),
    ]


def test_reload_preserves_values_only_for_the_exact_same_schema(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    settings.preferred_emotions[1].value = 0.82
    properties_module.apply_mixed_emotions(
        settings,
        ("Neutral", "Joy"),
        (0.3, 0.7),
    )
    for name, value in AUDIO2FACE_TUNING.items():
        setattr(settings, name, value)

    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    assert [(item.name, item.value) for item in settings.preferred_emotions] == [
        ("Neutral", 0.5),
        ("Joy", 0.82),
    ]
    assert [(item.name, item.value) for item in settings.mixed_emotions] == [
        ("Neutral", 0.0),
        ("Joy", 0.0),
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
    settings.preferred_emotions[1].value = 0.82
    properties_module.apply_mixed_emotions(
        settings,
        ("Neutral", "Joy"),
        (0.3, 0.7),
    )
    for name, value in AUDIO2FACE_TUNING.items():
        setattr(settings, name, value)

    schema = _schema()
    changed_defaults = AUDIO2FACE_DEFAULTS.copy()
    changed_defaults["input_strength"] = 1.5
    schema["audio2face_defaults"] = changed_defaults
    properties_module.apply_model_schema(settings, schema, MODEL_SIGNATURE)

    assert [(item.name, item.value) for item in settings.preferred_emotions] == [
        ("Neutral", 0.5),
        ("Joy", 0.2),
    ]
    assert [(item.name, item.value) for item in settings.mixed_emotions] == [
        ("Neutral", 0.0),
        ("Joy", 0.0),
    ]
    assert {
        name: getattr(settings, name) for name in AUDIO2FACE_DEFAULTS
    } == changed_defaults
    assert (
        settings.a2e_emotion_strength,
        settings.a2e_emotion_contrast,
        settings.a2e_max_emotions,
        settings.a2e_live_blend_coef,
        settings.a2e_transition_smoothing,
        settings.a2e_preferred_emotion_strength,
    ) == (0.6, 1.0, 6, 0.7, 0.5, 0.5)


@pytest.mark.parametrize("generated_active", (False, True))
def test_inference_settings_composes_emotion_drivers(
    properties_module: ModuleType,
    generated_active: bool,
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    settings.preferred_emotions[0].value = 0.25
    settings.preferred_emotions[1].value = 0.75
    settings.auto_audio2emotion = generated_active
    settings.a2e_emotion_strength = 0.8
    settings.a2e_emotion_contrast = 1.4
    settings.a2e_max_emotions = 3
    settings.a2e_live_blend_coef = 0.4
    settings.a2e_transition_smoothing = 0.9
    settings.a2e_preferred_emotion_strength = 0.35

    driver = properties_module.inference_settings(settings)["emotion_driver"]

    assert driver == {
        "emotion_strength": 0.8,
        "generated": (
            {
                "emotion_contrast": 1.4,
                "max_emotions": 3,
                "live_blend_coef": 0.4,
                "transition_smoothing": 0.9,
            }
            if generated_active
            else None
        ),
        "preferred": {
            "values": {"Neutral": 0.25, "Joy": 0.75},
            "strength": 0.35,
        },
    }


@pytest.mark.parametrize(
    "corrupt",
    (
        lambda settings: settings.preferred_emotions.pop(),
        lambda settings: setattr(settings.preferred_emotions[0], "name", "Renamed"),
        lambda settings: settings.preferred_emotions.reverse(),
    ),
)
def test_exact_schema_repairs_invalid_saved_preferred_collection(
    properties_module: ModuleType,
    corrupt: Callable[[SimpleNamespace], object],
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    corrupt(settings)

    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)

    assert [(item.name, item.value) for item in settings.preferred_emotions] == [
        ("Neutral", 0.5),
        ("Joy", 0.2),
    ]


def test_same_schema_rebuilds_transient_mixed_emotions(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    settings.mixed_emotions.pop()

    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)

    assert [(item.name, item.value) for item in settings.mixed_emotions] == [
        ("Neutral", 0.0),
        ("Joy", 0.0),
    ]


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
    before_preferred = copy.deepcopy(
        [(item.name, item.value) for item in settings.preferred_emotions]
    )
    before_mixed = copy.deepcopy(
        [(item.name, item.value) for item in settings.mixed_emotions]
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
        [(item.name, item.value) for item in settings.preferred_emotions]
        == before_preferred
    )
    assert (
        [(item.name, item.value) for item in settings.mixed_emotions]
        == before_mixed
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
