from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class _EmotionCollection(list[SimpleNamespace]):
    def add(self) -> SimpleNamespace:
        item = SimpleNamespace(name="", value=0.0)
        self.append(item)
        return item

    def clear(self) -> None:
        super().clear()


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

    module_name = "a2f_blender._properties_test"
    source = Path(__file__).resolve().parents[1] / "a2f_blender" / "properties.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        input_strength=1.0,
        lower_face_smoothing=0.0,
        upper_face_smoothing=0.0,
        lower_face_strength=1.0,
        upper_face_strength=1.0,
        face_mask_level=0.5,
        face_mask_softness=0.1,
        skin_strength=1.0,
        blink_strength=1.0,
        blink_offset=0.0,
        eyelid_open_offset=0.0,
        lip_open_offset=0.0,
        auto_audio2emotion=False,
        manual_emotions=_EmotionCollection(),
        emotion_strength=0.6,
        emotion_contrast=1.0,
        emotion_smoothing=0.7,
        emotion_transition_time=0.5,
        max_emotions=6,
    )


def _defaults(manual_values: dict[str, float]) -> dict[str, object]:
    return {
        "input_strength": 0.9,
        "skin": {
            "lower_face_smoothing": 0.1,
            "upper_face_smoothing": 0.2,
            "lower_face_strength": 0.8,
            "upper_face_strength": 0.7,
            "face_mask_level": 0.6,
            "face_mask_softness": 0.3,
            "skin_strength": 0.95,
            "blink_strength": 0.85,
            "blink_offset": -0.1,
            "eyelid_open_offset": 0.05,
            "lip_open_offset": 0.15,
        },
        "emotion": {
            "manual_values": manual_values,
            "auto": {
                "strength": 0.55,
                "contrast": 1.25,
                "smoothing": 0.65,
                "transition_time": 0.4,
                "max_emotions": 4,
            },
        },
    }


def test_model_emotions_preserve_configured_values_by_name(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    settings.manual_emotions.extend(
        [
            SimpleNamespace(name="Joy", value=0.91),
            SimpleNamespace(name="RemovedByNewModel", value=0.4),
        ]
    )

    properties_module.apply_model_defaults(
        settings,
        _defaults({"Neutral": 0.5, "Joy": 0.2, "Sadness": 0.1}),
        ["Neutral", "Joy", "Sadness"],
    )

    assert [(item.name, item.value) for item in settings.manual_emotions] == [
        ("Neutral", 0.5),
        ("Joy", 0.91),
        ("Sadness", 0.1),
    ]
    assert settings.emotion_strength == pytest.approx(0.55)
    assert settings.emotion_contrast == pytest.approx(1.25)
    assert settings.emotion_smoothing == pytest.approx(0.65)
    assert settings.emotion_transition_time == pytest.approx(0.4)
    assert settings.max_emotions == 4


def test_tuning_parameters_always_emit_complete_emotion_document(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    settings.auto_audio2emotion = True
    settings.manual_emotions.extend(
        [
            SimpleNamespace(name="Neutral", value=0.4),
            SimpleNamespace(name="Joy", value=0.75),
        ]
    )

    payload = properties_module.tuning_parameters(settings)

    assert payload["emotion"] == {
        "auto_audio2emotion": True,
        "manual_values": {"Neutral": 0.4, "Joy": 0.75},
        "auto": {
            "strength": 0.6,
            "contrast": 1.0,
            "smoothing": 0.7,
            "transition_time": 0.5,
            "max_emotions": 6,
        },
    }


def test_reloading_the_same_emotion_schema_preserves_user_controls(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    settings.input_strength = 1.7
    settings.lower_face_strength = 1.6
    settings.emotion_strength = 0.23
    settings.emotion_transition_time = 0.8
    settings.max_emotions = 2
    settings.manual_emotions.extend(
        [
            SimpleNamespace(name="Neutral", value=0.31),
            SimpleNamespace(name="Joy", value=0.82),
        ]
    )

    properties_module.apply_model_defaults(
        settings,
        _defaults({"Neutral": 0.5, "Joy": 0.2}),
        ["Neutral", "Joy"],
    )

    assert settings.input_strength == pytest.approx(1.7)
    assert settings.lower_face_strength == pytest.approx(1.6)
    assert settings.emotion_strength == pytest.approx(0.23)
    assert settings.emotion_transition_time == pytest.approx(0.8)
    assert settings.max_emotions == 2
    assert [(item.name, item.value) for item in settings.manual_emotions] == [
        ("Neutral", 0.31),
        ("Joy", 0.82),
    ]


def test_model_emotion_names_must_exactly_match_defaults(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    settings.manual_emotions.append(SimpleNamespace(name="Joy", value=0.8))

    with pytest.raises(ValueError, match="do not match"):
        properties_module.apply_model_defaults(
            settings,
            _defaults({"Neutral": 0.5}),
            ["Neutral", "Joy"],
        )

    assert [(item.name, item.value) for item in settings.manual_emotions] == [
        ("Joy", 0.8)
    ]
