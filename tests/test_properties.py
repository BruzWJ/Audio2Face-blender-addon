from __future__ import annotations

import copy
import importlib.util
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
        "emotion_channels": [
            {"name": "Neutral", "default": 0.5},
            {"name": "Joy", "default": 0.2},
        ],
    }


def test_emotion_configuration_is_not_hidden_by_auto_mode() -> None:
    assert "if settings.auto_audio2emotion:" not in UI_SOURCE
    for name in (
        "a2e_preferred_emotion_strength",
        "a2e_emotion_strength",
        "a2e_emotion_contrast",
        "a2e_max_emotions",
        "a2e_live_blend_coef",
        "a2e_transition_smoothing",
    ):
        assert f'auto_controls.prop(settings, "{name}")' in UI_SOURCE
    assert '"a2f.load_preferred_emotion"' in UI_SOURCE
    assert '"a2f.clear_preferred_emotion"' in UI_SOURCE


def test_schema_materializes_dynamic_emotions(
    properties_module: ModuleType,
) -> None:
    assert not hasattr(properties_module, "A2FModelParameterItem")
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


def test_changed_schema_resets_every_control_to_advertised_defaults(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    settings.manual_emotions[1].value = 0.82
    properties_module.load_preferred_emotion(settings)

    schema = _schema()
    schema["emotion_channels"] = [
        {"name": "Joy", "default": 0.1},
        {"name": "Sadness", "default": 0.3},
    ]
    properties_module.apply_model_schema(settings, schema, MODEL_SIGNATURE)
    settings.auto_audio2emotion = True

    assert not settings.preferred_emotions
    assert properties_module.emotion_settings(settings) == {
        "auto_audio2emotion": True,
        "manual_emotions": {"Joy": 0.1, "Sadness": 0.3},
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


def test_emotion_settings_freezes_manual_and_automatic_controls(
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
    properties_module.load_preferred_emotion(settings)
    settings.manual_emotions[0].value = 0.1
    settings.manual_emotions[1].value = 0.9

    assert properties_module.emotion_settings(settings) == {
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
        properties_module.emotion_settings(settings)["audio2emotion"][
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
    ),
)
def test_invalid_schema_does_not_mutate_controls(
    properties_module: ModuleType,
    mutate: Callable[[dict[str, object]], object],
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema(), MODEL_SIGNATURE)
    before = copy.deepcopy(
        [(item.name, item.value) for item in settings.manual_emotions]
    )
    schema = _schema()
    mutate(schema)

    with pytest.raises(ValueError):
        properties_module.apply_model_schema(settings, schema, MODEL_SIGNATURE)

    assert [(item.name, item.value) for item in settings.manual_emotions] == before
