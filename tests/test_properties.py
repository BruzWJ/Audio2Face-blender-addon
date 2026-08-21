from __future__ import annotations

import copy
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


CHANNELS = tuple(f"modelChannel{index}" for index in range(52))


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
        identity_index=0,
        model_identities=_Collection(lambda: SimpleNamespace(name="")),
        auto_audio2emotion=False,
        manual_emotions=_Collection(lambda: SimpleNamespace(name="", value=0.0)),
        model_parameters=_Collection(
            lambda: SimpleNamespace(path="", kind="", float_value=0.0, int_value=0)
        ),
    )


def _schema() -> dict[str, object]:
    return {
        "identities": ["Aki", "Mark"],
        "channels": list(CHANNELS),
        "parameters": {
            "/input_strength": 0.9,
            "/audio2emotion/max_emotions": 4,
        },
        "emotion_channels": [
            {"name": "Neutral", "default": 0.5},
            {"name": "Joy", "default": 0.2},
        ],
    }


def _parameter_values(settings: SimpleNamespace) -> list[tuple[object, ...]]:
    return [
        (item.path, item.kind, item.int_value if item.kind == "integer" else item.float_value)
        for item in settings.model_parameters
    ]


def test_schema_materializes_model_controls(properties_module: ModuleType) -> None:
    settings = _settings()
    settings.identity_index = 9

    assert properties_module.apply_model_schema(settings, _schema()) == CHANNELS
    assert [item.name for item in settings.model_identities] == ["Aki", "Mark"]
    assert settings.identity_index == 0
    assert _parameter_values(settings) == [
        ("/input_strength", "float", 0.9),
        ("/audio2emotion/max_emotions", "integer", 4),
    ]
    assert [(item.name, item.value) for item in settings.manual_emotions] == [
        ("Neutral", 0.5),
        ("Joy", 0.2),
    ]


def test_reload_preserves_matching_values_and_serializes_exact_schema(
    properties_module: ModuleType,
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema())
    settings.model_parameters[0].float_value = 1.7
    settings.model_parameters[1].int_value = 2
    settings.manual_emotions[1].value = 0.82

    schema = _schema()
    schema["parameters"] = {
        "/audio2emotion/max_emotions": 3.5,
        "/input_strength": 0.5,
        "/skin/skin_strength": 1.1,
    }
    schema["emotion_channels"] = [
        {"name": "Joy", "default": 0.1},
        {"name": "Sadness", "default": 0.3},
    ]
    properties_module.apply_model_schema(settings, schema)
    settings.auto_audio2emotion = True

    assert _parameter_values(settings) == [
        ("/audio2emotion/max_emotions", "float", 3.5),
        ("/input_strength", "float", 1.7),
        ("/skin/skin_strength", "float", 1.1),
    ]
    assert properties_module.tuning_parameters(settings) == {
        "auto_audio2emotion": True,
        "manual_emotions": {"Joy": 0.82, "Sadness": 0.3},
        "parameters": {
            "/audio2emotion/max_emotions": 3.5,
            "/input_strength": 1.7,
            "/skin/skin_strength": 1.1,
        },
    }


@pytest.mark.parametrize(
    "mutate",
    (
        lambda schema: schema.update(extra=True),
        lambda schema: schema.update(channels=[]),
        lambda schema: schema.update(parameters={"/invalid": True}),
        lambda schema: schema["emotion_channels"].append(
            {"name": "Joy", "default": 0.1}
        ),
        lambda schema: schema.update(identities=[]),
    ),
)
def test_invalid_schema_does_not_mutate_controls(
    properties_module: ModuleType,
    mutate: Callable[[dict[str, object]], object],
) -> None:
    settings = _settings()
    properties_module.apply_model_schema(settings, _schema())
    before = copy.deepcopy(
        (_parameter_values(settings), [(item.name, item.value) for item in settings.manual_emotions])
    )
    schema = _schema()
    mutate(schema)

    with pytest.raises(ValueError):
        properties_module.apply_model_schema(settings, schema)

    assert (
        _parameter_values(settings),
        [(item.name, item.value) for item in settings.manual_emotions],
    ) == before
