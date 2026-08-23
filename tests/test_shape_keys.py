from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from audio2face.frame_stream import sample_linear


class _ShapeKey:
    def __init__(self) -> None:
        self.value = 0.0


class _ShapeKeys:
    def __init__(self) -> None:
        self.key_blocks = {"jawOpen": _ShapeKey()}

    def as_pointer(self) -> int:
        return id(self)


@pytest.fixture
def shape_keys_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    bpy = ModuleType("bpy")
    bpy.types = SimpleNamespace(Object=object)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    name = "audio2face._shape_keys_test"
    source = Path(__file__).resolve().parents[1] / "audio2face" / "shape_keys.py"
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_sampled_frame_drives_matching_shape_key_only(
    shape_keys_module: ModuleType,
) -> None:
    shape_keys = _ShapeKeys()
    target = SimpleNamespace(data=SimpleNamespace(shape_keys=shape_keys))
    frame = sample_linear([0, 100], [[0.0], [1.0]], 25.0)

    shape_keys_module.apply_shape_key_frame((target,), ("jawOpen",), frame)

    assert shape_keys.key_blocks["jawOpen"].value == pytest.approx(0.25)


def test_output_channels_require_exact_json_array_of_52_unique_names(
    shape_keys_module: ModuleType,
) -> None:
    channels = [f"channel{index}" for index in range(52)]
    assert shape_keys_module.validate_output_channels(channels) == tuple(channels)

    with pytest.raises(shape_keys_module.ShapeKeyStreamError, match="JSON array"):
        shape_keys_module.validate_output_channels(tuple(channels))
    with pytest.raises(shape_keys_module.ShapeKeyStreamError, match="exactly 52"):
        shape_keys_module.validate_output_channels(channels[:-1])
    channels[-1] = channels[0]
    with pytest.raises(shape_keys_module.ShapeKeyStreamError, match="duplicate"):
        shape_keys_module.validate_output_channels(channels)


def test_all_registered_meshes_subscribe_and_missing_objects_are_ignored(
    shape_keys_module: ModuleType,
) -> None:
    first_target = SimpleNamespace(
        type="MESH",
        data=SimpleNamespace(shape_keys=None),
        as_pointer=lambda: 1,
    )
    second_target = SimpleNamespace(
        type="MESH",
        data=SimpleNamespace(shape_keys=None),
        as_pointer=lambda: 2,
    )
    settings = SimpleNamespace(
        target_meshes=[
            SimpleNamespace(object=first_target),
            SimpleNamespace(object=None),
            SimpleNamespace(object=second_target),
        ]
    )

    subscriptions = shape_keys_module.resolve_target_meshes(settings)
    shape_keys_module.apply_shape_key_frame(
        subscriptions,
        tuple(f"channel{index}" for index in range(52)),
        (0.0,) * 52,
    )

    assert subscriptions == (first_target, second_target)
