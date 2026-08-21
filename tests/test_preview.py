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
def preview_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    bpy = ModuleType("bpy")
    bpy.types = SimpleNamespace(Object=object, Scene=object)  # type: ignore[attr-defined]
    bpy.data = SimpleNamespace(scenes={})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    module_name = "audio2face._preview_test"
    source = Path(__file__).resolve().parents[1] / "audio2face" / "preview.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_sampled_frame_is_the_exact_internal_preview_tuple(
    preview_module: ModuleType,
) -> None:
    shape_keys = _ShapeKeys()
    target = SimpleNamespace(data=SimpleNamespace(shape_keys=shape_keys))
    frame = sample_linear([0, 100], [[0.0], [1.0]], 25.0)

    assert type(frame) is tuple
    preview_module.apply_shape_key_frame((target,), ("jawOpen",), frame)

    assert shape_keys.key_blocks["jawOpen"].value == pytest.approx(0.25)


def test_shape_key_delivery_rejects_mutable_frame_alias(
    preview_module: ModuleType,
) -> None:
    target = SimpleNamespace(data=SimpleNamespace(shape_keys=_ShapeKeys()))

    with pytest.raises(preview_module.PreviewError, match="frozen frame tuple"):
        preview_module.apply_shape_key_frame((target,), ("jawOpen",), [0.5])
