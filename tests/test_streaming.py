from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from audio2face import streaming


@pytest.mark.parametrize("requirements", [None, (16_000, 60_000)])
def test_public_pcm_requirements_use_explicit_or_context_scene(
    monkeypatch: pytest.MonkeyPatch,
    requirements: tuple[int, int] | None,
) -> None:
    context_scene = object()
    explicit_scene = object()
    bpy = ModuleType("bpy")
    bpy.context = SimpleNamespace(scene=context_scene)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    calls: list[object] = []
    controller = SimpleNamespace(
        pcm_stream_requirements=lambda scene: calls.append(scene) or requirements
    )
    runtime = ModuleType("audio2face.runtime")
    runtime.get_controller = lambda: controller  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)

    assert streaming.get_pcm_stream_requirements() == requirements
    assert streaming.get_pcm_stream_requirements(explicit_scene) == requirements
    assert calls == [context_scene, explicit_scene]
    assert "get_pcm_stream_requirements" in streaming.__all__
