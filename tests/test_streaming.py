from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from audio2face import streaming


@pytest.mark.parametrize("requirements", [None, (16_000, 60_000)])
def test_public_pcm_requirements_use_explicit_scene(
    monkeypatch: pytest.MonkeyPatch,
    requirements: tuple[int, int] | None,
) -> None:
    scene = object()
    calls: list[object] = []
    controller = SimpleNamespace(
        pcm_stream_requirements=lambda scene: calls.append(scene) or requirements
    )
    runtime = ModuleType("audio2face.runtime")
    runtime.get_controller = lambda: controller  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)

    assert streaming.get_pcm_stream_requirements(scene) == requirements
    assert calls == [scene]
    assert "get_pcm_stream_requirements" in streaming.__all__
