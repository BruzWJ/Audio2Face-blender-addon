from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from audio2face import streaming


@pytest.mark.parametrize("requirements", [(16_000, None), (16_000, 60_000)])
def test_public_pcm_requirements_use_explicit_scene(
    monkeypatch: pytest.MonkeyPatch,
    requirements: tuple[int, int | None],
) -> None:
    scene = object()
    calls: list[object] = []
    controller = SimpleNamespace(
        pcm_stream_requirements=lambda value: calls.append(value) or requirements
    )
    runtime = ModuleType("audio2face.runtime")
    runtime.get_controller = lambda: controller  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)

    assert streaming.get_pcm_stream_requirements(scene) == requirements
    assert calls == [scene]


def test_first_public_pcm_push_needs_no_start_or_operation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bytes, str]] = []
    controller = SimpleNamespace(
        queue_pcm_audio=lambda payload, *, scene_name: calls.append(
            (payload, scene_name)
        )
    )
    runtime = ModuleType("audio2face.runtime")
    runtime.get_controller = lambda: controller  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)

    payload = bytes(4)
    assert streaming.push_audio_f32le(payload, scene_name="Scene") is None
    assert calls == [(payload, "Scene")]


def test_end_marks_the_same_scene_input_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    controller = SimpleNamespace(
        finish_pcm_audio=lambda *, scene_name: calls.append(scene_name)
    )
    runtime = ModuleType("audio2face.runtime")
    runtime.get_controller = lambda: controller  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)

    assert streaming.end_pcm_stream(scene_name="Scene") is None
    assert calls == ["Scene"]
