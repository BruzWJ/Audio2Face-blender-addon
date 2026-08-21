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


def test_public_pcm_push_forwards_exact_bytes_and_operation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bytes, str]] = []
    controller = SimpleNamespace(
        push_stream_audio=lambda payload, *, operation_id: (
            calls.append((payload, operation_id)) or "request-1"
        )
    )
    runtime = ModuleType("audio2face.runtime")
    runtime.get_controller = lambda: controller  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)

    payload = b"\x00\x00\x00\x00"
    assert (
        streaming.push_audio_f32le(payload, operation_id="stream-1")
        == "request-1"
    )
    assert calls == [(payload, "stream-1")]


@pytest.mark.parametrize("payload", [bytearray(4), memoryview(bytes(4))])
def test_public_pcm_push_rejects_bytes_aliases(payload: object) -> None:
    with pytest.raises(TypeError, match="exact bytes"):
        streaming.push_audio_f32le(payload, operation_id="stream-1")


@pytest.mark.parametrize(
    ("operation_id", "error_type"),
    [(b"stream-1", TypeError), ("", ValueError)],
)
def test_public_pcm_push_requires_nonempty_exact_operation_id(
    operation_id: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        streaming.push_audio_f32le(bytes(4), operation_id=operation_id)
