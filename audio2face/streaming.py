"""Public ingress for automatic mono-float32 live audio inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bpy


def get_pcm_stream_requirements(
    scene: bpy.types.Scene,
) -> tuple[int, int | None]:
    """Return ``(sample_rate, prebuffer_samples)`` for the loaded model."""

    from .runtime import get_controller

    return get_controller().pcm_stream_requirements(scene)


def push_audio_f32le(audio_f32le: bytes, *, scene_name: str) -> None:
    """Queue mono f32le audio; the first chunk automatically starts inference."""

    from .runtime import get_controller

    get_controller().queue_pcm_audio(audio_f32le, scene_name=scene_name)


def end_pcm_stream(*, scene_name: str) -> None:
    """End live input after every chunk already queued for this scene."""

    from .runtime import get_controller

    get_controller().finish_pcm_audio(scene_name=scene_name)


__all__ = [
    "end_pcm_stream",
    "get_pcm_stream_requirements",
    "push_audio_f32le",
]
