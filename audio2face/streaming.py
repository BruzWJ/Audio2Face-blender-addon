"""Public in-Blender ingress for source-agnostic mono float32 PCM streams."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bpy


def start_pcm_stream(scene: bpy.types.Scene) -> str:
    """Open one stream and return its operation ID; call from Blender's main thread."""

    from .runtime import get_controller

    return get_controller().start_pcm_stream(scene)


def get_pcm_stream_requirements(
    scene: bpy.types.Scene,
) -> tuple[int, int] | None:
    """Return ``(sample_rate, prebuffer_samples)`` or ``None`` while starting."""

    from .runtime import get_controller

    return get_controller().pcm_stream_requirements(scene)


def push_audio_f32le(
    audio_f32le: bytes,
    *,
    operation_id: str,
) -> str:
    """Queue one finite mono f32le chunk; audio-source threads may call this."""

    if type(audio_f32le) is not bytes:
        raise TypeError("audio_f32le must be an exact bytes payload")
    if type(operation_id) is not str:
        raise TypeError("operation_id must be an exact string")
    if not operation_id:
        raise ValueError("operation_id must not be empty")

    from .runtime import get_controller

    return get_controller().push_stream_audio(audio_f32le, operation_id=operation_id)


def end_pcm_stream(scene: bpy.types.Scene) -> None:
    """Close input normally and drain final model frames on Blender's main thread."""

    from .runtime import get_controller

    get_controller().end_stream(scene)


def stop_pcm_stream(scene: bpy.types.Scene) -> None:
    """Cancel the stream immediately while leaving the GPU model loaded."""

    from .runtime import get_controller

    get_controller().stop_stream(scene)


__all__ = [
    "end_pcm_stream",
    "get_pcm_stream_requirements",
    "push_audio_f32le",
    "start_pcm_stream",
    "stop_pcm_stream",
]
