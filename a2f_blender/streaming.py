"""Public in-Blender ingress for source-agnostic mono float32 PCM streams."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bpy


def _scene_or_context(scene: bpy.types.Scene | None) -> bpy.types.Scene:
    import bpy

    resolved = scene if scene is not None else bpy.context.scene
    if resolved is None:
        raise RuntimeError("an editable Blender scene is required")
    return resolved


def start_pcm_stream(scene: bpy.types.Scene | None = None) -> str:
    """Open one stream and return its ID; call from Blender's main thread."""

    from .runtime import get_controller

    return get_controller().start_pcm_stream(_scene_or_context(scene))


def get_pcm_stream_requirements(
    scene: bpy.types.Scene | None = None,
) -> tuple[int, int] | None:
    """Return ``(sample_rate, prebuffer_samples)`` or ``None`` while starting."""

    from .runtime import get_controller

    return get_controller().pcm_stream_requirements(_scene_or_context(scene))


def push_audio_f32le(
    audio_f32le: bytes | bytearray | memoryview,
    *,
    stream_id: str | None = None,
) -> str:
    """Queue one finite mono f32le chunk; audio-source threads may call this."""

    from .runtime import get_controller

    return get_controller().push_stream_audio(audio_f32le, stream_id=stream_id)


def end_pcm_stream(scene: bpy.types.Scene | None = None) -> None:
    """Close input normally and drain final model frames on Blender's main thread."""

    from .runtime import get_controller

    get_controller().end_stream(_scene_or_context(scene))


def stop_pcm_stream(scene: bpy.types.Scene | None = None) -> None:
    """Cancel the stream immediately while leaving the GPU model loaded."""

    from .runtime import get_controller

    get_controller().stop_stream(_scene_or_context(scene))


__all__ = [
    "end_pcm_stream",
    "get_pcm_stream_requirements",
    "push_audio_f32le",
    "start_pcm_stream",
    "stop_pcm_stream",
]
