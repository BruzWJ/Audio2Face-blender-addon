"""Map Selected Audio onto Blender's native timeline."""

from __future__ import annotations

import math
from typing import Any

SELECTED_AUDIO_STRIP_NAME = "Audio2Face Selected Audio"
SELECTED_AUDIO_OWNER_KEY = "audio2face_selected_audio_owner"
SELECTED_AUDIO_OWNER_VALUE = "audio2face/selected-wav/1"


class SelectedAudioTimelineError(ValueError):
    """Raised when selected audio cannot be mapped onto Blender's timeline."""


def frames_per_second(fps: int, fps_base: float = 1.0) -> float:
    """Return Blender's effective frame rate."""

    if fps <= 0 or not math.isfinite(fps_base) or fps_base <= 0.0:
        raise SelectedAudioTimelineError("scene frame rate must be positive")
    return fps / fps_base


def frame_to_audio_sample(
    frame: int,
    *,
    frame_start: int,
    sample_rate: int,
    fps: int,
    fps_base: float = 1.0,
    prediction_delay: float = 0.0,
    audio_samples: int,
) -> int:
    """Map one Blender frame to the nearest available model sample."""

    if sample_rate <= 0 or audio_samples <= 0:
        raise SelectedAudioTimelineError(
            "sample rate and audio sample count must be positive"
        )
    position = (
        (frame - frame_start) / frames_per_second(fps, fps_base)
        + prediction_delay
    ) * sample_rate
    nearest = (
        math.floor(position + 0.5)
        if position >= 0.0
        else math.ceil(position - 0.5)
    )
    return min(max(nearest, 0), audio_samples - 1)


def is_selected_audio_strip(strip: Any) -> bool:
    """Return whether this add-on owns a Sequencer strip."""

    return strip.get(SELECTED_AUDIO_OWNER_KEY) == SELECTED_AUDIO_OWNER_VALUE


def _strip_frame_span(strip: Any) -> tuple[int, int]:
    """Return the native SoundStrip content span with an inclusive end."""

    return int(strip.content_start), int(strip.content_end) - 1


def selected_audio_frame_span(scene: Any) -> tuple[int, int] | None:
    """Return the owned SoundStrip's native inclusive frame span."""

    sequence_editor = scene.sequence_editor
    if sequence_editor is None:
        return None
    strip = next(
        (item for item in sequence_editor.strips if is_selected_audio_strip(item)),
        None,
    )
    return None if strip is None else _strip_frame_span(strip)


def remove_selected_audio_strips(scene: Any) -> None:
    """Remove only the sound strips owned by Selected Audio mode."""

    sequence_editor = scene.sequence_editor
    if sequence_editor is None:
        return
    for strip in tuple(sequence_editor.strips):
        if is_selected_audio_strip(strip):
            sequence_editor.strips.remove(strip)


def configure_selected_audio(
    scene: Any,
    audio_path: str,
    *,
    first_frame: int,
) -> tuple[int, int]:
    """Synchronize the selected WAV and return its native inclusive frame span."""

    strips = scene.sequence_editor_create().strips
    strip = next((item for item in strips if is_selected_audio_strip(item)), None)
    if strip is not None and strip.sound.filepath != audio_path:
        strips.remove(strip)
        strip = None

    channels = [item.channel for item in strips if not is_selected_audio_strip(item)]
    channel = max(channels, default=0) + 1
    if strip is None:
        strip = strips.new_sound(
            name=SELECTED_AUDIO_STRIP_NAME,
            filepath=audio_path,
            channel=channel,
            frame_start=first_frame,
        )

    strip[SELECTED_AUDIO_OWNER_KEY] = SELECTED_AUDIO_OWNER_VALUE
    strip.name = SELECTED_AUDIO_STRIP_NAME
    strip.channel = channel
    strip.content_start = first_frame
    strip.left_handle_offset = 0.0
    strip.right_handle_offset = 0.0
    scene.sync_mode = "AUDIO_SYNC"
    return _strip_frame_span(strip)
