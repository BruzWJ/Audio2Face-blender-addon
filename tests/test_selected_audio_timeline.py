from __future__ import annotations

from types import SimpleNamespace

import pytest

from audio2face.selected_audio_timeline import (
    SELECTED_AUDIO_OWNER_KEY,
    SELECTED_AUDIO_OWNER_VALUE,
    SELECTED_AUDIO_STRIP_NAME,
    SelectedAudioTimelineError,
    configure_selected_audio,
    frame_to_audio_sample,
    frames_per_second,
    is_selected_audio_strip,
    remove_selected_audio_strips,
    selected_audio_frame_span,
)


class _Strip(dict[str, object]):
    def __init__(
        self,
        name: str,
        filepath: str,
        channel: int,
        *,
        content_duration: int = 24,
    ) -> None:
        super().__init__()
        self.name = name
        self.sound = SimpleNamespace(filepath=filepath)
        self.channel = channel
        self.content_start = 1
        self.content_duration = content_duration
        self.left_handle_offset = 0.0
        self.right_handle_offset = 0.0

    @property
    def content_end(self) -> int:
        return int(self.content_start) + self.content_duration

    @property
    def duration(self) -> int:
        return int(
            self.content_duration
            - self.left_handle_offset
            - self.right_handle_offset
        )


class _Strips(list[_Strip]):
    def __init__(self, strips: tuple[_Strip, ...], sound_duration: int) -> None:
        super().__init__(strips)
        self.sound_duration = sound_duration

    def new_sound(
        self,
        *,
        name: str,
        filepath: str,
        channel: int,
        frame_start: int,
    ) -> _Strip:
        strip = _Strip(
            name,
            filepath,
            channel,
            content_duration=self.sound_duration,
        )
        strip.content_start = frame_start
        self.append(strip)
        return strip


class _Scene:
    def __init__(self, *strips: _Strip, sound_duration: int = 24) -> None:
        self.frame_start = 1
        self.frame_end = 1
        self.sync_mode = "NONE"
        self.render = SimpleNamespace(fps=24, fps_base=1.0)
        self.sequence_editor = SimpleNamespace(
            strips=_Strips(strips, sound_duration)
        )

    def sequence_editor_create(self) -> SimpleNamespace:
        return self.sequence_editor


def _unrelated(name: str, channel: int) -> _Strip:
    return _Strip(name, f"/{name}.wav", channel)


def _owned(path: str, channel: int = 2) -> _Strip:
    strip = _Strip(SELECTED_AUDIO_STRIP_NAME, path, channel)
    strip[SELECTED_AUDIO_OWNER_KEY] = SELECTED_AUDIO_OWNER_VALUE
    return strip


def test_frame_math_uses_effective_fps_delay_rounding_and_bounds() -> None:
    assert frames_per_second(30, 1.001) == pytest.approx(29.97002997002997)
    assert frame_to_audio_sample(
        2,
        frame_start=1,
        sample_rate=12,
        fps=24,
        audio_samples=10,
    ) == 1
    assert frame_to_audio_sample(
        1,
        frame_start=1,
        sample_rate=8,
        fps=24,
        prediction_delay=0.0625,
        audio_samples=10,
    ) == 1
    assert frame_to_audio_sample(
        100,
        frame_start=1,
        sample_rate=48_000,
        fps=24,
        audio_samples=8,
    ) == 7
    assert frame_to_audio_sample(
        -100,
        frame_start=1,
        sample_rate=48_000,
        fps=24,
        audio_samples=8,
    ) == 0


def test_frame_math_rejects_invalid_scene_or_audio_rates() -> None:
    with pytest.raises(SelectedAudioTimelineError):
        frames_per_second(24, 0.0)
    with pytest.raises(SelectedAudioTimelineError):
        frame_to_audio_sample(
            1,
            frame_start=1,
            sample_rate=0,
            fps=24,
            audio_samples=1,
        )


def test_configure_uses_one_native_strip_span_and_audio_sync() -> None:
    unrelated = _unrelated("music", 3)
    scene = _Scene(unrelated, sound_duration=23)
    scene.frame_start = -10
    scene.frame_end = 80

    span = configure_selected_audio(
        scene,
        "/audio/voice.wav",
        first_frame=12,
    )

    strip = scene.sequence_editor.strips[-1]
    assert tuple(scene.sequence_editor.strips) == (unrelated, strip)
    assert is_selected_audio_strip(strip)
    assert strip.channel == 4
    assert strip.content_start == 12
    assert strip.content_end == 35
    assert strip.duration == 23
    assert span == (12, 34)
    assert selected_audio_frame_span(scene) == span
    assert scene.sync_mode == "AUDIO_SYNC"
    assert (scene.frame_start, scene.frame_end) == (-10, 80)

    strip.left_handle_offset = 3.0
    strip.right_handle_offset = -2.0
    span = configure_selected_audio(
        scene,
        "/audio/voice.wav",
        first_frame=42,
    )

    assert scene.sequence_editor.strips[-1] is strip
    assert strip.content_start == 42
    assert strip.content_end == 65
    assert strip.duration == 23
    assert strip.left_handle_offset == 0.0
    assert strip.right_handle_offset == 0.0
    assert span == (42, 64)
    assert (scene.frame_start, scene.frame_end) == (-10, 80)


def test_changed_source_replaces_only_the_owned_strip() -> None:
    unrelated = _unrelated("music", 4)
    old = _owned("/audio/old.wav")
    scene = _Scene(unrelated, old, sound_duration=12)

    span = configure_selected_audio(
        scene,
        "/audio/new.WAV",
        first_frame=-20,
    )

    replacement = scene.sequence_editor.strips[-1]
    assert unrelated in scene.sequence_editor.strips
    assert all(item is not old for item in scene.sequence_editor.strips)
    assert replacement.sound.filepath == "/audio/new.WAV"
    assert replacement.content_start == -20
    assert replacement.content_end == -8
    assert span == (-20, -9)


def test_selected_audio_span_is_absent_without_an_owned_strip() -> None:
    scene = _Scene(_unrelated("music", 1))
    assert selected_audio_frame_span(scene) is None

    scene.sequence_editor = None
    assert selected_audio_frame_span(scene) is None


def test_remove_owned_strips_does_not_touch_other_media_or_create_editor() -> None:
    unrelated = _unrelated("music", 1)
    scene = _Scene(unrelated, _owned("/audio/voice.wav"))
    remove_selected_audio_strips(scene)
    assert tuple(scene.sequence_editor.strips) == (unrelated,)

    scene.sequence_editor = None
    remove_selected_audio_strips(scene)
