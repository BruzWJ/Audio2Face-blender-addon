from __future__ import annotations

from types import SimpleNamespace

import pytest

from audio2face.selected_audio_timeline import (
    SELECTED_AUDIO_OWNER_KEY,
    SELECTED_AUDIO_OWNER_VALUE,
    SELECTED_AUDIO_SOURCE_KEY,
    SELECTED_AUDIO_STRIP_NAME,
    SelectedAudioTimelineError,
    duration_frame_count,
    duration_frame_end,
    ensure_selected_audio_strip,
    frame_to_bake_sample,
    frames_per_second,
    is_selected_audio_strip,
    remove_selected_audio_strips,
)


class _Strip(dict[str, object]):
    def __init__(self, name: str, filepath: str, channel: int) -> None:
        super().__init__()
        self.name = name
        self.filepath = filepath
        self.channel = channel
        self.frame_start = 1
        self.frame_final_duration = 1


class _Strips(list[_Strip]):
    def remove(self, strip: _Strip) -> None:
        for index, item in enumerate(self):
            if item is strip:
                del self[index]
                return
        raise ValueError("strip is not in collection")

    def new_sound(
        self,
        *,
        name: str,
        filepath: str,
        channel: int,
        frame_start: int,
    ) -> _Strip:
        strip = _Strip(name, filepath, channel)
        strip.frame_start = frame_start
        self.append(strip)
        return strip


class _Scene:
    def __init__(self, *strips: _Strip) -> None:
        self.frame_start = 1
        self.frame_end = 1
        self.render = SimpleNamespace(fps=24, fps_base=1.0)
        self.sequence_editor = SimpleNamespace(strips=_Strips(strips))

    def sequence_editor_create(self) -> SimpleNamespace:
        return self.sequence_editor


def _unrelated(name: str, channel: int) -> _Strip:
    return _Strip(name, f"/{name}.wav", channel)


def _owned(path: str, channel: int = 2) -> _Strip:
    strip = _Strip(SELECTED_AUDIO_STRIP_NAME, path, channel)
    strip[SELECTED_AUDIO_OWNER_KEY] = SELECTED_AUDIO_OWNER_VALUE
    strip[SELECTED_AUDIO_SOURCE_KEY] = path
    return strip


def test_frame_math_uses_effective_fps_delay_rounding_and_bounds() -> None:
    assert frames_per_second(30, 1.001) == pytest.approx(29.97002997002997)
    assert duration_frame_count(1.000001, 24) == 25
    assert duration_frame_end(10, 1.25, 24) == 39
    assert frame_to_bake_sample(
        2,
        frame_start=1,
        sample_rate=12,
        fps=24,
        audio_samples=10,
    ) == 1
    assert frame_to_bake_sample(
        1,
        frame_start=1,
        sample_rate=8,
        fps=24,
        prediction_delay=0.0625,
        audio_samples=10,
    ) == 1
    assert frame_to_bake_sample(
        100,
        frame_start=1,
        sample_rate=48_000,
        fps=24,
        audio_samples=8,
    ) == 7
    assert frame_to_bake_sample(
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
        duration_frame_count(0.0, 24)
    with pytest.raises(SelectedAudioTimelineError):
        frame_to_bake_sample(
            1,
            frame_start=1,
            sample_rate=0,
            fps=24,
            audio_samples=1,
        )


def test_create_and_update_one_owned_strip_preserves_other_media() -> None:
    unrelated = _unrelated("music", 3)
    scene = _Scene(unrelated)

    strip = ensure_selected_audio_strip(scene, "/audio/voice.wav", 1.0)

    assert tuple(scene.sequence_editor.strips) == (unrelated, strip)
    assert is_selected_audio_strip(strip)
    assert strip.channel == 4
    assert strip.frame_final_duration == 24
    assert scene.frame_end == 24

    duplicate = _owned("/audio/voice.wav", 6)
    scene.sequence_editor.strips.append(duplicate)
    scene.frame_start = 10
    scene.frame_end = 100
    scene.render.fps = 30
    scene.render.fps_base = 1.001

    same = ensure_selected_audio_strip(scene, "/audio/voice.wav", 2.01)

    assert same is strip
    assert all(item is not duplicate for item in scene.sequence_editor.strips)
    assert strip.frame_start == 10
    assert strip.frame_final_duration == 61
    assert scene.frame_end == 100


def test_changed_source_replaces_only_owned_strips() -> None:
    unrelated = _unrelated("music", 4)
    old = _owned("/audio/old.wav")
    duplicate = _owned("/audio/old.wav", 3)
    scene = _Scene(unrelated, old, duplicate)

    replacement = ensure_selected_audio_strip(scene, "/audio/new.WAV", 0.5)

    assert unrelated in scene.sequence_editor.strips
    assert old not in scene.sequence_editor.strips
    assert duplicate not in scene.sequence_editor.strips
    assert replacement[SELECTED_AUDIO_SOURCE_KEY] == "/audio/new.WAV"


def test_remove_owned_strips_does_not_touch_other_media_or_create_editor() -> None:
    unrelated = _unrelated("music", 1)
    scene = _Scene(unrelated, _owned("/audio/voice.wav"))
    remove_selected_audio_strips(scene)
    assert tuple(scene.sequence_editor.strips) == (unrelated,)

    scene.sequence_editor = None
    remove_selected_audio_strips(scene)


@pytest.mark.parametrize("path", ["", "/audio/voice.mp3"])
def test_strip_requires_wav_path(path: str) -> None:
    with pytest.raises(SelectedAudioTimelineError):
        ensure_selected_audio_strip(_Scene(), path, 1.0)
