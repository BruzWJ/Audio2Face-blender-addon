from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import wave

import pytest

from audio2face.selected_audio_timeline import (
    ChannelAudioSource,
    SelectedAudioTimelineError,
    frame_to_audio_sample,
    frames_per_second,
    selected_audio_frame_span,
    selected_channel_snapshot,
)


class _Strip:
    def __init__(
        self, filepath: str | Path, channel: int = 1, *, start: int = 1,
        duration: int = 24, left_trim: int = 0, right_trim: int = 0,
        volume: float = 1.0, mute: bool = False,
    ) -> None:
        self.type = "SOUND"
        self.sound = SimpleNamespace(filepath=str(filepath), library=None)
        self.channel = channel
        self.content_start = start
        self.content_duration = duration
        self.left_handle_offset = left_trim
        self.right_handle_offset = right_trim
        self.volume = volume
        self.mute = mute

    @property
    def content_end(self) -> int:
        return self.content_start + self.content_duration

    @property
    def left_handle(self) -> int:
        return self.content_start + self.left_handle_offset

    @property
    def right_handle(self) -> int:
        return self.content_end - self.right_handle_offset


def _scene(*strips: _Strip, channel: int = 1, fps: int = 24) -> SimpleNamespace:
    return SimpleNamespace(
        audio2face=SimpleNamespace(audio_channel=channel),
        render=SimpleNamespace(fps=fps, fps_base=1.0),
        sequence_editor=SimpleNamespace(strips=list(strips), channels=[]),
        frame_start=-10, frame_end=200, sync_mode="NONE",
    )


def _wav(path: Path, values: list[int], *, rate: int = 8_000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack(f"<{len(values)}h", *values))
    return path


def _decode(scene: SimpleNamespace, *, chunk_frames: int = 257) -> tuple[float, ...]:
    snapshot = selected_channel_snapshot(scene)
    assert snapshot is not None
    source = ChannelAudioSource(snapshot, output_sample_rate=8_000, chunk_frames=chunk_frames)
    chunks = list(source)
    assert all(0 < len(chunk) <= chunk_frames * 4 for chunk in chunks)
    payload = b"".join(chunks)
    assert len(payload) == source.metadata.output_frames * 4
    return struct.unpack(f"<{len(payload) // 4}f", payload)


def test_frame_math_uses_effective_fps_delay_rounding_and_bounds() -> None:
    assert frames_per_second(30, 1.001) == pytest.approx(29.97002997002997)
    assert frame_to_audio_sample(2, frame_start=1, sample_rate=12, fps=24, audio_samples=10) == 1
    assert frame_to_audio_sample(1, frame_start=1, sample_rate=8, fps=24, prediction_delay=0.0625, audio_samples=10) == 1
    assert frame_to_audio_sample(100, frame_start=1, sample_rate=48_000, fps=24, audio_samples=8) == 7
    assert frame_to_audio_sample(-100, frame_start=1, sample_rate=48_000, fps=24, audio_samples=8) == 0


def test_frame_math_rejects_invalid_scene_or_audio_rates() -> None:
    with pytest.raises(SelectedAudioTimelineError):
        frames_per_second(24, 0.0)
    with pytest.raises(SelectedAudioTimelineError):
        frame_to_audio_sample(1, frame_start=1, sample_rate=0, fps=24, audio_samples=1)


def test_channel_span_uses_every_visible_sound_strip_without_mutation() -> None:
    first = _Strip("/first.wav", 3, start=-20, duration=24, left_trim=2, right_trim=3)
    last = _Strip("/last.wav", 3, start=70, duration=12, right_trim=2)
    other = _Strip("/other.wav", 1, start=-100, duration=300)
    image = _Strip("/image.png", 3, start=-100, duration=300)
    image.type = "IMAGE"
    scene = _scene(last, first, other, image, channel=3)
    before = [vars(strip).copy() for strip in scene.sequence_editor.strips]
    assert selected_audio_frame_span(scene) == (-18, 79)
    snapshot = selected_channel_snapshot(scene)
    assert snapshot is not None
    assert (snapshot.frame_start, snapshot.frame_end) == (-18, 79)
    assert [clip.path for clip in snapshot.clips] == ["/first.wav", "/last.wav"]
    assert snapshot.clips[0].source_start == pytest.approx(2 / 24)
    assert [vars(strip) for strip in scene.sequence_editor.strips] == before
    assert (scene.frame_start, scene.frame_end, scene.sync_mode) == (-10, 200, "NONE")
    with pytest.raises(FrozenInstanceError):
        snapshot.frame_start = 1


def test_empty_channel_does_not_create_an_editor() -> None:
    scene = _scene(_Strip("/other.wav", 2))
    assert selected_audio_frame_span(scene) is None
    assert selected_channel_snapshot(scene) is None
    scene.sequence_editor = None
    assert selected_audio_frame_span(scene) is None
    assert selected_channel_snapshot(scene) is None


def test_snapshot_tracks_timeline_input_changes() -> None:
    clip = _Strip("/voice.wav")
    scene = _scene(clip)
    previous = selected_channel_snapshot(scene)
    assert selected_channel_snapshot(scene) == previous
    for field, value in (
        ("content_start", 10), ("left_handle_offset", 2),
        ("right_handle_offset", 3), ("volume", 0.25), ("mute", True),
        ("sound_offset", 0.25), ("content_trim_start", 2),
    ):
        setattr(clip, field, value)
        changed = selected_channel_snapshot(scene)
        assert changed != previous, field
        previous = changed
    scene.render.fps_base = 1.001
    assert selected_channel_snapshot(scene) != previous
    scene.audio2face.audio_channel = 2
    assert selected_channel_snapshot(scene) is None


def test_snapshot_resolves_blender_relative_path_with_sound_library(monkeypatch: pytest.MonkeyPatch) -> None:
    library = object()
    calls = []
    def abspath(path: str, *, library: object) -> str:
        calls.append((path, library))
        return "/library/voice.wav"
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace(path=SimpleNamespace(abspath=abspath)))
    clip = _Strip("//voice.wav")
    clip.sound.library = library
    snapshot = selected_channel_snapshot(_scene(clip))
    assert snapshot.clips[0].path == "/library/voice.wav"
    assert calls == [("//voice.wav", library)]


def test_stream_preserves_gaps_trimmed_content_and_other_channels(tmp_path: Path) -> None:
    # At 8 fps, each frame contributes exactly 1000 samples.
    first = _wav(tmp_path / "first.wav", [1000] * 1000 + [8000] * 1000 + [2000] * 1000)
    last = _wav(tmp_path / "last.wav", [-4000] * 1000)
    scene = _scene(
        _Strip(first, start=-5, duration=3, left_trim=1, right_trim=1),
        _Strip(last, start=-1, duration=1),
        _Strip("/missing-file-on-other-channel.wav", channel=2),
        fps=8,
    )
    result = _decode(scene)
    assert result == pytest.approx([8000 / 32768] * 1000 + [0.0] * 2000 + [-4000 / 32768] * 1000)


def test_overlap_gain_muted_clips_and_channel_mute(tmp_path: Path) -> None:
    path = _wav(tmp_path / "voice.wav", [16000] * 2000)
    scene = _scene(
        _Strip(path, start=1, duration=2, volume=0.5),
        _Strip(path, start=2, duration=1, volume=1.0),
        _Strip("/missing-muted.wav", start=1, duration=3, mute=True),
        fps=8,
    )
    assert _decode(scene) == pytest.approx([8000 / 32768] * 1000 + [24000 / 32768] * 1000 + [0.0] * 1000)
    scene.sequence_editor.channels = [SimpleNamespace(channel=1, mute=True)]
    assert _decode(scene) == (0.0,) * 3000


def test_source_offsets_hard_trim_and_eof_are_silent(tmp_path: Path) -> None:
    path = _wav(tmp_path / "voice.wav", [8000] * 1000 + [16000] * 1000)
    strip = _Strip(path, start=5, duration=4)
    strip.sound_offset = 0.125
    scene = _scene(strip, fps=8)
    assert _decode(scene) == pytest.approx([0.0] * 1000 + [8000 / 32768] * 1000 + [16000 / 32768] * 1000 + [0.0] * 1000)
    strip.sound_offset = 0.0
    strip.content_trim_start = 1
    assert _decode(scene) == pytest.approx([16000 / 32768] * 1000 + [0.0] * 3000)


def test_float_stream_clips_mix_and_uses_effective_fps(tmp_path: Path) -> None:
    path = _wav(tmp_path / "voice.wav", [24000] * 4000)
    scene = _scene(_Strip(path, duration=1, volume=2.0), fps=8)
    scene.render.fps_base = 2.0
    assert _decode(scene) == (1.0,) * 2000


def test_channel_source_is_one_shot_and_close_is_idempotent(tmp_path: Path) -> None:
    path = _wav(tmp_path / "voice.wav", [8000] * 1000)
    source = ChannelAudioSource(selected_channel_snapshot(_scene(_Strip(path, duration=1), fps=8)), chunk_frames=100)
    iterator = iter(source)
    next(iterator)
    with pytest.raises(SelectedAudioTimelineError, match="only be iterated once"):
        iter(source)
    source.close()
    source.close()
    with pytest.raises(SelectedAudioTimelineError, match="closed"):
        next(iterator)


def test_compressed_audio_uses_bounded_blender_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    class Sound:
        def __init__(self, path: str) -> None:
            assert path == "/voice.mp3"
        def rechannel(self, channels: int) -> Sound:
            assert channels == 1
            return self
        def resample(self, rate: int, quality: int) -> Sound:
            assert (rate, quality) == (8000, 2)
            return self
        def limit(self, start: float, end: float) -> Sound:
            assert start == 0.0
            assert end == pytest.approx(0.25 + 2 / 8000)
            return self
        def write(self, path: str, rate: int, channels: int, fmt: int, container: int, codec: int, bitrate: int, buffer: int) -> None:
            calls.append(path)
            assert buffer == 4096
            _wav(Path(path), [8192] * 2000)
    monkeypatch.setitem(sys.modules, "aud", SimpleNamespace(
        Sound=Sound, CHANNELS_MONO=1, FORMAT_FLOAT32=2, CONTAINER_WAV=3, CODEC_PCM=4,
    ))
    result = _decode(_scene(_Strip("/voice.mp3", duration=2), fps=8), chunk_frames=100)
    assert result == (0.25,) * 2000
    assert len(calls) == 1
    assert not Path(calls[0]).exists()


def test_native_decoder_errors_are_wrapped_and_temp_files_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    class AudioError(Exception):
        pass
    def broken_sound(path: str) -> None:
        raise AudioError("unsupported source")
    monkeypatch.setitem(sys.modules, "aud", SimpleNamespace(Sound=broken_sound, error=AudioError))
    with pytest.raises(SelectedAudioTimelineError, match="could not decode channel audio: unsupported source"):
        _decode(_scene(_Strip("/broken.mp3")))


def test_retiming_and_audio_modifiers_fail_clearly_unless_muted() -> None:
    strip = _Strip("/voice.wav")
    scene = _scene(strip)
    strip.retiming_keys = [SimpleNamespace(timeline_frame=1)]
    with pytest.raises(SelectedAudioTimelineError, match="retiming keys"):
        selected_channel_snapshot(scene)
    strip.retiming_keys = []
    strip.modifiers = [SimpleNamespace(mute=False)]
    with pytest.raises(SelectedAudioTimelineError, match="audio modifiers"):
        selected_channel_snapshot(scene)
    strip.mute = True
    assert selected_channel_snapshot(scene).clips[0].muted


def test_decode_error_closes_all_active_readers(tmp_path: Path) -> None:
    path = _wav(tmp_path / "voice.wav", [8000] * 1000)
    snapshot = selected_channel_snapshot(_scene(_Strip(path, start=1), _Strip("/missing.wav", start=2)))
    source = ChannelAudioSource(snapshot)
    with pytest.raises(ValueError, match="missing or inaccessible"):
        list(source)
    assert source._readers == {}
    with pytest.raises(SelectedAudioTimelineError, match="closed"):
        iter(source)


def test_packed_audio_is_extracted_once_and_cleaned_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    encoded = _wav(tmp_path / "packed.wav", [8192] * 1000).read_bytes()
    decoded_paths = []
    class Sound:
        def __init__(self, path: str) -> None:
            assert Path(path).read_bytes() == encoded
            decoded_paths.append(path)
        def rechannel(self, channels: int) -> Sound:
            assert channels == 1
            return self
        def resample(self, rate: int, quality: int) -> Sound:
            assert (rate, quality) == (8000, 2)
            return self
        def limit(self, start: float, end: float) -> Sound:
            return self
        def write(self, path: str, *args: object) -> None:
            _wav(Path(path), [8192] * 1000)
    monkeypatch.setitem(sys.modules, "aud", SimpleNamespace(
        Sound=Sound, CHANNELS_MONO=1, FORMAT_FLOAT32=2, CONTAINER_WAV=3, CODEC_PCM=4,
    ))
    strip = _Strip("/deleted-packed-file.mp3", duration=1)
    strip.sound.packed_file = SimpleNamespace(data=encoded, size=len(encoded))
    scene = _scene(strip, fps=8)
    first = selected_channel_snapshot(scene)
    assert first == selected_channel_snapshot(scene)
    assert _decode(scene, chunk_frames=100) == (0.25,) * 1000
    assert len(decoded_paths) == 1
    assert not Path(decoded_paths[0]).exists()
