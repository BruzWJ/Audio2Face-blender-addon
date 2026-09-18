from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

from audio2face.selected_audio_timeline import (
    _AudioClipReader,
    ChannelAudioSource,
    SelectedAudioTimelineError,
    frame_to_audio_sample,
    frames_per_second,
    selected_audio_frame_span,
    selected_channel_snapshot,
)


class _AudioError(Exception):
    pass


def _float_wav(path: Path, samples: list[float], *, leading_frames: int = 0) -> None:
    payload = struct.pack(f"<{len(samples)}f", *samples)
    size = len(payload) + leading_frames * 4
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + size, b"WAVE",
        b"fmt ", 16, 3, 1, 8000, 8000 * 4, 4, 32, b"data", size,
    )
    with path.open("wb") as handle:
        handle.write(header)
        handle.seek(leading_frames * 4, 1)
        handle.write(payload)


class _Factory:
    """A native mono factory with known PCM; real codecs are covered in Blender."""

    def __init__(self, samples: list[float] | None) -> None:
        self.samples = samples
        self.paths: list[Path] = []
        self.end = 0.0

    def rechannel(self, channels: int) -> _Factory:
        assert channels == 1
        return self

    def resample(self, rate: int, quality: int) -> _Factory:
        assert (rate, quality) == (8000, 2)
        return self

    def limit(self, start: float, end: float) -> _Factory:
        assert start == 0.0
        self.end = end
        return self

    def write(
        self, path: str, rate: int, channels: int, fmt: int,
        container: int, codec: int, bitrate: int, buffer: int,
    ) -> None:
        assert (rate, channels, fmt, container, codec, bitrate, buffer) == (
            8000, 1, 2, 3, 4, 0, 4096,
        )
        self.paths.append(Path(path))
        if self.samples is None:
            raise _AudioError("unsupported source")
        samples = self.samples[:round(self.end * rate)]
        _float_wav(Path(path), samples)


class _Sound:
    """The source properties exposed by Blender 5.2's Sound datablock."""

    def __init__(self, filepath: str, samples: list[float] | None = None) -> None:
        self.filepath = filepath
        self.library = None
        self.packed_file = None
        self.use_mono = False
        self.factory = _Factory(samples)
        self.events: list[str] = []

    def as_pointer(self) -> int:
        return id(self)

    def update_tag(self) -> None:
        self.events.append("refresh")

    def evaluated_get(self, depsgraph: object) -> SimpleNamespace:
        assert depsgraph is _GRAPH
        self.events.append("evaluate")
        return SimpleNamespace(factory=self.factory)


class _Strip:
    def __init__(
        self, filepath: str, channel: int = 1, *, start: int = 1,
        duration: int = 24, left_trim: int = 0, right_trim: int = 0,
        volume: float = 1.0, mute: bool = False,
        samples: list[float] | None = None,
    ) -> None:
        self.type = "SOUND"
        self.sound = _Sound(filepath, samples)
        self.channel = channel
        self.content_start = start
        self.content_duration = duration
        self.left_handle_offset = left_trim
        self.right_handle_offset = right_trim
        self.volume = volume
        self.mute = mute
        self.content_trim_start = 0
        self.content_trim_end = 0
        self.sound_offset = 0.0
        self.retiming_keys = []
        self.modifiers = []

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
        frame_start=-10, frame_end=200, sync_mode="NONE", view_layers=[object()],
    )


_GRAPH = object()


@pytest.fixture(autouse=True)
def _blender(monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def temp_override(*, scene: SimpleNamespace, view_layer: object):
        assert view_layer is scene.view_layers[0]
        yield

    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace(
        path=SimpleNamespace(abspath=lambda path, *, library: str(Path(path).absolute())),
        context=SimpleNamespace(
            temp_override=temp_override, evaluated_depsgraph_get=lambda: _GRAPH,
        ),
    ))
    monkeypatch.setitem(sys.modules, "aud", SimpleNamespace(
        error=_AudioError, CHANNELS_MONO=1, FORMAT_FLOAT32=2, CONTAINER_WAV=3, CODEC_PCM=4,
    ))


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
    assert [clip.sound for clip in snapshot.clips] == [first.sound, last.sound]
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
    strip = _Strip("/voice.wav")
    scene = _scene(strip)
    previous = selected_channel_snapshot(scene)
    assert selected_channel_snapshot(scene) == previous
    for field, value in (
        ("content_start", 10), ("left_handle_offset", 2),
        ("right_handle_offset", 3), ("volume", 0.25), ("mute", True),
        ("sound_offset", 0.25), ("content_trim_start", 2),
    ):
        setattr(strip, field, value)
        changed = selected_channel_snapshot(scene)
        assert changed != previous, field
        previous = changed
    scene.render.fps_base = 1.001
    assert selected_channel_snapshot(scene) != previous
    scene.audio2face.audio_channel = 2
    assert selected_channel_snapshot(scene) is None


def test_snapshot_tracks_sound_identity_without_native_evaluation() -> None:
    strip = _Strip("/two-streams.mka")
    scene = _scene(strip)
    original = selected_channel_snapshot(scene)
    # Native factories use transient Python wrappers; the Sound identity is stable.
    strip.sound.factory = _Factory(None)
    assert selected_channel_snapshot(scene) == original
    assert strip.sound.events == []
    # Blender stores each immutable stream index in its own Sound datablock.
    strip.sound = _Sound("/two-streams.mka")
    changed = selected_channel_snapshot(scene)
    assert changed != original
    strip.sound.use_mono = True
    assert selected_channel_snapshot(scene) != changed
    mono = selected_channel_snapshot(scene)
    strip.sound.filepath = "/replacement.mka"
    assert selected_channel_snapshot(scene) != mono


def test_snapshot_resolves_blender_relative_path_with_sound_library() -> None:
    library = object()
    calls = []

    def abspath(path: str, *, library: object) -> str:
        calls.append((path, library))
        return "/library/voice.wav"

    sys.modules["bpy"].path.abspath = abspath
    strip = _Strip("//voice.wav")
    strip.sound.library = library
    selected_channel_snapshot(_scene(strip))
    assert calls == [("//voice.wav", library)]


def test_stream_preserves_gaps_trimmed_content_and_other_channels() -> None:
    scene = _scene(
        _Strip("/first.wav", start=-5, duration=3, left_trim=1, right_trim=1,
               samples=[0.125] * 1000 + [0.5] * 1000 + [0.25] * 1000),
        _Strip("/last.wav", start=-1, duration=1, samples=[-0.25] * 1000),
        _Strip("/missing-other-channel.wav", channel=2), fps=8,
    )
    assert _decode(scene) == (0.5,) * 1000 + (0.0,) * 2000 + (-0.25,) * 1000


def test_overlap_gain_muted_clips_and_channel_mute() -> None:
    scene = _scene(
        _Strip("/voice.wav", start=1, duration=2, volume=0.5, samples=[0.5] * 2000),
        _Strip("/voice.wav", start=2, duration=1, samples=[0.5] * 2000),
        _Strip("/missing-muted.wav", start=1, duration=3, mute=True), fps=8,
    )
    assert _decode(scene) == (0.25,) * 1000 + (0.75,) * 1000 + (0.0,) * 1000
    before = selected_channel_snapshot(scene)
    scene.sequence_editor.channels = [SimpleNamespace(number=2, mute=True)]
    assert selected_channel_snapshot(scene) == before
    scene.sequence_editor.channels.append(SimpleNamespace(number=1, mute=True))
    assert selected_channel_snapshot(scene) != before
    assert _decode(scene) == (0.0,) * 3000
    scene.sequence_editor.channels[-1].mute = False
    assert selected_channel_snapshot(scene) == before


def test_source_offsets_hard_trim_and_eof_are_silent() -> None:
    strip = _Strip("/voice.wav", start=5, duration=4, samples=[0.25] * 1000 + [0.5] * 1000)
    strip.sound_offset = 0.125
    scene = _scene(strip, fps=8)
    assert _decode(scene) == (0.0,) * 1000 + (0.25,) * 1000 + (0.5,) * 1000 + (0.0,) * 1000
    strip.sound_offset = 0.0
    strip.content_trim_start = 1
    assert _decode(scene) == (0.5,) * 1000 + (0.0,) * 3000


def test_float_stream_clips_mix_and_uses_effective_fps() -> None:
    scene = _scene(_Strip("/voice.wav", duration=1, volume=2.0, samples=[0.75] * 4000), fps=8)
    scene.render.fps_base = 2.0
    assert _decode(scene) == (1.0,) * 2000


def test_channel_source_is_one_shot_and_close_is_idempotent() -> None:
    strip = _Strip("/voice.wav", duration=1, samples=[0.25] * 1000)
    source = ChannelAudioSource(
        selected_channel_snapshot(_scene(strip, fps=8)),
        output_sample_rate=8000, chunk_frames=100,
    )
    iterator = iter(source)
    next(iterator)
    with pytest.raises(SelectedAudioTimelineError, match="only be iterated once"):
        iter(source)
    source.close()
    source.close()
    with pytest.raises(SelectedAudioTimelineError, match="closed"):
        next(iterator)
    assert not strip.sound.factory.paths[0].exists()


@pytest.mark.parametrize("packed", [False, True])
def test_native_factory_is_lazy_bounded_and_cleaned_up(packed: bool) -> None:
    strip = _Strip("/voice.mp3", duration=2, samples=[0.25] * 2000)
    if packed:
        strip.sound.packed_file = SimpleNamespace(size=123, as_pointer=lambda: 42)
    scene = _scene(strip, fps=8)
    first = selected_channel_snapshot(scene)
    assert first == selected_channel_snapshot(scene)
    assert strip.sound.events == []
    assert _decode(scene, chunk_frames=100) == (0.25,) * 2000
    assert strip.sound.events == (["refresh", "evaluate"] if packed else ["evaluate"])
    assert first == selected_channel_snapshot(scene)
    factory = strip.sound.factory
    assert factory.end == pytest.approx(0.25 + 4096 / 8000)
    assert len(factory.paths) == 1
    assert not factory.paths[0].exists()


@pytest.mark.parametrize(("field", "value", "message"), [
    ("retiming_keys", [SimpleNamespace(timeline_frame=1)], "retiming keys"),
    ("modifiers", [SimpleNamespace(mute=False)], "audio modifiers"),
])
def test_unsupported_strip_settings_fail_clearly_unless_muted(
    field: str, value: object, message: str,
) -> None:
    strip = _Strip("/voice.wav")
    scene = _scene(strip)
    setattr(strip, field, value)
    with pytest.raises(SelectedAudioTimelineError, match=message):
        selected_channel_snapshot(scene)
    strip.mute = True
    assert selected_channel_snapshot(scene).clips[0].muted


def test_decode_error_closes_all_active_readers_and_wraps_native_error() -> None:
    first = _Strip("/voice.wav", start=1, samples=[0.25] * 8000)
    broken = _Strip("/missing.wav", start=2)
    source = ChannelAudioSource(
        selected_channel_snapshot(_scene(first, broken)), output_sample_rate=8000,
    )
    with pytest.raises(SelectedAudioTimelineError, match="could not decode channel audio: unsupported source"):
        list(source)
    assert source._readers == {}
    assert all(not path.exists() for strip in (first, broken) for path in strip.sound.factory.paths)
    with pytest.raises(SelectedAudioTimelineError, match="closed"):
        iter(source)


def test_missing_native_factory_fails_without_reopening_stream_zero() -> None:
    strip = _Strip("/two-streams.mka")
    strip.sound.factory = None
    with pytest.raises(SelectedAudioTimelineError, match="no evaluated audio source"):
        _decode(_scene(strip))


@pytest.mark.parametrize(("sample_position", "expected"), [
    (-0.5, [0.0, 0.5, 0.25, -0.125, 0.0]),
    (0.0, [0.25, 0.75, -0.25, 0.0, 0.0]),
    (2.5, [-0.125, 0.0, 0.0, 0.0, 0.0]),
    (100_000.0, [0.0] * 5),
])
def test_pcm_reader_handles_fractional_positions_negative_offsets_and_eof(
    sample_position: float, expected: list[float],
) -> None:
    strip = _Strip("/voice.wav", samples=[0.25, 0.75, -0.25])
    reader = _AudioClipReader(selected_channel_snapshot(_scene(strip)).clips[0], 8000, 1.0)
    try:
        assert reader.read(sample_position / 8000, len(expected)) == expected
    finally:
        reader.close()


def test_heavily_trimmed_clip_seeks_directly_and_reads_only_requested_pcm() -> None:
    strip = _Strip("/long.wav", duration=1)
    strip.sound_offset = -1800.0

    def write_sparse(path: str, *args: object) -> None:
        _float_wav(Path(path), [0.5] * 1000, leading_frames=1800 * 8000)

    strip.sound.factory.write = write_sparse
    reader = _AudioClipReader(
        selected_channel_snapshot(_scene(strip, fps=8)).clips[0], 8000, 0.125,
    )
    handle = reader._file
    reads = []
    seeks = []

    def read(size: int) -> bytes:
        reads.append(size)
        return handle.read(size)

    def seek(offset: int) -> int:
        seeks.append(offset)
        return handle.seek(offset)

    reader._file = SimpleNamespace(read=read, seek=seek, close=handle.close)
    try:
        assert reader.read(1800.0, 32) == [0.5] * 32
        assert seeks == [reader._data_offset + 1800 * 8000 * 4]
        assert reads == [33 * 4]
    finally:
        reader.close()


@pytest.mark.parametrize("failure", ["truncated", "non-finite"])
def test_pcm_reader_rejects_incomplete_or_nonfinite_decoder_output(failure: str) -> None:
    strip = _Strip("/voice.wav", samples=[0.25, float("nan") if failure == "non-finite" else 0.5])
    reader = _AudioClipReader(selected_channel_snapshot(_scene(strip)).clips[0], 8000, 1.0)
    if failure == "truncated":
        with strip.sound.factory.paths[0].open("r+b") as handle:
            handle.truncate(reader._data_offset + 4)
        # Discard the buffered header read before checking the truncated file.
        reader._file.close()
        reader._file = strip.sound.factory.paths[0].open("rb")
    try:
        with pytest.raises(SelectedAudioTimelineError, match=failure):
            reader.read(0.0, 2)
    finally:
        reader.close()
