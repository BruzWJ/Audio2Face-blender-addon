"""Read one Sequencer channel without changing its strips or scene timing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import os
import struct
import tempfile
from typing import Any, Iterator

from .wav_stream import (
    MAX_CHUNK_FRAMES,
    MAX_OUTPUT_FRAMES,
    MAX_SAMPLE_RATE,
    MIN_SAMPLE_RATE,
    WavStreamSource,
)


class SelectedAudioTimelineError(ValueError):
    """Raised when a selected channel cannot be used as model input."""


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


def _property(item: Any, current: str, previous: str, default: Any = 0) -> Any:
    value = getattr(item, current, None)
    return getattr(item, previous, default) if value is None else value


def _channel_strips(scene: Any) -> tuple[Any, ...]:
    editor = scene.sequence_editor
    if editor is None:
        return ()
    channel = int(getattr(scene.audio2face, "audio_channel", 1))
    # Channel numbers belong to this timeline, not to nested meta timelines.
    strips = _property(editor, "strips", "sequences", ())
    return tuple(
        strip for strip in strips
        if strip.type == "SOUND" and strip.channel == channel
        and getattr(strip, "sound", None) is not None
    )


def _strip_frame_span(strip: Any) -> tuple[int, int]:
    """Return visible handle bounds, with an exclusive end."""

    start = _property(strip, "left_handle", "frame_final_start", None)
    end = _property(strip, "right_handle", "frame_final_end", None)
    if start is None:
        start = _property(strip, "content_start", "frame_start") + _property(
            strip, "left_handle_offset", "frame_offset_start"
        )
    if end is None:
        end = _property(strip, "content_end", "frame_end") - _property(
            strip, "right_handle_offset", "frame_offset_end"
        )
    return int(start), int(end)


def selected_audio_frame_span(scene: Any) -> tuple[int, int] | None:
    """Return the inclusive span of all sound strips on the selected channel."""

    spans = [
        (start, end) for strip in _channel_strips(scene)
        for start, end in (_strip_frame_span(strip),) if end > start
    ]
    if not spans:
        return None
    return min(start for start, _ in spans), max(end for _, end in spans) - 1


@dataclass(frozen=True, slots=True)
class ChannelAudioClip:
    """A sound strip copied out of Blender; frame_end is exclusive."""

    path: str
    frame_start: int
    frame_end: int
    source_start: float
    volume: float
    muted: bool
    speed: float
    signature: tuple[Any, ...]
    packed_file: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class SelectedChannelSnapshot:
    """Comparable input description; frame_end is inclusive."""

    channel: int
    fps: int
    fps_base: float
    clips: tuple[ChannelAudioClip, ...]
    frame_start: int
    frame_end: int


def _sound_path(sound: Any) -> str:
    path = str(sound.filepath)
    try:
        import bpy
    except ImportError:
        pass
    else:
        abspath = getattr(getattr(bpy, "path", None), "abspath", None)
        if abspath is not None:
            path = abspath(path, library=getattr(sound, "library", None))
    return os.path.abspath(path)


def selected_channel_snapshot(scene: Any) -> SelectedChannelSnapshot | None:
    """Copy selected-channel input settings without touching Blender data.

    No audio file is opened or stat'ed here: comparisons run during playback.
    Muted clips retain their timing and contribute silence to model input.
    """

    strips = _channel_strips(scene)
    if not strips:
        return None
    channel = int(getattr(scene.audio2face, "audio_channel", 1))
    fps, fps_base = int(scene.render.fps), float(scene.render.fps_base)
    rate = frames_per_second(fps, fps_base)
    channels = getattr(scene.sequence_editor, "channels", ())
    channel_muted = any(
        getattr(item, "channel", None) == channel and getattr(item, "mute", False)
        for item in channels
    )
    clips = []
    for strip in strips:
        start, end = _strip_frame_span(strip)
        if end <= start:
            continue
        origin = float(_property(strip, "content_start", "frame_start"))
        trim = int(_property(strip, "content_trim_start", "animation_offset_start"))
        trim_end = int(_property(strip, "content_trim_end", "animation_offset_end"))
        offset = float(getattr(strip, "sound_offset", 0.0)) + float(
            getattr(strip.sound, "offset_time", 0.0)
        )
        volume = float(getattr(strip, "volume", 1.0))
        speed = float(getattr(strip, "speed_factor", getattr(strip, "pitch", 1.0)))
        if not all(math.isfinite(value) for value in (origin, offset, volume, speed)):
            raise SelectedAudioTimelineError("sound strip settings must be finite")
        if volume < 0.0 or speed <= 0.0:
            raise SelectedAudioTimelineError("sound strip volume or speed is invalid")
        muted = bool(channel_muted or getattr(strip, "mute", False))
        if not muted and volume > 0.0:
            if speed != 1.0 or len(getattr(strip, "retiming_keys", ())) > 0:
                raise SelectedAudioTimelineError(
                    "selected channel sound strips must use normal playback speed without retiming keys"
                )
            if any(not getattr(modifier, "mute", False) for modifier in getattr(strip, "modifiers", ())):
                raise SelectedAudioTimelineError(
                    "selected channel sound strips with audio modifiers are not supported"
                )
        pointer = getattr(strip.sound, "as_pointer", None)
        identity = pointer() if callable(pointer) else id(strip.sound)
        packed = getattr(strip.sound, "packed_file", None)
        packed_pointer = getattr(packed, "as_pointer", None)
        packed_signature = (
            (packed_pointer() if callable(packed_pointer) else id(packed), getattr(packed, "size", 0))
            if packed is not None else None
        )
        clips.append(ChannelAudioClip(
            path=_sound_path(strip.sound),
            frame_start=start,
            frame_end=end,
            source_start=(start - origin + trim) / rate - offset,
            volume=volume,
            muted=muted,
            speed=speed,
            signature=(identity, origin, trim, trim_end, offset, packed_signature),
            packed_file=packed,
        ))
    if not clips:
        return None
    clips.sort(key=lambda clip: (clip.frame_start, clip.frame_end, clip.path))
    return SelectedChannelSnapshot(
        channel=channel,
        fps=fps,
        fps_base=fps_base,
        clips=tuple(clips),
        frame_start=min(clip.frame_start for clip in clips),
        frame_end=max(clip.frame_end for clip in clips) - 1,
    )


@dataclass(frozen=True, slots=True)
class ChannelAudioMetadata:
    output_sample_rate: int
    output_frames: int


class _WavClipReader:
    """Streaming fallback for tests and tools running outside Blender."""

    def __init__(self, clip: ChannelAudioClip, rate: int, chunk_frames: int) -> None:
        self.source = WavStreamSource(
            os.path.realpath(clip.path), output_sample_rate=rate, chunk_frames=chunk_frames
        )
        self.samples = self._samples()
        self.rate, self.speed = rate, clip.speed
        self.index = -1
        self.previous = self.current = 0.0

    def _samples(self) -> Iterator[float]:
        for chunk in self.source:
            yield from struct.unpack(f"<{len(chunk) // 4}f", chunk)

    def read(self, position: float, count: int) -> list[float]:
        result = []
        for offset in range(count):
            point = position * self.rate + offset * self.speed
            if point < 0.0:
                result.append(0.0)
                continue
            left = math.floor(point + 1e-7)
            while self.index < left + 1:
                self.previous = self.current
                self.current = next(self.samples, 0.0)
                self.index += 1
            fraction = max(0.0, point - left)
            result.append(self.previous + (self.current - self.previous) * fraction)
        return result

    def close(self) -> None:
        self.samples.close()
        self.source.close()


class _AudClipReader(_WavClipReader):
    """Decode with Blender into a temporary WAV, then stream without seeks.

    Native aud.limit().data() recreates a decoder for each chunk; compressed
    seeks can lose samples. Native write() instead keeps its decoder alive and
    uses a bounded buffer. Temporary files are removed as soon as a clip ends.
    """

    def __init__(
        self, clip: ChannelAudioClip, rate: int, chunk_frames: int,
        aud: Any, duration: float,
    ) -> None:
        end = max(0.0, clip.source_start) + duration + 2 / rate
        if end * rate > MAX_OUTPUT_FRAMES:
            raise SelectedAudioTimelineError("sound source trim exceeds the decoded-audio limit")
        self._temporary = tempfile.TemporaryDirectory(prefix="audio2face-channel-")
        path = os.path.join(self._temporary.name, "decoded.wav")
        error_type = getattr(aud, "error", RuntimeError)
        try:
            source_path = clip.path
            if clip.packed_file is not None:
                source_path = os.path.join(self._temporary.name, "packed-source")
                with open(source_path, "wb") as packed_handle:
                    packed_handle.write(clip.packed_file.data)
            sound = aud.Sound(source_path).rechannel(aud.CHANNELS_MONO).resample(rate, 2)
            sound.limit(0.0, end).write(
                path, rate, aud.CHANNELS_MONO, aud.FORMAT_FLOAT32,
                aud.CONTAINER_WAV, aud.CODEC_PCM, 0, 4_096,
            )
            super().__init__(replace(clip, path=path), rate, chunk_frames)
        except error_type as exc:
            self._temporary.cleanup()
            raise SelectedAudioTimelineError(f"could not decode channel audio: {exc}") from exc
        except Exception:
            self._temporary.cleanup()
            raise

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._temporary.cleanup()


class ChannelAudioSource:
    """One-shot bounded mono f32le stream of a channel's existing timeline.

    The stream begins at its first visible sound-strip frame. Gaps are silence;
    overlapping strips are summed and clipped to the model's [-1, 1] range.
    """

    def __init__(
        self,
        snapshot: SelectedChannelSnapshot,
        *,
        output_sample_rate: int = 16_000,
        chunk_frames: int = 4_096,
    ) -> None:
        for name, value, minimum, maximum in (
            ("output_sample_rate", output_sample_rate, MIN_SAMPLE_RATE, MAX_SAMPLE_RATE),
            ("chunk_frames", chunk_frames, 1, MAX_CHUNK_FRAMES),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise SelectedAudioTimelineError(f"{name} must be between {minimum} and {maximum}")
        self.snapshot = snapshot
        self.chunk_frames = chunk_frames
        self._fps = frames_per_second(snapshot.fps, snapshot.fps_base)
        output_frames = round(
            (snapshot.frame_end + 1 - snapshot.frame_start) * output_sample_rate / self._fps
        )
        if not 0 < output_frames <= MAX_OUTPUT_FRAMES:
            raise SelectedAudioTimelineError("selected channel duration exceeds the audio limit")
        self.metadata = ChannelAudioMetadata(output_sample_rate, output_frames)
        self._closed = False
        self._iteration_started = False
        self._readers: dict[int, Any] = {}

    def close(self) -> None:
        self._closed = True
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> ChannelAudioSource:
        if self._closed:
            raise SelectedAudioTimelineError("channel source is closed")
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def _reader(self, index: int, clip: ChannelAudioClip) -> Any:
        if index not in self._readers:
            try:
                import aud
            except ImportError:
                reader = _WavClipReader(clip, self.metadata.output_sample_rate, self.chunk_frames)
            else:
                reader = _AudClipReader(
                    clip, self.metadata.output_sample_rate, self.chunk_frames, aud,
                    (clip.frame_end - clip.frame_start) / self._fps,
                )
            self._readers[index] = reader
        return self._readers[index]

    def __iter__(self) -> Iterator[bytes]:
        if self._closed:
            raise SelectedAudioTimelineError("channel source is closed")
        if self._iteration_started:
            raise SelectedAudioTimelineError("channel source can only be iterated once")
        self._iteration_started = True

        def chunks() -> Iterator[bytes]:
            rate = self.metadata.output_sample_rate
            spans = [
                (
                    round((clip.frame_start - self.snapshot.frame_start) * rate / self._fps),
                    round((clip.frame_end - self.snapshot.frame_start) * rate / self._fps),
                )
                for clip in self.snapshot.clips
            ]
            try:
                for start in range(0, self.metadata.output_frames, self.chunk_frames):
                    if self._closed:
                        raise SelectedAudioTimelineError("channel source was closed while streaming")
                    count = min(self.chunk_frames, self.metadata.output_frames - start)
                    mixed = [0.0] * count
                    for index, (clip, (left, right)) in enumerate(zip(self.snapshot.clips, spans)):
                        begin, end = max(start, left), min(start + count, right)
                        if clip.muted or clip.volume == 0.0 or end <= begin:
                            continue
                        reader = self._reader(index, clip)
                        position = clip.source_start + (begin - left) * clip.speed / rate
                        samples = reader.read(position, end - begin)
                        for offset, value in enumerate(samples, begin - start):
                            mixed[offset] += value * clip.volume
                        if end >= right:
                            self._readers.pop(index).close()
                    yield struct.pack(f"<{count}f", *(max(-1.0, min(1.0, value)) for value in mixed))
            finally:
                self.close()

        return chunks()
