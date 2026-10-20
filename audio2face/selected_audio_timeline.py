"""Read one Sequencer channel without changing its strips or scene timing."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import struct
import tempfile
from typing import Any, Iterator

MAX_CHUNK_FRAMES = 65_536
MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 384_000
MAX_OUTPUT_FRAMES = 512 * 1024 * 1024 // 4
_DECODE_BUFFER_FRAMES = 4_096


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


def _channel_strips(scene: Any) -> tuple[Any, ...]:
    editor = scene.sequence_editor
    if editor is None:
        return ()
    # Channel numbers belong to this timeline, not nested meta timelines.
    return tuple(
        strip for strip in editor.strips
        if strip.type == "SOUND" and strip.channel == scene.audio2face.audio_channel
        and strip.sound is not None
    )


def _strip_frame_span(strip: Any) -> tuple[int, int]:
    """Return Blender 5.2's visible handle bounds, with an exclusive end."""

    return int(strip.left_handle), int(strip.right_handle)


def selected_channel_frame_span(scene: Any) -> tuple[int, int]:
    """Cover the scene's inclusive range and every selected sound strip."""

    frame_start, frame_end = int(scene.frame_start), int(scene.frame_end)
    for strip in _channel_strips(scene):
        start, end = _strip_frame_span(strip)
        if end > start:
            frame_start = min(frame_start, start)
            frame_end = max(frame_end, end - 1)
    return frame_start, frame_end


@dataclass(frozen=True, slots=True)
class ChannelAudioClip:
    """Frozen strip settings and native source references; frame_end is exclusive."""

    frame_start: int
    frame_end: int
    source_start: float
    volume: float
    muted: bool
    signature: tuple[Any, ...]
    sound: Any = field(compare=False, repr=False)
    scene: Any = field(compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class SelectedChannelSnapshot:
    """Comparable input description; frame_end is inclusive."""

    channel: int
    fps: int
    fps_base: float
    clips: tuple[ChannelAudioClip, ...]
    frame_start: int
    frame_end: int


def selected_channel_snapshot(scene: Any) -> SelectedChannelSnapshot:
    """Copy selected-channel input settings without touching Blender data.

    No audio file is opened or stat'ed here: comparisons run during playback.
    Empty regions and muted clips contribute silence throughout the scene range.
    Sound strips beyond that range extend the input without changing the scene.
    """

    import bpy

    strips = _channel_strips(scene)
    frame_start, frame_end = selected_channel_frame_span(scene)
    channel = scene.audio2face.audio_channel
    fps, fps_base = int(scene.render.fps), float(scene.render.fps_base)
    rate = frames_per_second(fps, fps_base)
    channels = scene.sequence_editor.channels if scene.sequence_editor is not None else ()
    channel_muted = any(
        item.number == channel and item.mute
        for item in channels
    )
    clips = []
    for strip in strips:
        start, end = _strip_frame_span(strip)
        if end <= start:
            continue
        sound = strip.sound
        origin = float(strip.content_start)
        trim, trim_end = int(strip.content_trim_start), int(strip.content_trim_end)
        offset, volume = float(strip.sound_offset), float(strip.volume)
        if not all(math.isfinite(value) for value in (origin, offset, volume)):
            raise SelectedAudioTimelineError("sound strip settings must be finite")
        if volume < 0.0:
            raise SelectedAudioTimelineError("sound strip volume must not be negative")
        muted = bool(channel_muted or strip.mute)
        if not muted and volume > 0.0:
            if strip.retiming_keys:
                raise SelectedAudioTimelineError(
                    "selected channel sound strips must use normal playback speed without retiming keys"
                )
            if any(not modifier.mute for modifier in strip.modifiers):
                raise SelectedAudioTimelineError(
                    "selected channel sound strips with audio modifiers are not supported"
                )
        packed = sound.packed_file
        clips.append(ChannelAudioClip(
            frame_start=start,
            frame_end=end,
            source_start=(start - origin + trim) / rate - offset,
            volume=volume,
            muted=muted,
            signature=(
                sound.as_pointer(), bpy.path.abspath(sound.filepath, library=sound.library),
                origin, trim, trim_end, offset,
                (packed.as_pointer(), packed.size) if packed is not None else None,
                sound.use_mono,
            ),
            sound=sound,
            scene=scene,
        ))
    clips.sort(key=lambda clip: (clip.frame_start, clip.frame_end, clip.signature[0]))
    return SelectedChannelSnapshot(
        channel=channel,
        fps=fps,
        fps_base=fps_base,
        clips=tuple(clips),
        frame_start=frame_start,
        frame_end=frame_end,
    )


@dataclass(frozen=True, slots=True)
class ChannelAudioMetadata:
    output_sample_rate: int
    output_frames: int


class _AudioClipReader:
    """Read native-decoded mono float32 samples with bounded PCM seeks."""

    def __init__(
        self, clip: ChannelAudioClip, rate: int, duration: float,
    ) -> None:
        import aud
        import bpy

        # Audaspace's FFmpeg writer fails on empty reads at exact buffer
        # boundaries and on short first buffers. Finish halfway through a
        # padded buffer, safely beyond the audio and limit()'s float32 rounding.
        required_frames = (max(0.0, clip.source_start) + duration) * rate
        decode_frames = (
            (math.ceil(required_frames / _DECODE_BUFFER_FRAMES) + 1)
            * _DECODE_BUFFER_FRAMES + _DECODE_BUFFER_FRAMES // 2
        )
        # Reserve a final buffer for the native writer's flush.
        if decode_frames + _DECODE_BUFFER_FRAMES > MAX_OUTPUT_FRAMES:
            raise SelectedAudioTimelineError("sound source trim exceeds the decoded-audio limit")
        self._rate = rate
        self._file = None
        self._temporary = tempfile.TemporaryDirectory(prefix="audio2face-channel-")
        path = os.path.join(self._temporary.name, "decoded.wav")
        try:
            if clip.sound.packed_file is not None:
                # Refresh packed bytes without changing authored Sound settings.
                clip.sound.update_tag()
            with bpy.context.temp_override(
                scene=clip.scene, view_layer=clip.scene.view_layers[0],
            ):
                depsgraph = bpy.context.evaluated_depsgraph_get()
                factory = clip.sound.evaluated_get(depsgraph).factory
            if factory is None:
                raise SelectedAudioTimelineError("selected sound has no evaluated audio source")
            # The native factory preserves the chosen media stream and packed
            # audio. Decode once to avoid compressed seeks losing samples.
            factory.rechannel(aud.CHANNELS_MONO).resample(rate, 2).join(
                aud.Sound.silence(rate),
            ).limit(0.0, decode_frames / rate).write(
                path, rate, aud.CHANNELS_MONO, aud.FORMAT_FLOAT32,
                aud.CONTAINER_WAV, aud.CODEC_PCM, 0, _DECODE_BUFFER_FRAMES,
            )
            self._file = open(path, "rb")
            self._data_offset, self._frames = self._pcm_data()
        except aud.error as exc:
            self.close()
            raise SelectedAudioTimelineError(f"could not decode channel audio: {exc}") from exc
        except Exception:
            self.close()
            raise

    def _pcm_data(self) -> tuple[int, int]:
        """Locate data in our native writer's float32 WAV, not arbitrary input."""

        header = self._file.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise SelectedAudioTimelineError("native decoder returned an invalid WAV header")
        while True:
            chunk = self._file.read(8)
            if len(chunk) != 8:
                raise SelectedAudioTimelineError("native decoder returned no complete audio data")
            kind, size = struct.unpack("<4sI", chunk)
            if kind == b"data":
                if size % 4:
                    raise SelectedAudioTimelineError("decoded audio must contain whole float32 samples")
                return self._file.tell(), size // 4
            self._file.seek(size + (size & 1), os.SEEK_CUR)

    def read(self, position: float, count: int) -> list[float]:
        result = [0.0] * count
        point = position * self._rate
        silence = min(count, max(0, math.ceil(-point)))
        point += silence
        first = math.floor(point + 1e-7)
        available = min(count - silence, self._frames - first)
        if available <= 0:
            return result
        size = min(available + 1, self._frames - first) * 4
        self._file.seek(self._data_offset + first * 4)
        payload = self._file.read(size)
        if len(payload) != size:
            raise SelectedAudioTimelineError("decoded audio was truncated")
        samples = struct.unpack(f"<{size // 4}f", payload)
        if not all(math.isfinite(value) for value in samples):
            raise SelectedAudioTimelineError("decoded audio contains a non-finite sample")
        samples = [max(-1.0, min(1.0, value)) for value in samples] + [0.0]
        fraction = max(0.0, point - first)
        for index in range(available):
            result[silence + index] = samples[index] + (
                samples[index + 1] - samples[index]
            ) * fraction
        return result

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        self._temporary.cleanup()


class ChannelAudioSource:
    """One-shot bounded mono f32le stream of a channel's existing timeline.

    Empty regions, including an entirely empty channel, are silence. Overlapping
    strips are summed and clipped to the model's [-1, 1] range.
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
        timeline_frames = snapshot.frame_end + 1 - snapshot.frame_start
        output_frames = max(1, round(timeline_frames * output_sample_rate / self._fps))
        if timeline_frames <= 0 or output_frames > MAX_OUTPUT_FRAMES:
            raise SelectedAudioTimelineError("selected channel duration exceeds the audio limit")
        self.metadata = ChannelAudioMetadata(output_sample_rate, output_frames)
        self._closed = False
        self._iteration_started = False
        self._readers: dict[int, _AudioClipReader] = {}

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

    def _reader(self, index: int, clip: ChannelAudioClip) -> _AudioClipReader:
        if index not in self._readers:
            self._readers[index] = _AudioClipReader(
                clip, self.metadata.output_sample_rate,
                (clip.frame_end - clip.frame_start) / self._fps,
            )
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
                        position = clip.source_start + (begin - left) / rate
                        samples = reader.read(position, end - begin)
                        for offset, value in enumerate(samples, begin - start):
                            mixed[offset] += value * clip.volume
                        if end >= right:
                            self._readers.pop(index).close()
                    yield struct.pack(f"<{count}f", *(max(-1.0, min(1.0, value)) for value in mixed))
            finally:
                self.close()

        return chunks()
