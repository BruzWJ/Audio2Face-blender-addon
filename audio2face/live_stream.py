"""Main-thread audio playback and streamed Shape Key delivery."""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path
import time
from typing import Any

import bpy

from .frame_stream import sample_linear
from .path_contract import require_unaliased_path
from .shape_keys import (
    ShapeKeyStreamError,
    apply_shape_key_frame,
    resolve_target_meshes,
    validate_output_channels,
)


MAX_BUFFER_SECONDS = 4.0


class LiveStreamError(RuntimeError):
    """Raised when a model output stream cannot be delivered safely."""


def validate_stream_frame(
    channels: tuple[str, ...],
    timestamp_sample: object,
    weights: object,
) -> tuple[float, ...]:
    """Validate one worker frame independently of its delivery state."""

    if type(timestamp_sample) is not int or not -(1 << 63) <= timestamp_sample < (1 << 63):
        raise LiveStreamError(
            "stream frame timestamp_sample must fit a signed 64-bit integer"
        )
    if type(weights) is not list:
        raise LiveStreamError("stream frame weights must be a JSON array")
    if len(weights) != len(channels):
        raise LiveStreamError(
            f"stream frame has {len(weights)} values; expected {len(channels)}"
        )
    validated: list[float] = []
    for index, weight in enumerate(weights):
        if type(weight) is not float:
            raise LiveStreamError(
                f"output channel {channels[index]} is not a JSON float"
            )
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise LiveStreamError(
                f"output channel {channels[index]} must be in [0, 1]"
            )
        validated.append(weight)
    return tuple(validated)


class LiveStreamController:
    """Buffer model frames and drive selected meshes from the audio clock."""

    def __init__(self) -> None:
        self._scene_name: str | None = None
        self._operation_id: str | None = None
        self._sample_rate = 0
        self._channels: tuple[str, ...] = ()
        self._timestamps: list[int] = []
        self._weights: list[tuple[float, ...]] = []
        self._audio_path: Path | None = None
        self._audio_start_position = 0.0
        self._start_paused = False
        self._device: Any = None
        self._sound: Any = None
        self._handle: Any = None
        self._duration = 0.0
        self._published_progress = 0.0
        self._playback_started: Callable[[], None] | None = None
        self._playback_paused: Callable[[], None] | None = None
        self._playback_resumed: Callable[[], None] | None = None
        self._playback_seeked: Callable[[float, bool], None] | None = None
        self._playback_stopped: Callable[[bool], None] | None = None
        self._terminal = False
        self._stream_clock_started: float | None = None
        self._stream_clock_origin = 0

    @property
    def active(self) -> bool:
        return self._operation_id is not None and self._scene_name is not None

    @property
    def operation_id(self) -> str | None:
        return self._operation_id

    @property
    def can_seek(self) -> bool:
        return (
            self.active
            and self._audio_path is not None
            and self._handle is not None
            and bool(self._handle.status)
            and self._duration > 0.0
        )

    def prepare(
        self,
        scene: bpy.types.Scene,
        operation_id: str,
        sample_rate: int,
        channels: list[str],
        *,
        audio_path: Path | None,
        audio_start_position: float = 0.0,
        start_paused: bool = False,
        playback_started: Callable[[], None] | None,
        playback_paused: Callable[[], None] | None = None,
        playback_resumed: Callable[[], None] | None = None,
        playback_seeked: Callable[[float, bool], None] | None = None,
        playback_stopped: Callable[[bool], None] | None,
    ) -> None:
        """Prepare one stream while resolving its target meshes at delivery time."""

        self.stop(reset=True, notify=False)
        if type(operation_id) is not str or not operation_id:
            raise LiveStreamError("operation ID must be a non-empty string")
        if type(sample_rate) is not int or sample_rate <= 0:
            raise LiveStreamError("stream sample rate must be a positive integer")
        if type(audio_start_position) is not float or not math.isfinite(
            audio_start_position
        ) or audio_start_position < 0.0:
            raise LiveStreamError("audio start position must be a finite non-negative float")
        if type(start_paused) is not bool:
            raise LiveStreamError("start_paused must be an exact bool")
        callbacks = (
            ("playback_started", playback_started),
            ("playback_paused", playback_paused),
            ("playback_resumed", playback_resumed),
            ("playback_seeked", playback_seeked),
            ("playback_stopped", playback_stopped),
        )
        for name, callback in callbacks:
            if callback is not None and not callable(callback):
                raise LiveStreamError(f"{name} must be callable or None")
        if audio_path is not None and not isinstance(audio_path, Path):
            raise LiveStreamError("stream audio path must be a Path or None")
        if audio_path is None and (audio_start_position != 0.0 or start_paused):
            raise LiveStreamError("external PCM cannot have selected-audio playback state")
        if type(channels) is not list:
            raise LiveStreamError(
                "invalid stream output contract: channels must be a JSON array"
            )
        try:
            validated_channels = validate_output_channels(channels)
        except ShapeKeyStreamError as exc:
            raise LiveStreamError(f"invalid stream output contract: {exc}") from exc

        resolved_audio: Path | None = None
        if audio_path is not None:
            resolved_audio = require_unaliased_path(
                audio_path,
                description="selected audio",
                error_type=LiveStreamError,
            )
            if not resolved_audio.is_file():
                raise LiveStreamError(f"audio file does not exist: {resolved_audio}")

        self._scene_name = scene.name
        self._operation_id = operation_id
        self._sample_rate = sample_rate
        self._channels = validated_channels
        self._audio_path = resolved_audio
        self._audio_start_position = audio_start_position
        self._start_paused = start_paused
        self._playback_started = playback_started
        self._playback_paused = playback_paused
        self._playback_resumed = playback_resumed
        self._playback_seeked = playback_seeked
        self._playback_stopped = playback_stopped

    @staticmethod
    def _sound_duration(sound: Any) -> float:
        length = sound.length
        specs = sound.specs
        if (
            isinstance(length, bool)
            or not isinstance(length, (int, float))
            or not math.isfinite(length)
            or length <= 0
            or not isinstance(specs, tuple)
            or len(specs) != 2
            or isinstance(specs[0], bool)
            or not isinstance(specs[0], (int, float))
            or not math.isfinite(specs[0])
            or specs[0] <= 0
        ):
            raise LiveStreamError("selected audio has no finite positive duration")
        duration = float(length) / float(specs[0])
        if not math.isfinite(duration) or duration <= 0.0:
            raise LiveStreamError("selected audio has no finite positive duration")
        return duration

    def _publish_position(self, settings: Any, position: float) -> None:
        position = min(max(0.0, position), self._duration)
        progress = position / self._duration if self._duration > 0.0 else 0.0
        settings.playback_progress = progress
        self._published_progress = float(settings.playback_progress)

    def _start_audio(self) -> None:
        try:
            import aud

            device = aud.Device()
            sound = aud.Sound(str(self._audio_path))
            duration = self._sound_duration(sound)
            start_position = min(self._audio_start_position, duration)
            handle = device.play(sound)
            handle.loop_count = 0
            handle.position = start_position
            if self._start_paused and not handle.pause():
                raise LiveStreamError("audio device could not pause selected audio")
        except LiveStreamError:
            raise
        except Exception as exc:
            raise LiveStreamError(f"could not play selected audio: {exc}") from exc

        scene = bpy.data.scenes.get(self._scene_name)
        if scene is None:
            handle.stop()
            raise LiveStreamError("stream scene no longer exists")
        self._device = device
        self._sound = sound
        self._handle = handle
        self._duration = duration
        settings = scene.audio2face
        settings.playback_duration = duration
        settings.playback_state = "PAUSED" if self._start_paused else "PLAYING"
        self._publish_position(settings, start_position)
        callback = self._playback_started
        self._playback_started = None
        if callback is not None:
            callback()
        if self._start_paused and self._playback_paused is not None:
            self._playback_paused()

    def receive(
        self,
        operation_id: str,
        timestamp_sample: int,
        weights: list[float],
    ) -> None:
        """Accept one worker frame and apply or buffer it."""

        if operation_id != self._operation_id or not self.active:
            raise LiveStreamError("received a frame for an inactive stream")
        frame = validate_stream_frame(self._channels, timestamp_sample, weights)
        if self._timestamps and timestamp_sample <= self._timestamps[-1]:
            raise LiveStreamError("stream frame timestamps must be strictly increasing")
        self._timestamps.append(timestamp_sample)
        self._weights.append(frame)

        scene = bpy.data.scenes.get(self._scene_name)
        if scene is None or not scene.is_editable:
            raise LiveStreamError("stream scene is no longer editable")
        settings = scene.audio2face
        if self._audio_path is None:
            settings.stream_time = max(0.0, timestamp_sample / self._sample_rate)
            if self._stream_clock_started is None:
                self._stream_clock_started = time.monotonic()
                self._stream_clock_origin = timestamp_sample
                self.tick()
            return

        delay = max(0.0, float(settings.prediction_delay))
        start_threshold = (self._audio_start_position + delay) * self._sample_rate
        if self._handle is None and timestamp_sample >= start_threshold:
            self._start_audio()

    def mark_terminal(self, operation_id: str) -> None:
        if operation_id != self._operation_id or not self.active:
            raise LiveStreamError("received a terminal event for an inactive stream")
        self._terminal = True
        if self._audio_path is None:
            if not self._timestamps or self._stream_sample_position() >= self._timestamps[-1]:
                self.stop(reset=False, notify=True, natural=True)
        elif self._handle is None:
            if self._timestamps:
                self._start_audio()
            else:
                self.stop(reset=False, notify=True, natural=True)

    def reset_frames(self, operation_id: str) -> None:
        """Discard cached frames before the worker replays current settings."""

        if operation_id != self._operation_id or not self.active:
            raise LiveStreamError("received a frame reset for an inactive stream")
        self._timestamps.clear()
        self._weights.clear()
        self._terminal = False

    def pause(self) -> None:
        scene = bpy.data.scenes.get(self._scene_name)
        if self._handle is None or scene is None or self._audio_path is None:
            raise LiveStreamError("selected audio is not playing")
        if not self._handle.pause():
            raise LiveStreamError("audio device could not pause selected audio")
        scene.audio2face.playback_state = "PAUSED"
        if self._playback_paused is not None:
            self._playback_paused()

    def resume(self) -> None:
        scene = bpy.data.scenes.get(self._scene_name)
        if self._handle is None or scene is None or self._audio_path is None:
            raise LiveStreamError("selected audio is not paused")
        if not self._handle.resume():
            raise LiveStreamError("audio device could not resume selected audio")
        scene.audio2face.playback_state = "PLAYING"
        if self._playback_resumed is not None:
            self._playback_resumed()

    def request_seek(self, position: float) -> None:
        if isinstance(position, bool) or not isinstance(position, (int, float)):
            raise TypeError("playback position must be a number")
        requested = float(position)
        if not math.isfinite(requested):
            raise LiveStreamError("playback position must be finite")
        scene = bpy.data.scenes.get(self._scene_name)
        if self._handle is None or scene is None or self._playback_seeked is None:
            raise LiveStreamError("selected audio playback is not active")
        # A stream seek needs at least one source sample to decode.  Blender's slider can
        # report exactly 1.0, so resolve that endpoint to the final model-rate sample.
        final_sample_position = max(
            0.0,
            self._duration - (1.0 / self._sample_rate),
        )
        requested = min(max(0.0, requested), final_sample_position)
        paused = scene.audio2face.playback_state == "PAUSED"
        self._playback_seeked(requested, paused)

    def _stream_sample_position(self) -> float:
        if self._stream_clock_started is None:
            return float(self._stream_clock_origin)
        return self._stream_clock_origin + (
            time.monotonic() - self._stream_clock_started
        ) * self._sample_rate

    def _drop_old_frames(self, sample_position: float) -> None:
        if len(self._timestamps) < 3:
            return
        cutoff = max(
            0.0,
            sample_position - self._sample_rate * 0.25,
            self._timestamps[-1] - self._sample_rate * MAX_BUFFER_SECONDS,
        )
        remove = 0
        while remove + 1 < len(self._timestamps) and self._timestamps[remove + 1] < cutoff:
            remove += 1
        if remove:
            del self._timestamps[:remove]
            del self._weights[:remove]

    def tick(self) -> bool:
        """Advance one audio-clocked stream; return whether fast polling is useful."""

        if not self.active:
            return False
        scene = bpy.data.scenes.get(self._scene_name)
        if scene is None or not scene.is_editable:
            self.stop(reset=False, notify=True)
            return False
        settings = scene.audio2face
        if self._audio_path is None:
            if not self._timestamps:
                return True
            sample_position = self._stream_sample_position()
            try:
                delay = float(settings.prediction_delay)
                if not math.isfinite(delay):
                    raise LiveStreamError("prediction delay must be finite")
                apply_shape_key_frame(
                    resolve_target_meshes(settings),
                    self._channels,
                    sample_linear(
                        self._timestamps,
                        self._weights,
                        sample_position + delay * self._sample_rate,
                    ),
                )
                settings.stream_time = max(0.0, sample_position / self._sample_rate)
                self._drop_old_frames(sample_position)
                if self._terminal and sample_position >= self._timestamps[-1]:
                    self.stop(reset=False, notify=True, natural=True)
                    return False
                return True
            except (LiveStreamError, RuntimeError, ValueError) as exc:
                self.stop(reset=False, notify=True)
                settings.status = "ERROR"
                settings.status_message = str(exc)
                return False

        if self._handle is None:
            return True
        try:
            if not self._handle.status:
                # Audio playback can finish a little before the worker has drained its
                # final inference frames.  Keep the operation registered until the
                # matching ``stream_ended`` event arrives; otherwise that valid terminal
                # event would look like an unknown operation and reject the worker.
                if self._terminal:
                    self.stop(reset=False, notify=True, natural=True)
                    return False
                self._publish_position(settings, self._duration)
                return True
            requested_progress = float(settings.playback_progress)
            if not math.isfinite(requested_progress) or not 0.0 <= requested_progress <= 1.0:
                raise LiveStreamError("playback progress must be in [0, 1]")
            if not math.isclose(
                requested_progress,
                self._published_progress,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                self.request_seek(requested_progress * self._duration)
                return True

            position = min(max(0.0, float(self._handle.position)), self._duration)
            delay = float(settings.prediction_delay)
            if not math.isfinite(delay):
                raise LiveStreamError("prediction delay must be finite")
            if self._timestamps:
                apply_shape_key_frame(
                    resolve_target_meshes(settings),
                    self._channels,
                    sample_linear(
                        self._timestamps,
                        self._weights,
                        (position + delay) * self._sample_rate,
                    ),
                )
            self._publish_position(settings, position)
            return True
        except (LiveStreamError, OSError, RuntimeError, ValueError) as exc:
            self.stop(reset=False, notify=True)
            settings.status = "ERROR"
            settings.status_message = str(exc)
            return False

    def stop_for_seek(self, position: float, *, paused: bool) -> None:
        """Stop the current stream while retaining its seek presentation."""

        if isinstance(position, bool) or not isinstance(position, (int, float)):
            raise TypeError("playback position must be a number")
        if type(paused) is not bool:
            raise TypeError("paused must be an exact bool")
        requested = float(position)
        if not math.isfinite(requested) or requested < 0.0:
            raise LiveStreamError(
                "playback position must be a finite non-negative number"
            )
        scene = bpy.data.scenes.get(self._scene_name)
        if (
            scene is None
            or not scene.is_editable
            or not self.active
            or self._audio_path is None
            or self._handle is None
            or self._duration <= 0.0
        ):
            raise LiveStreamError("selected audio playback is not active")
        duration = self._duration
        requested = min(requested, duration)
        self.stop(reset=False, notify=False)
        settings = scene.audio2face
        settings.playback_state = "PAUSED" if paused else "PLAYING"
        settings.playback_duration = duration
        settings.playback_progress = requested / duration

    def stop(
        self,
        *,
        reset: bool,
        notify: bool = True,
        natural: bool = False,
    ) -> None:
        if type(reset) is not bool or type(notify) is not bool or type(natural) is not bool:
            raise TypeError("reset, notify, and natural must be exact bool values")
        scene = bpy.data.scenes.get(self._scene_name) if self._scene_name else None
        settings = scene.audio2face if scene is not None and scene.is_editable else None
        if self._handle is not None:
            self._handle.stop()
        if reset and settings is not None:
            apply_shape_key_frame(
                resolve_target_meshes(settings),
                self._channels,
                (0.0,) * len(self._channels),
            )
        if settings is not None:
            settings.playback_state = "IDLE"
            settings.playback_duration = 0.0
            settings.playback_progress = 0.0
            settings.stream_time = 0.0

        stopped_callback = self._playback_stopped if notify else None
        self._scene_name = None
        self._operation_id = None
        self._sample_rate = 0
        self._channels = ()
        self._timestamps.clear()
        self._weights.clear()
        self._audio_path = None
        self._audio_start_position = 0.0
        self._start_paused = False
        self._device = None
        self._sound = None
        self._handle = None
        self._duration = 0.0
        self._published_progress = 0.0
        self._playback_started = None
        self._playback_paused = None
        self._playback_resumed = None
        self._playback_seeked = None
        self._playback_stopped = None
        self._terminal = False
        self._stream_clock_started = None
        self._stream_clock_origin = 0
        if stopped_callback is not None:
            stopped_callback(natural)


_LIVE_STREAM_CONTROLLER: LiveStreamController | None = None


def get_live_stream_controller() -> LiveStreamController:
    global _LIVE_STREAM_CONTROLLER
    if _LIVE_STREAM_CONTROLLER is None:
        _LIVE_STREAM_CONTROLLER = LiveStreamController()
    return _LIVE_STREAM_CONTROLLER


def unregister_live_stream() -> None:
    global _LIVE_STREAM_CONTROLLER
    if _LIVE_STREAM_CONTROLLER is not None:
        _LIVE_STREAM_CONTROLLER.stop(reset=True, notify=False)
        _LIVE_STREAM_CONTROLLER = None
