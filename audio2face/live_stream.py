"""Main-thread playback and Shape Key delivery for model-described frames."""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path
import time
from typing import Any

import bpy

from .frame_stream import sample_linear
from .path_contract import require_unaliased_path
from .preview import (
    apply_shape_key_frame,
    build_subscriptions,
)
from .result_io import ResultValidationError, validate_output_channels


MAX_BUFFER_SECONDS = 4.0


class LiveStreamError(RuntimeError):
    """Raised when a live output stream cannot be delivered safely."""


class LiveStreamController:
    """Buffer streamed frames and drive target Shape Keys on Blender's main thread."""

    def __init__(self) -> None:
        self._scene_name: str | None = None
        self._operation_id: str | None = None
        self._sample_rate = 0
        self._channels: tuple[str, ...] = ()
        self._timestamps: list[int] = []
        self._weights: list[tuple[float, ...]] = []
        self._subscriptions: tuple[bpy.types.Object, ...] = ()
        self._audio_path: Path | None = None
        self._device: Any = None
        self._sound: Any = None
        self._handle: Any = None
        self._aud: Any = None
        self._playback_started: Callable[[], None] | None = None
        self._playback_stopped: Callable[[], None] | None = None
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
    def plays_audio(self) -> bool:
        """Whether Blender owns audible playback for this stream."""

        return self.active and self._audio_path is not None

    def prepare(
        self,
        scene: bpy.types.Scene,
        operation_id: str,
        sample_rate: int,
        channels: list[str],
        *,
        audio_path: Path | None,
        playback_started: Callable[[], None] | None,
        playback_stopped: Callable[[], None] | None,
    ) -> None:
        """Freeze target subscriptions before the worker accepts stream audio."""

        self.stop(reset=True)
        if type(operation_id) is not str or not operation_id:
            raise LiveStreamError("operation ID must be a non-empty string")
        if type(sample_rate) is not int or sample_rate <= 0:
            raise LiveStreamError("stream sample rate must be a positive integer")
        if playback_started is not None and not callable(playback_started):
            raise LiveStreamError("playback_started must be callable or None")
        if playback_stopped is not None and not callable(playback_stopped):
            raise LiveStreamError("playback_stopped must be callable or None")
        if audio_path is not None and not isinstance(audio_path, Path):
            raise LiveStreamError("stream audio path must be a Path or None")
        if type(channels) is not list:
            raise LiveStreamError(
                "invalid stream output contract: channels must be a JSON array"
            )
        try:
            validated_channels = validate_output_channels(channels)
        except ResultValidationError as exc:
            raise LiveStreamError(f"invalid stream output contract: {exc}") from exc
        subscriptions = build_subscriptions(scene.audio2face)
        if not subscriptions:
            raise LiveStreamError("no enabled target mesh is selected")
        resolved_audio: Path | None = None
        if audio_path is not None:
            resolved_audio = require_unaliased_path(
                audio_path,
                description="stream audio",
                error_type=LiveStreamError,
            )
            if not resolved_audio.is_file():
                raise LiveStreamError(f"audio file does not exist: {resolved_audio}")

        self._scene_name = scene.name
        self._operation_id = operation_id
        self._sample_rate = sample_rate
        self._channels = validated_channels
        self._subscriptions = subscriptions
        self._audio_path = resolved_audio
        self._playback_started = playback_started
        self._playback_stopped = playback_stopped
        self._terminal = False

    def _start_audio(self) -> None:
        if self._audio_path is None or self._handle is not None:
            return
        try:
            import aud

            device = aud.Device()
            sound = aud.Sound(str(self._audio_path))
            handle = device.play(sound)
        except Exception as exc:
            raise LiveStreamError(f"could not play streamed WAV: {exc}") from exc
        scene = (
            bpy.data.scenes.get(self._scene_name)
            if self._scene_name is not None
            else None
        )
        if scene is None:
            handle.stop()
            raise LiveStreamError("stream scene no longer exists")
        self._device = device
        self._sound = sound
        self._handle = handle
        self._aud = aud
        if self._playback_started is not None:
            callback = self._playback_started
            self._playback_started = None
            callback()

    def _validated_weights(self, weights: list[float]) -> tuple[float, ...]:
        if type(weights) is not list:
            raise LiveStreamError("stream frame weights must be a JSON array")
        if len(weights) != len(self._channels):
            raise LiveStreamError(
                f"stream frame has {len(weights)} values; "
                f"expected {len(self._channels)}"
            )
        validated: list[float] = []
        for index, weight in enumerate(weights):
            if type(weight) is not float:
                raise LiveStreamError(
                    f"output channel {self._channels[index]} is not a JSON float"
                )
            value = weight
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise LiveStreamError(
                    f"output channel {self._channels[index]} must be in [0, 1]"
                )
            validated.append(value)
        return tuple(validated)

    def receive(
        self,
        operation_id: str,
        timestamp_sample: int,
        weights: list[float],
    ) -> None:
        """Accept one validated worker frame and apply or buffer it."""

        if operation_id != self._operation_id or not self.active:
            raise LiveStreamError("received a frame for an inactive stream")
        if (
            type(timestamp_sample) is not int
            or not -(1 << 63) <= timestamp_sample < (1 << 63)
        ):
            raise LiveStreamError(
                "stream frame timestamp_sample must fit a signed 64-bit integer"
            )
        if self._timestamps and timestamp_sample <= self._timestamps[-1]:
            raise LiveStreamError("stream frame timestamps must be strictly increasing")
        frame = self._validated_weights(weights)
        self._timestamps.append(timestamp_sample)
        self._weights.append(frame)

        scene = (
            bpy.data.scenes.get(self._scene_name)
            if self._scene_name is not None
            else None
        )
        if scene is None or not scene.is_editable:
            raise LiveStreamError("stream scene is no longer editable")
        settings = scene.audio2face
        settings.stream_time = max(0.0, timestamp_sample / self._sample_rate)

        if self._audio_path is None:
            if self._stream_clock_started is None:
                self._stream_clock_started = time.monotonic()
                self._stream_clock_origin = timestamp_sample
                self.tick()
        else:
            self._start_audio()

    def mark_terminal(self, operation_id: str) -> None:
        if operation_id != self._operation_id or not self.active:
            raise LiveStreamError("received a terminal event for an inactive stream")
        self._terminal = True
        if self._audio_path is None:
            if (
                not self._timestamps
                or self._stream_sample_position() >= self._timestamps[-1]
            ):
                self.stop(reset=False)
        elif self._handle is None:
            self.stop(reset=False)

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
        while (
            remove + 1 < len(self._timestamps)
            and self._timestamps[remove + 1] < cutoff
        ):
            remove += 1
        if remove:
            del self._timestamps[:remove]
            del self._weights[:remove]

    def tick(self) -> bool:
        """Advance an audio-clocked stream; return whether fast polling is useful."""

        if not self.active:
            return False
        if self._audio_path is None:
            if not self._timestamps:
                return True
            scene = bpy.data.scenes.get(self._scene_name)
            if scene is None or not scene.is_editable:
                self.stop(reset=False)
                return False
            sample_position = self._stream_sample_position()
            try:
                apply_shape_key_frame(
                    self._subscriptions,
                    self._channels,
                    sample_linear(self._timestamps, self._weights, sample_position),
                )
            except (RuntimeError, ValueError) as exc:
                self.stop(reset=False)
                scene.audio2face.status = "ERROR"
                scene.audio2face.status_message = str(exc)
                return False
            scene.audio2face.stream_time = max(0.0, sample_position / self._sample_rate)
            self._drop_old_frames(sample_position)
            if self._terminal and sample_position >= self._timestamps[-1]:
                self.stop(reset=False)
                return False
            return True
        if self._handle is None:
            return True
        scene = bpy.data.scenes.get(self._scene_name)
        if scene is None:
            self.stop(reset=False)
            return False
        settings = scene.audio2face
        try:
            status = self._handle.status
            if status == self._aud.STATUS_STOPPED:
                self.stop(reset=False)
                return False
            position = max(0.0, self._handle.position)
            sample_position = position * self._sample_rate
            if self._timestamps:
                apply_shape_key_frame(
                    self._subscriptions,
                    self._channels,
                    sample_linear(self._timestamps, self._weights, sample_position),
                )
            settings.stream_time = position
            self._drop_old_frames(sample_position)
            return True
        except (OSError, RuntimeError, ValueError) as exc:
            self.stop(reset=False)
            settings.status = "ERROR"
            settings.status_message = str(exc)
            return False

    def stop(self, *, reset: bool) -> None:
        if type(reset) is not bool:
            raise TypeError("reset must be an exact bool")
        scene = (
            bpy.data.scenes.get(self._scene_name)
            if self._scene_name is not None
            else None
        )
        settings = scene.audio2face if scene is not None and scene.is_editable else None
        if self._handle is not None:
            self._handle.stop()
        if reset and self._subscriptions:
            apply_shape_key_frame(
                self._subscriptions,
                self._channels,
                (0.0,) * len(self._channels),
            )
        if settings is not None:
            settings.stream_time = 0.0

        stopped_callback = self._playback_stopped

        self._scene_name = None
        self._operation_id = None
        self._sample_rate = 0
        self._channels = ()
        self._timestamps.clear()
        self._weights.clear()
        self._subscriptions = ()
        self._audio_path = None
        self._device = None
        self._sound = None
        self._handle = None
        self._aud = None
        self._playback_started = None
        self._playback_stopped = None
        self._terminal = False
        self._stream_clock_started = None
        self._stream_clock_origin = 0
        if stopped_callback is not None:
            stopped_callback()

_LIVE_STREAM_CONTROLLER: LiveStreamController | None = None


def get_live_stream_controller() -> LiveStreamController:
    global _LIVE_STREAM_CONTROLLER
    if _LIVE_STREAM_CONTROLLER is None:
        _LIVE_STREAM_CONTROLLER = LiveStreamController()
    return _LIVE_STREAM_CONTROLLER


def unregister_live_stream() -> None:
    global _LIVE_STREAM_CONTROLLER
    if _LIVE_STREAM_CONTROLLER is not None:
        _LIVE_STREAM_CONTROLLER.stop(reset=True)
        _LIVE_STREAM_CONTROLLER = None
