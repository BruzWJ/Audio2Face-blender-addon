"""Main-thread streamed Shape Key delivery."""

from __future__ import annotations

from collections.abc import Callable
import math
import time
from typing import Any

import bpy

from .frame_stream import sample_linear
from .properties import apply_mixed_emotions, reset_mixed_emotions
from .shape_keys import apply_shape_key_frame, resolve_target_objects


MAX_BUFFER_SECONDS = 4.0


class LiveStreamError(RuntimeError):
    """Raised when a model output stream cannot be delivered safely."""


def _validate_channel_values(
    names: tuple[str, ...],
    values: object,
    *,
    field: str,
    channel_kind: str,
    bounded: bool,
) -> tuple[float, ...]:
    if type(values) is not list:
        raise LiveStreamError(f"stream frame {field} must be a JSON array")
    if len(values) != len(names):
        raise LiveStreamError(
            f"stream frame has {len(values)} {field}; expected {len(names)}"
        )
    validated: list[float] = []
    for index, value in enumerate(values):
        if type(value) is not float:
            raise LiveStreamError(
                f"{channel_kind} channel {names[index]} is not a JSON float"
            )
        if not math.isfinite(value):
            raise LiveStreamError(
                f"{channel_kind} channel {names[index]} must be finite"
            )
        if bounded and not 0.0 <= value <= 1.0:
            raise LiveStreamError(
                f"{channel_kind} channel {names[index]} must be in [0, 1]"
            )
        validated.append(value)
    return tuple(validated)


def validate_stream_frame(
    channels: tuple[str, ...],
    emotion_channels: tuple[str, ...],
    timestamp_sample: object,
    weights: object,
    effective_emotions: object,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Validate one worker frame independently of its delivery state."""

    if (
        type(timestamp_sample) is not int
        or not -(1 << 63) <= timestamp_sample < (1 << 63)
    ):
        raise LiveStreamError(
            "stream frame timestamp_sample must fit a signed 64-bit integer"
        )
    return (
        _validate_channel_values(
            channels,
            weights,
            field="weights",
            channel_kind="output",
            bounded=True,
        ),
        _validate_channel_values(
            emotion_channels,
            effective_emotions,
            field="effective_emotions",
            channel_kind="emotion",
            bounded=False,
        ),
    )


def apply_model_frame(
    settings: Any,
    channels: tuple[str, ...],
    emotion_channels: tuple[str, ...],
    weights: tuple[float, ...],
    effective_emotions: tuple[float, ...],
) -> None:
    """Apply a frame already validated at the worker boundary."""

    apply_shape_key_frame(
        resolve_target_objects(settings),
        channels,
        weights,
    )
    apply_mixed_emotions(
        settings,
        emotion_channels,
        effective_emotions,
    )


class LiveStreamController:
    """Buffer external PCM model frames and present them from its source clock."""

    def __init__(self) -> None:
        self._scene_name: str | None = None
        self._operation_id: str | None = None
        self._sample_rate = 0
        self._channels: tuple[str, ...] = ()
        self._emotion_channels: tuple[str, ...] = ()
        self._timestamps: list[int] = []
        self._weights: list[tuple[float, ...]] = []
        self._effective_emotions: list[tuple[float, ...]] = []
        self._external_stopped: Callable[[], None] | None = None
        self._terminal = False
        self._stream_clock_started: float | None = None
        self._stream_clock_origin = 0

    @property
    def active(self) -> bool:
        return self._operation_id is not None and self._scene_name is not None

    def _prepare(
        self,
        scene: bpy.types.Scene,
        operation_id: str,
        sample_rate: int,
        channels: tuple[str, ...],
        emotion_channels: tuple[str, ...],
    ) -> None:
        self.stop(reset=True, notify=False)
        reset_mixed_emotions(scene.audio2face)
        self._scene_name = scene.name
        self._operation_id = operation_id
        self._sample_rate = sample_rate
        self._channels = channels
        self._emotion_channels = emotion_channels

    def prepare_external(
        self,
        scene: bpy.types.Scene,
        operation_id: str,
        sample_rate: int,
        channels: tuple[str, ...],
        emotion_channels: tuple[str, ...],
        presentation_stopped: Callable[[], None],
    ) -> None:
        """Prepare an external PCM stream driven by its monotonic audio clock."""

        self._prepare(scene, operation_id, sample_rate, channels, emotion_channels)
        self._external_stopped = presentation_stopped

    def receive(
        self,
        operation_id: str,
        timestamp_sample: int,
        weights: list[float],
        effective_emotions: list[float],
    ) -> None:
        """Accept one external worker frame and buffer it for presentation."""

        if operation_id != self._operation_id or not self.active:
            raise LiveStreamError("received a frame for an inactive stream")
        frame_weights, frame_effective_emotions = validate_stream_frame(
            self._channels,
            self._emotion_channels,
            timestamp_sample,
            weights,
            effective_emotions,
        )
        if self._timestamps and timestamp_sample <= self._timestamps[-1]:
            raise LiveStreamError("stream frame timestamps must be strictly increasing")
        self._timestamps.append(timestamp_sample)
        self._weights.append(frame_weights)
        self._effective_emotions.append(frame_effective_emotions)

        scene = bpy.data.scenes.get(self._scene_name)
        if scene is None or not scene.is_editable:
            raise LiveStreamError("stream scene is no longer editable")
        if self._stream_clock_started is None:
            self._stream_clock_started = time.monotonic()
            self._stream_clock_origin = timestamp_sample
            self.tick()

    def mark_terminal(self, operation_id: str) -> None:
        if operation_id != self._operation_id or not self.active:
            raise LiveStreamError("received a terminal event for an inactive stream")
        self._terminal = True
        scene = bpy.data.scenes.get(self._scene_name)
        if (
            not self._timestamps
            or scene is None
            or not scene.is_editable
            or self._terminal_reached(scene)
        ):
            self.stop(reset=False, notify=True)

    def _terminal_reached(self, scene: bpy.types.Scene) -> bool:
        return self._stream_sample_position() >= self._timestamps[-1]

    def _stream_sample_position(self) -> float:
        if self._stream_clock_started is None:
            return float(self._stream_clock_origin)
        return self._stream_clock_origin + (
            time.monotonic() - self._stream_clock_started
        ) * self._sample_rate

    def _requested_sample_position(self, scene: bpy.types.Scene) -> float:
        delay = float(scene.audio2face.prediction_delay)
        if not math.isfinite(delay):
            raise LiveStreamError("prediction delay must be finite")
        return self._stream_sample_position() + delay * self._sample_rate

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
            del self._effective_emotions[:remove]

    def _apply_sampled_frame(self, settings: Any, sample_position: float) -> None:
        apply_model_frame(
            settings,
            self._channels,
            self._emotion_channels,
            sample_linear(
                self._timestamps,
                self._weights,
                sample_position,
            ),
            sample_linear(
                self._timestamps,
                self._effective_emotions,
                sample_position,
            ),
        )

    def tick(self) -> bool:
        """Present external PCM from its monotonic source clock."""

        if not self.active:
            return False
        scene = bpy.data.scenes.get(self._scene_name)
        if scene is None or not scene.is_editable:
            self.stop(reset=False, notify=True)
            return False
        settings = scene.audio2face
        if not self._timestamps:
            return True
        try:
            sample_position = self._stream_sample_position()
            requested_sample = self._requested_sample_position(scene)
            self._apply_sampled_frame(settings, requested_sample)
            settings.stream_time = max(0.0, sample_position / self._sample_rate)
            self._drop_old_frames(min(sample_position, requested_sample))
            if self._terminal and self._terminal_reached(scene):
                self.stop(reset=False, notify=True)
                return False
            return True
        except (LiveStreamError, RuntimeError, ValueError) as exc:
            self.stop(reset=False, notify=True)
            settings.status = "ERROR"
            settings.status_message = str(exc)
            return False

    def stop(
        self,
        *,
        reset: bool,
        notify: bool = True,
    ) -> None:
        if type(reset) is not bool or type(notify) is not bool:
            raise TypeError("reset and notify must be exact bool values")
        scene = bpy.data.scenes.get(self._scene_name) if self._scene_name else None
        settings = scene.audio2face if scene is not None and scene.is_editable else None
        if reset and settings is not None:
            apply_shape_key_frame(
                resolve_target_objects(settings),
                self._channels,
                (0.0,) * len(self._channels),
            )
        if settings is not None:
            settings.stream_time = 0.0

        stopped_callback = self._external_stopped if notify else None
        self._scene_name = None
        self._operation_id = None
        self._sample_rate = 0
        self._channels = ()
        self._emotion_channels = ()
        self._timestamps.clear()
        self._weights.clear()
        self._effective_emotions.clear()
        self._external_stopped = None
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
        _LIVE_STREAM_CONTROLLER.stop(reset=True, notify=False)
        _LIVE_STREAM_CONTROLLER = None
