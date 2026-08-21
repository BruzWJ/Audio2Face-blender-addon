"""Main-thread playback and Shape Key delivery for incremental ARKit frames."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from pathlib import Path
import time
from typing import Any

import bpy

from .arkit import ARKIT_52_CHANNELS
from .frame_stream import sample_linear
from .preview import (
    PreviewError,
    TargetSubscription,
    apply_arkit_frame,
    build_subscriptions,
)


MAX_BUFFER_SECONDS = 4.0


class LiveStreamError(RuntimeError):
    """Raised when a live ARKit stream cannot be delivered safely."""


class LiveStreamController:
    """Buffer streamed frames and drive target Shape Keys on Blender's main thread."""

    def __init__(self) -> None:
        self._scene_name: str | None = None
        self._stream_id: str | None = None
        self._sample_rate = 0
        self._timestamps: list[int] = []
        self._weights: list[list[float]] = []
        self._subscriptions: tuple[TargetSubscription, ...] = ()
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
        return self._stream_id is not None and self._scene_name is not None

    @property
    def stream_id(self) -> str | None:
        return self._stream_id

    @property
    def plays_audio(self) -> bool:
        """Whether Blender owns audible playback for this stream."""

        return self.active and self._audio_path is not None

    def prepare(
        self,
        scene: bpy.types.Scene,
        stream_id: str,
        sample_rate: int,
        *,
        audio_path: str | Path | None = None,
        playback_started: Callable[[], None] | None = None,
        playback_stopped: Callable[[], None] | None = None,
    ) -> None:
        """Freeze target subscriptions before the worker accepts stream audio."""

        self.stop(reset=True)
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise LiveStreamError("stream sample rate must be a positive integer")
        subscriptions = build_subscriptions(scene.a2f_blender)
        if not subscriptions:
            raise LiveStreamError(
                "no enabled target mesh has an exact-name ARKit-52 shape key"
            )
        resolved_audio: Path | None = None
        if audio_path is not None:
            resolved_audio = Path(audio_path).expanduser().resolve(strict=False)
            if not resolved_audio.is_file():
                raise LiveStreamError(f"audio file does not exist: {resolved_audio}")

        self._scene_name = scene.name
        self._stream_id = stream_id
        self._sample_rate = sample_rate
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
        scene = bpy.data.scenes.get(self._scene_name) if self._scene_name else None
        if scene is None:
            try:
                handle.stop()
            except Exception:
                pass
            raise LiveStreamError("stream scene no longer exists")
        handle.volume = float(scene.a2f_blender.preview_volume)
        self._device = device
        self._sound = sound
        self._handle = handle
        self._aud = aud
        if self._playback_started is not None:
            callback = self._playback_started
            self._playback_started = None
            callback()

    @staticmethod
    def _validated_weights(weights: Sequence[float]) -> list[float]:
        if len(weights) != len(ARKIT_52_CHANNELS):
            raise LiveStreamError(
                f"ARKit frame has {len(weights)} values; expected {len(ARKIT_52_CHANNELS)}"
            )
        validated: list[float] = []
        for index, weight in enumerate(weights):
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise LiveStreamError(
                    f"ARKit channel {ARKIT_52_CHANNELS[index]} is not numeric"
                )
            value = float(weight)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise LiveStreamError(
                    f"ARKit channel {ARKIT_52_CHANNELS[index]} must be in [0, 1]"
                )
            validated.append(value)
        return validated

    def receive(
        self,
        stream_id: str,
        timestamp_sample: int,
        weights: Sequence[float],
    ) -> None:
        """Accept one validated worker frame and apply or buffer it."""

        if stream_id != self._stream_id or not self.active:
            raise LiveStreamError("received a frame for an inactive stream")
        if (
            isinstance(timestamp_sample, bool)
            or not isinstance(timestamp_sample, int)
            or not -(1 << 63) <= timestamp_sample < (1 << 63)
        ):
            raise LiveStreamError("stream frame timestamp_sample must fit a signed 64-bit integer")
        if self._timestamps and timestamp_sample <= self._timestamps[-1]:
            raise LiveStreamError("stream frame timestamps must be strictly increasing")
        frame = self._validated_weights(weights)
        self._timestamps.append(timestamp_sample)
        self._weights.append(frame)

        scene = bpy.data.scenes.get(self._scene_name) if self._scene_name else None
        if scene is None or not scene.is_editable:
            raise LiveStreamError("stream scene is no longer editable")
        settings = scene.a2f_blender
        settings.stream_time = max(0.0, timestamp_sample / self._sample_rate)

        if self._audio_path is None:
            if self._stream_clock_started is None:
                self._stream_clock_started = time.monotonic()
                self._stream_clock_origin = timestamp_sample
                self.tick()
        else:
            self._start_audio()

    def mark_terminal(self, stream_id: str) -> None:
        if stream_id != self._stream_id or not self.active:
            return
        self._terminal = True
        if self._audio_path is None:
            if (
                not self._timestamps
                or self._stream_sample_position() >= self._timestamps[-1]
            ):
                self.stop()
        elif self._handle is None:
            self.stop()

    def _stream_sample_position(self) -> float:
        if self._stream_clock_started is None:
            return float(self._stream_clock_origin)
        return self._stream_clock_origin + (
            time.monotonic() - self._stream_clock_started
        ) * self._sample_rate

    def _drop_old_frames(self, sample_position: float) -> None:
        if len(self._timestamps) < 3:
            return
        cutoff = max(0.0, sample_position - self._sample_rate * 0.25)
        remove = 0
        while remove + 1 < len(self._timestamps) and self._timestamps[remove + 1] < cutoff:
            remove += 1
        if remove:
            del self._timestamps[:remove]
            del self._weights[:remove]

        maximum_frames = max(2, int(MAX_BUFFER_SECONDS * 120.0))
        if len(self._timestamps) > maximum_frames:
            excess = len(self._timestamps) - maximum_frames
            del self._timestamps[:excess]
            del self._weights[:excess]

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
                apply_arkit_frame(
                    self._subscriptions,
                    sample_linear(self._timestamps, self._weights, sample_position),
                )
            except (PreviewError, RuntimeError, ValueError) as exc:
                self.stop(reset=False)
                scene.a2f_blender.status = "ERROR"
                scene.a2f_blender.status_message = str(exc)
                return False
            scene.a2f_blender.stream_time = max(0.0, sample_position / self._sample_rate)
            self._drop_old_frames(sample_position)
            if self._terminal and sample_position >= self._timestamps[-1]:
                self.stop()
                return False
            return True
        if self._handle is None:
            return True
        scene = bpy.data.scenes.get(self._scene_name)
        if scene is None:
            self.stop(reset=False)
            return False
        settings = scene.a2f_blender
        try:
            status = self._handle.status
            if status == self._aud.STATUS_STOPPED:
                self.stop()
                return False
            self._handle.volume = float(settings.preview_volume)
            position = max(0.0, float(self._handle.position))
            sample_position = position * self._sample_rate
            if self._timestamps:
                apply_arkit_frame(
                    self._subscriptions,
                    sample_linear(self._timestamps, self._weights, sample_position),
                )
            settings.stream_time = position
            self._drop_old_frames(sample_position)
            return True
        except (LiveStreamError, PreviewError, OSError, RuntimeError, ValueError) as exc:
            self.stop(reset=False)
            settings.status = "ERROR"
            settings.status_message = str(exc)
            return False

    def stop(self, *, reset: bool | None = None) -> None:
        scene = bpy.data.scenes.get(self._scene_name) if self._scene_name else None
        settings = scene.a2f_blender if scene is not None and scene.is_editable else None
        should_reset = (
            bool(settings.stream_reset_on_stop)
            if reset is None and settings
            else bool(reset)
        )
        if self._handle is not None:
            try:
                self._handle.stop()
            except Exception:
                pass
        if should_reset and self._subscriptions:
            apply_arkit_frame(self._subscriptions, [0.0] * len(ARKIT_52_CHANNELS))
        if settings is not None:
            settings.stream_time = 0.0

        stopped_callback = self._playback_stopped

        self._scene_name = None
        self._stream_id = None
        self._sample_rate = 0
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

    def close(self) -> None:
        self.stop(reset=True)


_LIVE_STREAM_CONTROLLER: LiveStreamController | None = None


def get_live_stream_controller() -> LiveStreamController:
    global _LIVE_STREAM_CONTROLLER
    if _LIVE_STREAM_CONTROLLER is None:
        _LIVE_STREAM_CONTROLLER = LiveStreamController()
    return _LIVE_STREAM_CONTROLLER


def unregister_live_stream() -> None:
    global _LIVE_STREAM_CONTROLLER
    if _LIVE_STREAM_CONTROLLER is not None:
        _LIVE_STREAM_CONTROLLER.close()
        _LIVE_STREAM_CONTROLLER = None


__all__ = [
    "LiveStreamController",
    "LiveStreamError",
    "get_live_stream_controller",
    "unregister_live_stream",
]
