"""Timestamp-clocked model output preview onto Blender Shape Key values."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import bpy

from .frame_stream import sample_linear
from .result_io import AnimationResult

if TYPE_CHECKING:
    from .properties import A2FSceneSettings


class PreviewError(RuntimeError):
    """Raised when a result cannot be previewed safely."""


def build_subscriptions(settings: A2FSceneSettings) -> tuple[bpy.types.Object, ...]:
    """Freeze each enabled target mesh once without inspecting Shape Keys."""

    targets: list[bpy.types.Object] = []
    seen: set[int] = set()
    for item in settings.target_meshes:
        target = item.object
        if not item.enabled or target is None:
            continue
        try:
            pointer = target.as_pointer()
        except ReferenceError:
            continue
        if pointer in seen or target.type != "MESH":
            continue
        seen.add(pointer)
        targets.append(target)
    return tuple(targets)


def apply_shape_key_frame(
    targets: tuple[bpy.types.Object, ...],
    channels: Sequence[str],
    weights: Sequence[float],
) -> None:
    """Assign one model-described frame to exact-name Shape Key properties."""

    if len(weights) != len(channels):
        raise PreviewError(
            f"shape-key frame has {len(weights)} values; expected {len(channels)}"
        )

    # Resolve Shape Keys for every delivery rather than prebinding names. This
    # lets a subscribed mesh gain, lose, or replace Shape Keys while previewing.
    # Multiple objects can share one Key datablock, so assign that datablock once.
    seen_shape_keys: set[int] = set()
    for target in targets:
        try:
            shape_keys = target.data.shape_keys
        except ReferenceError:
            continue
        if shape_keys is None:
            continue
        try:
            pointer = shape_keys.as_pointer()
        except ReferenceError:
            continue
        if pointer in seen_shape_keys:
            continue
        seen_shape_keys.add(pointer)
        for channel_index, channel_name in enumerate(channels):
            key = shape_keys.key_blocks.get(channel_name)
            if key is None:
                continue
            key.value = weights[channel_index]


class PreviewController:
    """Own audio playback and apply its synchronized model output stream."""

    def __init__(self) -> None:
        self._scene_name: str | None = None
        self._timestamps: list[int] = []
        self._weights: list[list[float]] = []
        self._sample_rate = 0
        self._channels: tuple[str, ...] = ()
        self._subscriptions: tuple[bpy.types.Object, ...] = ()
        self._device: Any = None
        self._sound: Any = None
        self._handle: Any = None
        self._aud: Any = None

    @property
    def active(self) -> bool:
        return self._handle is not None and self._scene_name is not None

    def start(
        self,
        scene: bpy.types.Scene,
        result: AnimationResult,
        audio_path: str | Path,
    ) -> None:
        self.stop(reset=True)
        settings = scene.audio2face
        subscriptions = build_subscriptions(settings)
        if not subscriptions:
            raise PreviewError("no enabled target mesh is selected")
        resolved_audio = Path(audio_path).expanduser().resolve(strict=False)
        if not resolved_audio.is_file():
            raise PreviewError(f"audio file does not exist: {resolved_audio}")
        try:
            import aud

            device = aud.Device()
            sound = aud.Sound(str(resolved_audio))
            handle = device.play(sound)
            handle.loop_count = -1 if bool(settings.preview_loop) else 0
            handle.volume = float(settings.preview_volume)
        except Exception as exc:
            raise PreviewError(f"could not play selected audio: {exc}") from exc

        self._scene_name = scene.name
        self._timestamps = result.timestamps
        self._weights = result.weights
        self._sample_rate = int(result.sample_rate)
        self._channels = tuple(result.channels)
        self._subscriptions = subscriptions
        self._device = device
        self._sound = sound
        self._handle = handle
        self._aud = aud

        settings.preview_state = "PLAYING"
        settings.preview_time = 0.0
        settings.preview_duration = max(0.0, self._timestamps[-1] / self._sample_rate)
        settings.status_message = (
            f"Playing {len(result.channels)} model channels on "
            f"{len(subscriptions)} mesh target(s)"
        )
        self.tick()

    def pause(self) -> None:
        scene = bpy.data.scenes.get(self._scene_name) if self._scene_name else None
        if self._handle is None or scene is None:
            raise PreviewError("audio preview is not playing")
        if not self._handle.pause():
            raise PreviewError("audio device could not pause preview")
        scene.audio2face.preview_state = "PAUSED"

    def resume(self) -> None:
        scene = bpy.data.scenes.get(self._scene_name) if self._scene_name else None
        if self._handle is None or scene is None:
            raise PreviewError("audio preview is not paused")
        if not self._handle.resume():
            raise PreviewError("audio device could not resume preview")
        scene.audio2face.preview_state = "PLAYING"

    def stop(self, *, reset: bool | None = None) -> None:
        scene = bpy.data.scenes.get(self._scene_name) if self._scene_name else None
        settings = scene.audio2face if scene is not None else None
        should_reset = (
            bool(settings.preview_reset_on_stop)
            if reset is None
            else reset
        )
        if self._handle is not None:
            try:
                self._handle.stop()
            except Exception:
                pass
        if should_reset and self._subscriptions:
            apply_shape_key_frame(
                self._subscriptions,
                self._channels,
                [0.0] * len(self._channels),
            )
        if settings is not None:
            settings.preview_state = "IDLE"
            settings.preview_time = 0.0

        self._scene_name = None
        self._timestamps = []
        self._weights = []
        self._sample_rate = 0
        self._channels = ()
        self._subscriptions = ()
        self._handle = None
        self._sound = None
        self._device = None
        self._aud = None

    def tick(self) -> bool:
        if not self.active:
            return False
        scene = bpy.data.scenes.get(self._scene_name)
        if scene is None or self._handle is None:
            self.stop(reset=False)
            return False
        settings = scene.audio2face
        try:
            status = self._handle.status
            if status == self._aud.STATUS_STOPPED:
                self.stop()
                return False
            position = max(0.0, float(self._handle.position))
            self._handle.loop_count = -1 if bool(settings.preview_loop) else 0
            self._handle.volume = float(settings.preview_volume)
            frame = sample_linear(
                self._timestamps,
                self._weights,
                position * self._sample_rate,
            )
            apply_shape_key_frame(
                self._subscriptions,
                self._channels,
                frame,
            )
            settings.preview_time = position
            if status == self._aud.STATUS_PLAYING:
                settings.preview_state = "PLAYING"
            elif status == self._aud.STATUS_PAUSED:
                settings.preview_state = "PAUSED"
            else:
                raise PreviewError(f"audio device returned invalid preview status {status!r}")
            return True
        except Exception as exc:
            self.stop(reset=False)
            settings.status = "ERROR"
            settings.status_message = str(exc)
            return False

_PREVIEW_CONTROLLER: PreviewController | None = None


def get_preview_controller() -> PreviewController:
    global _PREVIEW_CONTROLLER
    if _PREVIEW_CONTROLLER is None:
        _PREVIEW_CONTROLLER = PreviewController()
    return _PREVIEW_CONTROLLER


def unregister_preview() -> None:
    global _PREVIEW_CONTROLLER
    if _PREVIEW_CONTROLLER is not None:
        _PREVIEW_CONTROLLER.stop(reset=True)
        _PREVIEW_CONTROLLER = None
