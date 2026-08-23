"""Timestamp-clocked model output preview onto Blender Shape Key values."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import bpy

from .frame_stream import sample_linear
from .path_contract import require_unaliased_path
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
    channels: tuple[str, ...],
    weights: tuple[float, ...],
) -> None:
    """Assign one model-described frame to exact-name Shape Key properties."""

    if type(targets) is not tuple:
        raise PreviewError("shape-key targets must be a frozen tuple")
    if type(channels) is not tuple:
        raise PreviewError("shape-key channels must be a frozen tuple")
    if type(weights) is not tuple:
        raise PreviewError("shape-key weights must be a frozen frame tuple")
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
        self._timestamps: tuple[int, ...] = ()
        self._weights: tuple[tuple[float, ...], ...] = ()
        self._sample_rate = 0
        self._channels: tuple[str, ...] = ()
        self._subscriptions: tuple[bpy.types.Object, ...] = ()
        self._device: Any = None
        self._sound: Any = None
        self._handle: Any = None
        self._aud: Any = None
        self._duration = 0.0
        self._published_progress = 0.0

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
        resolved_audio = require_unaliased_path(
            audio_path,
            description="preview audio",
            error_type=PreviewError,
        )
        if not resolved_audio.is_file():
            raise PreviewError(f"audio file does not exist: {resolved_audio}")
        try:
            import aud

            device = aud.Device()
            sound = aud.Sound(str(resolved_audio))
            sound_length = sound.length
            sound_specs = sound.specs
            if (
                isinstance(sound_length, bool)
                or not isinstance(sound_length, (int, float))
                or not math.isfinite(sound_length)
                or sound_length <= 0
                or not isinstance(sound_specs, tuple)
                or len(sound_specs) != 2
                or isinstance(sound_specs[0], bool)
                or not isinstance(sound_specs[0], (int, float))
                or not math.isfinite(sound_specs[0])
                or sound_specs[0] <= 0
            ):
                raise PreviewError("selected audio has no finite positive duration")
            duration = float(sound_length) / float(sound_specs[0])
            if not math.isfinite(duration) or duration <= 0.0:
                raise PreviewError("selected audio has no finite positive duration")
            handle = device.play(sound)
            handle.loop_count = -1 if settings.preview_loop else 0
        except Exception as exc:
            raise PreviewError(f"could not play selected audio: {exc}") from exc

        self._scene_name = scene.name
        self._timestamps = tuple(result.timestamps)
        self._weights = tuple(tuple(frame) for frame in result.weights)
        self._sample_rate = result.sample_rate
        self._channels = tuple(result.channels)
        self._subscriptions = subscriptions
        self._device = device
        self._sound = sound
        self._handle = handle
        self._aud = aud
        self._duration = duration

        settings.preview_state = "PLAYING"
        settings.preview_duration = duration
        self._publish_position(settings, 0.0)
        settings.status_message = (
            f"Playing {len(result.channels)} model channels on "
            f"{len(subscriptions)} mesh target(s)"
        )
        self.tick()

    def _publish_position(self, settings: A2FSceneSettings, position: float) -> None:
        position = min(max(0.0, position), self._duration)
        progress = position / self._duration if self._duration > 0.0 else 0.0
        settings.preview_time = position
        settings.preview_progress = progress
        self._published_progress = float(settings.preview_progress)

    def _apply_position(self, settings: A2FSceneSettings, position: float) -> None:
        delay = float(settings.prediction_delay)
        if not math.isfinite(delay):
            raise PreviewError("prediction delay must be finite")
        frame = sample_linear(
            self._timestamps,
            self._weights,
            (position + delay) * self._sample_rate,
        )
        apply_shape_key_frame(
            self._subscriptions,
            self._channels,
            frame,
        )
        self._publish_position(settings, position)

    def seek(self, position: float) -> None:
        """Seek audible playback and model delivery to one audio time."""

        if isinstance(position, bool) or not isinstance(position, (int, float)):
            raise TypeError("preview position must be a number")
        requested = float(position)
        if not math.isfinite(requested):
            raise PreviewError("preview position must be finite")
        scene = (
            bpy.data.scenes.get(self._scene_name)
            if self._scene_name is not None
            else None
        )
        if self._handle is None or scene is None:
            raise PreviewError("audio preview is not active")
        requested = min(max(0.0, requested), self._duration)
        try:
            self._handle.position = requested
            actual = min(max(0.0, float(self._handle.position)), self._duration)
            self._apply_position(scene.audio2face, actual)
        except (OSError, RuntimeError, ValueError) as exc:
            raise PreviewError(f"could not seek audio preview: {exc}") from exc

    def rewind(self) -> None:
        """Return to the beginning without changing play/pause state."""

        self.seek(0.0)

    def pause(self) -> None:
        scene = (
            bpy.data.scenes.get(self._scene_name)
            if self._scene_name is not None
            else None
        )
        if self._handle is None or scene is None:
            raise PreviewError("audio preview is not playing")
        if not self._handle.pause():
            raise PreviewError("audio device could not pause preview")
        scene.audio2face.preview_state = "PAUSED"

    def resume(self) -> None:
        scene = (
            bpy.data.scenes.get(self._scene_name)
            if self._scene_name is not None
            else None
        )
        if self._handle is None or scene is None:
            raise PreviewError("audio preview is not paused")
        if not self._handle.resume():
            raise PreviewError("audio device could not resume preview")
        scene.audio2face.preview_state = "PLAYING"

    def stop(self, *, reset: bool) -> None:
        if type(reset) is not bool:
            raise TypeError("reset must be an exact bool")
        scene = (
            bpy.data.scenes.get(self._scene_name)
            if self._scene_name is not None
            else None
        )
        settings = scene.audio2face if scene is not None else None
        if self._handle is not None:
            self._handle.stop()
        if reset and self._subscriptions:
            apply_shape_key_frame(
                self._subscriptions,
                self._channels,
                (0.0,) * len(self._channels),
            )
        if settings is not None:
            settings.preview_state = "IDLE"
            settings.preview_time = 0.0
            settings.preview_progress = 0.0

        self._scene_name = None
        self._timestamps = ()
        self._weights = ()
        self._sample_rate = 0
        self._channels = ()
        self._subscriptions = ()
        self._handle = None
        self._sound = None
        self._device = None
        self._aud = None
        self._duration = 0.0
        self._published_progress = 0.0

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
                self.stop(reset=False)
                return False
            requested_progress = float(settings.preview_progress)
            if (
                not math.isfinite(requested_progress)
                or requested_progress < 0.0
                or requested_progress > 1.0
            ):
                raise PreviewError("preview progress must be in [0, 1]")
            if not math.isclose(
                requested_progress,
                self._published_progress,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                self._handle.position = requested_progress * self._duration
            position = min(
                max(0.0, float(self._handle.position)),
                self._duration,
            )
            self._handle.loop_count = -1 if settings.preview_loop else 0
            self._apply_position(settings, position)
            if status == self._aud.STATUS_PLAYING:
                settings.preview_state = "PLAYING"
            elif status == self._aud.STATUS_PAUSED:
                settings.preview_state = "PAUSED"
            else:
                raise PreviewError(
                    f"audio device returned invalid preview status {status!r}"
                )
            return True
        except (OSError, RuntimeError, ValueError) as exc:
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
