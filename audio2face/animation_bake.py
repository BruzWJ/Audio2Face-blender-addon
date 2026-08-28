"""Create Blender 5.2 Shape Key Actions from baked Audio2Face frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


ACTION_OWNER_KEY = "audio2face_animation_bake_owner"
ACTION_OWNER_VALUE = "audio2face/animation-bake/1"
ACTION_NAME = "Audio2Face Shape Key Bake"
ACTION_LAYER_NAME = "Audio2Face Bake"


class AnimationBakeError(ValueError):
    """Raised when a bake cannot be assigned without replacing artist data."""


@dataclass(frozen=True, slots=True)
class BakeTarget:
    """One unique Shape Key datablock and its matching model channels."""

    shape_keys: Any
    channels: tuple[tuple[int, str], ...]


def is_addon_bake_action(action: Any) -> bool:
    return action.get(ACTION_OWNER_KEY) == ACTION_OWNER_VALUE


def plan_bake_targets(
    channels: tuple[str, ...],
    targets: Iterable[Any],
) -> tuple[BakeTarget, ...]:
    """Resolve each unique compatible Shape Key datablock once before baking."""

    plans: list[BakeTarget] = []
    seen: set[int] = set()
    for target in targets:
        try:
            shape_keys = target.data.shape_keys
            if shape_keys is None:
                continue
            identity = shape_keys.as_pointer()
            if identity in seen:
                continue
            seen.add(identity)
            matched = tuple(
                (index, shape_key.path_from_id("value"))
                for index, name in enumerate(channels)
                if (shape_key := shape_keys.key_blocks.get(name)) is not None
            )
            if not matched:
                continue
            animation_data = shape_keys.animation_data
            active_action = animation_data.action if animation_data is not None else None
            if active_action is not None and not is_addon_bake_action(active_action):
                raise AnimationBakeError(
                    f"{shape_keys.name!r} has a non-Audio2Face active Action; "
                    "clear it before baking"
                )
            plans.append(BakeTarget(shape_keys, matched))
        except ReferenceError:
            continue
    return tuple(plans)


def _build_action(
    actions: Any,
    plan: BakeTarget,
    frames: tuple[int, ...],
    weights: tuple[tuple[float, ...], ...],
) -> tuple[Any, Any]:
    shape_keys = plan.shape_keys
    action = actions.new(name=f"{ACTION_NAME} - {shape_keys.name}")
    action[ACTION_OWNER_KEY] = ACTION_OWNER_VALUE
    slot = action.slots.new(id_type="KEY", name=shape_keys.name)
    layer = action.layers.new(name=ACTION_LAYER_NAME)
    strip = layer.strips.new(type="KEYFRAME")
    channelbag = strip.channelbag(slot, ensure=True)

    for channel_index, data_path in plan.channels:
        fcurve = channelbag.fcurves.new(data_path=data_path, index=0)
        points = fcurve.keyframe_points
        points.add(len(frames))
        for point, frame, row in zip(points, frames, weights, strict=True):
            point.co = (float(frame), row[channel_index])
            point.interpolation = "LINEAR"
        fcurve.update()
    return action, slot


def bake_shape_key_actions(
    frames: Iterable[int],
    weights: Iterable[tuple[float, ...]],
    targets: tuple[BakeTarget, ...],
    actions: Any,
) -> tuple[Any, ...]:
    """Create and assign one new layered Action per planned Shape Key ID."""

    frame_values = tuple(frames)
    weight_values = tuple(weights)
    if not frame_values or len(frame_values) != len(weight_values):
        raise AnimationBakeError("bake frames and weights must be nonempty and aligned")
    if not targets:
        return ()

    largest_channel = max(
        channel_index
        for target in targets
        for channel_index, _data_path in target.channels
    )
    if any(len(row) <= largest_channel for row in weight_values):
        raise AnimationBakeError("bake weight rows do not match the model channels")

    built = tuple(
        (target.shape_keys, *_build_action(actions, target, frame_values, weight_values))
        for target in targets
    )
    for shape_keys, action, slot in built:
        animation_data = shape_keys.animation_data_create()
        animation_data.action = action
        animation_data.action_slot = slot
    return tuple(action for _shape_keys, action, _slot in built)
