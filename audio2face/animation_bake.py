"""Write baked Audio2Face frames as Blender 5.2 Shape Key animation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


ACTION_NAME = "Audio2Face Shape Key Bake"
ACTION_GROUP_NAME = "Audio2Face Bake"
ACTION_OWNER_KEY = "audio2face_shape_key_bake"


class AnimationBakeError(ValueError):
    """Raised when baked frames cannot form a Shape Key Action."""


@dataclass(frozen=True, slots=True)
class BakeTarget:
    """One unique Shape Key datablock and its matching model channels."""

    shape_keys: Any
    channels: tuple[tuple[int, str], ...]


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
            plans.append(BakeTarget(shape_keys, matched))
        except ReferenceError:
            continue
    return tuple(plans)


def _write_action(
    actions: Any,
    plan: BakeTarget,
    frames: tuple[int, ...],
    weights: tuple[tuple[float, ...], ...],
) -> Any:
    shape_keys = plan.shape_keys
    animation_data = shape_keys.animation_data_create()
    action = animation_data.action
    if action is None or action.get(ACTION_OWNER_KEY) is not True:
        action = actions.new(name=f"{ACTION_NAME} - {shape_keys.name}")
        action[ACTION_OWNER_KEY] = True
        animation_data.action = action

    for layer in tuple(action.layers):
        action.layers.remove(layer)

    for channel_index, data_path in plan.channels:
        fcurve = action.fcurve_ensure_for_datablock(
            shape_keys,
            data_path,
            index=0,
            group_name=ACTION_GROUP_NAME,
        )
        points = fcurve.keyframe_points
        points.add(len(frames))
        for offset, (frame, row) in enumerate(zip(frames, weights, strict=True)):
            point = points[offset]
            point.co = (float(frame), row[channel_index])
            point.interpolation = "LINEAR"
        fcurve.update()
    return action


def bake_shape_key_actions(
    frames: Iterable[int],
    weights: Iterable[tuple[float, ...]],
    targets: tuple[BakeTarget, ...],
    actions: Any,
) -> tuple[Any, ...]:
    """Write baked curves into each planned Shape Key datablock's Action."""

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

    return tuple(
        _write_action(actions, target, frame_values, weight_values)
        for target in targets
    )
