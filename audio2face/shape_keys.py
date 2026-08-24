"""Model-output validation and delivery to Blender Shape Keys."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import bpy

if TYPE_CHECKING:
    from .properties import A2FSceneSettings


OUTPUT_CHANNEL_COUNT = 52
SHAPE_KEY_OBJECT_TYPES = frozenset({"MESH", "CURVE", "SURFACE", "LATTICE"})


class ShapeKeyStreamError(ValueError):
    """Raised when a model frame cannot be delivered safely."""


def validate_output_channels(channels: Any) -> tuple[str, ...]:
    """Validate and freeze the model-provided ARKit output description."""

    if not isinstance(channels, list):
        raise ShapeKeyStreamError("channels must be a JSON array")
    if len(channels) != OUTPUT_CHANNEL_COUNT:
        raise ShapeKeyStreamError(
            f"channels must contain exactly {OUTPUT_CHANNEL_COUNT} names"
        )

    validated: list[str] = []
    seen: set[str] = set()
    for index, name in enumerate(channels):
        if not isinstance(name, str) or not name:
            raise ShapeKeyStreamError(
                f"channels[{index}] must be a non-empty string"
            )
        if name in seen:
            raise ShapeKeyStreamError(
                f"channels contains duplicate name {name!r}"
            )
        validated.append(name)
        seen.add(name)
    return tuple(validated)


def supports_shape_keys(target: bpy.types.Object) -> bool:
    """Return whether Blender 5.2 supports Shape Keys on this object type."""

    return target.type in SHAPE_KEY_OBJECT_TYPES


def resolve_target_objects(
    settings: A2FSceneSettings,
) -> tuple[bpy.types.Object, ...]:
    """Resolve the Shape Key-capable objects listed as live frame targets."""

    targets: list[bpy.types.Object] = []
    seen: set[int] = set()
    for item in settings.target_objects:
        target = item.object
        if target is None:
            continue
        try:
            pointer = target.as_pointer()
        except ReferenceError:
            continue
        if pointer in seen or not supports_shape_keys(target):
            continue
        seen.add(pointer)
        targets.append(target)
    return tuple(targets)


def apply_shape_key_frame(
    targets: tuple[bpy.types.Object, ...],
    channels: tuple[str, ...],
    weights: tuple[float, ...],
) -> None:
    """Assign one model-described frame to matching Shape Keys when present."""

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
        for index, name in enumerate(channels):
            key = shape_keys.key_blocks.get(name)
            if key is not None:
                key.value = weights[index]
