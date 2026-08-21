"""Extension preferences for the managed local GPU worker."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty


class A2FAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    runtime_license_accepted: BoolProperty(
        name="I accept the NVIDIA runtime and both model license terms",
        description=(
            "Required before downloading Audio2Face, Audio2Emotion, CUDA, "
            "and TensorRT files"
        ),
        default=False,
    )

    def draw(self, _context: bpy.types.Context) -> None:
        layout = self.layout
        layout.label(
            text="Audio2X, TensorRT/CUDA, Audio2Face, and Audio2Emotion are managed by the add-on.",
            icon="INFO",
        )
        layout.prop(self, "runtime_license_accepted")


def get_preferences(context: bpy.types.Context | None = None) -> A2FAddonPreferences | None:
    context = context or bpy.context
    addon = context.preferences.addons.get(__package__)
    return addon.preferences if addon is not None else None


CLASSES = (A2FAddonPreferences,)


__all__ = ["A2FAddonPreferences", "CLASSES", "get_preferences"]
