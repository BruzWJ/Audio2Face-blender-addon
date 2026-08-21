"""Extension-level setup for the managed local GPU worker."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty


def _uninstall_target(context: bpy.types.Context) -> tuple[str, str] | None:
    """Return the installed extension repository and package identifier."""

    parts = __package__.split(".")
    if len(parts) != 3 or parts[0] != "bl_ext":
        return None
    repo_module, package_id = parts[1:]
    for repository in context.preferences.extensions.repos:
        if (
            repository.enabled
            and repository.module == repo_module
            and repository.source == "USER"
        ):
            return repository.directory, package_id
    return None


class A2FAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    nvidia_terms_accepted: BoolProperty(
        name="I accept the NVIDIA terms above",
        description=(
            "One acknowledgment covering the linked runtime, Audio2Face, "
            "and Audio2Emotion terms"
        ),
        default=False,
    )

    def draw(self, _context: bpy.types.Context) -> None:
        from .runtime import get_controller

        layout = self.layout
        if _uninstall_target(_context) is not None:
            removal = layout.row()
            removal.alignment = "RIGHT"
            removal.operator(
                "a2f.uninstall",
                text="Uninstall",
            )
            layout.separator(type="LINE")

        controller = get_controller()
        runtime_ready, runtime_message = controller.runtime_availability()
        can_install, blocked_reason = controller.install_eligibility()

        setup = layout.box()
        setup.label(text="Managed Runtime & Models", icon="PREFERENCES")
        setup.label(
            text="One install manages the GPU runtime and both models.",
            icon="INFO",
        )
        setup.label(text="No executable or model paths need to be configured.")

        terms = setup.operator("wm.url_open", text="NVIDIA Terms", icon="URL")
        terms.url = (
            "https://www.nvidia.com/en-us/agreements/enterprise-software/"
            "nvidia-open-model-license/"
        )
        setup.prop(self, "nvidia_terms_accepted")

        setup.label(text="Model Sources")
        sources = setup.row(align=True)
        for label, url in (
            ("Audio2Face", "https://huggingface.co/nvidia/Audio2Face-3D-v3.0"),
            ("Audio2Emotion", "https://huggingface.co/nvidia/Audio2Emotion-v3.0"),
        ):
            source = sources.operator("wm.url_open", text=label, icon="URL")
            source.url = url

        if controller.install_in_progress:
            setup.label(text=controller.install_message, icon="TIME")
            setup.progress(
                factor=controller.install_progress,
                type="BAR",
                text=f"Install Progress {controller.install_progress:.0%}",
            )
            setup.operator(
                "a2f.cancel_runtime_install",
                text="Cancel Install",
                icon="CANCEL",
            )
            return

        if runtime_ready:
            setup.label(
                text="Runtime and GPU-optimized models are ready",
                icon="CHECKMARK",
            )
        else:
            unavailable = setup.row()
            unavailable.alert = True
            unavailable.label(text="Runtime and models are not ready", icon="ERROR")
            setup.label(text=runtime_message)

        action = setup.row()
        action.enabled = can_install
        action.scale_y = 1.2
        action.operator(
            "a2f.install_runtime",
            text=(
                "Repair / Rebuild Runtime"
                if runtime_ready
                else "Install Runtime & Models"
            ),
            icon="FILE_REFRESH" if runtime_ready else "IMPORT",
        )
        if not can_install:
            reason = setup.row()
            reason.alert = True
            reason.label(text=blocked_reason, icon="ERROR")
        elif controller.install_message:
            setup.label(text=controller.install_message, icon="INFO")


def get_preferences(
    context: bpy.types.Context | None = None,
) -> A2FAddonPreferences | None:
    context = context or bpy.context
    addon = context.preferences.addons.get(__package__)
    return addon.preferences if addon is not None else None


CLASSES = (A2FAddonPreferences,)
