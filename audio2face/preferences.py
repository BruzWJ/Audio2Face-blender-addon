"""Extension-level setup for the managed local GPU worker."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, StringProperty


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
        description="Acknowledge the NVIDIA terms linked above",
        default=False,
    )
    audio2face_model_directory: StringProperty(
        name="Audio2Face model folder",
        description=(
            "Select the complete Audio2Face Hugging Face clone folder containing "
            "model.json"
        ),
        subtype="DIR_PATH",
        default="",
    )
    audio2emotion_model_directory: StringProperty(
        name="Audio2Emotion model folder",
        description=(
            "Select the complete Audio2Emotion Hugging Face clone folder containing "
            "model.json"
        ),
        subtype="DIR_PATH",
        default="",
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
        models_ready, model_message = controller.model_availability(
            require_engine=False
        )
        can_install, blocked_reason = controller.install_eligibility()

        setup = layout.box()
        setup.label(text="GPU Worker & Models", icon="PREFERENCES")
        setup.label(
            text="Setup installs this OS's worker and optimizes both selected models.",
            icon="INFO",
        )
        setup.label(text="The native worker path is managed automatically.")

        terms = setup.operator("wm.url_open", text="NVIDIA Terms", icon="URL")
        terms.url = (
            "https://www.nvidia.com/en-us/agreements/enterprise-software/"
            "nvidia-open-model-license/"
        )
        setup.prop(self, "nvidia_terms_accepted")

        setup.label(text="Clone or download both complete model repositories.")
        sources = setup.row(align=True)
        for label, url in (
            ("Download Audio2Face", "https://huggingface.co/nvidia/Audio2Face-3D-v3.0"),
            ("Download Audio2Emotion", "https://huggingface.co/nvidia/Audio2Emotion-v3.0"),
        ):
            source = sources.operator("wm.url_open", text=label, icon="URL")
            source.url = url
        model_directories = setup.column()
        model_directories.enabled = not controller.install_in_progress
        model_directories.prop(self, "audio2face_model_directory")
        model_directories.prop(self, "audio2emotion_model_directory")
        model_status = setup.row()
        model_status.alert = not models_ready
        model_status.label(
            text=model_message,
            icon="CHECKMARK" if models_ready else "ERROR",
        )

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
                text="GPU worker and optimized models are ready",
                icon="CHECKMARK",
            )
        else:
            unavailable = setup.row()
            unavailable.alert = True
            unavailable.label(text="GPU worker and models are not ready", icon="ERROR")
            setup.label(text=runtime_message)

        action = setup.row()
        action.enabled = can_install
        action.scale_y = 1.2
        action.operator(
            "a2f.install_runtime",
            text=(
                "Repair Worker / Rebuild Models"
                if runtime_ready
                else "Install Worker & Optimize Models"
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
