"""Audio2Face extension preferences and model setup."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, StringProperty

from .ui_text import draw_wrapped_label


PREFERENCES_TEXT_WIDTH = 88


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
        snapshot = controller.setup_snapshot()
        runtime_status = snapshot.runtime_status
        model_status = snapshot.model_status
        engine_status = snapshot.engine_status
        can_optimize, blocked_reason = controller.optimization_eligibility(snapshot)

        setup = layout.box()
        setup.label(text="Bundled GPU Runtime & Models", icon="PREFERENCES")
        setup.label(
            text="The native worker and CUDA/TensorRT libraries ship with this add-on.",
            icon="INFO",
        )
        runtime_row = setup.column()
        runtime_row.alert = not runtime_status.ready
        draw_wrapped_label(
            runtime_row,
            runtime_status.message,
            width=PREFERENCES_TEXT_WIDTH,
            icon="CHECKMARK" if runtime_status.ready else "ERROR",
        )

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
            (
                "Download Audio2Emotion",
                "https://huggingface.co/nvidia/Audio2Emotion-v3.0",
            ),
        ):
            source = sources.operator("wm.url_open", text=label, icon="URL")
            source.url = url
        model_directories = setup.column()
        model_directories.enabled = not controller.optimization_in_progress
        model_directories.prop(self, "audio2face_model_directory")
        model_directories.prop(self, "audio2emotion_model_directory")
        model_row = setup.column()
        model_row.alert = not model_status.ready
        draw_wrapped_label(
            model_row,
            model_status.message,
            width=PREFERENCES_TEXT_WIDTH,
            icon="CHECKMARK" if model_status.ready else "ERROR",
        )

        if controller.optimization_in_progress:
            draw_wrapped_label(
                setup,
                controller.optimization_message,
                width=PREFERENCES_TEXT_WIDTH,
                icon="TIME",
            )
            setup.progress(
                factor=controller.optimization_progress,
                type="BAR",
                text=f"Optimization {controller.optimization_progress:.0%}",
            )
            setup.operator(
                "a2f.cancel_model_optimization",
                text="Cancel Optimization",
                icon="CANCEL",
            )
            return

        if model_status.ready:
            optimized_status = setup.column()
            draw_wrapped_label(
                optimized_status,
                engine_status.message,
                width=PREFERENCES_TEXT_WIDTH,
                icon="CHECKMARK" if engine_status.ready else "INFO",
            )

        action = setup.row()
        action.enabled = can_optimize
        action.scale_y = 1.2
        action.operator(
            "a2f.optimize_models",
            text="Rebuild Models" if engine_status.ready else "Optimize Models",
            icon="FILE_REFRESH" if engine_status.ready else "MODIFIER",
        )
        if not can_optimize:
            reason = setup.column()
            reason.alert = True
            draw_wrapped_label(
                reason,
                blocked_reason,
                width=PREFERENCES_TEXT_WIDTH,
                icon="ERROR",
            )
        elif controller.optimization_message:
            message = setup.column()
            message.alert = controller.optimization_failed
            draw_wrapped_label(
                message,
                controller.optimization_message,
                width=PREFERENCES_TEXT_WIDTH,
                icon="ERROR" if controller.optimization_failed else "INFO",
            )
            if controller.optimization_failed:
                open_logs = message.operator(
                    "wm.path_open",
                    text="Open Optimization Logs",
                    icon="FILE_FOLDER",
                )
                open_logs.filepath = str(controller.log_directory())


def get_preferences() -> A2FAddonPreferences | None:
    addon = bpy.context.preferences.addons.get(__package__)
    return addon.preferences if addon is not None else None


CLASSES = (A2FAddonPreferences,)
