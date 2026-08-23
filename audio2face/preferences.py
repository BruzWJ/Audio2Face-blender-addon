"""Audio2Face extension preferences and model setup."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, StringProperty

from .ui_text import context_wrap_width, draw_wrapped_label


class A2FAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    nvidia_terms_accepted: BoolProperty(
        name="I accept the NVIDIA terms",
        description="Acknowledge the linked NVIDIA terms",
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
        text_width = context_wrap_width(_context)
        controller = get_controller()
        snapshot = controller.setup_snapshot()
        engine_status = snapshot.engine_status
        can_optimize, blocked_reason = controller.optimization_eligibility(snapshot)

        setup = layout.column()
        setup.label(text="GPU Runtime & Models", icon="PREFERENCES")

        terms_row = setup.row(align=True)
        terms_row.prop(self, "nvidia_terms_accepted")
        terms = terms_row.operator("wm.url_open", text="NVIDIA Terms", icon="URL")
        terms.url = (
            "https://www.nvidia.com/en-us/agreements/enterprise-software/"
            "nvidia-open-model-license/"
        )

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

        if controller.optimization_in_progress:
            draw_wrapped_label(
                setup,
                controller.optimization_message,
                width=text_width,
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
                width=text_width,
                icon="ERROR",
            )
        elif controller.optimization_failed:
            message = setup.column()
            message.alert = True
            draw_wrapped_label(
                message,
                controller.optimization_message,
                width=text_width,
                icon="ERROR",
            )
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
