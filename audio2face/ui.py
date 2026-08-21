"""View3D sidebar UI for the Audio2Face workflow."""

from __future__ import annotations

from pathlib import Path

import bpy

from .live_stream import get_live_stream_controller
from .preferences import get_preferences
from .properties import PARAMETER_GROUPS
from .runtime import get_controller
from .sidecar import Lifecycle


class A2F_UL_target_meshes(bpy.types.UIList):
    def draw_item(
        self,
        _context: bpy.types.Context,
        layout: bpy.types.UILayout,
        _data: object,
        item: object,
        _icon: int,
        _active_data: object,
        _active_property: str,
        _index: int = 0,
        _flt_flag: int = 0,
    ) -> None:
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.prop(item, "object", text="")


class A2F_PT_main(bpy.types.Panel):
    bl_label = "Audio2Face"
    bl_idname = "A2F_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Audio2Face"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        settings = context.scene.audio2face
        if not context.scene.is_editable:
            warning = layout.box()
            warning.alert = True
            warning.label(
                text="Use a local scene or editable library override",
                icon="ERROR",
            )
            return
        controller = get_controller()
        preferences = get_preferences(context)
        runtime_ready, runtime_message = controller.runtime_availability()
        install_available, install_message = controller.install_availability()

        status_box = layout.box()
        header = status_box.row(align=True)
        status_icon = "ERROR" if settings.status == "ERROR" else "INFO"
        header.label(text=settings.status.replace("_", " ").title(), icon=status_icon)
        header.label(text=f"PID {controller.client.pid}" if controller.client.pid else "Stopped")
        status_box.label(text=settings.status_message)
        if settings.status in {"GENERATING", "CANCELLING"}:
            status_box.prop(settings, "progress", text="Progress", slider=True)

        runtime_box = layout.box()
        runtime_box.label(text="Managed GPU Runtime", icon="PREFERENCES")
        if controller.install_in_progress:
            runtime_box.label(
                text=controller.install_message,
                icon="TIME",
            )
            runtime_box.prop(
                settings,
                "runtime_install_progress",
                text="Install Progress",
                slider=True,
            )
            runtime_box.operator(
                "a2f.cancel_runtime_install",
                text="Cancel Install",
                icon="CANCEL",
            )
        elif runtime_ready:
            runtime_box.label(text="Runtime and GPU-optimized models ready", icon="CHECKMARK")
            runtime_box.label(text="Managed automatically; no executable or model paths needed")
            repair_row = runtime_box.row()
            repair_row.enabled = bool(
                install_available
                and bpy.app.online_access
                and preferences is not None
                and preferences.runtime_license_accepted
                and controller.client.state in {Lifecycle.STOPPED, Lifecycle.FAILED}
            )
            repair_row.operator(
                "a2f.install_runtime",
                text="Repair / Rebuild Runtime",
                icon="FILE_REFRESH",
            )
        else:
            warning = runtime_box.row()
            warning.alert = True
            warning.label(text="Runtime and models are not ready", icon="ERROR")
            runtime_box.label(text=runtime_message)
            if not install_available:
                release_warning = runtime_box.row()
                release_warning.alert = True
                release_warning.label(text=install_message, icon="ERROR")
            if preferences is not None:
                runtime_box.prop(preferences, "runtime_license_accepted")
            model_links = runtime_box.row(align=True)
            face_terms = model_links.operator(
                "wm.url_open", text="Audio2Face Terms", icon="URL"
            )
            face_terms.url = "https://huggingface.co/nvidia/Audio2Face-3D-v3.0"
            emotion_terms = model_links.operator(
                "wm.url_open", text="Audio2Emotion Terms", icon="URL"
            )
            emotion_terms.url = "https://huggingface.co/nvidia/Audio2Emotion-v3.0"
            runtime_terms = runtime_box.operator(
                "wm.url_open", text="NVIDIA Runtime Terms", icon="URL"
            )
            runtime_terms.url = (
                "https://www.nvidia.com/en-us/agreements/enterprise-software/"
                "nvidia-software-license-agreement/"
            )
            can_install = bool(
                bpy.app.online_access
                and install_available
                and preferences is not None
                and preferences.runtime_license_accepted
            )
            if not bpy.app.online_access:
                runtime_box.label(
                    text="Enable Online Access in Blender Preferences first",
                    icon="ERROR",
                )
            install_row = runtime_box.row()
            install_row.enabled = can_install
            install_row.scale_y = 1.2
            install_row.operator(
                "a2f.install_runtime",
                text="Install Runtime & Models",
                icon="IMPORT",
            )

        worker_row = layout.row(align=True)
        worker_state = controller.client.state
        if worker_state == Lifecycle.STOPPING:
            worker_row.enabled = False
            worker_row.label(text="Worker is stopping", icon="TIME")
        elif worker_state == Lifecycle.RUNNING:
            worker_row.operator("a2f.stop_worker", text="Stop", icon="CANCEL")
        else:
            worker_row.enabled = runtime_ready and not controller.install_in_progress
            worker_row.operator("a2f.start_worker", text="Start Worker", icon="PLAY")

        input_box = layout.box()
        input_box.label(text="Inputs", icon="SOUND")
        mode_row = input_box.row()
        mode_row.enabled = bool(
            not controller.operation_in_progress
            and not settings.stream_id
            and settings.preview_state == "IDLE"
        )
        mode_row.prop(settings, "input_mode", expand=True)
        input_box.prop(settings, "audio_path")
        if settings.input_mode == "STREAM":
            input_box.label(
                text="Built-in WAV source sends incremental mono PCM",
                icon="INFO",
            )
            input_box.label(
                text="Blender integrations may also push live f32le PCM",
                icon="INFO",
            )
        input_box.label(text="Model: managed Audio2Face ARKit resolver", icon="SHAPEKEY_DATA")
        input_box.prop(settings, "identity_index")

        emotion_box = layout.box()
        emotion_box.enabled = not controller.operation_in_progress
        emotion_box.label(text="Emotion Driver", icon="SOUND")
        emotion_box.prop(settings, "auto_audio2emotion")
        if settings.auto_audio2emotion:
            emotion_box.label(text="Input audio overrides manual emotion values", icon="INFO")
            auto_column = emotion_box.column(align=True)
            auto_column.prop(settings, "emotion_strength")
            auto_column.prop(settings, "emotion_contrast")
            auto_column.prop(settings, "emotion_smoothing")
            auto_column.prop(settings, "emotion_transition_time")
            auto_column.prop(settings, "max_emotions")
        elif settings.manual_emotions:
            manual_column = emotion_box.column(align=True)
            for emotion in settings.manual_emotions:
                manual_column.prop(emotion, "value", text=emotion.name, slider=True)
        else:
            emotion_box.label(
                text="Start the worker to load model emotion channels",
                icon="INFO",
            )

        tuning_box = layout.box()
        tuning_box.enabled = not controller.operation_in_progress
        tuning_box.prop(
            settings,
            "show_tuning",
            icon="TRIA_DOWN" if settings.show_tuning else "TRIA_RIGHT",
            emboss=False,
        )
        if settings.show_tuning:
            tuning_box.label(text="Values refresh from the managed model", icon="INFO")
            tuning_box.prop(settings, "input_strength")
            for group, names in PARAMETER_GROUPS:
                column = tuning_box.column(align=True)
                column.label(text=group.title())
                for name in names:
                    column.prop(settings, name)

        operation_ready = bool(
            runtime_ready
            and not controller.install_in_progress
            and not controller.operation_in_progress
            and controller.client.state == Lifecycle.RUNNING
            and controller.negotiated
        )
        if settings.stream_id:
            stream_row = layout.row(align=True)
            stream_row.scale_y = 1.3
            stream_row.operator("a2f.stop_stream", text="Stop Stream", icon="CANCEL")
        elif settings.status in {"GENERATING", "CANCELLING"}:
            cancel_row = layout.row(align=True)
            cancel_row.scale_y = 1.3
            cancel_row.operator("a2f.cancel", text="Cancel Generation", icon="CANCEL")
        elif settings.input_mode == "SELECTED":
            generate_row = layout.row(align=True)
            generate_row.scale_y = 1.3
            generate_button = generate_row.row(align=True)
            generate_button.enabled = operation_ready
            generate_button.operator("a2f.generate", icon="OUTLINER_OB_FORCE_FIELD")
        else:
            stream_row = layout.row(align=True)
            stream_row.scale_y = 1.3
            stream_start = stream_row.row(align=True)
            stream_start.enabled = operation_ready
            stream_start.operator("a2f.stream_wav", text="Start WAV Stream", icon="PLAY")

        target_box = layout.box()
        target_box.label(text="ARKit Shape-Key Targets", icon="SHAPEKEY_DATA")
        target_box.label(text="Exact ARKit-52 names; each mesh may have a subset", icon="INFO")
        target_row = target_box.row(align=True)
        target_row.operator("a2f.add_selected_targets", icon="ADD")
        target_row.operator("a2f.remove_target", text="Remove", icon="REMOVE")
        if settings.target_meshes:
            target_box.template_list(
                "A2F_UL_target_meshes",
                "",
                settings,
                "target_meshes",
                settings,
                "target_mesh_index",
                rows=3,
            )

        preview_box = layout.box()
        if settings.preview_state != "IDLE" or (
            settings.input_mode == "SELECTED" and not settings.stream_id
        ):
            preview_box.label(text="Generated ARKit-52 Stream", icon="PLAY")
            result_name = (
                Path(settings.result_path).name
                if settings.result_path
                else "No generated result"
            )
            preview_box.label(text=result_name, icon="FILE")
            if settings.result_audio_path:
                preview_box.label(
                    text=f"Audio: {Path(settings.result_audio_path).name}",
                    icon="SOUND",
                )
            preview_row = preview_box.row(align=True)
            if settings.preview_state == "PLAYING":
                preview_row.operator("a2f.preview_pause", text="Pause", icon="PAUSE")
            else:
                label = "Resume" if settings.preview_state == "PAUSED" else "Play Selected Audio"
                preview_row.operator("a2f.preview_play", text=label, icon="PLAY")
            preview_row.operator("a2f.preview_stop", text="Stop", icon="CANCEL")
            preview_box.label(
                text=f"{settings.preview_time:.2f}s / {settings.preview_duration:.2f}s"
            )
            controls = preview_box.row(align=True)
            controls.prop(settings, "preview_loop")
            controls.prop(settings, "preview_volume")
            preview_box.prop(settings, "preview_reset_on_stop")
        else:
            preview_box.label(text="Live ARKit-52 Stream", icon="PLAY")
            rate = settings.stream_sample_rate
            preview_box.label(
                text=(
                    f"{rate} Hz mono f32le PCM"
                    if rate
                    else "Stream is stopped"
                )
            )
            preview_box.label(text=f"Audio time: {settings.stream_time:.2f}s")
            live_stream = get_live_stream_controller()
            if not settings.stream_id or live_stream.plays_audio:
                preview_box.prop(settings, "preview_volume")
            else:
                preview_box.label(
                    text="External PCM source owns audio playback",
                    icon="INFO",
                )
            preview_box.prop(settings, "stream_reset_on_stop")


CLASSES = (A2F_UL_target_meshes, A2F_PT_main)


__all__ = ["CLASSES"]
