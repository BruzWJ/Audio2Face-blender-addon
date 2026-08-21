"""View3D sidebar UI for the Audio2Face workflow."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import bpy

from .live_stream import get_live_stream_controller
from .runtime import get_controller
from .sidecar import Lifecycle

if TYPE_CHECKING:
    from .properties import A2FModelParameterItem, A2FSceneSettings


def _draw_audio_playback(
    layout: bpy.types.UILayout,
    settings: A2FSceneSettings,
) -> None:
    """Draw mode-specific audio controls beside their source selection."""

    playback_box = layout.box()
    playback_box.label(text="Audio Playback", icon="SPEAKER")
    if settings.input_mode == "SELECTED":
        result_name = (
            Path(settings.result_path).name
            if settings.result_path
            else "No generated result yet"
        )
        playback_box.label(text=result_name, icon="FILE")
        if settings.result_audio_path:
            playback_box.label(
                text=f"Audio: {Path(settings.result_audio_path).name}",
                icon="SOUND",
            )
        else:
            playback_box.label(
                text="Generate the selected WAV to enable playback",
                icon="INFO",
            )

        playback_row = playback_box.row(align=True)
        if settings.preview_state == "PLAYING":
            playback_row.operator("a2f.preview_pause", text="Pause", icon="PAUSE")
        elif settings.preview_state == "PAUSED":
            playback_row.operator("a2f.preview_play", text="Resume", icon="PLAY")
        elif settings.preview_state == "IDLE":
            playback_row.operator("a2f.preview_play", text="Play Result", icon="PLAY")
        else:
            raise RuntimeError(f"invalid preview state {settings.preview_state!r}")
        playback_row.operator("a2f.preview_stop", text="Stop", icon="CANCEL")
        playback_box.label(
            text=f"{settings.preview_time:.2f}s / {settings.preview_duration:.2f}s"
        )
        controls = playback_box.row(align=True)
        controls.prop(settings, "preview_loop")
        controls.prop(settings, "preview_volume")
        playback_box.prop(settings, "preview_reset_on_stop")
        playback_box.label(
            text="Plays audio and delivers model channels to targets in sync",
            icon="INFO",
        )
        return

    if settings.input_mode != "STREAM":
        raise RuntimeError(f"invalid input mode {settings.input_mode!r}")
    rate = settings.stream_sample_rate
    playback_box.label(
        text=(f"{rate} Hz mono PCM" if rate > 0 else "Stream audio is stopped"),
        icon="SOUND",
    )
    playback_box.label(text=f"Audio time: {settings.stream_time:.2f}s")
    live_stream = get_live_stream_controller()
    if not settings.stream_operation_id or live_stream.plays_audio:
        playback_box.prop(settings, "preview_volume")
    else:
        playback_box.label(
            text="External PCM source owns audio playback",
            icon="INFO",
        )
    playback_box.prop(settings, "stream_reset_on_stop")


def _draw_model_parameters(
    layout: bpy.types.UILayout,
    parameters: Iterable[A2FModelParameterItem],
) -> None:
    """Draw the exact opaque parameter IDs advertised by the worker."""

    for parameter in parameters:
        if parameter.kind == "integer":
            value_property = "int_value"
        elif parameter.kind == "float":
            value_property = "float_value"
        else:
            raise RuntimeError(f"invalid model parameter kind {parameter.kind!r}")
        layout.prop(
            parameter,
            value_property,
            text=parameter.path,
        )


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
        setup = controller.setup_snapshot()
        runtime_ready = setup.model_spec is not None and setup.engine_status.ready
        if runtime_ready:
            runtime_message = "Bundled GPU worker and models are ready"
        elif not setup.runtime_status.ready:
            runtime_message = setup.runtime_status.message
        elif not setup.model_status.ready:
            runtime_message = setup.model_status.message
        elif not setup.engine_status.ready:
            runtime_message = setup.engine_status.message
        else:
            raise RuntimeError("invalid Audio2Face setup snapshot")

        status_box = layout.box()
        header = status_box.row(align=True)
        status_icon = "ERROR" if settings.status == "ERROR" else "INFO"
        header.label(text=settings.status.replace("_", " ").title(), icon=status_icon)
        worker_pid = controller.client.pid
        header.label(text=f"PID {worker_pid}" if worker_pid is not None else "Stopped")
        status_box.label(text=settings.status_message)
        if settings.status in {"GENERATING", "CANCELLING"}:
            status_box.prop(settings, "progress", text="Progress", slider=True)

        runtime_box = layout.box()
        runtime_box.label(text="GPU Worker & Models", icon="PREFERENCES")
        if controller.optimization_in_progress:
            runtime_box.label(text="Model optimization in progress", icon="TIME")
            runtime_box.label(text="Manage optimization in Add-on Preferences")
        elif runtime_ready:
            runtime_box.label(
                text="Bundled GPU worker and models ready",
                icon="CHECKMARK",
            )
        else:
            warning = runtime_box.row()
            warning.alert = True
            warning.label(text="GPU worker and models are not ready", icon="ERROR")
            runtime_box.label(text=runtime_message)
            runtime_box.label(text="Open this add-on's Preferences", icon="INFO")

        worker_row = layout.row(align=True)
        worker_state = controller.client.state
        if worker_state == Lifecycle.STOPPING:
            worker_row.enabled = False
            worker_row.label(text="Worker is stopping", icon="TIME")
        elif worker_state == Lifecycle.RUNNING:
            worker_row.operator("a2f.stop_worker", text="Stop", icon="CANCEL")
        elif worker_state == Lifecycle.STOPPED:
            worker_row.enabled = (
                runtime_ready and not controller.optimization_in_progress
            )
            worker_row.operator("a2f.start_worker", text="Start Worker", icon="PLAY")
        elif worker_state == Lifecycle.FAILED:
            worker_row.enabled = (
                runtime_ready and not controller.optimization_in_progress
            )
            worker_row.operator("a2f.start_worker", text="Restart Worker", icon="PLAY")
        else:
            raise RuntimeError(f"invalid worker lifecycle {worker_state!r}")

        input_box = layout.box()
        input_box.label(text="Inputs", icon="SOUND")
        mode_row = input_box.row()
        mode_row.enabled = (
            not controller.operation_in_progress
            and not settings.stream_operation_id
            and settings.preview_state == "IDLE"
        )
        mode_row.prop(settings, "input_mode", expand=True)
        _draw_audio_playback(input_box, settings)
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
        input_box.label(
            text="Output: model-provided ARKit-52 channels",
            icon="SHAPEKEY_DATA",
        )
        if len(settings.model_identities) > 1:
            input_box.label(text="Identity")
            input_box.template_list(
                "UI_UL_list",
                "",
                settings,
                "model_identities",
                settings,
                "identity_index",
                rows=min(4, len(settings.model_identities)),
            )
        elif len(settings.model_identities) == 1:
            input_box.label(
                text=f"Identity: {settings.model_identities[0].name}",
                icon="USER",
            )
        else:
            input_box.label(text="Identity loads from the model", icon="INFO")

        emotion_box = layout.box()
        emotion_box.label(text="Emotion Driver", icon="SOUND")
        mode_control = emotion_box.row()
        mode_control.enabled = not controller.operation_in_progress
        mode_control.prop(settings, "auto_audio2emotion")

        manual_box = emotion_box.box()
        manual_box.enabled = (
            not settings.auto_audio2emotion and not controller.operation_in_progress
        )
        manual_box.label(text="Manual Emotion Channels", icon="DRIVER")
        if settings.manual_emotions:
            manual_column = manual_box.column(align=True)
            for emotion in settings.manual_emotions:
                manual_column.prop(emotion, "value", text=emotion.name, slider=True)
        else:
            manual_box.label(
                text="Channels and defaults load dynamically from the model",
                icon="INFO",
            )
            manual_box.label(
                text="Start the GPU worker to make them available",
                icon="INFO",
            )

        if settings.auto_audio2emotion:
            emotion_box.label(
                text="Inferred values override the manual driver",
                icon="INFO",
            )

        tuning_box = layout.box()
        tuning_box.enabled = not controller.operation_in_progress
        tuning_box.label(text="Model & Emotion Controls", icon="MODIFIER")
        if settings.model_parameters:
            _draw_model_parameters(
                tuning_box.column(align=True),
                settings.model_parameters,
            )
        else:
            tuning_box.label(
                text="Controls and defaults load from the worker",
                icon="INFO",
            )

        operation_ready = (
            runtime_ready
            and not controller.optimization_in_progress
            and not controller.operation_in_progress
            and controller.client.state == Lifecycle.RUNNING
            and controller.negotiated
        )
        if settings.stream_operation_id:
            stream_row = layout.row(align=True)
            stream_row.scale_y = 1.3
            stream_row.operator("a2f.stop_stream", text="Stop Stream", icon="CANCEL")
        elif settings.status == "GENERATING":
            cancel_row = layout.row(align=True)
            cancel_row.scale_y = 1.3
            cancel_row.operator("a2f.cancel", text="Cancel Generation", icon="CANCEL")
        elif settings.status == "CANCELLING":
            cancel_row = layout.row(align=True)
            cancel_row.enabled = False
            cancel_row.scale_y = 1.3
            cancel_row.label(text="Cancellation requested", icon="TIME")
        elif settings.input_mode == "SELECTED":
            generate_row = layout.row(align=True)
            generate_row.scale_y = 1.3
            generate_button = generate_row.row(align=True)
            generate_button.enabled = operation_ready
            generate_button.operator("a2f.generate", icon="OUTLINER_OB_FORCE_FIELD")
        elif settings.input_mode == "STREAM":
            stream_row = layout.row(align=True)
            stream_row.scale_y = 1.3
            stream_start = stream_row.row(align=True)
            stream_start.enabled = operation_ready
            stream_start.operator(
                "a2f.stream_wav",
                text="Start WAV Stream",
                icon="PLAY",
            )
        else:
            raise RuntimeError(f"invalid input mode {settings.input_mode!r}")

        target_box = layout.box()
        target_box.label(text="Mesh Targets", icon="SHAPEKEY_DATA")
        target_box.label(
            text="Any mesh can receive the model channel stream",
            icon="INFO",
        )
        target_box.label(
            text="Missing Shape Keys are ignored during delivery",
            icon="INFO",
        )
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

CLASSES = (A2F_UL_target_meshes, A2F_PT_main)
