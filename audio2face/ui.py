"""View3D sidebar UI for the Audio2Face workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import bpy

from .live_stream import get_live_stream_controller
from .runtime import get_controller
from .sidecar import Lifecycle
from .ui_text import context_wrap_width, draw_wrapped_label

if TYPE_CHECKING:
    from .properties import A2FSceneSettings


def _draw_audio_playback(
    layout: bpy.types.UILayout,
    settings: A2FSceneSettings,
) -> None:
    """Draw playback controls that apply to the current input mode."""

    if settings.input_mode == "SELECTED":
        playback = get_live_stream_controller()
        playback_box = layout.box()
        playback_box.label(text="Playback", icon="SPEAKER")
        playback_row = playback_box.row(align=True)
        play_button = playback_row.row(align=True)
        if settings.playback_state == "PLAYING":
            play_button.operator(
                "a2f.play_pause", text="Pause", icon="PAUSE"
            )
        elif settings.playback_state in {"IDLE", "PAUSED"}:
            play_button.operator(
                "a2f.play_pause", text="Play", icon="PLAY"
            )
        else:
            raise RuntimeError(f"invalid playback state {settings.playback_state!r}")
        playback_row.operator("a2f.rewind", text="", icon="REW")
        playback_row.prop(settings, "playback_loop", text="Loop", toggle=True)

        if playback.plays_audio:
            seek_row = playback_box.row()
            seek_row.prop(settings, "playback_progress", text="", slider=True)
            playback_box.label(
                text=(
                    f"{_timecode(settings.playback_time)} / "
                    f"{_timecode(settings.playback_duration)}"
                )
            )
        playback_box.prop(settings, "prediction_delay")
        return

    if settings.input_mode != "STREAM":
        raise RuntimeError(f"invalid input mode {settings.input_mode!r}")
    if not settings.stream_operation_id:
        return
    playback_box = layout.box()
    playback_box.label(text="Stream", icon="SPEAKER")
    rate = settings.stream_sample_rate
    if rate > 0:
        playback_box.label(
            text=f"{_timecode(settings.stream_time)}  |  {rate} Hz mono PCM"
        )


def _timecode(seconds: float) -> str:
    centiseconds = max(0, round(float(seconds) * 100.0))
    minutes, remainder = divmod(centiseconds, 6000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{minutes}:{whole_seconds:02d}.{fraction:02d}"


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
        target = item.object
        layout.label(
            text=target.name if target is not None else "Missing Mesh",
            icon="OUTLINER_OB_MESH" if target is not None else "ERROR",
        )


class A2F_PT_main(bpy.types.Panel):
    bl_label = "Audio2Face"
    bl_idname = "A2F_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Audio2Face"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        text_width = context_wrap_width(context)
        settings = context.scene.audio2face
        if not context.scene.is_editable:
            warning = layout.box()
            warning.alert = True
            draw_wrapped_label(
                warning,
                "Use a local scene or editable library override",
                width=text_width,
                icon="ERROR",
            )
            return
        controller = get_controller()
        setup = controller.setup_snapshot()
        runtime_ready = setup.model_spec is not None and setup.engine_status.ready
        if not setup.runtime_status.ready:
            runtime_message = setup.runtime_status.message
        elif not setup.model_status.ready:
            runtime_message = setup.model_status.message
        elif not setup.engine_status.ready:
            runtime_message = setup.engine_status.message
        elif not runtime_ready:
            raise RuntimeError("invalid Audio2Face setup snapshot")

        visible_statuses = {
            "STARTING",
            "LOADING_MODEL",
            "STREAM_STARTING",
            "STREAM_ENDING",
            "STOPPING",
            "ERROR",
        }
        if settings.status in visible_statuses:
            status_box = layout.box()
            status_box.alert = settings.status == "ERROR"
            draw_wrapped_label(
                status_box,
                settings.status_message,
                width=text_width,
                icon="ERROR" if settings.status == "ERROR" else "TIME",
            )
        if controller.optimization_in_progress:
            runtime_box = layout.box()
            draw_wrapped_label(
                runtime_box,
                controller.optimization_message or "Optimizing models",
                width=text_width,
                icon="TIME",
            )
        elif not runtime_ready:
            runtime_box = layout.box()
            runtime_box.alert = True
            draw_wrapped_label(
                runtime_box,
                runtime_message,
                width=text_width,
                icon="ERROR",
            )
            runtime_box.label(text="Configure in Add-on Preferences", icon="PREFERENCES")

        target_box = layout.box()
        target_box.label(text="Target Meshes", icon="SHAPEKEY_DATA")
        if not settings.target_meshes:
            target_box.operator(
                "a2f.add_selected_targets",
                text="Add Selected Meshes",
                icon="ADD",
            )
        else:
            target_row = target_box.row()
            target_row.template_list(
                "A2F_UL_target_meshes",
                "",
                settings,
                "target_meshes",
                settings,
                "target_mesh_index",
                rows=3,
            )
            target_controls = target_row.column(align=True)
            target_controls.operator(
                "a2f.add_selected_targets",
                text="",
                icon="ADD",
            )
            target_controls.operator(
                "a2f.remove_target",
                text="",
                icon="REMOVE",
            )

        worker_row = layout.row(align=True)
        worker_state = controller.client.state
        if worker_state == Lifecycle.STOPPING:
            worker_row.enabled = False
            worker_row.label(text="Worker is stopping", icon="TIME")
        elif worker_state == Lifecycle.RUNNING:
            worker_row.operator("a2f.stop_worker", text="Stop Worker", icon="CANCEL")
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
            and settings.playback_state == "IDLE"
        )
        mode_row.prop(settings, "input_mode", expand=True)
        _draw_audio_playback(input_box, settings)
        if settings.input_mode == "SELECTED":
            input_box.prop(settings, "audio_path")

        emotion_box = layout.box()
        emotion_box.label(text="Emotion", icon="DRIVER")
        emotion_controls = emotion_box.column(align=True)
        emotion_controls.enabled = not controller.operation_in_progress
        emotion_controls.prop(settings, "auto_audio2emotion")
        auto_controls = emotion_controls.column(align=True)
        auto_controls.prop(settings, "a2e_emotion_strength")
        auto_controls.prop(settings, "a2e_max_emotions")
        auto_controls.prop(settings, "a2e_emotion_contrast")
        auto_controls.prop(settings, "a2e_live_blend_coef")
        auto_controls.prop(settings, "a2e_transition_smoothing")
        preferred_row = auto_controls.row(align=True)
        preferred_row.label(
            text=(
                "Preferred Emotion: is set"
                if settings.preferred_emotions
                else "Preferred Emotion: is not set"
            )
        )
        preferred_row.operator("a2f.load_preferred_emotion", text="Load")
        preferred_row.operator("a2f.clear_preferred_emotion", text="Clear")
        auto_controls.prop(settings, "a2e_preferred_emotion_strength")

        if settings.manual_emotions:
            emotion_controls.separator()
            emotion_controls.label(text="Manual Emotion")
            for emotion in settings.manual_emotions:
                emotion_controls.prop(
                    emotion,
                    "value",
                    text=emotion.name,
                    slider=True,
                )

CLASSES = (A2F_UL_target_meshes, A2F_PT_main)
