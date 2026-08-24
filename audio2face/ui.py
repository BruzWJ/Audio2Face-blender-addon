"""View3D sidebar UI for the Audio2Face workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import bpy

from .live_stream import (
    PLAYBACK_POSITION_KEY,
    PLAYBACK_POSITION_PATH,
    get_live_stream_controller,
)
from .properties import AUDIO2FACE_SETTING_GROUPS
from .runtime import RuntimeController, get_controller
from .sidecar import Lifecycle
from .ui_text import context_wrap_width, draw_wrapped_label

if TYPE_CHECKING:
    from .properties import A2FSceneSettings


def _draw_audio_playback(
    layout: bpy.types.UILayout,
    settings: A2FSceneSettings,
    controller: RuntimeController,
    scene_name: str,
) -> None:
    """Draw playback controls that apply to the current input mode."""

    if settings.input_mode == "SELECTED":
        playback = get_live_stream_controller()
        playback_box = layout.box()
        playback_box.label(text="Playback", icon="SPEAKER")
        playback_row = playback_box.row(align=True)
        play_button = playback_row.row(align=True)
        if playback.can_seek and settings.playback_state == "PLAYING":
            play_button.operator(
                "a2f.play_pause", text="Pause", icon="PAUSE"
            )
        else:
            play_button.operator(
                "a2f.play_pause", text="Play", icon="PLAY"
            )
        playback_row.prop(settings, "playback_loop", text="Loop", toggle=True)

        if PLAYBACK_POSITION_KEY in settings:
            playback_box.prop(
                settings,
                PLAYBACK_POSITION_PATH,
                text="",
                slider=True,
            )
        playback_box.prop(settings, "prediction_delay", slider=True)
        return

    if settings.input_mode != "STREAM":
        raise RuntimeError(f"invalid input mode {settings.input_mode!r}")
    stream = controller.active_stream
    if (
        stream is None
        or stream.scene_name != scene_name
        or stream.wav_source is not None
    ):
        return
    playback_box = layout.box()
    playback_box.label(text="Stream", icon="SPEAKER")
    rate = controller.model_sample_rate
    if rate > 0:
        playback_box.label(
            text=f"{_timecode(settings.stream_time)}  |  {rate} Hz mono PCM"
        )


def _timecode(seconds: float) -> str:
    centiseconds = max(0, round(float(seconds) * 100.0))
    minutes, remainder = divmod(centiseconds, 6000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{minutes}:{whole_seconds:02d}.{fraction:02d}"


def _draw_model_tuning(
    layout: bpy.types.UILayout,
    settings: A2FSceneSettings,
) -> None:
    tuning_box = layout.box()
    tuning_box.label(text="Model Tuning", icon="MODIFIER")
    tuning_box.use_property_split = True
    tuning_box.use_property_decorate = False

    for index, (label, fields) in enumerate(AUDIO2FACE_SETTING_GROUPS):
        if index:
            tuning_box.separator()
        tuning_box.label(text=label)
        controls = tuning_box.column(align=True)
        for name, slider in fields:
            controls.prop(settings, name, slider=slider)


class A2F_UL_target_objects(bpy.types.UIList):
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
        if target is None:
            layout.label(text="Missing Object", icon="ERROR")
        else:
            layout.label(text=target.name, icon_value=layout.icon(target))


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
        runtime_ready = setup.engine_status.ready
        if not setup.runtime_status.ready:
            runtime_message = setup.runtime_status.message
        elif not setup.model_status.ready:
            runtime_message = setup.model_status.message
        elif not setup.engine_status.ready:
            runtime_message = setup.engine_status.message

        status_notice = controller.status_notice(context.scene)
        if status_notice is not None:
            status, message = status_notice
            status_box = layout.box()
            status_box.alert = status == "ERROR"
            draw_wrapped_label(
                status_box,
                message,
                width=text_width,
                icon="ERROR" if status == "ERROR" else "TIME",
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
        target_box.label(text="Target Objects", icon="SHAPEKEY_DATA")
        if not settings.target_objects:
            target_box.operator(
                "a2f.add_selected_targets",
                text="Add Selected Objects",
                icon="ADD",
            )
        else:
            target_row = target_box.row()
            target_row.template_list(
                "A2F_UL_target_objects",
                "",
                settings,
                "target_objects",
                settings,
                "target_object_index",
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
        input_mode_row = input_box.row(align=True)
        input_mode_row.prop(settings, "input_mode", expand=True)
        _draw_audio_playback(input_box, settings, controller, context.scene.name)
        if settings.input_mode == "SELECTED":
            input_box.prop(settings, "audio_path")

        _draw_model_tuning(layout, settings)

        emotion_box = layout.box()
        emotion_box.label(text="Emotion", icon="DRIVER")
        emotion_controls = emotion_box.column(align=True)
        emotion_controls.prop(settings, "auto_audio2emotion")
        auto_controls = emotion_controls.column(align=True)
        auto_controls.prop(settings, "a2e_emotion_strength", slider=True)
        auto_controls.prop(settings, "a2e_max_emotions", slider=True)
        auto_controls.prop(settings, "a2e_emotion_contrast", slider=True)
        auto_controls.prop(settings, "a2e_live_blend_coef", slider=True)
        auto_controls.prop(settings, "a2e_transition_smoothing", slider=True)
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
        auto_controls.prop(
            settings,
            "a2e_preferred_emotion_strength",
            slider=True,
        )

        if settings.manual_emotions:
            emotion_controls.separator()
            for emotion in settings.manual_emotions:
                emotion_controls.prop(
                    emotion,
                    "value",
                    text=emotion.name,
                    slider=True,
                )

CLASSES = (A2F_UL_target_objects, A2F_PT_main)
