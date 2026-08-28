"""View3D sidebar UI for the Audio2Face workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import bpy

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
    screen: bpy.types.Screen | None,
) -> None:
    """Draw playback controls that apply to the current input mode."""

    if settings.input_mode == "SELECTED":
        playback_box = layout.box()
        playback_box.label(text="Playback", icon="SPEAKER")
        playback_row = playback_box.row(align=True)
        if screen is not None and screen.is_animation_playing:
            playback_row.operator(
                "a2f.play_pause", text="Pause", icon="PAUSE"
            )
        else:
            playback_row.operator(
                "a2f.play_pause", text="Play", icon="PLAY"
            )
        bake = controller.active_bake
        if bake is not None and bake.scene_name == scene_name:
            bake_row = playback_box.row(align=True)
            bake_row.operator(
                "a2f.cancel_bake",
                text="Cancel Bake",
                icon="CANCEL",
            )
        else:
            playback_box.operator(
                "a2f.bake_animation",
                text="Bake Shape Key Animation",
                icon="ACTION",
            )
        playback_box.prop(settings, "prediction_delay", slider=True)
        return

    if settings.input_mode != "STREAM":
        raise RuntimeError(f"invalid input mode {settings.input_mode!r}")
    stream = controller.active_stream
    if stream is None or stream.scene_name != scene_name:
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
    tuning_header, tuning_body = layout.panel(
        "audio2face_model_tuning",
        default_closed=True,
    )
    tuning_header.label(text="Model Tuning", icon="MODIFIER")
    tuning_header.operator(
        "a2f.reset_model_tuning",
        text="Reset",
        icon="FILE_REFRESH",
    )
    if tuning_body is None:
        return
    tuning_body.use_property_split = True
    tuning_body.use_property_decorate = True

    for index, (label, fields) in enumerate(AUDIO2FACE_SETTING_GROUPS):
        if index:
            tuning_body.separator()
        tuning_body.label(text=label)
        controls = tuning_body.column(align=True)
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
        _draw_audio_playback(
            input_box,
            settings,
            controller,
            context.scene.name,
            context.screen,
        )
        if settings.input_mode == "SELECTED":
            input_box.prop(settings, "audio_path")

        _draw_model_tuning(layout, settings)

        preferred_header, preferred_body = layout.panel(
            "audio2face_preferred_emotion",
            default_closed=True,
        )
        preferred_header.label(text="Preferred Emotion", icon="BOOKMARKS")
        if preferred_body is not None:
            preferred_body.use_property_split = True
            preferred_body.use_property_decorate = True
            preferred_body.prop(
                settings,
                "a2e_preferred_emotion_strength",
                slider=True,
            )
            if settings.preferred_emotions:
                for emotion in settings.preferred_emotions:
                    preferred_body.prop(
                        emotion,
                        "value",
                        text=emotion.name,
                        slider=True,
                    )
            else:
                preferred_body.label(
                    text="Load the Audio2Face model to edit emotion values",
                    icon="INFO",
                )

        emotion_header, emotion_body = layout.panel(
            "audio2face_emotion_tuning",
            default_closed=True,
        )
        emotion_header.label(text="Emotion Tuning", icon="PREFERENCES")
        emotion_header.operator(
            "a2f.reset_emotion_settings",
            text="Reset",
            icon="FILE_REFRESH",
        )
        if emotion_body is not None:
            emotion_body.use_property_split = True
            emotion_body.use_property_decorate = True
            emotion_controls = emotion_body.column(align=True)
            emotion_controls.prop(settings, "auto_audio2emotion")
            emotion_controls.prop(settings, "a2e_emotion_strength", slider=True)
            emotion_controls.prop(settings, "a2e_max_emotions", slider=True)
            emotion_controls.prop(settings, "a2e_emotion_contrast", slider=True)
            emotion_controls.prop(settings, "a2e_live_blend_coef", slider=True)
            emotion_controls.prop(settings, "a2e_transition_smoothing", slider=True)

            mixed_box = emotion_body.box()
            mixed_box.label(text="Mixed Emotion", icon="DRIVER")
            mixed_controls = mixed_box.column(align=True)
            mixed_controls.enabled = False
            for emotion in settings.mixed_emotions:
                mixed_controls.prop(
                    emotion,
                    "value",
                    text=emotion.name,
                    slider=True,
                )

CLASSES = (A2F_UL_target_objects, A2F_PT_main)
