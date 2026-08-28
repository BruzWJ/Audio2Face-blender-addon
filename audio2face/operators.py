"""User-facing operators; all Blender mutations run on the main thread."""

from __future__ import annotations

from typing import Callable

import bpy

from .properties import (
    reset_emotion_settings,
    reset_model_tuning,
)
from .runtime import RuntimeController, get_controller
from .selected_audio_timeline import (
    configure_selected_audio,
)
from .shape_keys import supports_shape_keys
from .sidecar import Lifecycle, SidecarError


def _run_runtime(
    operator: bpy.types.Operator,
    operation: Callable[[RuntimeController], None],
) -> set[str]:
    try:
        operation(get_controller())
    except (OSError, SidecarError, ValueError) as exc:
        operator.report({"ERROR"}, str(exc))
        return {"CANCELLED"}
    return {"FINISHED"}


class A2F_OT_start_worker(bpy.types.Operator):
    bl_idname = "a2f.start_worker"
    bl_label = "Start Worker"
    bl_description = "Start the bundled local Audio2Face GPU worker"

    def execute(self, context: bpy.types.Context) -> set[str]:
        return _run_runtime(self, lambda controller: controller.start(context.scene))


class A2F_OT_optimize_models(bpy.types.Operator):
    bl_idname = "a2f.optimize_models"
    bl_label = "Optimize Models"
    bl_description = "Build both selected NVIDIA models for CUDA device 0"

    @classmethod
    def poll(cls, _context: bpy.types.Context) -> bool:
        controller = get_controller()
        can_optimize, reason = controller.optimization_eligibility(
            controller.setup_snapshot()
        )
        if not can_optimize:
            cls.poll_message_set(reason)
        return can_optimize

    def execute(self, _context: bpy.types.Context) -> set[str]:
        return _run_runtime(self, lambda controller: controller.optimize_models())


class A2F_OT_cancel_model_optimization(bpy.types.Operator):
    bl_idname = "a2f.cancel_model_optimization"
    bl_label = "Cancel Model Optimization"
    bl_description = "Cancel the current TensorRT model optimization"

    @classmethod
    def poll(cls, _context: bpy.types.Context) -> bool:
        in_progress = get_controller().optimization_in_progress
        if not in_progress:
            cls.poll_message_set("model optimization is not running")
        return in_progress

    def execute(self, _context: bpy.types.Context) -> set[str]:
        return _run_runtime(
            self,
            lambda controller: controller.cancel_model_optimization(),
        )


class A2F_OT_stop_worker(bpy.types.Operator):
    bl_idname = "a2f.stop_worker"
    bl_label = "Stop Worker"
    bl_description = "Request graceful worker shutdown without blocking Blender"

    def execute(self, context: bpy.types.Context) -> set[str]:
        return _run_runtime(self, lambda controller: controller.stop(context.scene))


class A2F_OT_reset_model_tuning(bpy.types.Operator):
    bl_idname = "a2f.reset_model_tuning"
    bl_label = "Reset Model Tuning"
    bl_description = "Reset all Model Tuning controls to their default values"

    def execute(self, context: bpy.types.Context) -> set[str]:
        reset_model_tuning(context.scene.audio2face)
        get_controller().refresh_inference_settings(context.scene)
        return {"FINISHED"}


class A2F_OT_reset_emotion_settings(bpy.types.Operator):
    bl_idname = "a2f.reset_emotion_settings"
    bl_label = "Reset Emotion Tuning"
    bl_description = "Reset Emotion Tuning controls to their default values"

    def execute(self, context: bpy.types.Context) -> set[str]:
        reset_emotion_settings(context.scene.audio2face)
        get_controller().refresh_inference_settings(context.scene)
        return {"FINISHED"}


class A2F_OT_add_selected_targets(bpy.types.Operator):
    bl_idname = "a2f.add_selected_targets"
    bl_label = "Add Selected Objects"
    bl_description = "Add selected Shape Key-capable objects as Audio2Face targets"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        has_selected_target = any(
            supports_shape_keys(obj) for obj in context.selected_objects
        )
        if not has_selected_target:
            cls.poll_message_set("select a Mesh, Curve, Surface, or Lattice object")
        return has_selected_target

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = context.scene.audio2face
        selected = [
            obj for obj in context.selected_objects if supports_shape_keys(obj)
        ]
        if not selected:
            self.report(
                {"ERROR"},
                "select a Mesh, Curve, Surface, or Lattice object",
            )
            return {"CANCELLED"}

        last_added_index: int | None = None
        existing = {
            item.object.as_pointer()
            for item in settings.target_objects
            if item.object is not None
        }
        for target in selected:
            if target.as_pointer() in existing:
                continue
            item = settings.target_objects.add()
            item.object = target
            existing.add(target.as_pointer())
            last_added_index = len(settings.target_objects) - 1
        if last_added_index is None:
            self.report({"INFO"}, "Selected objects are already targets")
            return {"FINISHED"}
        settings.target_object_index = last_added_index
        self.report({"INFO"}, "Added selected objects as targets")
        return {"FINISHED"}


class A2F_OT_remove_target(bpy.types.Operator):
    bl_idname = "a2f.remove_target"
    bl_label = "Remove Target"
    bl_description = (
        "Stop driving the active object by removing it from the target list"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = context.scene.audio2face
        valid_index = (
            0 <= settings.target_object_index < len(settings.target_objects)
        )
        if not valid_index:
            cls.poll_message_set("select a target object to remove")
        return valid_index

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = context.scene.audio2face
        index = settings.target_object_index
        if not 0 <= index < len(settings.target_objects):
            self.report({"ERROR"}, "selected target object index is invalid")
            return {"CANCELLED"}
        settings.target_objects.remove(index)
        if settings.target_objects:
            settings.target_object_index = min(
                index,
                len(settings.target_objects) - 1,
            )
        else:
            settings.target_object_index = 0
        return {"FINISHED"}


class A2F_OT_play_pause(bpy.types.Operator):
    bl_idname = "a2f.play_pause"
    bl_label = "Play/Pause Selected Audio"
    bl_description = (
        "Play or pause Blender's native timeline while the selected WAV drives "
        "live Shape Keys"
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = context.scene.audio2face
        if settings.input_mode != "SELECTED":
            return False
        if not settings.audio_path:
            cls.poll_message_set("select a WAV file first")
            return False
        if context.screen is None:
            cls.poll_message_set("native playback requires a Blender screen")
            return False
        return True

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = context.scene.audio2face
        screen = context.screen
        if screen is None:
            self.report({"ERROR"}, "native playback requires a Blender screen")
            return {"CANCELLED"}
        try:
            if screen.is_animation_playing:
                bpy.ops.screen.animation_pause()
                return {"FINISHED"}

            scene = context.scene
            audio_path = bpy.path.abspath(settings.audio_path)
            _strip, audio_end = configure_selected_audio(scene, audio_path)
            controller = get_controller()
            controller.start_selected_audio(
                scene,
                timeline_frame_end=audio_end,
            )
            try:
                playback_result = bpy.ops.screen.animation_play()
                if playback_result != {"FINISHED"}:
                    raise RuntimeError(
                        "Blender could not start native timeline playback"
                    )
            except Exception:
                controller.cancel_selected_audio(scene)
                raise
        except (OSError, RuntimeError, SidecarError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class A2F_OT_bake_animation(bpy.types.Operator):
    bl_idname = "a2f.bake_animation"
    bl_label = "Bake Shape Key Animation"
    bl_description = (
        "Evaluate the selected WAV and animated tuning controls at each "
        "Blender frame, then write native Shape Key animation"
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = context.scene.audio2face
        controller = get_controller()
        if settings.input_mode != "SELECTED":
            cls.poll_message_set("baking requires Selected Audio mode")
            return False
        if not settings.audio_path:
            cls.poll_message_set("select a WAV file first")
            return False
        if controller.client.state != Lifecycle.RUNNING or not controller.negotiated:
            cls.poll_message_set("start the Audio2Face worker first")
            return False
        if controller.operation_in_progress:
            cls.poll_message_set("wait for the current Audio2Face operation to finish")
            return False
        return True

    def execute(self, context: bpy.types.Context) -> set[str]:
        return _run_runtime(
            self,
            lambda controller: controller.bake_selected_audio(context.scene),
        )


class A2F_OT_cancel_bake(bpy.types.Operator):
    bl_idname = "a2f.cancel_bake"
    bl_label = "Cancel Animation Bake"
    bl_description = "Cancel the active frame-based Shape Key animation bake"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        bake = get_controller().active_bake
        available = (
            bake is not None
            and bake.scene_name == context.scene.name
            and not bake.cancel_requested
        )
        if not available:
            cls.poll_message_set("there is no active animation bake")
        return available

    def execute(self, context: bpy.types.Context) -> set[str]:
        return _run_runtime(
            self,
            lambda controller: controller.cancel_bake(context.scene),
        )


CLASSES = (
    A2F_OT_optimize_models,
    A2F_OT_cancel_model_optimization,
    A2F_OT_start_worker,
    A2F_OT_stop_worker,
    A2F_OT_reset_model_tuning,
    A2F_OT_reset_emotion_settings,
    A2F_OT_add_selected_targets,
    A2F_OT_remove_target,
    A2F_OT_play_pause,
    A2F_OT_bake_animation,
    A2F_OT_cancel_bake,
)
