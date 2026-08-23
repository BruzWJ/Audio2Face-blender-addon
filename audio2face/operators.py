"""User-facing operators; all Blender mutations run on the main thread."""

from __future__ import annotations

from typing import Callable

import bpy

from .live_stream import get_live_stream_controller
from .properties import (
    clear_preferred_emotion,
    load_preferred_emotion,
)
from .runtime import RuntimeController, get_controller
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


class A2F_OT_load_preferred_emotion(bpy.types.Operator):
    bl_idname = "a2f.load_preferred_emotion"
    bl_label = "Load Preferred Emotion"
    bl_description = "Load preferred emotion from the current manual emotion settings"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        available = bool(context.scene.audio2face.manual_emotions)
        if not available:
            cls.poll_message_set("load the Audio2Face model first")
        return available

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            load_preferred_emotion(context.scene.audio2face)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        get_controller().refresh_inference_settings(context.scene)
        return {"FINISHED"}


class A2F_OT_clear_preferred_emotion(bpy.types.Operator):
    bl_idname = "a2f.clear_preferred_emotion"
    bl_label = "Clear Preferred Emotion"
    bl_description = "Clear the loaded preferred emotion"

    def execute(self, context: bpy.types.Context) -> set[str]:
        clear_preferred_emotion(context.scene.audio2face)
        get_controller().refresh_inference_settings(context.scene)
        return {"FINISHED"}


class A2F_OT_add_selected_targets(bpy.types.Operator):
    bl_idname = "a2f.add_selected_targets"
    bl_label = "Add Selected Meshes"
    bl_description = "Add selected mesh objects to receive Audio2Face Shape Key values"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        has_selected_mesh = any(
            obj.type == "MESH" for obj in context.selected_objects
        )
        if not has_selected_mesh:
            cls.poll_message_set("select at least one mesh object")
        return has_selected_mesh

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = context.scene.audio2face
        selected = [obj for obj in context.selected_objects if obj.type == "MESH"]
        if not selected:
            self.report({"ERROR"}, "select at least one mesh object")
            return {"CANCELLED"}

        last_added_index: int | None = None
        existing = {
            item.object.as_pointer()
            for item in settings.target_meshes
            if item.object is not None
        }
        for target in selected:
            if target.as_pointer() in existing:
                continue
            item = settings.target_meshes.add()
            item.object = target
            existing.add(target.as_pointer())
            last_added_index = len(settings.target_meshes) - 1
        if last_added_index is None:
            self.report({"INFO"}, "Selected meshes are already targets")
            return {"FINISHED"}
        settings.target_mesh_index = last_added_index
        self.report({"INFO"}, "Added selected meshes as targets")
        return {"FINISHED"}


class A2F_OT_remove_target(bpy.types.Operator):
    bl_idname = "a2f.remove_target"
    bl_label = "Remove Target"
    bl_description = "Stop driving the active mesh by removing it from the target list"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = context.scene.audio2face
        valid_index = 0 <= settings.target_mesh_index < len(settings.target_meshes)
        if not valid_index:
            cls.poll_message_set("select a mesh target to remove")
        return valid_index

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = context.scene.audio2face
        index = settings.target_mesh_index
        if not 0 <= index < len(settings.target_meshes):
            self.report({"ERROR"}, "selected mesh target index is invalid")
            return {"CANCELLED"}
        settings.target_meshes.remove(index)
        if settings.target_meshes:
            settings.target_mesh_index = min(index, len(settings.target_meshes) - 1)
        else:
            settings.target_mesh_index = 0
        return {"FINISHED"}


class A2F_OT_play_pause(bpy.types.Operator):
    bl_idname = "a2f.play_pause"
    bl_label = "Play/Pause Audio2Face"
    bl_description = "Play or pause selected audio and its live ARKit-52 stream"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = context.scene.audio2face
        if settings.input_mode != "SELECTED":
            return False
        live = get_live_stream_controller()
        if live.can_seek:
            return settings.playback_state in {"PLAYING", "PAUSED"}
        runtime = get_controller()
        return bool(
            settings.playback_state == "IDLE"
            and settings.audio_path
            and runtime.client.state == Lifecycle.RUNNING
            and runtime.negotiated
            and not runtime.operation_in_progress
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = context.scene.audio2face
        if settings.playback_state == "PLAYING":
            return _run_runtime(
                self,
                lambda controller: controller.pause_selected_audio(context.scene),
            )
        if settings.playback_state == "PAUSED":
            return _run_runtime(
                self,
                lambda controller: controller.resume_selected_audio(context.scene),
            )
        if settings.playback_state == "IDLE":
            return _run_runtime(
                self,
                lambda controller: controller.start_selected_audio(context.scene),
            )
        self.report({"ERROR"}, f"invalid playback state {settings.playback_state!r}")
        return {"CANCELLED"}


CLASSES = (
    A2F_OT_optimize_models,
    A2F_OT_cancel_model_optimization,
    A2F_OT_start_worker,
    A2F_OT_stop_worker,
    A2F_OT_load_preferred_emotion,
    A2F_OT_clear_preferred_emotion,
    A2F_OT_add_selected_targets,
    A2F_OT_remove_target,
    A2F_OT_play_pause,
)
