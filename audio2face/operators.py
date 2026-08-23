"""User-facing operators; all Blender mutations run on the main thread."""

from __future__ import annotations

from typing import Callable

import bpy

from .live_stream import get_live_stream_controller
from .preview import PreviewError, get_preview_controller
from .properties import (
    A2FSceneSettings,
    clear_preferred_emotion,
    load_preferred_emotion,
)
from .result_io import AnimationResult, ResultValidationError, load_animation_result
from .runtime import RuntimeController, get_controller
from .sidecar import SidecarError


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


def _load_selected_result(settings: A2FSceneSettings) -> AnimationResult:
    if (
        not settings.result_path
        or not settings.result_operation_id
        or not settings.result_audio_path
    ):
        raise ResultValidationError("generate an animation result first")
    result = load_animation_result(
        settings.result_path,
        allowed_directory=str(get_controller().result_directory()),
    )
    if result.operation_id != settings.result_operation_id:
        raise ResultValidationError(
            f"stale result operation {result.operation_id!r}; "
            f"active result operation is {settings.result_operation_id!r}"
        )
    return result


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


class A2F_OT_generate(bpy.types.Operator):
    bl_idname = "a2f.generate"
    bl_label = "Generate ARKit Values"
    bl_description = (
        "Generate a timestamped ARKit-52 value stream through the GPU worker"
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        return _run_runtime(self, lambda controller: controller.generate(context.scene))


class A2F_OT_cancel(bpy.types.Operator):
    bl_idname = "a2f.cancel"
    bl_label = "Cancel Generation"
    bl_description = "Request cancellation of the current generation operation"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.scene.audio2face.status == "GENERATING"

    def execute(self, context: bpy.types.Context) -> set[str]:
        return _run_runtime(self, lambda controller: controller.cancel(context.scene))


class A2F_OT_stream_wav(bpy.types.Operator):
    bl_idname = "a2f.stream_wav"
    bl_label = "Start WAV Stream"
    bl_description = (
        "Decode the selected WAV incrementally and stream PCM through the "
        "bundled GPU model"
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        return _run_runtime(
            self, lambda controller: controller.start_wav_stream(context.scene)
        )


class A2F_OT_stop_stream(bpy.types.Operator):
    bl_idname = "a2f.stop_stream"
    bl_label = "Stop Stream"
    bl_description = "Stop the current PCM stream while keeping the GPU model ready"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.scene.audio2face.stream_operation_id != ""

    def execute(self, context: bpy.types.Context) -> set[str]:
        return _run_runtime(
            self,
            lambda controller: controller.stop_stream(context.scene),
        )


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
        return {"FINISHED"}


class A2F_OT_clear_preferred_emotion(bpy.types.Operator):
    bl_idname = "a2f.clear_preferred_emotion"
    bl_label = "Clear Preferred Emotion"
    bl_description = "Clear the loaded preferred emotion"

    def execute(self, context: bpy.types.Context) -> set[str]:
        clear_preferred_emotion(context.scene.audio2face)
        return {"FINISHED"}


class A2F_OT_add_selected_targets(bpy.types.Operator):
    bl_idname = "a2f.add_selected_targets"
    bl_label = "Add Selected Meshes"
    bl_description = "Add selected mesh objects as model-channel targets"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = context.scene.audio2face
        selected = [obj for obj in context.selected_objects if obj.type == "MESH"]
        if not selected:
            self.report({"ERROR"}, "select at least one mesh object")
            return {"CANCELLED"}

        added = 0
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
            item.enabled = True
            existing.add(target.as_pointer())
            added += 1
        settings.target_mesh_index = max(0, len(settings.target_meshes) - 1)
        self.report({"INFO"}, f"Added {added} mesh target(s)")
        return {"FINISHED"}


class A2F_OT_remove_target(bpy.types.Operator):
    bl_idname = "a2f.remove_target"
    bl_label = "Remove Target"
    bl_description = "Remove the active mesh target"
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


class A2F_OT_preview_play_pause(bpy.types.Operator):
    bl_idname = "a2f.preview_play_pause"
    bl_label = "Play/Pause Audio and Animation"
    bl_description = (
        "Toggle generated audio and synchronized target Shape Key playback"
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = context.scene.audio2face
        controller = get_preview_controller()
        if settings.stream_operation_id or get_live_stream_controller().active:
            return False
        if controller.active:
            return settings.preview_state in {"PLAYING", "PAUSED"}
        return (
            settings.preview_state == "IDLE"
            and settings.result_path != ""
            and settings.result_operation_id != ""
            and settings.result_audio_path != ""
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = context.scene.audio2face
        controller = get_preview_controller()
        try:
            if settings.preview_state == "PLAYING":
                controller.pause()
            elif settings.preview_state == "PAUSED":
                controller.resume()
            elif settings.preview_state == "IDLE":
                result = _load_selected_result(settings)
                controller.start(
                    context.scene,
                    result,
                    settings.result_audio_path,
                )
            else:
                raise PreviewError(
                    f"cannot toggle preview from state {settings.preview_state!r}"
                )
        except (PreviewError, ResultValidationError, OSError, ValueError) as exc:
            settings.status = "ERROR"
            settings.status_message = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class A2F_OT_preview_rewind(bpy.types.Operator):
    bl_idname = "a2f.preview_rewind"
    bl_label = "Rewind Audio and Animation"
    bl_description = (
        "Rewind audio and synchronized Shape Keys without changing playback state"
    )

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return (
            context.scene.audio2face.preview_state in {"PLAYING", "PAUSED"}
            and get_preview_controller().active
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            get_preview_controller().rewind()
        except PreviewError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


CLASSES = (
    A2F_OT_optimize_models,
    A2F_OT_cancel_model_optimization,
    A2F_OT_start_worker,
    A2F_OT_stop_worker,
    A2F_OT_generate,
    A2F_OT_cancel,
    A2F_OT_stream_wav,
    A2F_OT_stop_stream,
    A2F_OT_load_preferred_emotion,
    A2F_OT_clear_preferred_emotion,
    A2F_OT_add_selected_targets,
    A2F_OT_remove_target,
    A2F_OT_preview_play_pause,
    A2F_OT_preview_rewind,
)
