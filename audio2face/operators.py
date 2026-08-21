"""User-facing operators; all Blender mutations run on the main thread."""

from __future__ import annotations

from typing import Callable

import bpy

from .live_stream import get_live_stream_controller
from .preview import PreviewError, get_preview_controller
from .properties import A2FSceneSettings
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
        or not settings.current_job_id
        or not settings.result_audio_path
    ):
        raise ResultValidationError("generate an animation result first")
    result = load_animation_result(
        bpy.path.abspath(settings.result_path),
        allowed_directory=str(get_controller().result_directory()),
    )
    if result.job_id != settings.current_job_id:
        raise ResultValidationError(
            f"stale result job ID {result.job_id!r}; active job is {settings.current_job_id!r}"
        )
    return result


class A2F_OT_start_worker(bpy.types.Operator):
    bl_idname = "a2f.start_worker"
    bl_label = "Start Worker"
    bl_description = "Start the add-on-managed local Audio2Face GPU worker"

    def execute(self, context: bpy.types.Context) -> set[str]:
        return _run_runtime(self, lambda controller: controller.start(context.scene))


class A2F_OT_install_runtime(bpy.types.Operator):
    bl_idname = "a2f.install_runtime"
    bl_label = "Install Runtime & Models"
    bl_description = (
        "Download, verify, and install the managed NVIDIA GPU runtime and both models"
    )

    @classmethod
    def poll(cls, _context: bpy.types.Context) -> bool:
        can_install, reason = get_controller().install_eligibility()
        if not can_install:
            cls.poll_message_set(reason)
        return can_install

    def execute(self, _context: bpy.types.Context) -> set[str]:
        return _run_runtime(self, lambda controller: controller.install_runtime())


class A2F_OT_cancel_runtime_install(bpy.types.Operator):
    bl_idname = "a2f.cancel_runtime_install"
    bl_label = "Cancel Runtime Install"
    bl_description = "Cancel the current managed-runtime download or model optimization"

    @classmethod
    def poll(cls, _context: bpy.types.Context) -> bool:
        in_progress = get_controller().install_in_progress
        if not in_progress:
            cls.poll_message_set("managed-runtime installation is not running")
        return in_progress

    def execute(self, _context: bpy.types.Context) -> set[str]:
        return _run_runtime(self, lambda controller: controller.cancel_runtime_install())


class A2F_OT_stop_worker(bpy.types.Operator):
    bl_idname = "a2f.stop_worker"
    bl_label = "Stop Worker"
    bl_description = "Request graceful worker shutdown without blocking Blender"

    def execute(self, context: bpy.types.Context) -> set[str]:
        return _run_runtime(self, lambda controller: controller.stop(context.scene))


class A2F_OT_generate(bpy.types.Operator):
    bl_idname = "a2f.generate"
    bl_label = "Generate ARKit Values"
    bl_description = "Generate a timestamped ARKit-52 value stream through the GPU worker"

    def execute(self, context: bpy.types.Context) -> set[str]:
        return _run_runtime(self, lambda controller: controller.generate(context.scene))


class A2F_OT_cancel(bpy.types.Operator):
    bl_idname = "a2f.cancel"
    bl_label = "Cancel Generation"
    bl_description = "Request cancellation of the current worker job"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.scene.audio2face.status in {"GENERATING", "CANCELLING"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        return _run_runtime(self, lambda controller: controller.cancel(context.scene))


class A2F_OT_stream_wav(bpy.types.Operator):
    bl_idname = "a2f.stream_wav"
    bl_label = "Start WAV Stream"
    bl_description = (
        "Decode the selected WAV incrementally and stream PCM through the managed GPU model"
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
        return bool(context.scene.audio2face.stream_id)

    def execute(self, context: bpy.types.Context) -> set[str]:
        return _run_runtime(self, lambda controller: controller.stop_stream(context.scene))


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
        return bool(context.scene.audio2face.target_meshes)

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = context.scene.audio2face
        index = min(settings.target_mesh_index, len(settings.target_meshes) - 1)
        settings.target_meshes.remove(index)
        settings.target_mesh_index = max(0, min(index, len(settings.target_meshes) - 1))
        return {"FINISHED"}


class A2F_OT_preview_play(bpy.types.Operator):
    bl_idname = "a2f.preview_play"
    bl_label = "Play Audio and Animation"
    bl_description = "Play generated audio and deliver model channels to target Shape Keys in sync"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = context.scene.audio2face
        return (
            bool(settings.result_path)
            and bool(settings.current_job_id)
            and bool(settings.result_audio_path)
            and not settings.stream_id
            and not get_live_stream_controller().active
            and settings.preview_state != "PLAYING"
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = context.scene.audio2face
        controller = get_preview_controller()
        try:
            if settings.preview_state == "PAUSED" and controller.active:
                controller.resume()
            else:
                result = _load_selected_result(settings)
                controller.start(
                    context.scene,
                    result,
                    settings.result_audio_path,
                )
        except (PreviewError, ResultValidationError, OSError, ValueError) as exc:
            settings.status = "ERROR"
            settings.status_message = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class A2F_OT_preview_pause(bpy.types.Operator):
    bl_idname = "a2f.preview_pause"
    bl_label = "Pause Audio and Animation"
    bl_description = "Pause generated audio and hold the current Shape Key values"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return context.scene.audio2face.preview_state == "PLAYING"

    def execute(self, context: bpy.types.Context) -> set[str]:
        try:
            get_preview_controller().pause()
        except PreviewError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class A2F_OT_preview_stop(bpy.types.Operator):
    bl_idname = "a2f.preview_stop"
    bl_label = "Stop Audio and Animation"
    bl_description = "Stop audio playback and synchronized Shape Key updates"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        settings = context.scene.audio2face
        return settings.preview_state != "IDLE"

    def execute(self, context: bpy.types.Context) -> set[str]:
        get_preview_controller().stop()
        return {"FINISHED"}


CLASSES = (
    A2F_OT_install_runtime,
    A2F_OT_cancel_runtime_install,
    A2F_OT_start_worker,
    A2F_OT_stop_worker,
    A2F_OT_generate,
    A2F_OT_cancel,
    A2F_OT_stream_wav,
    A2F_OT_stop_stream,
    A2F_OT_add_selected_targets,
    A2F_OT_remove_target,
    A2F_OT_preview_play,
    A2F_OT_preview_pause,
    A2F_OT_preview_stop,
)
