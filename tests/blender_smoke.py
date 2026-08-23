"""Blender 5.2 headless smoke test for model-channel Shape Key streaming.

Run from the project root with::

    blender --factory-startup --background --python tests/blender_smoke.py

This is intentionally a Blender script rather than a pytest test: it validates
real ``bpy`` RNA registration and multi-mesh ``ShapeKey.value`` assignment.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import Mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bpy  # noqa: E402  (available only inside Blender)

import audio2face  # noqa: E402
from audio2face.preview import (  # noqa: E402
    apply_shape_key_frame,
    build_subscriptions,
)
from audio2face.live_stream import LiveStreamController  # noqa: E402
from audio2face import runtime  # noqa: E402
from audio2face.preferences import A2FAddonPreferences  # noqa: E402
from audio2face.properties import (  # noqa: E402
    A2FSceneSettings,
    apply_model_schema,
    tuning_parameters,
)
from audio2face.ui_text import draw_wrapped_label  # noqa: E402


MODEL_CHANNELS = ["jawOpen", *(f"modelChannel{index}" for index in range(51))]


def _assert_close(actual: float, expected: float, *, label: str) -> None:
    assert math.isclose(actual, expected, rel_tol=1.0e-6, abs_tol=1.0e-6), (
        f"{label}: expected {expected}, got {actual}"
    )


def _make_shape_key_target(
    scene: bpy.types.Scene,
    *,
    object_name: str = "A2FSmokeTarget",
    shape_name: str = "jawOpen",
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{object_name}Mesh")
    mesh.from_pydata(
        [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        [],
        [(0, 1, 2)],
    )
    mesh.update()
    target = bpy.data.objects.new(object_name, mesh)
    scene.collection.objects.link(target)
    target.shape_key_add(name="Basis")
    target.shape_key_add(name=shape_name)
    return target


def main() -> None:
    assert bpy.app.version[:2] == (5, 2), (
        f"this smoke test targets Blender 5.2, got {bpy.app.version_string}"
    )
    assert not hasattr(bpy.types.Scene, "audio2face"), (
        "factory-startup scene unexpectedly has Audio2Face registered"
    )

    registered = False
    try:
        audio2face.register()
        registered = True
        selected = runtime.RuntimeController._selected_directory_path(
            bpy.app.tempdir,
            "selected model directory",
        )
        assert selected == Path(bpy.app.tempdir)
        assert hasattr(bpy.types.Scene, "audio2face")
        assert bpy.app.timers.is_registered(runtime._timer_callback)
        assert runtime._load_pre_handler in bpy.app.handlers.load_pre
        assert runtime._load_post_handler in bpy.app.handlers.load_post
        notice_layout = Mock()
        draw_wrapped_label(
            notice_layout,
            "Click Optimize Models to generate the GPU-specific TensorRT "
            "engines from the downloaded ONNX models",
            width=42,
            icon="INFO",
        )
        notice_calls = notice_layout.label.call_args_list
        assert len(notice_calls) > 1
        assert notice_calls[0].kwargs["icon"] == "INFO"
        assert all(call.kwargs["icon"] == "BLANK1" for call in notice_calls[1:])
        assert "uninstall" not in dir(bpy.ops.a2f)
        preference_names = set(A2FAddonPreferences.bl_rna.properties.keys())
        assert set(A2FAddonPreferences.__annotations__) == {
            "nvidia_terms_accepted",
            "audio2face_model_directory",
            "audio2emotion_model_directory",
        }
        missing_preference_names = (
            set(A2FAddonPreferences.__annotations__) - preference_names
        )
        assert not missing_preference_names, (
            f"preferences missing registered RNA properties: "
            f"{sorted(missing_preference_names)}"
        )
        assert (
            A2FAddonPreferences.bl_rna.properties[
                "audio2face_model_directory"
            ].subtype
            == "DIR_PATH"
        )
        assert (
            A2FAddonPreferences.bl_rna.properties[
                "audio2emotion_model_directory"
            ].subtype
            == "DIR_PATH"
        )
        scene_property_names = set(A2FSceneSettings.bl_rna.properties.keys())
        missing_scene_property_names = (
            set(A2FSceneSettings.__annotations__) - scene_property_names
        )
        assert not missing_scene_property_names, (
            f"scene settings missing registered RNA properties: "
            f"{sorted(missing_scene_property_names)}"
        )

        scene = bpy.context.scene
        target = _make_shape_key_target(scene)

        settings = scene.audio2face
        model_schema = {
            "identities": ["Aki", "Mark"],
            "channels": MODEL_CHANNELS.copy(),
            "parameters": {
                "/input_strength": 1.0,
                "/audio2emotion/emotion_strength": 0.6,
                "/audio2emotion/max_emotions": 6,
            },
            "emotion_channels": [
                {"name": "Neutral", "default": 1.0},
                {"name": "Joy", "default": 0.0},
            ],
        }
        model_signature = ("/models/audio2face", "/models/audio2emotion", 0)
        apply_model_schema(settings, model_schema, model_signature)
        assert [item.name for item in settings.model_identities] == ["Aki", "Mark"]
        assert [(item.name, item.value) for item in settings.manual_emotions] == [
            ("Neutral", 1.0),
            ("Joy", 0.0),
        ]
        settings.manual_emotions[1].value = 0.75
        settings.model_parameters[1].float_value = 0.8
        apply_model_schema(settings, model_schema, model_signature)
        _assert_close(settings.manual_emotions[1].value, 0.75, label="preserved Joy")
        _assert_close(
            settings.model_parameters[1].float_value,
            0.8,
            label="preserved emotion strength",
        )
        settings.auto_audio2emotion = True
        tuning_payload = tuning_parameters(settings)
        assert set(tuning_payload) == {
            "auto_audio2emotion",
            "manual_emotions",
            "parameters",
        }
        assert tuning_payload["auto_audio2emotion"] is True
        manual_payload = tuning_payload["manual_emotions"]
        assert set(manual_payload) == {"Neutral", "Joy"}
        _assert_close(manual_payload["Neutral"], 1.0, label="manual Neutral")
        _assert_close(manual_payload["Joy"], 0.75, label="manual Joy")
        parameter_payload = tuning_payload["parameters"]
        assert set(parameter_payload) == {
            "/input_strength",
            "/audio2emotion/emotion_strength",
            "/audio2emotion/max_emotions",
        }
        _assert_close(parameter_payload["/input_strength"], 1.0, label="input strength")
        _assert_close(
            parameter_payload["/audio2emotion/emotion_strength"],
            0.8,
            label="emotion strength",
        )
        assert parameter_payload["/audio2emotion/max_emotions"] == 6

        extra_target = _make_shape_key_target(
            scene,
            object_name="A2FSmokeExtraTarget",
        )
        linked_target = bpy.data.objects.new("A2FSmokeLinkedTarget", target.data)
        scene.collection.objects.link(linked_target)
        plain_mesh = bpy.data.meshes.new("A2FSmokePlainMesh")
        plain_mesh.from_pydata(
            [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
            [],
            [(0, 1, 2)],
        )
        plain_target = bpy.data.objects.new("A2FSmokePlainTarget", plain_mesh)
        scene.collection.objects.link(plain_target)
        bpy.ops.object.select_all(action="DESELECT")
        target.select_set(True)
        extra_target.select_set(True)
        linked_target.select_set(True)
        plain_target.select_set(True)
        bpy.context.view_layer.objects.active = target
        assert bpy.ops.a2f.add_selected_targets() == {"FINISHED"}
        selected_targets = {item.object for item in settings.target_meshes}
        assert selected_targets == {target, extra_target, linked_target, plain_target}
        primary_vertices = [tuple(vertex.co) for vertex in target.data.vertices]
        extra_vertices = [tuple(vertex.co) for vertex in extra_target.data.vertices]

        subscriptions = build_subscriptions(settings)
        # Every mesh remains subscribed without Shape Key inspection; shared
        # Key datablocks are deduplicated only when a frame is delivered.
        assert len(subscriptions) == 4
        preview_frame = [0.0] * len(MODEL_CHANNELS)
        preview_frame[MODEL_CHANNELS.index("jawOpen")] = 0.625
        apply_shape_key_frame(subscriptions, tuple(MODEL_CHANNELS), tuple(preview_frame))
        _assert_close(
            target.data.shape_keys.key_blocks["jawOpen"].value,
            0.625,
            label="primary preview jawOpen",
        )
        _assert_close(
            extra_target.data.shape_keys.key_blocks["jawOpen"].value,
            0.625,
            label="extra preview jawOpen",
        )
        _assert_close(
            linked_target.data.shape_keys.key_blocks["jawOpen"].value,
            0.625,
            label="linked preview jawOpen",
        )
        apply_shape_key_frame(
            subscriptions,
            tuple(MODEL_CHANNELS),
            (0.0,) * len(MODEL_CHANNELS),
        )
        _assert_close(
            target.data.shape_keys.key_blocks["jawOpen"].value,
            0.0,
            label="primary reset jawOpen",
        )
        assert [tuple(vertex.co) for vertex in target.data.vertices] == primary_vertices
        assert [tuple(vertex.co) for vertex in extra_target.data.vertices] == extra_vertices
        assert target.data.shape_keys.animation_data is None
        assert extra_target.data.shape_keys.animation_data is None

        live = LiveStreamController()
        try:
            live.prepare(
                scene,
                "blender-smoke-stream",
                16_000,
                MODEL_CHANNELS.copy(),
                audio_path=None,
                playback_started=None,
                playback_stopped=None,
            )
            streamed_frame = [0.0] * len(MODEL_CHANNELS)
            streamed_frame[MODEL_CHANNELS.index("jawOpen")] = 0.375
            # The current SDK can emit receptive-field frames before sample zero.
            # They must be accepted and applied directly without animation data.
            live.receive("blender-smoke-stream", -160, streamed_frame)
            _assert_close(
                target.data.shape_keys.key_blocks["jawOpen"].value,
                0.375,
                label="primary live-stream jawOpen",
            )
            _assert_close(
                extra_target.data.shape_keys.key_blocks["jawOpen"].value,
                0.375,
                label="extra live-stream jawOpen",
            )
            _assert_close(settings.stream_time, 0.0, label="negative stream time clamp")
            assert [tuple(vertex.co) for vertex in target.data.vertices] == primary_vertices
            assert [tuple(vertex.co) for vertex in extra_target.data.vertices] == extra_vertices
            assert target.data.shape_keys.animation_data is None
            assert extra_target.data.shape_keys.animation_data is None
        finally:
            live.stop(reset=True)
        _assert_close(
            target.data.shape_keys.key_blocks["jawOpen"].value,
            0.0,
            label="primary live-stream reset",
        )
    finally:
        if registered:
            audio2face.unregister()
        assert not hasattr(bpy.types.Scene, "audio2face"), (
            "unregister left Scene.audio2face behind"
        )
        assert not bpy.app.timers.is_registered(runtime._timer_callback), (
            "unregister left the runtime timer registered"
        )
        assert runtime._load_pre_handler not in bpy.app.handlers.load_pre
        assert runtime._load_post_handler not in bpy.app.handlers.load_post

    print("Audio2Face 5.2 smoke test passed")


if __name__ == "__main__":
    main()
