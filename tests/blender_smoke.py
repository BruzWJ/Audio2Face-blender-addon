"""Blender 5.2 headless smoke test for ARKit shape-key value streaming.

Run from the project root with::

    blender --factory-startup --background --python tests/blender_smoke.py

This is intentionally a Blender script rather than a pytest test: it validates
real ``bpy`` RNA registration and multi-mesh ``ShapeKey.value`` assignment.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bpy  # noqa: E402  (available only inside Blender)

import audio2face  # noqa: E402
from audio2face.arkit import ARKIT_52_CHANNELS  # noqa: E402
from audio2face.preview import (  # noqa: E402
    PreviewError,
    apply_arkit_frame,
    build_subscriptions,
)
from audio2face.live_stream import LiveStreamController  # noqa: E402
from audio2face import runtime  # noqa: E402
from audio2face.preferences import A2FAddonPreferences  # noqa: E402
from audio2face.properties import (  # noqa: E402
    A2FSceneSettings,
    apply_model_defaults,
    tuning_parameters,
)


def _assert_close(actual: float, expected: float, *, label: str) -> None:
    assert math.isclose(actual, expected, rel_tol=1.0e-6, abs_tol=1.0e-6), (
        f"{label}: expected {expected}, got {actual}"
    )


def _make_shape_key_target(
    scene: bpy.types.Scene,
    *,
    object_name: str = "A2FSmokeTarget",
    shape_name: str = "JawOpen",
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
        assert hasattr(bpy.types.Scene, "audio2face")
        assert bpy.app.timers.is_registered(runtime._timer_callback)
        assert runtime._load_pre_handler in bpy.app.handlers.load_pre
        assert runtime._load_post_handler in bpy.app.handlers.load_post
        preference_names = set(A2FAddonPreferences.bl_rna.properties.keys())
        assert preference_names == {"rna_type"} | set(A2FAddonPreferences.__annotations__)
        scene_property_names = set(A2FSceneSettings.bl_rna.properties.keys())
        assert scene_property_names == {"rna_type"} | set(A2FSceneSettings.__annotations__)

        scene = bpy.context.scene
        target = _make_shape_key_target(scene)

        settings = scene.audio2face
        model_defaults = {
            "input_strength": 1.0,
            "skin": {
                "lower_face_smoothing": 0.0,
                "upper_face_smoothing": 0.0,
                "lower_face_strength": 1.0,
                "upper_face_strength": 1.0,
                "face_mask_level": 0.5,
                "face_mask_softness": 0.1,
                "skin_strength": 1.0,
                "blink_strength": 1.0,
                "blink_offset": 0.0,
                "eyelid_open_offset": 0.0,
                "lip_open_offset": 0.0,
            },
            "emotion": {
                "manual_values": {"Neutral": 1.0, "Joy": 0.0},
                "auto": {
                    "strength": 0.6,
                    "contrast": 1.0,
                    "smoothing": 0.7,
                    "transition_time": 0.5,
                    "max_emotions": 6,
                },
            },
        }
        apply_model_defaults(settings, model_defaults, ["Neutral", "Joy"])
        assert [(item.name, item.value) for item in settings.manual_emotions] == [
            ("Neutral", 1.0),
            ("Joy", 0.0),
        ]
        settings.manual_emotions[1].value = 0.75
        apply_model_defaults(settings, model_defaults, ["Neutral", "Joy"])
        _assert_close(settings.manual_emotions[1].value, 0.75, label="preserved Joy")
        settings.auto_audio2emotion = True
        emotion_payload = tuning_parameters(settings)["emotion"]
        assert emotion_payload == {
            "auto_audio2emotion": True,
            "manual_values": {"Neutral": 1.0, "Joy": 0.75},
            "auto": {
                "strength": 0.6,
                "contrast": 1.0,
                "smoothing": 0.7,
                "transition_time": 0.5,
                "max_emotions": 6,
            },
        }

        extra_target = _make_shape_key_target(
            scene,
            object_name="A2FSmokeExtraTarget",
        )
        linked_target = bpy.data.objects.new("A2FSmokeLinkedTarget", target.data)
        scene.collection.objects.link(linked_target)
        bpy.ops.object.select_all(action="DESELECT")
        target.select_set(True)
        extra_target.select_set(True)
        linked_target.select_set(True)
        bpy.context.view_layer.objects.active = target
        assert bpy.ops.a2f.add_selected_targets() == {"FINISHED"}
        selected_targets = {item.object for item in settings.target_meshes}
        assert selected_targets == {target, extra_target, linked_target}
        primary_vertices = [tuple(vertex.co) for vertex in target.data.vertices]
        extra_vertices = [tuple(vertex.co) for vertex in extra_target.data.vertices]

        subscriptions = build_subscriptions(settings)
        # target and linked_target share one Shape Key datablock, so the stream
        # subscribes it only once.
        assert len(subscriptions) == 2
        preview_frame = [0.0] * len(ARKIT_52_CHANNELS)
        preview_frame[ARKIT_52_CHANNELS.index("JawOpen")] = 0.625
        apply_arkit_frame(subscriptions, preview_frame)
        _assert_close(
            target.data.shape_keys.key_blocks["JawOpen"].value,
            0.625,
            label="primary preview JawOpen",
        )
        _assert_close(
            extra_target.data.shape_keys.key_blocks["JawOpen"].value,
            0.625,
            label="extra preview JawOpen",
        )
        _assert_close(
            linked_target.data.shape_keys.key_blocks["JawOpen"].value,
            0.625,
            label="linked preview JawOpen",
        )
        invalid_frame = preview_frame.copy()
        invalid_frame[ARKIT_52_CHANNELS.index("JawOpen")] = 1.01
        try:
            apply_arkit_frame(subscriptions, invalid_frame)
        except PreviewError:
            pass
        else:
            raise AssertionError("preview accepted an ARKit weight outside [0, 1]")
        apply_arkit_frame(
            subscriptions,
            [0.0] * len(ARKIT_52_CHANNELS),
        )
        _assert_close(
            target.data.shape_keys.key_blocks["JawOpen"].value,
            0.0,
            label="primary reset JawOpen",
        )
        assert [tuple(vertex.co) for vertex in target.data.vertices] == primary_vertices
        assert [tuple(vertex.co) for vertex in extra_target.data.vertices] == extra_vertices
        assert target.data.shape_keys.animation_data is None
        assert extra_target.data.shape_keys.animation_data is None

        live = LiveStreamController()
        try:
            live.prepare(scene, "blender-smoke-stream", 16_000)
            streamed_frame = [0.0] * len(ARKIT_52_CHANNELS)
            streamed_frame[ARKIT_52_CHANNELS.index("JawOpen")] = 0.375
            # The current SDK can emit receptive-field frames before sample zero.
            # They must be accepted and applied directly without animation data.
            live.receive("blender-smoke-stream", -160, streamed_frame)
            _assert_close(
                target.data.shape_keys.key_blocks["JawOpen"].value,
                0.375,
                label="primary live-stream JawOpen",
            )
            _assert_close(
                extra_target.data.shape_keys.key_blocks["JawOpen"].value,
                0.375,
                label="extra live-stream JawOpen",
            )
            _assert_close(settings.stream_time, 0.0, label="negative stream time clamp")
            assert [tuple(vertex.co) for vertex in target.data.vertices] == primary_vertices
            assert [tuple(vertex.co) for vertex in extra_target.data.vertices] == extra_vertices
            assert target.data.shape_keys.animation_data is None
            assert extra_target.data.shape_keys.animation_data is None
        finally:
            live.close()
        _assert_close(
            target.data.shape_keys.key_blocks["JawOpen"].value,
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
