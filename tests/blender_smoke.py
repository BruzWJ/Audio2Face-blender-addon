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
from _bpy_restrict_state import RestrictBlend  # noqa: E402

import audio2face  # noqa: E402
from audio2face.shape_keys import (  # noqa: E402
    apply_shape_key_frame,
    resolve_target_meshes,
)
from audio2face.live_stream import (  # noqa: E402
    PLAYBACK_POSITION_KEY,
    LiveStreamController,
    clear_playback_position,
    configure_playback_position,
    playback_position,
)
from audio2face import runtime  # noqa: E402
from audio2face.preferences import A2FAddonPreferences  # noqa: E402
from audio2face.properties import (  # noqa: E402
    AUDIO2FACE_SETTING_FIELDS,
    A2FSceneSettings,
    A2FTargetMeshItem,
    apply_model_schema,
    inference_settings,
)
from audio2face.ui_text import draw_wrapped_label  # noqa: E402


MODEL_CHANNELS = ["jawOpen", *(f"modelChannel{index}" for index in range(51))]
MODEL_DEFAULTS: dict[str, float | int] = {
    "input_strength": 1.0,
    "lower_face_smoothing": 0.006,
    "upper_face_smoothing": 0.001,
    "lower_face_strength": 1.0,
    "upper_face_strength": 1.0,
    "face_mask_level": 0.6,
    "face_mask_softness": 0.0085,
    "skin_strength": 1.0,
    "blink_strength": 1.0,
    "eyelid_open_offset": 0.0,
    "lip_open_offset": 0.0,
    "eyeballs_strength": 1.0,
    "saccade_strength": 0.6,
    "right_eye_rot_x_offset": 0.0,
    "right_eye_rot_y_offset": 0.0,
    "left_eye_rot_x_offset": 0.0,
    "left_eye_rot_y_offset": 0.0,
    "eye_saccade_seed": 0,
}


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
        with RestrictBlend():
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
        operator_names = set(dir(bpy.ops.a2f))
        assert {
            "play_pause",
            "load_preferred_emotion",
            "clear_preferred_emotion",
        } <= operator_names
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
        assert set(A2FTargetMeshItem.__annotations__) == {"object"}
        missing_scene_property_names = (
            set(A2FSceneSettings.__annotations__) - scene_property_names
        )
        assert not missing_scene_property_names, (
            f"scene settings missing registered RNA properties: "
            f"{sorted(missing_scene_property_names)}"
        )
        assert {
            "prediction_delay",
            "auto_audio2emotion",
            "manual_emotions",
            "preferred_emotions",
            "a2e_emotion_strength",
            "a2e_emotion_contrast",
            "a2e_max_emotions",
            "a2e_live_blend_coef",
            "a2e_transition_smoothing",
            "a2e_preferred_emotion_strength",
            *AUDIO2FACE_SETTING_FIELDS,
        } <= scene_property_names
        expected_model_ranges = {
            "input_strength": (0.0, 3.0),
            "lower_face_smoothing": (0.0, 0.1),
            "upper_face_smoothing": (0.0, 0.1),
            "lower_face_strength": (0.0, 2.0),
            "upper_face_strength": (0.0, 2.0),
            "face_mask_level": (0.0, 1.0),
            "face_mask_softness": (0.001, 0.5),
            "skin_strength": (0.0, 2.0),
            "blink_strength": (0.0, 2.0),
            "eyelid_open_offset": (-1.0, 1.0),
            "lip_open_offset": (-0.2, 0.2),
            "eyeballs_strength": (0.0, 2.0),
            "saccade_strength": (0.0, 2.0),
            "right_eye_rot_x_offset": (-10.0, 10.0),
            "right_eye_rot_y_offset": (-10.0, 10.0),
            "left_eye_rot_x_offset": (-10.0, 10.0),
            "left_eye_rot_y_offset": (-10.0, 10.0),
            "eye_saccade_seed": (0.0, 4999.0),
        }
        for name, expected_range in expected_model_ranges.items():
            prop = A2FSceneSettings.bl_rna.properties[name]
            _assert_close(prop.hard_min, expected_range[0], label=f"{name} min")
            _assert_close(prop.hard_max, expected_range[1], label=f"{name} max")
        assert {
            name: A2FSceneSettings.bl_rna.properties[name].name
            for name in (
                "eyelid_open_offset",
                "eyeballs_strength",
                "right_eye_rot_x_offset",
                "right_eye_rot_y_offset",
                "left_eye_rot_x_offset",
                "left_eye_rot_y_offset",
                "eye_saccade_seed",
            )
        } == {
            "eyelid_open_offset": "Eyelid Offset",
            "eyeballs_strength": "Offset Strength",
            "right_eye_rot_x_offset": "Right Eye Rotate X",
            "right_eye_rot_y_offset": "Right Eye Rotate Y",
            "left_eye_rot_x_offset": "Left Eye Rotate X",
            "left_eye_rot_y_offset": "Left Eye Rotate Y",
            "eye_saccade_seed": "Eye Saccade Data",
        }
        assert not A2FSceneSettings.bl_rna.properties[
            "preferred_emotions"
        ].is_skip_save
        scene = bpy.context.scene
        target = _make_shape_key_target(scene)

        settings = scene.audio2face
        configure_playback_position(settings, 1.25, 4.0)
        _assert_close(
            playback_position(settings),
            1.25,
            label="absolute playback position",
        )
        playback_ui = settings.id_properties_ui(PLAYBACK_POSITION_KEY).as_dict()
        assert playback_ui["subtype"] == "TIME"
        _assert_close(playback_ui["min"], 0.0, label="playback slider min")
        _assert_close(playback_ui["max"], 4.0, label="playback slider max")
        clear_playback_position(settings)
        assert PLAYBACK_POSITION_KEY not in settings
        model_schema = {
            "channels": MODEL_CHANNELS.copy(),
            "audio2face_defaults": MODEL_DEFAULTS.copy(),
            "emotion_channels": [
                {"name": "Neutral", "default": 1.0},
                {"name": "Joy", "default": 0.0},
            ],
        }
        model_signature = ("/models/audio2face", "/models/audio2emotion")
        apply_model_schema(settings, model_schema, model_signature)
        for name, expected in MODEL_DEFAULTS.items():
            if name == "eye_saccade_seed":
                assert getattr(settings, name) == expected
            else:
                _assert_close(getattr(settings, name), expected, label=name)
        assert [(item.name, item.value) for item in settings.manual_emotions] == [
            ("Neutral", 1.0),
            ("Joy", 0.0),
        ]
        for name, bounds in expected_model_ranges.items():
            if name == "eye_saccade_seed":
                continue
            original = getattr(settings, name)
            for endpoint in bounds:
                setattr(settings, name, endpoint)
                payload = inference_settings(settings)
                _assert_close(payload["audio2face"][name], endpoint, label=name)
            setattr(settings, name, original)
        settings.manual_emotions[1].value = 0.75
        settings.input_strength = 2.0
        settings.blink_strength = 1.5
        settings.eye_saccade_seed = 41
        apply_model_schema(settings, model_schema, model_signature)
        _assert_close(settings.manual_emotions[1].value, 0.75, label="preserved Joy")
        _assert_close(
            settings.input_strength,
            2.0,
            label="preserved input strength",
        )
        _assert_close(
            settings.blink_strength,
            1.5,
            label="preserved blink strength",
        )
        assert settings.eye_saccade_seed == 41
        settings.auto_audio2emotion = True
        settings.a2e_emotion_strength = 0.8
        settings.a2e_emotion_contrast = 1.4
        settings.a2e_max_emotions = 3
        settings.a2e_live_blend_coef = 0.4
        settings.a2e_transition_smoothing = 0.9
        settings.a2e_preferred_emotion_strength = 0.35
        assert bpy.ops.a2f.clear_preferred_emotion() == {"FINISHED"}
        assert bpy.ops.a2f.load_preferred_emotion() == {"FINISHED"}
        settings.manual_emotions[1].value = 0.5
        settings.input_strength = 2.0
        settings.blink_strength = 1.5
        settings.eye_saccade_seed = 41
        emotion_payload = inference_settings(settings)
        assert set(emotion_payload) == {
            "audio2face",
            "auto_audio2emotion",
            "manual_emotions",
            "audio2emotion",
        }
        assert emotion_payload["auto_audio2emotion"] is True
        assert set(emotion_payload["audio2face"]) == set(
            AUDIO2FACE_SETTING_FIELDS
        )
        _assert_close(
            emotion_payload["audio2face"]["input_strength"],
            2.0,
            label="input strength",
        )
        _assert_close(
            emotion_payload["audio2face"]["blink_strength"],
            1.5,
            label="blink strength",
        )
        assert emotion_payload["audio2face"]["eye_saccade_seed"] == 41
        manual_payload = emotion_payload["manual_emotions"]
        assert set(manual_payload) == {"Neutral", "Joy"}
        _assert_close(manual_payload["Neutral"], 1.0, label="manual Neutral")
        _assert_close(manual_payload["Joy"], 0.5, label="manual Joy")
        automatic_payload = emotion_payload["audio2emotion"]
        assert set(automatic_payload) == {
            "emotion_strength",
            "emotion_contrast",
            "max_emotions",
            "live_blend_coef",
            "transition_smoothing",
            "preferred_emotion",
            "preferred_emotion_strength",
        }
        assert automatic_payload["max_emotions"] == 3
        preferred_payload = automatic_payload["preferred_emotion"]
        assert set(preferred_payload) == {"Neutral", "Joy"}
        _assert_close(preferred_payload["Neutral"], 1.0, label="preferred Neutral")
        _assert_close(preferred_payload["Joy"], 0.75, label="preferred Joy")
        for name, expected in (
            ("emotion_strength", 0.8),
            ("emotion_contrast", 1.4),
            ("live_blend_coef", 0.4),
            ("transition_smoothing", 0.9),
            ("preferred_emotion_strength", 0.35),
        ):
            _assert_close(automatic_payload[name], expected, label=name)
        assert bpy.ops.a2f.clear_preferred_emotion() == {"FINISHED"}
        assert (
            inference_settings(settings)["audio2emotion"]["preferred_emotion"]
            is None
        )

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

        subscriptions = resolve_target_meshes(settings)
        # Every mesh remains subscribed without Shape Key inspection; shared
        # Key datablocks are deduplicated only when a frame is delivered.
        assert len(subscriptions) == 4
        streamed_values = [0.0] * len(MODEL_CHANNELS)
        streamed_values[MODEL_CHANNELS.index("jawOpen")] = 0.625
        apply_shape_key_frame(
            subscriptions,
            tuple(MODEL_CHANNELS),
            tuple(streamed_values),
        )
        _assert_close(
            target.data.shape_keys.key_blocks["jawOpen"].value,
            0.625,
            label="primary streamed jawOpen",
        )
        _assert_close(
            extra_target.data.shape_keys.key_blocks["jawOpen"].value,
            0.625,
            label="extra streamed jawOpen",
        )
        _assert_close(
            linked_target.data.shape_keys.key_blocks["jawOpen"].value,
            0.625,
            label="linked streamed jawOpen",
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
                ["Neutral", "Joy"],
                audio_path=None,
                playback_started=None,
                playback_stopped=None,
            )
            streamed_frame = [0.0] * len(MODEL_CHANNELS)
            streamed_frame[MODEL_CHANNELS.index("jawOpen")] = 0.375
            # The current SDK can emit receptive-field frames before sample zero.
            # They must be accepted and applied directly without animation data.
            live.receive(
                "blender-smoke-stream",
                -160,
                streamed_frame,
                [0.25, 0.75],
            )
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
            _assert_close(
                settings.manual_emotions[0].value,
                0.25,
                label="effective Neutral",
            )
            _assert_close(
                settings.manual_emotions[1].value,
                0.75,
                label="effective Joy",
            )
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
