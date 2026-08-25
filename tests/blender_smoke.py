"""Blender 5.2 headless smoke test for model-channel Shape Key streaming.

Run from the project root with::

    blender --factory-startup --background --python tests/blender_smoke.py

This is intentionally a Blender script rather than a pytest test: it validates
real ``bpy`` RNA registration and multi-object ``ShapeKey.value`` assignment.
"""

from __future__ import annotations

import math
import sys
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bpy  # noqa: E402  (available only inside Blender)
from _bpy_restrict_state import RestrictBlend  # noqa: E402

import audio2face  # noqa: E402
from audio2face.shape_keys import (  # noqa: E402
    apply_shape_key_frame,
    resolve_target_objects,
)
from audio2face.live_stream import (  # noqa: E402
    PLAYBACK_POSITION_KEY,
    clear_playback_position,
    configure_playback_position,
    playback_position,
)
from audio2face import runtime  # noqa: E402
from audio2face.preferences import A2FAddonPreferences  # noqa: E402
from audio2face.properties import (  # noqa: E402
    AUDIO2FACE_SETTING_FIELDS,
    A2FSceneSettings,
    A2FTargetObjectItem,
    apply_mixed_emotions,
    apply_model_schema,
    inference_settings,
)
from audio2face.ui_text import draw_wrapped_label  # noqa: E402
from audio2face.ui import A2F_PT_main  # noqa: E402


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


def _route_stream_event(
    controller: runtime.RuntimeController,
    operation_id: str,
    event: str,
    data: dict[str, object],
) -> None:
    controller._handle_event(
        {
            "event": event,
            "operation_id": operation_id,
            "data": data,
        }
    )


def _make_shape_key_target(
    scene: bpy.types.Scene,
    *,
    object_name: str = "A2FSmokeTarget",
    object_type: str = "MESH",
    shape_name: str = "jawOpen",
) -> bpy.types.Object:
    if object_type == "MESH":
        data = bpy.data.meshes.new(f"{object_name}Mesh")
        data.from_pydata(
            [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
            [],
            [(0, 1, 2)],
        )
        data.update()
    elif object_type in {"CURVE", "SURFACE"}:
        data = bpy.data.curves.new(f"{object_name}Curve", object_type)
        spline = data.splines.new("POLY" if object_type == "CURVE" else "NURBS")
        spline.points.add(2)
    elif object_type == "LATTICE":
        data = bpy.data.lattices.new(f"{object_name}Lattice")
    else:
        raise ValueError(f"unsupported Shape Key object type {object_type!r}")
    target = bpy.data.objects.new(object_name, data)
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
        scene = bpy.context.scene
        settings = scene.audio2face
        settings.playback_state = "PAUSED"
        clear_playback_position(settings)

        stale_layout = Mock()
        stale_layout.panel.return_value = (Mock(), None)
        panel_context = SimpleNamespace(
            scene=scene,
            region=SimpleNamespace(width=320),
            preferences=SimpleNamespace(
                system=SimpleNamespace(ui_scale=1.0),
            ),
        )
        A2F_PT_main.draw(SimpleNamespace(layout=stale_layout), panel_context)
        stale_layout.panel.assert_called_once_with(
            "audio2face_preferred_emotion",
            default_closed=True,
        )
        stale_labels = {
            call.kwargs.get("text")
            for call in stale_layout.mock_calls
            if call[0].endswith(".label")
        }
        assert {"Inputs", "Model Tuning", "Emotion"} <= stale_labels
        visible_properties = {
            call.args[1]
            for call in stale_layout.mock_calls
            if call[0].endswith(".prop") and len(call.args) > 1
        }
        assert {
            "audio_path",
            "input_strength",
            "auto_audio2emotion",
            "a2e_emotion_strength",
        } <= (
            visible_properties
        )
        settings.playback_state = "IDLE"
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
        assert set(A2FTargetObjectItem.__annotations__) == {"object"}
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
            "preferred_emotions",
            "preferred_emotion_active",
            "mixed_emotions",
            "a2e_emotion_strength",
            "a2e_emotion_contrast",
            "a2e_max_emotions",
            "a2e_live_blend_coef",
            "a2e_transition_smoothing",
            "a2e_preferred_emotion_strength",
            *AUDIO2FACE_SETTING_FIELDS,
        } <= scene_property_names
        emotion_strength_property = A2FSceneSettings.bl_rna.properties[
            "a2e_emotion_strength"
        ]
        _assert_close(
            emotion_strength_property.hard_max,
            2.0,
            label="Emotion Strength max",
        )
        assert emotion_strength_property.subtype == "NONE"
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
                "a2e_preferred_emotion_strength",
            )
        } == {
            "eyelid_open_offset": "Eyelid Offset",
            "eyeballs_strength": "Offset Strength",
            "right_eye_rot_x_offset": "Right Eye Rotate X",
            "right_eye_rot_y_offset": "Right Eye Rotate Y",
            "left_eye_rot_x_offset": "Left Eye Rotate X",
            "left_eye_rot_y_offset": "Left Eye Rotate Y",
            "eye_saccade_seed": "Eye Saccade Data",
            "a2e_preferred_emotion_strength": "Preferred Emotion Strength",
        }
        assert not A2FSceneSettings.bl_rna.properties[
            "preferred_emotions"
        ].is_skip_save
        assert A2FSceneSettings.bl_rna.properties[
            "preferred_emotion_active"
        ].is_hidden
        assert not A2FSceneSettings.bl_rna.properties[
            "preferred_emotion_active"
        ].is_skip_save
        assert A2FSceneSettings.bl_rna.properties["mixed_emotions"].is_hidden
        assert A2FSceneSettings.bl_rna.properties[
            "mixed_emotions"
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
        with tempfile.TemporaryDirectory() as temporary_directory:
            wav_path = Path(temporary_directory) / "selected.wav"
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16_000)
                wav_file.writeframes(bytes(16_000 * 2))
            settings.audio_path = str(wav_path)
            assert PLAYBACK_POSITION_KEY in settings
            playback_ui = settings.id_properties_ui(
                PLAYBACK_POSITION_KEY
            ).as_dict()
            _assert_close(playback_ui["max"], 1.0, label="selected WAV duration")
            settings.audio_path = ""
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
        expected_preferred_emotions = [
            ("Neutral", 1.0),
            ("Joy", 0.0),
        ]
        assert [
            (item.name, item.value) for item in settings.preferred_emotions
        ] == expected_preferred_emotions
        assert [
            (item.name, item.value) for item in settings.mixed_emotions
        ] == [("Neutral", 0.0), ("Joy", 0.0)]
        assert settings.preferred_emotion_active is False
        for name, bounds in expected_model_ranges.items():
            if name == "eye_saccade_seed":
                continue
            original = getattr(settings, name)
            for endpoint in bounds:
                setattr(settings, name, endpoint)
                payload = inference_settings(settings)
                _assert_close(payload["audio2face"][name], endpoint, label=name)
            setattr(settings, name, original)
        settings.preferred_emotions[1].value = 0.75
        assert settings.preferred_emotion_active is False
        apply_mixed_emotions(
            settings,
            ("Neutral", "Joy"),
            (0.3, 0.7),
        )
        settings.input_strength = 2.0
        apply_model_schema(settings, model_schema, model_signature)
        _assert_close(
            settings.preferred_emotions[1].value,
            0.75,
            label="preserved preferred Joy",
        )
        _assert_close(
            settings.mixed_emotions[1].value,
            0.0,
            label="reset transient mixed Joy",
        )
        assert settings.preferred_emotion_active is False
        _assert_close(
            settings.input_strength,
            2.0,
            label="preserved input strength",
        )
        settings.auto_audio2emotion = True
        settings.a2e_preferred_emotion_strength = 0.35
        controller = runtime.get_controller()
        original_refresh = controller.refresh_inference_settings
        refresh = Mock()
        controller.refresh_inference_settings = refresh
        try:
            settings.preferred_emotions[1].value = 0.5
            refresh.assert_not_called()
            assert bpy.ops.a2f.toggle_preferred_emotion() == {"FINISHED"}
            refresh.assert_called_once_with(scene)
            refresh.reset_mock()
            settings.preferred_emotions[0].value = 0.6
            refresh.assert_called_once_with(scene)
        finally:
            controller.refresh_inference_settings = original_refresh
        assert settings.preferred_emotion_active is True
        for item, expected in zip(settings.preferred_emotions, (0.6, 0.5)):
            _assert_close(item.value, expected, label=f"authored {item.name}")
        apply_mixed_emotions(
            settings,
            ("Neutral", "Joy"),
            (0.2, 1.2),
        )
        for collection, expected_values in (
            (settings.preferred_emotions, (0.6, 0.5)),
            (settings.mixed_emotions, (0.2, 1.2)),
        ):
            for item, expected in zip(collection, expected_values):
                _assert_close(item.value, expected, label=item.name)
        automatic_payload = inference_settings(settings)["emotion_driver"]
        assert automatic_payload["mode"] == "automatic"
        preferred_payload = automatic_payload["preferred"]
        for name, expected in (("Neutral", 0.6), ("Joy", 0.5)):
            _assert_close(preferred_payload["values"][name], expected, label=name)
        _assert_close(preferred_payload["strength"], 0.35, label="preferred strength")
        assert bpy.ops.a2f.toggle_preferred_emotion() == {"FINISHED"}
        assert settings.preferred_emotion_active is False
        for item, expected in zip(settings.preferred_emotions, (0.6, 0.5)):
            _assert_close(item.value, expected, label=item.name)
        assert inference_settings(settings)["emotion_driver"]["preferred"] is None
        settings.auto_audio2emotion = False
        manual_driver = inference_settings(settings)["emotion_driver"]
        assert manual_driver["mode"] == "manual"
        for name, expected in (("Neutral", 0.6), ("Joy", 0.5)):
            _assert_close(manual_driver["values"][name], expected, label=name)
        extra_target = _make_shape_key_target(
            scene,
            object_name="A2FSmokeExtraTarget",
        )
        curve_target = _make_shape_key_target(
            scene,
            object_name="A2FSmokeCurveTarget",
            object_type="CURVE",
        )
        surface_target = _make_shape_key_target(
            scene,
            object_name="A2FSmokeSurfaceTarget",
            object_type="SURFACE",
        )
        lattice_target = _make_shape_key_target(
            scene,
            object_name="A2FSmokeLatticeTarget",
            object_type="LATTICE",
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
        unsupported_target = bpy.data.objects.new("A2FSmokeUnsupportedTarget", None)
        scene.collection.objects.link(unsupported_target)
        bpy.ops.object.select_all(action="DESELECT")
        unsupported_target.select_set(True)
        bpy.context.view_layer.objects.active = unsupported_target
        assert not bpy.ops.a2f.add_selected_targets.poll()

        bpy.ops.object.select_all(action="DESELECT")
        expected_targets = {
            target,
            extra_target,
            curve_target,
            surface_target,
            lattice_target,
            linked_target,
            plain_target,
        }
        for selected_target in (*expected_targets, unsupported_target):
            selected_target.select_set(True)
        bpy.context.view_layer.objects.active = target
        assert bpy.ops.a2f.add_selected_targets() == {"FINISHED"}
        selected_targets = {item.object for item in settings.target_objects}
        assert selected_targets == expected_targets
        primary_vertices = [tuple(vertex.co) for vertex in target.data.vertices]
        extra_vertices = [tuple(vertex.co) for vertex in extra_target.data.vertices]

        subscriptions = resolve_target_objects(settings)
        # Every supported object remains subscribed without Shape Key inspection;
        # shared Key datablocks are deduplicated only when a frame is delivered.
        assert len(subscriptions) == len(expected_targets)
        streamed_values = [0.0] * len(MODEL_CHANNELS)
        streamed_values[MODEL_CHANNELS.index("jawOpen")] = 0.625
        apply_shape_key_frame(
            subscriptions,
            tuple(MODEL_CHANNELS),
            tuple(streamed_values),
        )
        for keyed_target in (
            target,
            extra_target,
            curve_target,
            surface_target,
            lattice_target,
            linked_target,
        ):
            _assert_close(
                keyed_target.data.shape_keys.key_blocks["jawOpen"].value,
                0.625,
                label=f"{keyed_target.name} streamed jawOpen",
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

        live = runtime.get_live_stream_controller()
        routed_operation = "blender-smoke-stream"
        try:
            live.prepare(
                scene,
                routed_operation,
                16_000,
                tuple(MODEL_CHANNELS),
                ("Neutral", "Joy"),
                audio_path=None,
                playback_started=None,
                playback_stopped=None,
            )
            controller.active_stream = runtime.ActiveStream(
                operation_id=routed_operation,
                scene_name=scene.name,
                wav_source=None,
            )
            streamed_frame = [0.0] * len(MODEL_CHANNELS)
            streamed_frame[MODEL_CHANNELS.index("jawOpen")] = 0.375
            # The current SDK can emit receptive-field frames before sample zero.
            # They must be accepted and applied directly without animation data.
            _route_stream_event(
                controller,
                routed_operation,
                "stream_frame",
                {
                    "timestamp_sample": -160,
                    "weights": streamed_frame,
                    "effective_emotions": [0.25, 1.25],
                },
            )
            for keyed_target in (
                target,
                extra_target,
                curve_target,
                surface_target,
                lattice_target,
                linked_target,
            ):
                _assert_close(
                    keyed_target.data.shape_keys.key_blocks["jawOpen"].value,
                    0.375,
                    label=f"{keyed_target.name} live-stream jawOpen",
                )
            _assert_close(settings.stream_time, 0.0, label="negative stream time clamp")
            _assert_close(
                settings.mixed_emotions[0].value,
                0.25,
                label="mixed Neutral",
            )
            _assert_close(
                settings.mixed_emotions[1].value,
                1.25,
                label="mixed Joy",
            )
            _route_stream_event(
                controller,
                routed_operation,
                "stream_reset",
                {},
            )
            _route_stream_event(
                controller,
                routed_operation,
                "stream_frame",
                {
                    "timestamp_sample": 0,
                    "weights": streamed_frame,
                    "effective_emotions": [0.75, 0.25],
                },
            )
            assert live.tick() is True
            _assert_close(
                settings.mixed_emotions[0].value,
                0.75,
                label="mixed Neutral after reset",
            )
            _assert_close(
                settings.mixed_emotions[1].value,
                0.25,
                label="mixed Joy after reset",
            )
            assert [tuple(vertex.co) for vertex in target.data.vertices] == primary_vertices
            assert [tuple(vertex.co) for vertex in extra_target.data.vertices] == extra_vertices
            assert target.data.shape_keys.animation_data is None
            assert extra_target.data.shape_keys.animation_data is None
        finally:
            controller._release_active_stream(routed_operation)
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
