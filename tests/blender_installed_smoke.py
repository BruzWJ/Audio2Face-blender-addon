"""Smoke-test the built extension after Blender installs and enables its ZIP."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import bpy


AUDIO2FACE_DEFAULTS: dict[str, float | int] = {
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


def main() -> None:
    assert bpy.app.version[:2] == (5, 2)
    addon_names = [
        addon.module
        for addon in bpy.context.preferences.addons
        if addon.module.endswith(".audio2face")
    ]
    assert len(addon_names) == 1, f"expected one enabled Audio2Face extension, got {addon_names}"
    package = importlib.import_module(addon_names[0])
    runtime = importlib.import_module(f"{addon_names[0]}.runtime")
    preferences = importlib.import_module(f"{addon_names[0]}.preferences")
    properties = importlib.import_module(f"{addon_names[0]}.properties")
    ui = importlib.import_module(f"{addon_names[0]}.ui")

    assert hasattr(bpy.types.Scene, "audio2face")
    assert bpy.app.timers.is_registered(runtime._timer_callback)
    assert runtime._load_pre_handler in bpy.app.handlers.load_pre
    assert runtime._load_post_handler in bpy.app.handlers.load_post
    operator_names = set(dir(bpy.ops.a2f))
    assert {
        "play_pause",
        "load_preferred_emotion",
        "clear_preferred_emotion",
    } <= operator_names
    runtime.get_controller().poll()
    assert bpy.context.scene.audio2face.status == "IDLE"

    preference_names = set(preferences.A2FAddonPreferences.bl_rna.properties.keys())
    assert set(preferences.A2FAddonPreferences.__annotations__) == {
        "nvidia_terms_accepted",
        "audio2face_model_directory",
        "audio2emotion_model_directory",
    }
    missing_preference_names = (
        set(preferences.A2FAddonPreferences.__annotations__) - preference_names
    )
    assert not missing_preference_names, (
        "preferences missing registered RNA properties: "
        f"{sorted(missing_preference_names)}"
    )
    assert (
        preferences.A2FAddonPreferences.bl_rna.properties[
            "audio2face_model_directory"
        ].subtype
        == "DIR_PATH"
    )
    assert (
        preferences.A2FAddonPreferences.bl_rna.properties[
            "audio2emotion_model_directory"
        ].subtype
        == "DIR_PATH"
    )
    scene_property_names = set(properties.A2FSceneSettings.bl_rna.properties.keys())
    assert set(properties.A2FTargetMeshItem.__annotations__) == {"object"}
    missing_scene_property_names = (
        set(properties.A2FSceneSettings.__annotations__) - scene_property_names
    )
    assert not missing_scene_property_names, (
        "scene settings missing registered RNA properties: "
        f"{sorted(missing_scene_property_names)}"
    )
    assert set(properties.AUDIO2FACE_SETTING_FIELDS) == set(AUDIO2FACE_DEFAULTS)
    assert len(properties.AUDIO2FACE_SETTING_FIELDS) == 18
    emotion_property_defaults = {
        "a2e_emotion_strength": 0.6,
        "a2e_emotion_contrast": 1.0,
        "a2e_max_emotions": 6,
        "a2e_live_blend_coef": 0.7,
        "a2e_transition_smoothing": 0.5,
        "a2e_preferred_emotion_strength": 0.5,
    }
    assert {
        "prediction_delay",
        "playback_progress",
        "auto_audio2emotion",
        "manual_emotions",
        "preferred_emotions",
        *emotion_property_defaults,
        *AUDIO2FACE_DEFAULTS,
    } <= scene_property_names
    for name, default in AUDIO2FACE_DEFAULTS.items():
        assert (
            properties.A2FSceneSettings.bl_rna.properties[name].default
            == default
        )
    for name, default in emotion_property_defaults.items():
        assert properties.A2FSceneSettings.bl_rna.properties[name].default == default
    assert not properties.A2FSceneSettings.bl_rna.properties[
        "preferred_emotions"
    ].is_skip_save
    panel_source = inspect.getsource(ui.A2F_PT_main.draw)
    assert "if settings.manual_emotions:" in panel_source
    assert (
        "settings.manual_emotions and not settings.auto_audio2emotion"
        not in panel_source
    )
    assert "if settings.auto_audio2emotion:" not in panel_source

    settings = bpy.context.scene.audio2face
    properties.apply_model_schema(
        settings,
        {
            "channels": [f"modelChannel{index}" for index in range(52)],
            "emotion_channels": [],
            "audio2face_defaults": AUDIO2FACE_DEFAULTS.copy(),
        },
        ("/models/audio2face/model.json", "/models/audio2emotion/model.json"),
    )
    settings.input_strength = 2.0
    settings.blink_strength = 1.5
    settings.eye_saccade_seed = 41
    tuned_audio2face = AUDIO2FACE_DEFAULTS.copy()
    tuned_audio2face.update(
        input_strength=2.0,
        blink_strength=1.5,
        eye_saccade_seed=41,
    )
    assert properties.inference_settings(settings) == {
        "audio2face": tuned_audio2face,
        "auto_audio2emotion": False,
        "manual_emotions": {},
        "audio2emotion": {
            "emotion_strength": 0.6,
            "emotion_contrast": 1.0,
            "max_emotions": 6,
            "live_blend_coef": 0.7,
            "transition_smoothing": 0.5,
            "preferred_emotion": None,
            "preferred_emotion_strength": 0.5,
        },
    }

    bundle = runtime.resolve_runtime_bundle()
    package_directory = Path(package.__file__).resolve().parent
    assert bundle.root == package_directory / "runtime"
    assert bundle.executable.is_file()
    assert bundle.trtexec.is_file()
    layout = Mock()
    draw_context = SimpleNamespace(
        region=SimpleNamespace(width=800),
        preferences=SimpleNamespace(
            system=SimpleNamespace(ui_scale=1.0),
        ),
    )
    preferences.A2FAddonPreferences.draw(
        SimpleNamespace(layout=layout),
        draw_context,
    )
    setup = runtime.get_controller().setup_snapshot()
    assert setup.model_spec is None
    assert "model folder" in setup.model_status.message
    assert package.__package__.startswith("bl_ext.")
    print(f"Installed Audio2Face smoke test passed ({bundle.platform})")


if __name__ == "__main__":
    main()
