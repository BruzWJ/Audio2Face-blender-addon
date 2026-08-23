"""Smoke-test the built extension after Blender installs and enables its ZIP."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import bpy


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
    assert "uninstall" not in operator_names
    assert {
        "preview_play_pause",
        "preview_rewind",
        "load_preferred_emotion",
        "clear_preferred_emotion",
    } <= operator_names
    assert {"preview_play", "preview_pause", "preview_stop"}.isdisjoint(
        operator_names
    )
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
    assert not hasattr(properties, "A2FModelParameterItem")
    missing_scene_property_names = (
        set(properties.A2FSceneSettings.__annotations__) - scene_property_names
    )
    assert not missing_scene_property_names, (
        "scene settings missing registered RNA properties: "
        f"{sorted(missing_scene_property_names)}"
    )
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
        "preview_progress",
        "auto_audio2emotion",
        "manual_emotions",
        "preferred_emotions",
        *emotion_property_defaults,
    } <= scene_property_names
    for name, default in emotion_property_defaults.items():
        assert properties.A2FSceneSettings.bl_rna.properties[name].default == default
    assert not properties.A2FSceneSettings.bl_rna.properties[
        "preferred_emotions"
    ].is_skip_save
    assert {
        "preview_volume",
        "preview_reset_on_stop",
        "stream_reset_on_stop",
        "model_parameters",
        "identity_index",
        "model_identities",
    }.isdisjoint(scene_property_names)
    panel_source = inspect.getsource(ui.A2F_PT_main.draw)
    assert "if settings.manual_emotions:" in panel_source
    assert "settings.manual_emotions and not settings.auto_audio2emotion" not in panel_source
    assert "if settings.auto_audio2emotion:" not in panel_source

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
