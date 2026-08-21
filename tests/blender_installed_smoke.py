"""Smoke-test the built extension after Blender installs and enables its ZIP."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

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
    operators = importlib.import_module(f"{addon_names[0]}.operators")
    preferences = importlib.import_module(f"{addon_names[0]}.preferences")
    properties = importlib.import_module(f"{addon_names[0]}.properties")

    assert hasattr(bpy.types.Scene, "audio2face")
    assert bpy.app.timers.is_registered(runtime._timer_callback)
    assert runtime._load_pre_handler in bpy.app.handlers.load_pre
    assert runtime._load_post_handler in bpy.app.handlers.load_post
    assert bpy.ops.a2f.uninstall.poll()
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
    missing_scene_property_names = (
        set(properties.A2FSceneSettings.__annotations__) - scene_property_names
    )
    assert not missing_scene_property_names, (
        "scene settings missing registered RNA properties: "
        f"{sorted(missing_scene_property_names)}"
    )

    bundle = runtime.resolve_runtime_bundle()
    package_directory = Path(package.__file__).resolve().parent
    assert bundle.root == package_directory / "runtime"
    assert bundle.executable.is_file()
    assert bundle.trtexec.is_file()
    repo_directory, package_id = preferences._uninstall_target(bpy.context)
    assert package_id == "audio2face"
    assert Path(repo_directory).is_dir()
    layout = Mock()
    preferences.A2FAddonPreferences.draw(
        SimpleNamespace(layout=layout),
        bpy.context,
    )
    layout.row.assert_called_once_with()
    removal = layout.row.return_value
    assert removal.alignment == "RIGHT"
    removal.operator.assert_called_once_with("a2f.uninstall", text="Uninstall")
    layout.separator.assert_called_once_with(type="LINE")

    dialog_layout = Mock()
    operators.A2F_OT_uninstall.draw(
        SimpleNamespace(layout=dialog_layout),
        bpy.context,
    )
    assert dialog_layout.label.call_args_list == [
        call(text="Remove Add-on: 'Audio2Face'?", translate=False),
        call(
            text=f"Path: {str(Path(repo_directory, package_id))!r}",
            translate=False,
        ),
    ]
    window_manager = Mock()
    window_manager.invoke_props_dialog.return_value = {"RUNNING_MODAL"}
    dialog_operator = SimpleNamespace()
    result = operators.A2F_OT_uninstall.invoke(
        dialog_operator,
        SimpleNamespace(window_manager=window_manager),
        SimpleNamespace(),
    )
    assert result == {"RUNNING_MODAL"}
    window_manager.invoke_props_dialog.assert_called_once_with(
        dialog_operator,
        width=600,
    )
    setup = runtime.get_controller().setup_snapshot()
    assert setup.model_spec is None
    assert "model folder" in setup.model_status.message
    assert package.__package__.startswith("bl_ext.")
    print(f"Installed Audio2Face smoke test passed ({bundle.platform})")


if __name__ == "__main__":
    main()
