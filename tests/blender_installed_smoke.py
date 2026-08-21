"""Smoke-test the built extension after Blender installs and enables its ZIP."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

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

    assert hasattr(bpy.types.Scene, "audio2face")
    assert bpy.app.timers.is_registered(runtime._timer_callback)
    assert runtime._load_pre_handler in bpy.app.handlers.load_pre
    assert runtime._load_post_handler in bpy.app.handlers.load_post
    runtime.get_controller().poll()
    assert bpy.context.scene.audio2face.status == "IDLE"

    preference_names = set(preferences.A2FAddonPreferences.bl_rna.properties.keys())
    assert preference_names == {"rna_type"} | set(
        preferences.A2FAddonPreferences.__annotations__
    )
    assert set(properties.A2FSceneSettings.bl_rna.properties.keys()) == {
        "rna_type"
    } | set(properties.A2FSceneSettings.__annotations__)

    data_root = runtime.get_controller().data_root(create=True)
    isolated_extensions = os.environ.get("BLENDER_USER_EXTENSIONS")
    if isolated_extensions:
        data_root.relative_to(Path(isolated_extensions).resolve())
    ready, reason = runtime.get_controller().runtime_availability()
    assert not ready
    assert "published" in reason or "runtime" in reason
    assert package.__package__.startswith("bl_ext.")
    print(f"Installed Audio2Face smoke test passed ({data_root})")


if __name__ == "__main__":
    main()
