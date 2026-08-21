"""Blender 5.2 extension entry point for Audio2Face.

The protocol, result parsing, and process-client modules intentionally avoid
importing :mod:`bpy`, which keeps their contracts testable by ordinary Python
test runners.
"""

from __future__ import annotations

_REGISTERED = False


def register() -> None:
    """Register Blender classes, scene properties, and the main-thread timer."""

    global _REGISTERED
    if _REGISTERED:
        return

    try:
        import bpy
    except ModuleNotFoundError as exc:  # pragma: no cover - Blender-only path
        raise RuntimeError("Audio2Face must be registered inside Blender") from exc

    from . import operators, preferences, properties, runtime, ui

    modules = (preferences, properties, operators, ui)
    registered: list[type] = []
    try:
        for module in modules:
            for cls in module.CLASSES:
                bpy.utils.register_class(cls)
                registered.append(cls)

        bpy.types.Scene.audio2face = bpy.props.PointerProperty(
            type=properties.A2FSceneSettings
        )
        runtime.register_runtime()
    except Exception:
        if hasattr(bpy.types.Scene, "audio2face"):
            del bpy.types.Scene.audio2face
        for cls in reversed(registered):
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass
        raise

    _REGISTERED = True


def unregister() -> None:
    """Stop the worker and remove every Blender registration cleanly."""

    global _REGISTERED
    if not _REGISTERED:
        return

    import bpy

    from . import operators, preferences, properties, runtime, ui

    runtime.unregister_runtime()

    if hasattr(bpy.types.Scene, "audio2face"):
        del bpy.types.Scene.audio2face

    for module in (ui, operators, properties, preferences):
        for cls in reversed(module.CLASSES):
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass

    _REGISTERED = False


__all__ = ["register", "unregister"]
