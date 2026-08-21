# Audio2Face distribution

This directory contains generated Blender extension packages. Do not edit a
ZIP directly; rebuild it from the repository's `audio2face/` source directory.

## Current package

`audio2face-0.1.0.zip` is the installable Audio2Face extension for Blender
5.2.x on Windows x64 and Linux x64.

- Display name: `Audio2Face`
- Extension ID: `audio2face`
- Version: `0.1.0`
- SHA-256: `4fff15b6a5c716ed3ef1a44f917cb7bb219daa81c63b7c28e3605fc6bddb7341`

The ZIP contains the Blender add-on only. It does not embed CUDA, TensorRT, the
native worker, Audio2Face, or Audio2Emotion. Add-on Preferences provides one
NVIDIA terms acceptance, source buttons for both models, and one managed
install action that downloads a reviewed artifact containing the runtime and
both models. The same Preferences page provides **Uninstall Audio2Face**, which
shows a legacy-style path confirmation before using Blender's native extension
removal to delete the add-on and all of its managed runtime, model, temporary,
log, and result files.

> [!IMPORTANT]
> The checked-in runtime catalog currently publishes no platform archives.
> This ZIP is therefore a development package: the interface can be installed
> and tested, but managed GPU inference cannot be installed until reviewed
> runtime archives are published and added to `audio2face/runtime_catalog.json`.

## Install in Blender 5.2

1. Open **Edit > Preferences > Extensions**.
2. Open the Extensions menu and choose **Install from Disk**.
3. Select `audio2face-0.1.0.zip`.
4. Enable **Audio2Face** and Blender Online Access.
5. Open **Edit > Preferences > Add-ons > Audio2Face**, review the linked NVIDIA
   terms, use the single acceptance checkbox, and click
   **Install Runtime & Models** when a runtime artifact is published for the
   platform. The Audio2Face and Audio2Emotion buttons show the exact model
   sources used by that bundle.
6. Open the **Audio2Face** tab in the 3D View sidebar. Runtime setup controls
   stay in Add-on Preferences; the sidebar only reports readiness.

To remove Audio2Face cleanly, return to its Add-on Preferences and click
**Uninstall Audio2Face**. This does not remove external WAV or `.blend` files.

## Build and verify

Run these commands from the repository root. PowerShell example:

```powershell
$Blender = "C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe"

& $Blender --command extension validate audio2face
& $Blender --command extension build --source-dir audio2face --output-dir dist
& $Blender --factory-startup --background --python tests/blender_smoke.py
python -m pytest -q
Get-FileHash dist/audio2face-0.1.0.zip -Algorithm SHA256
```

The validated archive must contain `blender_manifest.toml` and `__init__.py` at
its root, not inside another directory. Blender derives the output filename
from the manifest's `id` and `version` fields.

The current package was built with Blender 5.2.0 LTS. Its extension validation,
headless Blender smoke test, and Python suite passed.
