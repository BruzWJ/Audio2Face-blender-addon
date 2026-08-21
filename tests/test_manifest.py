from __future__ import annotations

import re
from pathlib import Path


def test_extension_is_limited_to_blender_52() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "a2f_blender"
        / "blender_manifest.toml"
    )
    manifest = manifest_path.read_text(encoding="utf-8")
    fields = dict(
        re.findall(r'^([a-z_]+)\s*=\s*"([^"]+)"\s*$', manifest, flags=re.MULTILINE)
    )

    assert fields["id"] == "a2f_blender"
    assert fields["name"] == "Audio2Face"
    assert fields["blender_version_min"] == "5.2.0"
    # Blender interprets blender_version_max as the first unsupported version.
    assert fields["blender_version_max"] == "5.3.0"
    assert 'platforms = ["windows-x64", "linux-x64"]' in manifest
