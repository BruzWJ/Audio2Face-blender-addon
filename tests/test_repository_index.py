from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPOSITORY_ROOT / "tools" / "build_repository_index.py"
SPEC = importlib.util.spec_from_file_location("build_repository_index", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)

VERSION = "2026.8.27"
DIGEST = "0123456789abcdef" * 4


def manifest() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "id": "audio2face",
        "version": VERSION,
        "name": "Audio2Face",
        "tagline": "Drive Shape Key objects with GPU-generated ARKit-52 values",
        "maintainer": "x.com/BruzWJ",
        "type": "add-on",
        "tags": ["Animation"],
        "blender_version_min": "5.2.0",
        "blender_version_max": "5.3.0",
        "platforms": ["windows-x64", "linux-x64"],
        "license": ["SPDX:GPL-3.0-or-later"],
        "permissions": {"files": "not part of the listing"},
    }


def assets() -> dict[str, tuple[str, int, str]]:
    return {
        platform: (f"audio2face-{VERSION}-{platform}.zip", 1234, DIGEST)
        for platform in tool.PLATFORMS
    }


def test_build_index_creates_the_two_native_release_entries() -> None:
    index = tool.build_index(
        manifest(),
        repository="BruzWJ/Audio2Face-blender-addon",
        release_tag=f"v{VERSION}",
        assets=assets(),
    )

    assert index["version"] == "v1"
    assert index["blocklist"] == []
    assert [entry["platforms"] for entry in index["data"]] == [
        ["windows-x64"],
        ["linux-x64"],
    ]
    for entry in index["data"]:
        platform = entry["platforms"][0]
        filename = f"audio2face-{VERSION}-{platform}.zip"
        assert entry["archive_url"] == (
            "https://github.com/BruzWJ/Audio2Face-blender-addon/"
            f"releases/download/v{VERSION}/{filename}"
        )
        assert entry["archive_size"] == 1234
        assert entry["archive_hash"] == f"sha256:{DIGEST}"
        assert "permissions" not in entry


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value.update(platforms=["linux-x64"]), "platforms"),
        (lambda value: value.update(version="2026.08.27"), "manifest version"),
        (lambda value: value.pop("tagline"), "missing listing fields"),
    ],
)
def test_build_index_rejects_invalid_manifest(change, message: str) -> None:
    value = manifest()
    change(value)
    with pytest.raises(tool.IndexError, match=message):
        tool.build_index(
            value,
            repository="BruzWJ/Audio2Face-blender-addon",
            release_tag=f"v{VERSION}",
            assets=assets(),
        )


def test_cli_writes_valid_index(tmp_path: Path) -> None:
    output = tmp_path / "index.json"
    manifest_path = REPOSITORY_ROOT / "audio2face" / "blender_manifest.toml"
    current_version = tomllib.loads(manifest_path.read_text(encoding="utf-8"))[
        "version"
    ]
    result = tool.main(
        [
            "--manifest",
            str(manifest_path),
            "--repository",
            "BruzWJ/Audio2Face-blender-addon",
            "--release-tag",
            f"v{current_version}",
            "--windows-name",
            f"audio2face-{current_version}-windows-x64.zip",
            "--windows-size",
            "1234",
            "--windows-sha256",
            DIGEST,
            "--linux-name",
            f"audio2face-{current_version}-linux-x64.zip",
            "--linux-size",
            "1234",
            "--linux-sha256",
            DIGEST,
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8"))["version"] == "v1"
