#!/usr/bin/env python3
"""Build the Blender v1 repository index for one Audio2Face release."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import quote


PLATFORMS = ("windows-x64", "linux-x64")
LISTING_FIELDS = (
    "schema_version",
    "id",
    "name",
    "tagline",
    "version",
    "type",
    "maintainer",
    "license",
    "blender_version_min",
    "blender_version_max",
    "tags",
)
SEMVER = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
SHA256 = re.compile(r"[0-9a-f]{64}")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


class IndexError(ValueError):
    """The release metadata cannot form a Blender repository index."""


def build_index(
    manifest: Mapping[str, object],
    *,
    repository: str,
    release_tag: str,
    assets: Mapping[str, tuple[str, int, str]],
) -> dict[str, object]:
    """Return one index containing the verified Windows and Linux packages."""

    if REPOSITORY.fullmatch(repository) is None:
        raise IndexError(f"invalid GitHub repository name: {repository!r}")
    missing = [field for field in LISTING_FIELDS if field not in manifest]
    if missing:
        raise IndexError(f"manifest is missing listing fields: {missing}")

    extension_id = manifest["id"]
    version = manifest["version"]
    if not isinstance(extension_id, str) or not extension_id:
        raise IndexError("manifest id must be a non-empty string")
    if not isinstance(version, str) or SEMVER.fullmatch(version) is None:
        raise IndexError(f"invalid manifest version: {version!r}")
    if release_tag != f"v{version}":
        raise IndexError(f"release tag must be v{version}, got {release_tag!r}")
    if manifest.get("platforms") != list(PLATFORMS):
        raise IndexError(f"manifest platforms must be exactly {list(PLATFORMS)!r}")
    if set(assets) != set(PLATFORMS):
        raise IndexError(f"assets must cover exactly {list(PLATFORMS)!r}")

    common = {field: manifest[field] for field in LISTING_FIELDS}
    entries: list[dict[str, object]] = []
    for platform in PLATFORMS:
        filename, size, digest = assets[platform]
        expected_name = f"{extension_id}-{version}-{platform}.zip"
        if filename != expected_name:
            raise IndexError(
                f"{platform} asset must be {expected_name!r}, got {filename!r}"
            )
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise IndexError(f"{platform} asset size must be a positive integer")
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise IndexError(f"{platform} asset SHA-256 must be 64 lowercase hex digits")
        entry = dict(common)
        entry.update(
            platforms=[platform],
            archive_url=(
                f"https://github.com/{repository}/releases/download/"
                f"{quote(release_tag, safe='')}/{quote(filename, safe='')}"
            ),
            archive_size=size,
            archive_hash=f"sha256:{digest}",
        )
        entries.append(entry)

    return {"version": "v1", "blocklist": [], "data": entries}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--windows-name", required=True)
    parser.add_argument("--windows-size", required=True, type=int)
    parser.add_argument("--windows-sha256", required=True)
    parser.add_argument("--linux-name", required=True)
    parser.add_argument("--linux-size", required=True, type=int)
    parser.add_argument("--linux-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = create_parser().parse_args(argv)
    try:
        manifest = tomllib.loads(arguments.manifest.read_text(encoding="utf-8"))
        index = build_index(
            manifest,
            repository=arguments.repository,
            release_tag=arguments.release_tag,
            assets={
                "windows-x64": (
                    arguments.windows_name,
                    arguments.windows_size,
                    arguments.windows_sha256,
                ),
                "linux-x64": (
                    arguments.linux_name,
                    arguments.linux_size,
                    arguments.linux_sha256,
                ),
            },
        )
        arguments.output.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except (IndexError, OSError, tomllib.TOMLDecodeError) as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
