"""Validated release catalog for this add-on's native GPU worker packages.

The catalog is intentionally independent of :mod:`bpy`.  Release automation
fills it with one pinned, checksummed archive per supported platform.  Blender
never guesses a URL, follows a mutable ``latest`` link, or searches the host for
an existing worker.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlparse


CATALOG_SCHEMA = "audio2face-runtime-catalog/1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_ARCHIVE_BYTES = 20 * 1024 * 1024 * 1024
MAX_UNPACKED_BYTES = 30 * 1024 * 1024 * 1024
_CATALOG_FIELDS = frozenset({"schema", "release", "artifacts"})
_ARTIFACT_FIELDS = frozenset({"url", "sha256", "size", "unpacked_size"})


class RuntimeCatalogError(ValueError):
    """Raised when a runtime release catalog is incomplete or unsafe."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeCatalogError(f"runtime catalog contains duplicate field {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class RuntimeArtifact:
    platform: str
    url: str
    sha256: str
    size: int
    unpacked_size: int


@dataclass(frozen=True, slots=True)
class RuntimeCatalog:
    release: str
    artifacts: Mapping[str, RuntimeArtifact]

    def artifact_for(self, platform: str) -> RuntimeArtifact:
        try:
            return self.artifacts[platform]
        except KeyError as exc:
            raise RuntimeCatalogError(
                "this Audio2Face add-on release does not include a verified "
                f"GPU worker package for {platform}"
            ) from exc


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeCatalogError(f"{field} must be an object")
    return value


def _require_fields(
    value: dict[str, Any],
    expected: frozenset[str],
    field: str,
) -> None:
    keys = frozenset(value)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise RuntimeCatalogError(f"invalid {field} ({'; '.join(details)})")


def _positive_int(value: Any, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeCatalogError(f"{field} must be a positive integer")
    if value > maximum:
        raise RuntimeCatalogError(f"{field} exceeds the managed-runtime safety limit")
    return value


def validate_runtime_catalog(document: Any) -> RuntimeCatalog:
    if not isinstance(document, dict):
        raise RuntimeCatalogError("catalog must be an object")
    _require_fields(document, _CATALOG_FIELDS, "catalog")
    if document["schema"] != CATALOG_SCHEMA:
        raise RuntimeCatalogError(f"catalog schema must be {CATALOG_SCHEMA!r}")
    release = document["release"]
    if not isinstance(release, str) or not release.strip():
        raise RuntimeCatalogError("catalog release must be a non-empty string")

    raw_artifacts = _object(document["artifacts"], "catalog.artifacts")
    artifacts: dict[str, RuntimeArtifact] = {}
    for platform, raw_artifact in raw_artifacts.items():
        if platform not in {"linux-x64", "windows-x64"}:
            raise RuntimeCatalogError(f"unsupported catalog platform {platform!r}")
        artifact = _object(raw_artifact, f"catalog.artifacts[{platform!r}]")
        _require_fields(artifact, _ARTIFACT_FIELDS, f"catalog artifact {platform!r}")
        url = artifact["url"]
        if not isinstance(url, str) or not url:
            raise RuntimeCatalogError(f"runtime URL for {platform} must be non-empty")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise RuntimeCatalogError(
                f"runtime URL for {platform} must be an HTTPS URL without credentials"
            )
        checksum = artifact["sha256"]
        if not isinstance(checksum, str) or not SHA256_PATTERN.fullmatch(checksum):
            raise RuntimeCatalogError(f"runtime SHA-256 for {platform} is invalid")
        artifacts[platform] = RuntimeArtifact(
            platform=platform,
            url=url,
            sha256=checksum,
            size=_positive_int(
                artifact["size"],
                f"catalog.artifacts[{platform!r}].size",
                MAX_ARCHIVE_BYTES,
            ),
            unpacked_size=_positive_int(
                artifact["unpacked_size"],
                f"catalog.artifacts[{platform!r}].unpacked_size",
                MAX_UNPACKED_BYTES,
            ),
        )
    return RuntimeCatalog(
        release=release,
        artifacts=MappingProxyType(dict(artifacts)),
    )


def load_runtime_catalog(path: str | Path | None = None) -> RuntimeCatalog:
    source = Path(path) if path is not None else Path(__file__).with_name("runtime_catalog.json")
    try:
        with source.open("r", encoding="utf-8") as handle:
            document = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except RuntimeCatalogError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeCatalogError(f"cannot read managed-runtime catalog {source}: {exc}") from exc
    return validate_runtime_catalog(document)
