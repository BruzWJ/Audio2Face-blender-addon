from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio2face.runtime_catalog import (
    CATALOG_SCHEMA,
    MAX_ARCHIVE_BYTES,
    MAX_UNPACKED_BYTES,
    RuntimeCatalogError,
    load_runtime_catalog,
    validate_runtime_catalog,
)


def _catalog_document(**artifact_overrides: object) -> dict[str, object]:
    artifact: dict[str, object] = {
        "url": "https://downloads.example.test/a2f/runtime-linux-x64-0.1.0.zip",
        "sha256": "a" * 64,
        "size": 1024,
        "unpacked_size": 4096,
    }
    artifact.update(artifact_overrides)
    return {
        "schema": CATALOG_SCHEMA,
        "release": "0.1.0",
        "artifacts": {"linux-x64": artifact},
    }


def test_catalog_validates_pinned_https_artifact_and_release() -> None:
    catalog = validate_runtime_catalog(_catalog_document())

    artifact = catalog.artifact_for("linux-x64")
    assert catalog.release == "0.1.0"
    assert artifact.platform == "linux-x64"
    assert artifact.url.startswith("https://")
    assert artifact.sha256 == "a" * 64
    assert artifact.size == 1024
    assert artifact.unpacked_size == 4096


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"url": "http://downloads.example.test/runtime.zip"}, "HTTPS"),
        ({"url": "https://user@example.test/runtime.zip"}, "without credentials"),
        ({"url": "https://user:secret@example.test/runtime.zip"}, "without credentials"),
        ({"url": "https:///runtime.zip"}, "HTTPS"),
        ({"sha256": "A" * 64}, "SHA-256"),
        ({"sha256": "a" * 63}, "SHA-256"),
        ({"sha256": "g" * 64}, "SHA-256"),
        ({"size": True}, "positive integer"),
        ({"size": 0}, "positive integer"),
        ({"size": MAX_ARCHIVE_BYTES + 1}, "safety limit"),
        ({"unpacked_size": False}, "positive integer"),
        ({"unpacked_size": -1}, "positive integer"),
        ({"unpacked_size": MAX_UNPACKED_BYTES + 1}, "safety limit"),
        ({"unexpected": True}, "unknown fields"),
    ],
)
def test_catalog_rejects_unpinned_or_unsafe_artifact_fields(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(RuntimeCatalogError, match=match):
        validate_runtime_catalog(_catalog_document(**overrides))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(schema="wrong"), "schema"),
        (lambda value: value.update(release=""), "release"),
        (lambda value: value.update(artifacts=[]), "artifacts must be an object"),
        (
            lambda value: value.update(
                artifacts={"macos-x64": value["artifacts"]["linux-x64"]}
            ),
            "unsupported catalog platform",
        ),
    ],
)
def test_catalog_rejects_invalid_top_level_contract(mutation: object, match: str) -> None:
    document = _catalog_document()
    mutation(document)  # type: ignore[operator]
    with pytest.raises(RuntimeCatalogError, match=match):
        validate_runtime_catalog(document)


def test_catalog_missing_platform_reports_unpublished_asset_not_host_rejection() -> None:
    catalog = validate_runtime_catalog(_catalog_document())
    with pytest.raises(
        RuntimeCatalogError,
        match="release has not published.*windows-x64 worker asset.*host was not rejected",
    ):
        catalog.artifact_for("windows-x64")


def test_load_runtime_catalog_wraps_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RuntimeCatalogError, match="cannot read managed-runtime catalog"):
        load_runtime_catalog(path)


def test_catalog_document_must_be_an_object() -> None:
    with pytest.raises(RuntimeCatalogError, match="catalog must be an object"):
        validate_runtime_catalog([])


def test_load_runtime_catalog_rejects_duplicate_fields(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(
        '{"schema":"audio2face-runtime-catalog/1",'
        '"schema":"audio2face-runtime-catalog/1",'
        '"release":"0.1.0","artifacts":{}}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeCatalogError, match="duplicate field"):
        load_runtime_catalog(path)


def test_load_runtime_catalog_reads_a_valid_document(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(_catalog_document()), encoding="utf-8")
    assert load_runtime_catalog(path).artifact_for("linux-x64").size == 1024
