"""Validate user-selected Audio2Face model folders without importing Blender."""

from __future__ import annotations

import json
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterator


MAX_MODEL_JSON_BYTES = 64 * 1024
GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class ModelInputError(ValueError):
    """Raised when a selected model is incomplete, inaccessible, or unsafe."""


def _resolve(path: Path, description: str) -> Path:
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ModelInputError(
            f"{description} is missing or inaccessible: {path}"
        ) from exc


def _require_regular_file(
    path: Path,
    description: str,
    *,
    nonempty: bool,
) -> None:
    try:
        details = path.stat()
    except OSError as exc:
        raise ModelInputError(
            f"{description} is missing or inaccessible: {path}"
        ) from exc
    if not stat.S_ISREG(details.st_mode):
        raise ModelInputError(f"{description} must be a regular file: {path}")
    if nonempty and details.st_size == 0:
        raise ModelInputError(f"{description} must not be empty: {path}")
    if 0 < details.st_size <= 1024:
        try:
            with path.open("rb") as handle:
                prefix = handle.read(len(GIT_LFS_POINTER_PREFIX))
        except OSError as exc:
            raise ModelInputError(
                f"{description} is inaccessible: {path}"
            ) from exc
        if prefix == GIT_LFS_POINTER_PREFIX:
            raise ModelInputError(
                f"{description} is a Git LFS pointer, not the downloaded model file: {path}"
            )


def _require_inside(path: Path, directory: Path, description: str) -> None:
    try:
        path.relative_to(directory)
    except ValueError as exc:
        raise ModelInputError(
            f"{description} escapes the selected model directory: {path}"
        ) from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelInputError(f"model.json contains duplicate field {key!r}")
        result[key] = value
    return result


def _read_model_document(path: Path, label: str) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_MODEL_JSON_BYTES:
            raise ModelInputError(f"{label} model.json is unexpectedly large")
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ModelInputError(
                        f"{label} model.json contains invalid number {token}"
                    )
                ),
            )
    except ModelInputError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelInputError(f"cannot read {label} model.json: {exc}") from exc
    if not isinstance(document, dict):
        raise ModelInputError(f"{label} model.json must contain an object")
    return document


def _referenced_path_strings(value: Any, field: str) -> Iterator[str]:
    if isinstance(value, str):
        if not value:
            raise ModelInputError(f"model.json field {field!r} contains an empty path")
        yield value
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _referenced_path_strings(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for name, item in value.items():
            yield from _referenced_path_strings(item, f"{field}.{name}")
        return
    raise ModelInputError(f"model.json field {field!r} has an invalid path structure")


def _resolve_referenced_file(
    model_directory: Path,
    value: str,
    description: str,
) -> Path:
    if "\x00" in value or "\\" in value:
        raise ModelInputError(f"{description} must be a portable relative path")
    relative = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        relative.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or str(relative) != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ModelInputError(f"{description} must be a canonical relative path")
    resolved = _resolve(
        model_directory.joinpath(*relative.parts),
        description,
    )
    _require_inside(resolved, model_directory, description)
    _require_regular_file(resolved, description, nonempty=True)
    return resolved


def validate_model_directory(
    value: str | Path,
    label: str,
    require_engine: bool,
) -> Path:
    """Validate one exact model folder and return its resolved model card.

    ``value`` must be the complete Hugging Face clone folder with ``model.json``
    at its root. The ONNX network and TensorRT build metadata must be non-empty
    sibling files. A built TensorRT engine is also required when
    ``require_engine`` is true. Symlinks may point within the selected model
    directory, but may not escape it. Parent-directory searching is never used.
    """

    try:
        selected_directory = Path(value).expanduser()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ModelInputError(f"{label} model folder path is invalid") from exc

    model_directory = _resolve(selected_directory, f"{label} model folder")
    if not model_directory.is_dir():
        raise ModelInputError(
            f"{label} must select the complete model folder, not a file: "
            f"{selected_directory}"
        )
    resolved_model = _resolve(model_directory / "model.json", f"{label} model.json")
    _require_inside(resolved_model, model_directory, f"{label} model.json")
    _require_regular_file(
        resolved_model,
        f"{label} model.json",
        nonempty=True,
    )
    document = _read_model_document(resolved_model, label)

    network_path = document.get("networkPath")
    if network_path != "network.trt":
        raise ModelInputError(
            f"{label} model.json networkPath must be exactly 'network.trt'"
        )

    referenced: set[str] = set()
    for field, value in document.items():
        if field.casefold().endswith(("path", "paths")):
            referenced.update(_referenced_path_strings(value, field))
    for value in sorted(referenced):
        if value == "network.trt" and not require_engine:
            continue
        _resolve_referenced_file(
            model_directory,
            value,
            f"{label} model.json reference {value!r}",
        )

    companion_names = ["network.onnx", "trt_info.json"]
    if require_engine:
        companion_names.append("network.trt")

    for filename in companion_names:
        description = f"{label} companion {filename}"
        companion = _resolve(model_directory / filename, description)
        _require_inside(companion, model_directory, description)
        _require_regular_file(companion, description, nonempty=True)

    return resolved_model
