"""Validate user-selected Audio2Face model folders without importing Blender."""

from __future__ import annotations

import json
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from .path_contract import require_unaliased_path
from .strict_json import duplicate_key_hook, invalid_constant_hook


MAX_MODEL_JSON_BYTES = 64 * 1024
GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
ModelRole = Literal["audio2face", "audio2emotion"]

_ROLE_LABELS: dict[ModelRole, str] = {
    "audio2face": "Audio2Face",
    "audio2emotion": "Audio2Emotion",
}
_AUDIO2FACE_MODEL_FIELDS = frozenset(
    {
        "networkInfoPath",
        "networkPath",
        "modelConfigPaths",
        "modelDataPaths",
        "blendshapePaths",
    }
)
_AUDIO2EMOTION_MODEL_FIELDS = frozenset(
    {"networkInfoPath", "networkPath", "modelConfigPath"}
)
_BLENDSHAPE_IDENTITY_FIELDS = frozenset({"skin", "tongue"})
_BLENDSHAPE_PART_FIELDS = frozenset({"config", "data"})


class ModelInputError(ValueError):
    """Raised when a selected model is incomplete, inaccessible, or unsafe."""


def _require_existing_path(path: str | Path, description: str) -> Path:
    return require_unaliased_path(
        path,
        description=description,
        error_type=ModelInputError,
    )


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


def _read_model_document(path: Path, label: str) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_MODEL_JSON_BYTES:
            raise ModelInputError(f"{label} model.json is unexpectedly large")
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(
                handle,
                object_pairs_hook=duplicate_key_hook(ModelInputError, "model.json"),
                parse_constant=invalid_constant_hook(
                    ModelInputError,
                    f"{label} model.json",
                ),
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelInputError(f"cannot read {label} model.json: {exc}") from exc
    if not isinstance(document, dict):
        raise ModelInputError(f"{label} model.json must contain an object")
    return document


def _exact_fields(value: object, expected: frozenset[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != expected:
        raise ModelInputError(
            f"{location} must contain exactly {sorted(expected)}"
        )
    return value


def _path_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelInputError(f"{location} must be a non-empty path string")
    return value


def _path_array(value: object, location: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ModelInputError(f"{location} must be a non-empty path array")
    return [
        _path_string(item, f"{location}[{index}]")
        for index, item in enumerate(value)
    ]


def _audio2face_references(document: dict[str, Any]) -> list[str]:
    model = _exact_fields(document, _AUDIO2FACE_MODEL_FIELDS, "Audio2Face model.json")
    network = _path_string(model["networkPath"], "Audio2Face model.json.networkPath")
    if network != "network.trt":
        raise ModelInputError(
            "Audio2Face model.json networkPath must be exactly 'network.trt'"
        )
    network_info = _path_string(
        model["networkInfoPath"], "Audio2Face model.json.networkInfoPath"
    )
    model_configs = _path_array(
        model["modelConfigPaths"], "Audio2Face model.json.modelConfigPaths"
    )
    model_data = _path_array(
        model["modelDataPaths"], "Audio2Face model.json.modelDataPaths"
    )
    blendshape_descriptors = model["blendshapePaths"]
    if not isinstance(blendshape_descriptors, list) or not blendshape_descriptors:
        raise ModelInputError(
            "Audio2Face model.json.blendshapePaths must be a non-empty array"
        )
    if not (
        len(model_configs) == len(model_data) == len(blendshape_descriptors)
    ):
        raise ModelInputError(
            "Audio2Face identity path arrays must have the same length"
        )

    references = [network_info, network, *model_configs, *model_data]
    for identity_index, raw_identity in enumerate(blendshape_descriptors):
        identity_location = (
            f"Audio2Face model.json.blendshapePaths[{identity_index}]"
        )
        identity = _exact_fields(
            raw_identity,
            _BLENDSHAPE_IDENTITY_FIELDS,
            identity_location,
        )
        for part_name in ("skin", "tongue"):
            part_location = f"{identity_location}.{part_name}"
            part = _exact_fields(
                identity[part_name],
                _BLENDSHAPE_PART_FIELDS,
                part_location,
            )
            references.extend(
                (
                    _path_string(part["config"], f"{part_location}.config"),
                    _path_string(part["data"], f"{part_location}.data"),
                )
            )
    return references


def _audio2emotion_references(document: dict[str, Any]) -> list[str]:
    model = _exact_fields(
        document,
        _AUDIO2EMOTION_MODEL_FIELDS,
        "Audio2Emotion model.json",
    )
    network = _path_string(
        model["networkPath"], "Audio2Emotion model.json.networkPath"
    )
    if network != "network.trt":
        raise ModelInputError(
            "Audio2Emotion model.json networkPath must be exactly 'network.trt'"
        )
    return [
        _path_string(
            model["networkInfoPath"],
            "Audio2Emotion model.json.networkInfoPath",
        ),
        network,
        _path_string(
            model["modelConfigPath"],
            "Audio2Emotion model.json.modelConfigPath",
        ),
    ]


def _model_references(
    document: dict[str, Any],
    role: ModelRole,
) -> list[str]:
    if role == "audio2face":
        references = _audio2face_references(document)
    elif role == "audio2emotion":
        references = _audio2emotion_references(document)
    else:
        raise ModelInputError(f"unsupported model role {role!r}")
    duplicates = sorted(
        {reference for reference in references if references.count(reference) > 1}
    )
    if duplicates:
        raise ModelInputError(
            f"{_ROLE_LABELS[role]} model.json repeats file references: {duplicates}"
        )
    return references


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
    resolved = _require_existing_path(
        model_directory.joinpath(*relative.parts),
        description,
    )
    _require_regular_file(resolved, description, nonempty=True)
    return resolved


def _selected_model_directory(value: str | Path, role: ModelRole) -> Path:
    label = _ROLE_LABELS[role]
    model_directory = _require_existing_path(value, f"{label} model folder")
    if not model_directory.is_dir():
        raise ModelInputError(
            f"{label} must select the complete model folder, not a file: "
            f"{model_directory}"
        )
    return model_directory


def _validate_resolved_model_directory(
    model_directory: Path,
    role: ModelRole,
) -> Path:
    label = _ROLE_LABELS[role]
    resolved_model = _require_existing_path(
        model_directory / "model.json",
        f"{label} model.json",
    )
    _require_regular_file(
        resolved_model,
        f"{label} model.json",
        nonempty=True,
    )
    document = _read_model_document(resolved_model, label)

    for reference in _model_references(document, role):
        if reference == "network.trt":
            continue
        _resolve_referenced_file(
            model_directory,
            reference,
            f"{label} model.json reference {reference!r}",
        )

    for filename in ("network.onnx", "trt_info.json"):
        description = f"{label} companion {filename}"
        companion = _require_existing_path(model_directory / filename, description)
        _require_regular_file(companion, description, nonempty=True)

    return resolved_model


def validate_model_pair(
    audio2face_directory: str | Path,
    audio2emotion_directory: str | Path,
) -> tuple[Path, Path]:
    """Validate the one Audio2Face/Audio2Emotion model pairing contract."""

    face_directory = _selected_model_directory(audio2face_directory, "audio2face")
    emotion_directory = _selected_model_directory(
        audio2emotion_directory,
        "audio2emotion",
    )
    if face_directory == emotion_directory:
        raise ModelInputError(
            "Audio2Face and Audio2Emotion must use different model folders"
        )
    return (
        _validate_resolved_model_directory(
            face_directory,
            "audio2face",
        ),
        _validate_resolved_model_directory(
            emotion_directory,
            "audio2emotion",
        ),
    )


def _validate_model_engine(model: Path, role: ModelRole) -> None:
    label = _ROLE_LABELS[role]
    model_directory = model.parent
    if model != model_directory / "model.json":
        raise ModelInputError(
            f"{label} engine validation requires its validated model.json"
        )
    _resolve_referenced_file(
        model_directory,
        "network.trt",
        f"{label} model.json reference 'network.trt'",
    )


def validate_model_engines(
    audio2face_model: Path,
    audio2emotion_model: Path,
) -> None:
    """Validate the two TensorRT engines for one already validated model pair."""

    _validate_model_engine(audio2face_model, "audio2face")
    _validate_model_engine(audio2emotion_model, "audio2emotion")
