"""Strict I/O for model-described animation results."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .path_contract import require_unaliased_path
from .strict_json import duplicate_key_hook, invalid_constant_hook

RESULT_SCHEMA = "a2f-animation/2"
RESULT_CHANNEL_COUNT = 52
MAX_RESULT_FILE_BYTES = 512 * 1024 * 1024
MAX_SAMPLE_RATE = (1 << 32) - 1
MIN_SAMPLE_TIMESTAMP = -(1 << 63)
MAX_SAMPLE_TIMESTAMP = (1 << 63) - 1
MAX_OPERATION_ID_LENGTH = 128
_RESULT_FIELDS = frozenset(
    {
        "schema",
        "operation_id",
        "channels",
        "sample_rate",
        "timestamps_samples",
        "weights",
    }
)


class ResultValidationError(ValueError):
    """Raised when a result is unsafe or violates the output contract."""


@dataclass(slots=True)
class AnimationResult:
    timestamps: list[int]
    sample_rate: int
    weights: list[list[float]]
    operation_id: str
    channels: list[str]


def _array(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ResultValidationError(f"{field} must be a JSON array")
    return value


def _strict_fields(value: dict[str, Any]) -> None:
    keys = frozenset(value)
    missing = sorted(_RESULT_FIELDS - keys)
    unknown = sorted(keys - _RESULT_FIELDS)
    if not missing and not unknown:
        return
    details: list[str] = []
    if missing:
        details.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        details.append(f"unknown fields: {', '.join(unknown)}")
    raise ResultValidationError(f"invalid result document ({'; '.join(details)})")


def _timestamps(value: Any) -> list[int]:
    timestamps: list[int] = []
    for index, item in enumerate(_array(value, "timestamps_samples")):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ResultValidationError(
                f"timestamps_samples[{index}] must be an integer audio-sample index"
            )
        if item < MIN_SAMPLE_TIMESTAMP or item > MAX_SAMPLE_TIMESTAMP:
            raise ResultValidationError(
                f"timestamps_samples[{index}] must fit a signed 64-bit integer"
            )
        timestamps.append(item)
    if not timestamps:
        raise ResultValidationError("timestamps_samples must not be empty")
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ResultValidationError("timestamps_samples must be strictly increasing")
    return timestamps


def validate_output_channels(channels: Any) -> tuple[str, ...]:
    """Validate and freeze the model-provided output bus description."""

    if not isinstance(channels, list):
        raise ResultValidationError("channels must be a JSON array")
    if len(channels) != RESULT_CHANNEL_COUNT:
        raise ResultValidationError(
            f"channels must contain exactly {RESULT_CHANNEL_COUNT} names"
        )

    validated: list[str] = []
    seen: set[str] = set()
    for index, name in enumerate(channels):
        if not isinstance(name, str) or not name:
            raise ResultValidationError(f"channels[{index}] must be a non-empty string")
        if name in seen:
            raise ResultValidationError(f"channels contains duplicate name {name!r}")
        validated.append(name)
        seen.add(name)
    return tuple(validated)


def _weights(value: Any, frame_count: int, channel_count: int) -> list[list[float]]:
    rows = _array(value, "weights")
    if len(rows) != frame_count:
        raise ResultValidationError(
            f"weights has {len(rows)} rows but timestamps_samples has {frame_count} frames"
        )
    weights: list[list[float]] = []
    for frame_index, raw_row in enumerate(rows):
        row = _array(raw_row, f"weights[{frame_index}]")
        if len(row) != channel_count:
            raise ResultValidationError(
                f"weights[{frame_index}] must contain exactly {channel_count} values"
            )
        parsed: list[float] = []
        for channel_index, item in enumerate(row):
            field = f"weights[{frame_index}][{channel_index}]"
            if type(item) is not float:
                raise ResultValidationError(f"{field} must be a JSON float")
            coefficient = item
            if not math.isfinite(coefficient):
                raise ResultValidationError(f"{field} must be finite")
            if coefficient < 0.0 or coefficient > 1.0:
                raise ResultValidationError(f"{field} must be between 0.0 and 1.0")
            parsed.append(coefficient)
        weights.append(parsed)
    return weights


def validate_result_document(document: dict[str, Any]) -> AnimationResult:
    """Validate exactly one ``a2f-animation/2`` output-described document."""

    if not isinstance(document, dict):
        raise ResultValidationError("result must be a JSON object")
    _strict_fields(document)
    if document["schema"] != RESULT_SCHEMA:
        raise ResultValidationError(
            f"unsupported result schema {document['schema']!r}; expected {RESULT_SCHEMA!r}"
        )
    sample_rate = document["sample_rate"]
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate <= 0
        or sample_rate > MAX_SAMPLE_RATE
    ):
        raise ResultValidationError("sample_rate must be a positive uint32 integer")

    operation_id = document["operation_id"]
    if (
        not isinstance(operation_id, str)
        or not operation_id
        or len(operation_id) > MAX_OPERATION_ID_LENGTH
    ):
        raise ResultValidationError(
            "operation_id must be a non-empty string of at most "
            f"{MAX_OPERATION_ID_LENGTH} characters"
        )

    channels = validate_output_channels(_array(document["channels"], "channels"))
    timestamps = _timestamps(document["timestamps_samples"])
    return AnimationResult(
        timestamps=timestamps,
        sample_rate=sample_rate,
        weights=_weights(document["weights"], len(timestamps), len(channels)),
        operation_id=operation_id,
        channels=list(channels),
    )


def resolve_result_path(
    reference: str | os.PathLike[str], allowed_directory: str | os.PathLike[str]
) -> Path:
    """Validate one canonical result path and confine it to its canonical root."""

    root_lexical = require_unaliased_path(
        allowed_directory,
        description="controller-owned result directory",
        error_type=ResultValidationError,
    )
    path_lexical = require_unaliased_path(
        reference,
        description="result file",
        error_type=ResultValidationError,
    )
    if not root_lexical.is_dir():
        raise ResultValidationError(
            f"controller-owned result directory is not a directory: {root_lexical}"
        )
    if not path_lexical.is_file():
        raise ResultValidationError(f"result file is not a regular file: {path_lexical}")
    try:
        path_lexical.relative_to(root_lexical)
    except ValueError as exc:
        raise ResultValidationError(
            "result path is outside the controller-owned result directory: "
            f"{path_lexical}"
        ) from exc
    return path_lexical


def load_animation_result(
    path: str | os.PathLike[str],
    *,
    allowed_directory: str | os.PathLike[str],
) -> AnimationResult:
    """Load and validate one atomically published controller-owned result."""

    resolved = resolve_result_path(path, allowed_directory)
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise ResultValidationError(f"cannot stat result file {resolved}: {exc}") from exc
    if size > MAX_RESULT_FILE_BYTES:
        raise ResultValidationError(
            f"result file is larger than {MAX_RESULT_FILE_BYTES} bytes"
        )
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            document = json.load(
                handle,
                object_pairs_hook=duplicate_key_hook(
                    ResultValidationError,
                    "result",
                ),
                parse_constant=invalid_constant_hook(
                    ResultValidationError,
                    "result",
                ),
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResultValidationError(f"cannot read result file {resolved}: {exc}") from exc
    return validate_result_document(document)
