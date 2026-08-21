"""Strict I/O for the worker's canonical ARKit-52 animation result."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .arkit import ARKIT_52_CHANNELS


RESULT_SCHEMA = "a2f-animation/1"
MAX_RESULT_FILE_BYTES = 512 * 1024 * 1024
MAX_SAMPLE_RATE = (1 << 32) - 1
MIN_SAMPLE_TIMESTAMP = -(1 << 63)
MAX_SAMPLE_TIMESTAMP = (1 << 63) - 1
MAX_JOB_ID_LENGTH = 128
_RESULT_FIELDS = frozenset(
    {
        "schema",
        "job_id",
        "sample_rate",
        "timestamps_samples",
        "weights",
    }
)


class ResultValidationError(ValueError):
    """Raised when a result is unsafe or violates the fixed ARKit contract."""


@dataclass(slots=True)
class AnimationResult:
    timestamps: list[int]
    sample_rate: int
    weights: list[list[float]]
    job_id: str


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


def _weights(value: Any, frame_count: int) -> list[list[float]]:
    rows = _array(value, "weights")
    if len(rows) != frame_count:
        raise ResultValidationError(
            f"weights has {len(rows)} rows but timestamps_samples has {frame_count} frames"
        )
    weights: list[list[float]] = []
    for frame_index, raw_row in enumerate(rows):
        row = _array(raw_row, f"weights[{frame_index}]")
        if len(row) != len(ARKIT_52_CHANNELS):
            raise ResultValidationError(
                f"weights[{frame_index}] must contain exactly {len(ARKIT_52_CHANNELS)} values"
            )
        parsed: list[float] = []
        for channel_index, item in enumerate(row):
            field = f"weights[{frame_index}][{channel_index}]"
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ResultValidationError(f"{field} must be a number")
            coefficient = float(item)
            if not math.isfinite(coefficient):
                raise ResultValidationError(f"{field} must be finite")
            if coefficient < 0.0 or coefficient > 1.0:
                raise ResultValidationError(f"{field} must be between 0.0 and 1.0")
            parsed.append(coefficient)
        weights.append(parsed)
    return weights


def validate_result_document(document: dict[str, Any]) -> AnimationResult:
    """Validate exactly one ``a2f-animation/1`` ARKit-52 document."""

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

    job_id = document["job_id"]
    if not isinstance(job_id, str) or not job_id or len(job_id) > MAX_JOB_ID_LENGTH:
        raise ResultValidationError(
            f"job_id must be a non-empty string of at most {MAX_JOB_ID_LENGTH} characters"
        )

    timestamps = _timestamps(document["timestamps_samples"])
    return AnimationResult(
        timestamps=timestamps,
        sample_rate=sample_rate,
        weights=_weights(document["weights"], len(timestamps)),
        job_id=job_id,
    )


def resolve_result_path(
    reference: str | os.PathLike[str], allowed_directory: str | os.PathLike[str]
) -> Path:
    """Resolve a worker result and confine it to the managed result directory."""

    path = Path(reference).expanduser().resolve(strict=False)
    root = Path(allowed_directory).expanduser().resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ResultValidationError(
            f"result path is outside the managed result directory: {path}"
        ) from exc
    return path


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResultValidationError(f"result contains duplicate field {key!r}")
        result[key] = value
    return result


def load_animation_result(
    path: str | os.PathLike[str],
    *,
    allowed_directory: str | os.PathLike[str],
    max_bytes: int = MAX_RESULT_FILE_BYTES,
) -> AnimationResult:
    """Load and validate one atomically published managed result."""

    resolved = resolve_result_path(path, allowed_directory)
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise ResultValidationError(f"cannot stat result file {resolved}: {exc}") from exc
    if size > max_bytes:
        raise ResultValidationError(f"result file is larger than {max_bytes} bytes")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            document = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ResultValidationError(f"result contains invalid number {token}")
                ),
            )
    except ResultValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResultValidationError(f"cannot read result file {resolved}: {exc}") from exc
    return validate_result_document(document)


__all__ = [
    "AnimationResult",
    "RESULT_SCHEMA",
    "ResultValidationError",
    "load_animation_result",
    "resolve_result_path",
    "validate_result_document",
]
