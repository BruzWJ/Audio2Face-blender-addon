from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

import audio2face.result_io as result_io
from audio2face.result_io import (
    RESULT_CHANNEL_COUNT,
    RESULT_SCHEMA,
    ResultValidationError,
    load_animation_result,
    resolve_result_path,
    validate_result_document,
)


MODEL_CHANNELS = ["sdkJawOpen", *(f"sdkOutput{index:02d}" for index in range(51))]
WORKER_BACKEND_SOURCE = (
    Path(__file__).resolve().parents[1] / "worker" / "src" / "a2f_backend.cpp"
).read_text(encoding="utf-8")


def test_native_worker_mirrors_the_python_result_identity() -> None:
    assert f'{{"schema", "{RESULT_SCHEMA}"}}' in WORKER_BACKEND_SOURCE
    assert (
        f"constexpr std::size_t kArkit52ChannelCount = {RESULT_CHANNEL_COUNT};"
        in WORKER_BACKEND_SOURCE
    )


@pytest.fixture
def canonical_result_document() -> dict[str, object]:
    return {
        "schema": RESULT_SCHEMA,
        "operation_id": "operation-canonical",
        "channels": MODEL_CHANNELS.copy(),
        "sample_rate": 16_000,
        "timestamps_samples": [-800, 0, 267],
        "weights": [
            [0.0] * 52,
            [0.5] * 52,
            [1.0] * 52,
        ],
    }


def test_model_described_result_is_signed_exact_and_preserves_channel_order(
    canonical_result_document: dict[str, object],
) -> None:
    result = validate_result_document(canonical_result_document)

    assert result.timestamps == [-800, 0, 267]
    assert result.sample_rate == 16_000
    assert result.operation_id == "operation-canonical"
    assert result.channels == MODEL_CHANNELS
    assert result.weights[1] == [0.5] * 52


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(schema="a2f-animation/999"), "unsupported result schema"),
        (lambda value: value.update(sample_rate=0), "sample_rate"),
        (lambda value: value.update(sample_rate=True), "sample_rate"),
        (lambda value: value.update(sample_rate=16_000.0), "sample_rate"),
        (lambda value: value.update(sample_rate=1 << 32), "sample_rate"),
        (lambda value: value.update(timestamps_samples=[]), "must not be empty"),
        (
            lambda value: value.update(timestamps_samples=[-(1 << 63) - 1, 0, 1]),
            "signed 64-bit",
        ),
        (lambda value: value.update(timestamps_samples=[0, 0, 1]), "strictly increasing"),
        (lambda value: value.update(timestamps_samples=[0, -1, 1]), "strictly increasing"),
        (lambda value: value.update(timestamps_samples=[False, 0, 1]), "integer audio-sample"),
        (lambda value: value.update(weights=[[0.0] * 52]), "timestamps_samples has 3 frames"),
        (
            lambda value: value.update(weights=[[0.0] * 51] * 3),
            "exactly 52 values",
        ),
        (
            lambda value: value.update(channels=MODEL_CHANNELS[:-1]),
            "exactly 52 names",
        ),
        (
            lambda value: value.update(
                channels=[MODEL_CHANNELS[0], MODEL_CHANNELS[0], *MODEL_CHANNELS[2:]]
            ),
            "duplicate name",
        ),
        (
            lambda value: value.update(channels=["", *MODEL_CHANNELS[1:]]),
            r"channels\[0\].*non-empty string",
        ),
        (lambda value: value.update(operation_id=""), "operation_id"),
        (lambda value: value.update(operation_id="x" * 129), "operation_id"),
    ],
)
def test_malformed_canonical_documents_are_rejected(
    canonical_result_document: dict[str, object],
    mutate: object,
    message: str,
) -> None:
    document = copy.deepcopy(canonical_result_document)
    mutate(document)  # type: ignore[operator]

    with pytest.raises(ResultValidationError, match=message):
        validate_result_document(document)


@pytest.mark.parametrize(
    "coefficient",
    [-0.001, 1.001, math.nan, math.inf, -math.inf, 0, True, "0.5"],
)
def test_weights_require_finite_unit_interval_floats(
    canonical_result_document: dict[str, object], coefficient: object
) -> None:
    document = copy.deepcopy(canonical_result_document)
    document["weights"][1][7] = coefficient  # type: ignore[index]

    with pytest.raises(ResultValidationError, match=r"weights\[1\]\[7\]"):
        validate_result_document(document)


def test_missing_and_unknown_fields_are_rejected(
    canonical_result_document: dict[str, object],
) -> None:
    missing = copy.deepcopy(canonical_result_document)
    missing.pop("timestamps_samples")
    with pytest.raises(ResultValidationError, match="missing fields: timestamps_samples"):
        validate_result_document(missing)

    unknown = copy.deepcopy(canonical_result_document)
    unknown["unexpected"] = True
    with pytest.raises(ResultValidationError, match="unknown fields: unexpected"):
        validate_result_document(unknown)

    old_vocabulary = copy.deepcopy(canonical_result_document)
    old_vocabulary["job_id"] = old_vocabulary.pop("operation_id")
    with pytest.raises(
        ResultValidationError,
        match=r"missing fields: operation_id; unknown fields: job_id",
    ):
        validate_result_document(old_vocabulary)


def test_output_channels_require_the_model_json_array_type() -> None:
    with pytest.raises(ResultValidationError, match="must be a JSON array"):
        result_io.validate_output_channels(tuple(MODEL_CHANNELS))


def test_confined_load_round_trip(
    tmp_path: Path, canonical_result_document: dict[str, object]
) -> None:
    output_directory = tmp_path / "results"
    output_directory.mkdir()
    destination = output_directory / "operation-canonical.json"
    destination.write_text(json.dumps(canonical_result_document), encoding="utf-8")

    loaded = load_animation_result(destination, allowed_directory=output_directory)

    assert loaded.operation_id == "operation-canonical"
    assert loaded.timestamps == [-800, 0, 267]
    assert loaded.channels == MODEL_CHANNELS


def test_load_rejects_duplicate_fields_invalid_json_and_invalid_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"a2f-animation/2","schema":"a2f-animation/2"}',
        encoding="utf-8",
    )
    with pytest.raises(ResultValidationError, match="duplicate field 'schema'"):
        load_animation_result(duplicate, allowed_directory=tmp_path)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not json}", encoding="utf-8")
    with pytest.raises(ResultValidationError, match="cannot read result file"):
        load_animation_result(invalid, allowed_directory=tmp_path)

    invalid_number = tmp_path / "invalid-number.json"
    invalid_number.write_text('{"weight":NaN}', encoding="utf-8")
    with pytest.raises(ResultValidationError, match="invalid number NaN"):
        load_animation_result(invalid_number, allowed_directory=tmp_path)


def test_result_path_cannot_escape_controller_directory(tmp_path: Path) -> None:
    allowed = tmp_path / "results"
    allowed.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ResultValidationError,
        match="outside the controller-owned result directory",
    ):
        resolve_result_path(outside, allowed)


def test_result_path_rejects_noncanonical_parent_traversal(tmp_path: Path) -> None:
    allowed = tmp_path / "results"
    allowed.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ResultValidationError, match="canonical absolute path"):
        resolve_result_path(allowed / ".." / outside.name, allowed)


def test_result_path_prefix_sibling_is_not_a_child(tmp_path: Path) -> None:
    allowed = tmp_path / "results"
    sibling = tmp_path / "results-untrusted" / "operation.json"
    allowed.mkdir()
    sibling.parent.mkdir()
    sibling.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ResultValidationError,
        match="outside the controller-owned result directory",
    ):
        resolve_result_path(sibling, allowed)


def test_result_path_rejects_in_tree_alias_to_outside(tmp_path: Path) -> None:
    allowed = tmp_path / "results"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    outside_result = outside / "operation.json"
    outside_result.write_text("{}", encoding="utf-8")
    link = allowed / "redirect"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(
        ResultValidationError,
        match="filesystem alias",
    ):
        resolve_result_path(link / outside_result.name, allowed)


def test_result_path_rejects_outside_alias_to_in_tree_result(tmp_path: Path) -> None:
    allowed = tmp_path / "results"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    result = allowed / "operation.json"
    result.write_text("{}", encoding="utf-8")
    alias = outside / "operation.json"
    try:
        alias.symlink_to(result)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable")

    with pytest.raises(ResultValidationError, match="filesystem alias"):
        resolve_result_path(alias, allowed)


def test_result_path_rejects_aliased_allowed_root(tmp_path: Path) -> None:
    allowed = tmp_path / "results"
    allowed.mkdir()
    result = allowed / "operation.json"
    result.write_text("{}", encoding="utf-8")
    alias_root = tmp_path / "results-alias"
    try:
        alias_root.symlink_to(allowed, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(ResultValidationError, match="filesystem alias"):
        resolve_result_path(alias_root / result.name, alias_root)


def test_load_rejects_oversized_files(
    tmp_path: Path,
    canonical_result_document: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_text(json.dumps(canonical_result_document), encoding="utf-8")
    monkeypatch.setattr(result_io, "MAX_RESULT_FILE_BYTES", 8)
    with pytest.raises(ResultValidationError, match="larger than 8 bytes"):
        load_animation_result(oversized, allowed_directory=tmp_path)
