from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from audio2face.model_inputs import ModelInputError, validate_model_directory


def _make_model(directory: Path, *, engine: bool = False) -> Path:
    directory.mkdir(parents=True)
    model = directory / "model.json"
    model.write_text('{"networkPath":"network.trt"}', encoding="utf-8")
    (directory / "network.onnx").write_bytes(b"onnx")
    (directory / "trt_info.json").write_text("{}", encoding="utf-8")
    if engine:
        (directory / "network.trt").write_bytes(b"engine")
    return directory


@pytest.mark.parametrize("as_string", [False, True])
def test_validate_model_directory_returns_resolved_descriptor(
    tmp_path: Path, as_string: bool
) -> None:
    directory = _make_model(tmp_path / "models" / "face")
    value: str | Path = str(directory) if as_string else directory

    assert validate_model_directory(value, "Audio2Face model", False) == (
        directory / "model.json"
    ).resolve()


def test_validate_model_directory_requires_engine_only_after_build(tmp_path: Path) -> None:
    directory = _make_model(tmp_path / "face")

    with pytest.raises(ModelInputError, match=r"network\.trt.*missing or inaccessible"):
        validate_model_directory(directory, "Audio2Face model", True)

    (directory / "network.trt").write_bytes(b"engine")
    assert validate_model_directory(directory, "Audio2Face model", True) == (
        directory / "model.json"
    ).resolve()


@pytest.mark.parametrize("name", ["face.json", "Model.json", "model.json", "network.onnx"])
def test_validate_model_directory_rejects_file_selections(
    tmp_path: Path, name: str
) -> None:
    selected = tmp_path / name
    selected.write_text("{}", encoding="utf-8")

    with pytest.raises(ModelInputError, match="complete model folder, not a file"):
        validate_model_directory(selected, "Audio2Face model", False)


def test_validate_model_directory_rejects_missing_selected_path(tmp_path: Path) -> None:
    selected = tmp_path / "missing-model-folder"

    with pytest.raises(ModelInputError, match="missing or inaccessible"):
        validate_model_directory(selected, "Audio2Face model", False)


def test_validate_model_directory_requires_a_root_model_json(tmp_path: Path) -> None:
    selected = tmp_path / "model-folder"
    selected.mkdir()

    with pytest.raises(ModelInputError, match=r"model\.json.*missing or inaccessible"):
        validate_model_directory(selected, "Audio2Face model", False)


def test_validate_model_directory_does_not_search_a_selected_parent(
    tmp_path: Path,
) -> None:
    selected_parent = tmp_path / "downloads"
    _make_model(selected_parent / "Audio2Face-3D-v3.0")

    with pytest.raises(ModelInputError, match=r"model\.json.*missing or inaccessible"):
        validate_model_directory(selected_parent, "Audio2Face model", False)


def test_validate_model_directory_rejects_empty_model_json(tmp_path: Path) -> None:
    selected = _make_model(tmp_path / "model-folder")
    (selected / "model.json").write_bytes(b"")

    with pytest.raises(ModelInputError, match="must not be empty"):
        validate_model_directory(selected, "Audio2Face model", False)


@pytest.mark.parametrize("filename", ["network.onnx", "trt_info.json"])
def test_validate_model_directory_requires_each_build_companion(
    tmp_path: Path, filename: str
) -> None:
    directory = _make_model(tmp_path / "face")
    (directory / filename).unlink()

    with pytest.raises(ModelInputError, match=filename.replace(".", r"\.")):
        validate_model_directory(directory, "Audio2Face model", False)


@pytest.mark.parametrize(
    ("filename", "require_engine"),
    [
        ("network.onnx", False),
        ("trt_info.json", False),
        ("network.trt", True),
    ],
)
def test_validate_model_directory_rejects_empty_companions(
    tmp_path: Path, filename: str, require_engine: bool
) -> None:
    directory = _make_model(tmp_path / "face", engine=require_engine)
    (directory / filename).write_bytes(b"")

    with pytest.raises(ModelInputError, match=r"must not be empty"):
        validate_model_directory(directory, "Audio2Face model", require_engine)


@pytest.mark.parametrize(
    ("filename", "require_engine"),
    [
        ("network.onnx", False),
        ("trt_info.json", False),
        ("network.trt", True),
    ],
)
def test_validate_model_directory_rejects_non_file_companions(
    tmp_path: Path, filename: str, require_engine: bool
) -> None:
    directory = _make_model(tmp_path / "face", engine=require_engine)
    companion = directory / filename
    companion.unlink()
    companion.mkdir()

    with pytest.raises(ModelInputError, match="must be a regular file"):
        validate_model_directory(directory, "Audio2Face model", require_engine)


def test_validate_model_directory_includes_caller_label_in_errors(tmp_path: Path) -> None:
    directory = _make_model(tmp_path / "emotion")
    (directory / "network.onnx").unlink()

    with pytest.raises(ModelInputError, match="Audio2Emotion model companion"):
        validate_model_directory(directory, "Audio2Emotion model", False)


def test_validate_model_directory_rejects_descriptor_symlink_escape(tmp_path: Path) -> None:
    model_directory = tmp_path / "selected"
    model_directory.mkdir()
    outside = _make_model(tmp_path / "outside") / "model.json"
    (model_directory / "network.onnx").write_bytes(b"onnx")
    (model_directory / "trt_info.json").write_text("{}", encoding="utf-8")
    selected = model_directory / "model.json"
    try:
        selected.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable")

    with pytest.raises(ModelInputError, match="escapes the selected model directory"):
        validate_model_directory(model_directory, "Audio2Face model", False)


@pytest.mark.parametrize(
    ("filename", "require_engine"),
    [
        ("network.onnx", False),
        ("trt_info.json", False),
        ("network.trt", True),
    ],
)
def test_validate_model_directory_rejects_companion_symlink_escape(
    tmp_path: Path, filename: str, require_engine: bool
) -> None:
    directory = _make_model(tmp_path / "selected", engine=require_engine)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    companion = directory / filename
    companion.unlink()
    try:
        companion.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable")

    with pytest.raises(ModelInputError, match="escapes the selected model directory"):
        validate_model_directory(directory, "Audio2Face model", require_engine)


def test_validate_model_directory_accepts_companion_symlink_within_directory(
    tmp_path: Path,
) -> None:
    directory = _make_model(tmp_path / "selected")
    storage = directory / "payload"
    storage.mkdir()
    target = storage / "network.onnx"
    target.write_bytes(b"onnx")
    companion = directory / "network.onnx"
    companion.unlink()
    try:
        companion.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable")

    assert validate_model_directory(directory, "Audio2Face model", False) == (
        directory / "model.json"
    ).resolve()


def test_validate_model_directory_rejects_broken_companion_symlink(tmp_path: Path) -> None:
    directory = _make_model(tmp_path / "selected")
    companion = directory / "network.onnx"
    companion.unlink()
    try:
        companion.symlink_to(tmp_path / "missing.onnx")
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable")

    with pytest.raises(ModelInputError, match="missing or inaccessible"):
        validate_model_directory(directory, "Audio2Face model", False)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are unavailable")
def test_validate_model_directory_rejects_non_regular_descriptor(tmp_path: Path) -> None:
    directory = _make_model(tmp_path / "selected")
    selected = directory / "model.json"
    selected.unlink()
    os.mkfifo(selected)

    with pytest.raises(ModelInputError, match="must be a regular file"):
        validate_model_directory(directory, "Audio2Face model", False)


def test_validate_model_directory_rejects_invalid_value_type(tmp_path: Path) -> None:
    del tmp_path
    with pytest.raises(ModelInputError, match="path is invalid"):
        validate_model_directory(None, "Audio2Face model", False)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ("not-json", "cannot read"),
        ("[]", "must contain an object"),
        ('{"networkPath":"network.trt","networkPath":"other.trt"}', "duplicate"),
        ("{}", "networkPath"),
        ('{"networkPath":"other.trt"}', "networkPath"),
    ],
)
def test_validate_model_directory_rejects_invalid_model_documents(
    tmp_path: Path, contents: str, match: str
) -> None:
    directory = _make_model(tmp_path / "selected")
    (directory / "model.json").write_text(contents, encoding="utf-8")

    with pytest.raises(ModelInputError, match=match):
        validate_model_directory(directory, "Audio2Face", False)


def test_validate_model_directory_follows_all_nested_model_json_references(
    tmp_path: Path,
) -> None:
    directory = _make_model(tmp_path / "selected")
    document = {
        "networkPath": "network.trt",
        "networkInfoPath": "network_info.json",
        "modelConfigPaths": ["model_config.json"],
        "blendshapePaths": [
            {"skin": {"config": "skin.json", "data": "skin.npz"}}
        ],
    }
    (directory / "model.json").write_text(json.dumps(document), encoding="utf-8")
    for name in ("network_info.json", "model_config.json", "skin.json", "skin.npz"):
        (directory / name).write_bytes(b"payload")

    assert validate_model_directory(directory, "Audio2Face", False) == (
        directory / "model.json"
    ).resolve()
    (directory / "skin.npz").unlink()
    with pytest.raises(ModelInputError, match=r"skin\.npz"):
        validate_model_directory(directory, "Audio2Face", False)


@pytest.mark.parametrize("value", ["../outside.npz", "/outside.npz", r"C:\outside.npz"])
def test_validate_model_directory_rejects_unsafe_referenced_paths(
    tmp_path: Path, value: str
) -> None:
    directory = _make_model(tmp_path / "selected")
    (directory / "model.json").write_text(
        json.dumps({"networkPath": "network.trt", "modelDataPath": value}),
        encoding="utf-8",
    )

    with pytest.raises(ModelInputError, match="relative path"):
        validate_model_directory(directory, "Audio2Face", False)


def test_validate_model_directory_rejects_git_lfs_pointer_payloads(tmp_path: Path) -> None:
    directory = _make_model(tmp_path / "selected")
    (directory / "network.onnx").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "0" * 64 + "\nsize 123\n",
        encoding="ascii",
    )

    with pytest.raises(ModelInputError, match="Git LFS pointer"):
        validate_model_directory(directory, "Audio2Face", False)
