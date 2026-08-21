from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from audio2face.model_inputs import (
    ModelInputError,
    ModelRole,
    _selected_model_directory,
    _validate_model_engine,
    _validate_resolved_model_directory,
    validate_model_engines,
    validate_model_pair,
)


def _make_model(
    directory: Path,
    *,
    role: ModelRole = "audio2face",
    engine: bool = False,
) -> Path:
    directory.mkdir(parents=True)
    model = directory / "model.json"
    if role == "audio2face":
        document = {
            "networkInfoPath": "network_info.json",
            "networkPath": "network.trt",
            "modelConfigPaths": ["model_config.json"],
            "modelDataPaths": ["model_data.npz"],
            "blendshapePaths": [
                {
                    "skin": {"config": "skin.json", "data": "skin.npz"},
                    "tongue": {"config": "tongue.json", "data": "tongue.npz"},
                }
            ],
        }
        referenced = (
            "network_info.json",
            "model_config.json",
            "model_data.npz",
            "skin.json",
            "skin.npz",
            "tongue.json",
            "tongue.npz",
        )
    else:
        document = {
            "networkInfoPath": "network_info.json",
            "networkPath": "network.trt",
            "modelConfigPath": "model_config.json",
        }
        referenced = ("network_info.json", "model_config.json")
    model.write_text(json.dumps(document), encoding="utf-8")
    for filename in referenced:
        (directory / filename).write_bytes(b"model payload")
    (directory / "network.onnx").write_bytes(b"onnx")
    (directory / "trt_info.json").write_text("{}", encoding="utf-8")
    if engine:
        (directory / "network.trt").write_bytes(b"engine")
    return directory


def _validate_selected_repository(
    value: str | Path,
    role: ModelRole,
    require_engine: bool,
) -> Path:
    """Exercise the same two exact component phases used by the pair contract."""

    model = _validate_resolved_model_directory(
        _selected_model_directory(value, role),
        role,
    )
    if require_engine:
        _validate_model_engine(model, role)
    return model


def test_model_pair_is_the_only_public_selection_contract(tmp_path: Path) -> None:
    face = _make_model(tmp_path / "face")
    emotion = _make_model(tmp_path / "emotion", role="audio2emotion")

    assert validate_model_pair(face, emotion) == (
        (face / "model.json").resolve(),
        (emotion / "model.json").resolve(),
    )


def test_model_pair_rejects_one_folder_for_both_roles(tmp_path: Path) -> None:
    selected = _make_model(tmp_path / "selected")

    with pytest.raises(ModelInputError, match="different model folders"):
        validate_model_pair(selected, selected)


@pytest.mark.parametrize("as_string", [False, True])
def test_selected_model_repository_returns_resolved_descriptor(
    tmp_path: Path, as_string: bool
) -> None:
    directory = _make_model(tmp_path / "models" / "face")
    value: str | Path = str(directory) if as_string else directory

    assert _validate_selected_repository(value, "audio2face", False) == (
        directory / "model.json"
    ).resolve()


def test_selected_model_repository_requires_engine_only_after_build(tmp_path: Path) -> None:
    directory = _make_model(tmp_path / "face")

    with pytest.raises(ModelInputError, match=r"network\.trt.*missing or inaccessible"):
        _validate_selected_repository(directory, "audio2face", True)

    (directory / "network.trt").write_bytes(b"engine")
    assert _validate_selected_repository(directory, "audio2face", True) == (
        directory / "model.json"
    ).resolve()


def test_engine_check_reuses_already_validated_model_cards(tmp_path: Path) -> None:
    face = _make_model(tmp_path / "face", engine=True)
    emotion = _make_model(
        tmp_path / "emotion",
        role="audio2emotion",
        engine=True,
    )
    face_model = _validate_selected_repository(face, "audio2face", False)
    emotion_model = _validate_selected_repository(emotion, "audio2emotion", False)

    validate_model_engines(face_model, emotion_model)

    (emotion / "network.trt").unlink()
    with pytest.raises(ModelInputError, match=r"Audio2Emotion.*network\.trt"):
        validate_model_engines(face_model, emotion_model)


@pytest.mark.parametrize("name", ["face.json", "Model.json", "model.json", "network.onnx"])
def test_selected_model_repository_rejects_file_selections(
    tmp_path: Path, name: str
) -> None:
    selected = tmp_path / name
    selected.write_text("{}", encoding="utf-8")

    with pytest.raises(ModelInputError, match="complete model folder, not a file"):
        _validate_selected_repository(selected, "audio2face", False)


def test_selected_model_repository_rejects_missing_selected_path(tmp_path: Path) -> None:
    selected = tmp_path / "missing-model-folder"

    with pytest.raises(ModelInputError, match="missing or inaccessible"):
        _validate_selected_repository(selected, "audio2face", False)


def test_selected_model_repository_rejects_relative_path() -> None:
    with pytest.raises(ModelInputError, match="canonical absolute path"):
        _validate_selected_repository(Path("relative/model"), "audio2face", False)


def test_selected_model_repository_rejects_normalizable_string_spelling(
    tmp_path: Path,
) -> None:
    _make_model(tmp_path / "face")
    selected = f"{tmp_path}{os.sep}.{os.sep}face"

    with pytest.raises(ModelInputError, match="canonical absolute path"):
        _validate_selected_repository(selected, "audio2face", False)


def test_selected_model_repository_requires_a_root_model_json(tmp_path: Path) -> None:
    selected = tmp_path / "model-folder"
    selected.mkdir()

    with pytest.raises(ModelInputError, match=r"model\.json.*missing or inaccessible"):
        _validate_selected_repository(selected, "audio2face", False)


def test_selected_model_repository_does_not_search_a_selected_parent(
    tmp_path: Path,
) -> None:
    selected_parent = tmp_path / "downloads"
    _make_model(selected_parent / "Audio2Face-3D-v3.0")

    with pytest.raises(ModelInputError, match=r"model\.json.*missing or inaccessible"):
        _validate_selected_repository(selected_parent, "audio2face", False)


def test_selected_model_repository_rejects_empty_model_json(tmp_path: Path) -> None:
    selected = _make_model(tmp_path / "model-folder")
    (selected / "model.json").write_bytes(b"")

    with pytest.raises(ModelInputError, match="must not be empty"):
        _validate_selected_repository(selected, "audio2face", False)


@pytest.mark.parametrize("filename", ["network.onnx", "trt_info.json"])
def test_selected_model_repository_requires_each_build_companion(
    tmp_path: Path, filename: str
) -> None:
    directory = _make_model(tmp_path / "face")
    (directory / filename).unlink()

    with pytest.raises(ModelInputError, match=filename.replace(".", r"\.")):
        _validate_selected_repository(directory, "audio2face", False)


@pytest.mark.parametrize(
    ("filename", "require_engine"),
    [
        ("network.onnx", False),
        ("trt_info.json", False),
        ("network.trt", True),
    ],
)
def test_selected_model_repository_rejects_empty_companions(
    tmp_path: Path, filename: str, require_engine: bool
) -> None:
    directory = _make_model(tmp_path / "face", engine=require_engine)
    (directory / filename).write_bytes(b"")

    with pytest.raises(ModelInputError, match=r"must not be empty"):
        _validate_selected_repository(directory, "audio2face", require_engine)


@pytest.mark.parametrize(
    ("filename", "require_engine"),
    [
        ("network.onnx", False),
        ("trt_info.json", False),
        ("network.trt", True),
    ],
)
def test_selected_model_repository_rejects_non_file_companions(
    tmp_path: Path, filename: str, require_engine: bool
) -> None:
    directory = _make_model(tmp_path / "face", engine=require_engine)
    companion = directory / filename
    companion.unlink()
    companion.mkdir()

    with pytest.raises(ModelInputError, match="must be a regular file"):
        _validate_selected_repository(directory, "audio2face", require_engine)


def test_selected_model_repository_includes_caller_label_in_errors(tmp_path: Path) -> None:
    directory = _make_model(tmp_path / "emotion", role="audio2emotion")
    (directory / "network.onnx").unlink()

    with pytest.raises(ModelInputError, match="Audio2Emotion companion"):
        _validate_selected_repository(directory, "audio2emotion", False)


def test_selected_model_repository_rejects_descriptor_symlink_escape(tmp_path: Path) -> None:
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

    with pytest.raises(ModelInputError, match="filesystem alias"):
        _validate_selected_repository(model_directory, "audio2face", False)


def test_selected_model_repository_rejects_selected_folder_alias(tmp_path: Path) -> None:
    target = _make_model(tmp_path / "target")
    selected = tmp_path / "selected"
    try:
        selected.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(ModelInputError, match="filesystem alias"):
        _validate_selected_repository(selected, "audio2face", False)


@pytest.mark.parametrize(
    ("filename", "require_engine"),
    [
        ("network.onnx", False),
        ("trt_info.json", False),
        ("network.trt", True),
    ],
)
def test_selected_model_repository_rejects_companion_symlink_escape(
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

    with pytest.raises(ModelInputError, match="filesystem alias"):
        _validate_selected_repository(directory, "audio2face", require_engine)


def test_selected_model_repository_rejects_companion_symlink_within_directory(
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

    with pytest.raises(ModelInputError, match="filesystem alias"):
        _validate_selected_repository(directory, "audio2face", False)


def test_selected_model_repository_rejects_broken_companion_symlink(tmp_path: Path) -> None:
    directory = _make_model(tmp_path / "selected")
    companion = directory / "network.onnx"
    companion.unlink()
    try:
        companion.symlink_to(tmp_path / "missing.onnx")
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable")

    with pytest.raises(ModelInputError, match="filesystem alias"):
        _validate_selected_repository(directory, "audio2face", False)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are unavailable")
def test_selected_model_repository_rejects_non_regular_descriptor(tmp_path: Path) -> None:
    directory = _make_model(tmp_path / "selected")
    selected = directory / "model.json"
    selected.unlink()
    os.mkfifo(selected)

    with pytest.raises(ModelInputError, match="must be a regular file"):
        _validate_selected_repository(directory, "audio2face", False)


def test_selected_model_repository_rejects_invalid_value_type(tmp_path: Path) -> None:
    del tmp_path
    with pytest.raises(ModelInputError, match="path is invalid"):
        _validate_selected_repository(None, "audio2face", False)  # type: ignore[arg-type]


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
def test_selected_model_repository_rejects_invalid_model_documents(
    tmp_path: Path, contents: str, match: str
) -> None:
    directory = _make_model(tmp_path / "selected")
    (directory / "model.json").write_text(contents, encoding="utf-8")

    with pytest.raises(ModelInputError, match=match):
        _validate_selected_repository(directory, "audio2face", False)


def test_selected_model_repository_follows_all_nested_model_json_references(
    tmp_path: Path,
) -> None:
    directory = _make_model(tmp_path / "selected")
    document = {
        "networkInfoPath": "network_info.json",
        "networkPath": "network.trt",
        "modelConfigPaths": ["model_config.json"],
        "modelDataPaths": ["model_data.npz"],
        "blendshapePaths": [
            {
                "skin": {"config": "skin.json", "data": "skin.npz"},
                "tongue": {"config": "tongue.json", "data": "tongue.npz"},
            }
        ],
    }
    (directory / "model.json").write_text(json.dumps(document), encoding="utf-8")
    for name in (
        "network_info.json",
        "model_config.json",
        "model_data.npz",
        "skin.json",
        "skin.npz",
        "tongue.json",
        "tongue.npz",
    ):
        (directory / name).write_bytes(b"payload")

    assert _validate_selected_repository(directory, "audio2face", False) == (
        directory / "model.json"
    ).resolve()
    (directory / "skin.npz").unlink()
    with pytest.raises(ModelInputError, match=r"skin\.npz"):
        _validate_selected_repository(directory, "audio2face", False)


@pytest.mark.parametrize("value", ["../outside.npz", "/outside.npz", r"C:\outside.npz"])
def test_selected_model_repository_rejects_unsafe_referenced_paths(
    tmp_path: Path, value: str
) -> None:
    directory = _make_model(tmp_path / "selected")
    document = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    document["modelDataPaths"] = [value]
    (directory / "model.json").write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ModelInputError, match="relative path"):
        _validate_selected_repository(directory, "audio2face", False)


def test_selected_model_repository_rejects_git_lfs_pointer_payloads(tmp_path: Path) -> None:
    directory = _make_model(tmp_path / "selected")
    (directory / "network.onnx").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "0" * 64 + "\nsize 123\n",
        encoding="ascii",
    )

    with pytest.raises(ModelInputError, match="Git LFS pointer"):
        _validate_selected_repository(directory, "audio2face", False)


def test_validate_audio2emotion_requires_its_exact_singular_schema(
    tmp_path: Path,
) -> None:
    directory = _make_model(tmp_path / "emotion", role="audio2emotion")
    document = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    document["modelConfigPaths"] = [document.pop("modelConfigPath")]
    (directory / "model.json").write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ModelInputError, match="must contain exactly"):
        _validate_selected_repository(directory, "audio2emotion", False)


def test_selected_model_repository_rejects_duplicate_references(tmp_path: Path) -> None:
    directory = _make_model(tmp_path / "face")
    document = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    document["modelDataPaths"] = ["model_config.json"]
    (directory / "model.json").write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ModelInputError, match="repeats file references"):
        _validate_selected_repository(directory, "audio2face", False)
