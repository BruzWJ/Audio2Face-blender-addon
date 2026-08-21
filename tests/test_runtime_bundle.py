from __future__ import annotations

import json
import os
import stat
import struct
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from a2f_blender.runtime_bundle import (
    BundleError,
    RUNTIME_SCHEMA,
    current_platform_id,
    resolve_runtime_bundle,
)


def _write_elf_x64(path: Path, *, executable: bool = True) -> None:
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<H", header, 16, 3)
    struct.pack_into("<H", header, 18, 62)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header)
    mode = path.stat().st_mode
    if executable:
        path.chmod(mode | stat.S_IXUSR)
    else:
        path.chmod(mode & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))


def _write_pe_x64(path: Path) -> None:
    image = bytearray(256)
    image[:2] = b"MZ"
    pe_offset = 128
    struct.pack_into("<I", image, 0x3C, pe_offset)
    image[pe_offset : pe_offset + 4] = b"PE\x00\x00"
    struct.pack_into("<H", image, pe_offset + 4, 0x8664)
    struct.pack_into("<H", image, pe_offset + 20, 0xF0)
    struct.pack_into("<H", image, pe_offset + 24, 0x20B)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image)


def _make_bundle(
    data_root: Path,
    platform_id: str = "linux-x64",
    *,
    include_engine: bool = True,
) -> tuple[Path, dict[str, object]]:
    root = data_root / "runtime" / platform_id
    root.mkdir(parents=True)
    windows = platform_id == "windows-x64"
    worker_name = "a2f_blender_worker.exe" if windows else "a2f_blender_worker"
    trtexec_name = "trtexec.exe" if windows else "trtexec"
    worker = root / "bin" / worker_name
    trtexec = root / "bin" / trtexec_name
    if windows:
        _write_pe_x64(worker)
        _write_pe_x64(trtexec)
    else:
        _write_elf_x64(worker)
        _write_elf_x64(trtexec)

    (root / "lib" / "audio2x").mkdir(parents=True)
    for model_name in ("audio2face", "audio2emotion"):
        model_dir = root / "models" / model_name
        model_dir.mkdir(parents=True)
        (model_dir / "model.json").write_text("{}", encoding="utf-8")
        (model_dir / "network.onnx").write_bytes(b"onnx")
        (model_dir / "trt_info.json").write_text("{}", encoding="utf-8")
        if include_engine:
            (model_dir / "network.trt").write_bytes(b"engine")
    license_dir = root / "licenses"
    license_dir.mkdir()
    (license_dir / "THIRD_PARTY.txt").write_text("notices", encoding="utf-8")

    manifest: dict[str, object] = {
        "schema": RUNTIME_SCHEMA,
        "platform": platform_id,
        "worker": f"bin/{worker_name}",
        "trtexec": f"bin/{trtexec_name}",
        "audio2face_model": "models/audio2face/model.json",
        "audio2emotion_model": "models/audio2emotion/model.json",
        "library_directories": ["lib", "lib/audio2x"],
        "licenses": ["licenses/THIRD_PARTY.txt"],
    }
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest


def _rewrite_manifest(root: Path, manifest: dict[str, object]) -> None:
    (root / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("linux", "x86_64", "linux-x64"),
        ("win32", "AMD64", "windows-x64"),
    ],
)
def test_current_platform_id_identifies_supported_x64(
    system: str, machine: str, expected: str
) -> None:
    assert current_platform_id(system=system, machine=machine) == expected


@pytest.mark.parametrize(
    ("system", "machine"),
    [
        ("darwin", "x86_64"),
        ("linux", "aarch64"),
        ("win32", "ARM64"),
        ("linux", "amd64"),
        ("win32", "x86_64"),
        ("linux", "x64"),
        ("linux", "x86-64"),
    ],
)
def test_current_platform_id_rejects_unsupported_targets(system: str, machine: str) -> None:
    with pytest.raises(BundleError):
        current_platform_id(system=system, machine=machine)


def test_resolve_linux_bundle_returns_immutable_child_launch_spec(tmp_path: Path) -> None:
    root, _manifest = _make_bundle(tmp_path)
    source_environment = {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/system/lib", "KEEP": "yes"}
    original = dict(source_environment)

    spec = resolve_runtime_bundle(
        tmp_path,
        system="linux",
        machine="x86_64",
        environ=source_environment,
    )

    assert spec.platform == "linux-x64"
    assert spec.root == root.resolve()
    assert spec.executable == (root / "bin" / "a2f_blender_worker").resolve()
    assert spec.trtexec == (root / "bin" / "trtexec").resolve()
    assert spec.audio2face_model == (
        root / "models" / "audio2face" / "model.json"
    ).resolve()
    assert spec.audio2emotion_model == (
        root / "models" / "audio2emotion" / "model.json"
    ).resolve()
    expected_prefix = ":".join(
        str(path.resolve())
        for path in (root / "bin", root / "lib", root / "lib" / "audio2x")
    )
    assert spec.env["PATH"] == f"{expected_prefix}:/usr/bin"
    assert spec.env["LD_LIBRARY_PATH"] == f"{expected_prefix}:/system/lib"
    assert spec.env["KEEP"] == "yes"
    assert source_environment == original
    with pytest.raises(TypeError):
        spec.env["NEW"] = "value"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        spec.platform = "other"  # type: ignore[misc]


def test_resolve_windows_bundle_uses_windows_path_rules(tmp_path: Path) -> None:
    root, _manifest = _make_bundle(tmp_path, "windows-x64")
    source_environment = {"Path": r"C:\Windows\System32", "KEEP": "yes"}

    spec = resolve_runtime_bundle(
        tmp_path,
        system="win32",
        machine="AMD64",
        environ=source_environment,
    )

    expected_prefix = ";".join(
        str(path.resolve())
        for path in (root / "bin", root / "lib", root / "lib" / "audio2x")
    )
    assert spec.executable.suffix == ".exe"
    assert spec.trtexec.suffix == ".exe"
    assert spec.env["Path"] == expected_prefix + ";" + r"C:\Windows\System32"
    assert "PATH" not in spec.env
    assert "LD_LIBRARY_PATH" not in spec.env
    assert source_environment == {"Path": r"C:\Windows\System32", "KEEP": "yes"}


def test_resolver_requires_explicit_existing_writable_root(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="data root is missing"):
        resolve_runtime_bundle(tmp_path / "absent", system="linux", machine="x86_64")


def test_resolver_accepts_an_explicit_catalog_platform(tmp_path: Path) -> None:
    _make_bundle(tmp_path, "windows-x64")
    spec = resolve_runtime_bundle(
        tmp_path,
        platform="windows-x64",
        environ={},
    )
    assert spec.platform == "windows-x64"
    with pytest.raises(BundleError, match="unsupported runtime platform"):
        resolve_runtime_bundle(tmp_path, platform="darwin-x64", environ={})
    with pytest.raises(BundleError, match="does not match"):
        resolve_runtime_bundle(
            tmp_path,
            platform="windows-x64",
            system="linux",
            machine="x86_64",
            environ={},
        )


def test_prebuild_validation_allows_onnx_without_engine(tmp_path: Path) -> None:
    _make_bundle(tmp_path, include_engine=False)
    spec = resolve_runtime_bundle(
        tmp_path,
        system="linux",
        machine="x86_64",
        environ={},
        require_engine=False,
    )
    assert spec.audio2face_model.with_name("network.onnx").is_file()
    assert spec.audio2emotion_model.with_name("network.onnx").is_file()
    with pytest.raises(BundleError, match="audio2face_model"):
        resolve_runtime_bundle(
            tmp_path,
            system="linux",
            machine="x86_64",
            environ={},
        )


@pytest.mark.parametrize(
    ("model_name", "field", "missing"),
    [
        ("audio2face", "audio2face_model", "model.json"),
        ("audio2face", "audio2face_model", "network.onnx"),
        ("audio2face", "audio2face_model", "trt_info.json"),
        ("audio2emotion", "audio2emotion_model", "model.json"),
        ("audio2emotion", "audio2emotion_model", "network.onnx"),
        ("audio2emotion", "audio2emotion_model", "trt_info.json"),
    ],
)
def test_resolver_requires_every_file_for_both_models(
    tmp_path: Path,
    model_name: str,
    field: str,
    missing: str,
) -> None:
    root, _manifest = _make_bundle(tmp_path)
    (root / "models" / model_name / missing).unlink()

    with pytest.raises(BundleError, match=field):
        resolve_runtime_bundle(
            tmp_path,
            system="linux",
            machine="x86_64",
            environ={},
            require_engine=False,
        )


def test_resolver_requires_audio2emotion_engine_after_audio2face(tmp_path: Path) -> None:
    root, _manifest = _make_bundle(tmp_path)
    (root / "models/audio2emotion/network.trt").unlink()

    with pytest.raises(BundleError, match="audio2emotion_model"):
        resolve_runtime_bundle(
            tmp_path,
            system="linux",
            machine="x86_64",
            environ={},
        )


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("worker", "../bin/a2f_blender_worker"),
        ("worker", "/bin/a2f_blender_worker"),
        ("worker", r"bin\a2f_blender_worker"),
        ("worker", "bin/../a2f_blender_worker"),
        ("trtexec", "lib/trtexec"),
        ("audio2face_model", "bin/model.json"),
        ("audio2face_model", "models/other/model.json"),
        ("audio2emotion_model", "C:/models/model.json"),
        ("audio2emotion_model", "models/other/model.json"),
    ],
)
def test_manifest_rejects_unsafe_or_misplaced_paths(
    tmp_path: Path, field: str, unsafe: str
) -> None:
    root, manifest = _make_bundle(tmp_path)
    manifest[field] = unsafe
    _rewrite_manifest(root, manifest)
    with pytest.raises(BundleError, match=field):
        resolve_runtime_bundle(tmp_path, system="linux", machine="x86_64", environ={})


def test_manifest_rejects_unknown_and_missing_fields(tmp_path: Path) -> None:
    root, manifest = _make_bundle(tmp_path)
    manifest["unexpected"] = True
    del manifest["licenses"]
    _rewrite_manifest(root, manifest)
    with pytest.raises(BundleError, match="missing fields: licenses") as error:
        resolve_runtime_bundle(tmp_path, system="linux", machine="x86_64", environ={})
    assert "unknown fields: unexpected" in str(error.value)


def test_manifest_rejects_the_removed_single_model_field(tmp_path: Path) -> None:
    root, manifest = _make_bundle(tmp_path)
    del manifest["audio2face_model"]
    del manifest["audio2emotion_model"]
    manifest["default_model"] = "models/audio2face/model.json"
    _rewrite_manifest(root, manifest)

    with pytest.raises(BundleError) as error:
        resolve_runtime_bundle(
            tmp_path,
            system="linux",
            machine="x86_64",
            environ={},
        )
    assert "audio2face_model" in str(error.value)
    assert "audio2emotion_model" in str(error.value)
    assert "unknown fields: default_model" in str(error.value)


def test_manifest_rejects_schema_and_platform_mismatch(tmp_path: Path) -> None:
    root, manifest = _make_bundle(tmp_path)
    manifest["schema"] = "a2f-blender-runtime/99"
    _rewrite_manifest(root, manifest)
    with pytest.raises(BundleError, match="unsupported bundle schema"):
        resolve_runtime_bundle(tmp_path, system="linux", machine="x86_64", environ={})

    manifest["schema"] = RUNTIME_SCHEMA
    manifest["platform"] = "windows-x64"
    _rewrite_manifest(root, manifest)
    with pytest.raises(BundleError, match="does not match"):
        resolve_runtime_bundle(tmp_path, system="linux", machine="x86_64", environ={})


def test_manifest_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    root, _manifest = _make_bundle(tmp_path)
    (root / "bundle.json").write_text(
        '{"schema":"a2f-blender-runtime/2","schema":"again"}', encoding="utf-8"
    )
    with pytest.raises(BundleError, match="duplicate field 'schema'"):
        resolve_runtime_bundle(tmp_path, system="linux", machine="x86_64", environ={})


def test_linux_executables_require_x_bit_and_elf64_x64(tmp_path: Path) -> None:
    root, _manifest = _make_bundle(tmp_path)
    worker = root / "bin" / "a2f_blender_worker"
    worker.chmod(worker.stat().st_mode & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    with pytest.raises(BundleError, match="worker is not executable"):
        resolve_runtime_bundle(tmp_path, system="linux", machine="x86_64", environ={})

    _write_elf_x64(worker)
    image = bytearray(worker.read_bytes())
    struct.pack_into("<H", image, 18, 183)  # AArch64.
    worker.write_bytes(image)
    with pytest.raises(BundleError, match="ELF64 x86-64"):
        resolve_runtime_bundle(tmp_path, system="linux", machine="x86_64", environ={})


def test_windows_executables_require_pe32_plus_amd64(tmp_path: Path) -> None:
    root, _manifest = _make_bundle(tmp_path, "windows-x64")
    trtexec = root / "bin" / "trtexec.exe"
    image = bytearray(trtexec.read_bytes())
    struct.pack_into("<H", image, 128 + 4, 0xAA64)
    trtexec.write_bytes(image)
    with pytest.raises(BundleError, match=r"PE32\+ AMD64"):
        resolve_runtime_bundle(tmp_path, system="win32", machine="AMD64", environ={})


def test_manifest_members_cannot_escape_through_symlinks(tmp_path: Path) -> None:
    root, manifest = _make_bundle(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "license.txt").write_text("outside", encoding="utf-8")
    link = root / "licenses" / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    manifest["licenses"] = ["licenses/escape/license.txt"]
    _rewrite_manifest(root, manifest)
    with pytest.raises(BundleError, match="escapes"):
        resolve_runtime_bundle(tmp_path, system="linux", machine="x86_64", environ={})


def test_model_companions_cannot_escape_through_symlinks(tmp_path: Path) -> None:
    root, _manifest = _make_bundle(tmp_path)
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"outside")
    network = root / "models/audio2emotion/network.onnx"
    network.unlink()
    try:
        network.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(BundleError, match="audio2emotion_model.*escapes"):
        resolve_runtime_bundle(
            tmp_path,
            system="linux",
            machine="x86_64",
            environ={},
            require_engine=False,
        )


def test_runtime_root_cannot_escape_through_a_symlink(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside_root = tmp_path / "outside"
    _make_bundle(outside_root)
    try:
        (data_root / "runtime").symlink_to(
            outside_root / "runtime", target_is_directory=True
        )
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(BundleError, match="runtime root escapes"):
        resolve_runtime_bundle(
            data_root,
            system="linux",
            machine="x86_64",
            environ={},
        )


def test_environment_values_must_be_strings(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    with pytest.raises(BundleError, match="keys and values must be strings"):
        resolve_runtime_bundle(
            tmp_path,
            system="linux",
            machine="x86_64",
            environ={"PATH": 1},  # type: ignore[dict-item]
        )
