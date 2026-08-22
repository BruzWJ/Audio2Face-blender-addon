from __future__ import annotations

import json
import re
import stat
import struct
from dataclasses import FrozenInstanceError
from pathlib import Path, PureWindowsPath

import pytest

import audio2face.runtime_bundle as runtime_bundle
from audio2face.runtime_bundle import (
    BundleError,
    RUNTIME_SCHEMA,
    current_platform_id,
    resolve_runtime_bundle,
)
from audio2face.runtime_contract import runtime_contract


def _set_host(
    monkeypatch: pytest.MonkeyPatch,
    *,
    system: str,
    machine: str,
    environment: dict[object, object] | None = None,
) -> None:
    monkeypatch.setattr(runtime_bundle.sys, "platform", system)
    monkeypatch.setattr(runtime_bundle.host_platform, "machine", lambda: machine)
    if system == "win32" and runtime_bundle.os.name != "nt":
        monkeypatch.setattr(
            runtime_bundle,
            "_require_windows_directory",
            lambda value, _description: PureWindowsPath(value),
        )
    if environment is not None:
        monkeypatch.setattr(runtime_bundle.os, "environ", environment)


@pytest.fixture
def package_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "audio2face"
    root.mkdir()
    (root / "runtime_bundle.py").write_text("# test module\n", encoding="utf-8")
    monkeypatch.setattr(runtime_bundle, "__file__", str(root / "runtime_bundle.py"))
    _set_host(
        monkeypatch,
        system="linux",
        machine="x86_64",
        environment={
            "PATH": "/usr/local/cuda/bin:/usr/bin",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
            "LD_PRELOAD": "/ambient/libnvinfer.so",
            "LD_AUDIT": "/ambient/audit.so",
            "CUDA_HOME": "/ambient/cuda",
            "CUDA_PATH_V12_9": "/ambient/cuda-12.9",
            "CUDA_ROOT": "/ambient/cuda-root",
            "CUDA_BIN_PATH": "/ambient/cuda-bin",
            "CUDA_TOOLKIT_ROOT_DIR": "/ambient/cuda-toolkit",
            "CUDATOOLKIT_ROOT": "/ambient/cudatoolkit",
            "TENSORRT_ROOT": "/ambient/tensorrt",
            "TENSORRT_ROOT_DIR": "/ambient/tensorrt-root",
            "TRT_LIB_DIR": "/ambient/tensorrt-lib",
            "TRT_OUT_DIR": "/ambient/tensorrt-out",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_CACHE_PATH": "/driver/cache",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "KEEP": "yes",
        },
    )
    return root


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
    package_root: Path,
    platform_id: str = "linux-x64",
) -> tuple[Path, dict[str, object]]:
    root = package_root / "runtime"
    root.mkdir(parents=True)
    contract = runtime_contract(platform_id)
    windows = platform_id == "windows-x64"
    worker_name = Path(contract.worker).name
    trtexec_name = Path(contract.trtexec).name
    worker = root / "bin" / worker_name
    trtexec = root / "bin" / trtexec_name
    if windows:
        _write_pe_x64(worker)
        _write_pe_x64(trtexec)
    else:
        _write_elf_x64(worker)
        _write_elf_x64(trtexec)

    for packaged_file in contract.libraries:
        path = root / packaged_file.path
        if windows:
            _write_pe_x64(path)
        else:
            _write_elf_x64(path, executable=False)
    for packaged_file in contract.licenses:
        path = root / packaged_file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("notice", encoding="utf-8")

    manifest = contract.manifest()
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
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    expected: str,
) -> None:
    _set_host(monkeypatch, system=system, machine=machine)
    assert current_platform_id() == expected


@pytest.mark.parametrize(
    ("system", "machine"),
    [
        ("darwin", "x86_64"),
        ("linux", "aarch64"),
        ("win32", "ARM64"),
        ("win32", "amd64"),
        ("linux", "amd64"),
        ("win32", "x86_64"),
        ("linux", "x64"),
        ("linux", "x86-64"),
    ],
)
def test_current_platform_id_rejects_unsupported_targets(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
) -> None:
    _set_host(monkeypatch, system=system, machine=machine)
    with pytest.raises(BundleError):
        current_platform_id()


def test_resolve_linux_bundle_is_package_local_and_immutable(
    package_root: Path,
) -> None:
    root, _manifest = _make_bundle(package_root)
    source_environment = runtime_bundle.os.environ
    original = dict(source_environment)

    spec = resolve_runtime_bundle()

    assert spec.platform == "linux-x64"
    assert spec.root == root.resolve()
    assert spec.executable == (root / "bin" / "audio2face_worker").resolve()
    assert spec.trtexec == (root / "bin" / "trtexec").resolve()
    assert spec.env["PATH"] == str((root / "bin").resolve())
    assert spec.env["LD_LIBRARY_PATH"] == str((root / "lib").resolve())
    assert dict(spec.env) == {
        "PATH": str((root / "bin").resolve()),
        "LD_LIBRARY_PATH": str((root / "lib").resolve()),
    }
    assert source_environment == original
    with pytest.raises(TypeError):
        spec.env["NEW"] = "value"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        spec.platform = "other"  # type: ignore[misc]


def test_resolve_windows_bundle_overwrites_host_search_paths(
    package_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _manifest = _make_bundle(package_root, "windows-x64")
    source_environment = {
        "Path": r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
        "LD_LIBRARY_PATH": r"C:\TensorRT\lib",
        "LD_PRELOAD": r"C:\TensorRT\inject.dll",
        "LD_AUDIT": r"C:\TensorRT\audit.dll",
        "CUDA_HOME": r"C:\ambient\cuda",
        "CUDA_PATH_V12_9": r"C:\ambient\cuda-12.9",
        "TENSORRT_ROOT_DIR": r"C:\ambient\TensorRT",
        "TRT_LIB_DIR": r"C:\ambient\TensorRT\lib",
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_CACHE_PATH": r"C:\driver-cache",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        "SystemRoot": r"C:\Windows",
        "KEEP": "yes",
    }
    _set_host(
        monkeypatch,
        system="win32",
        machine="AMD64",
        environment=source_environment,
    )

    spec = resolve_runtime_bundle()

    assert spec.executable.suffix == ".exe"
    assert spec.trtexec.suffix == ".exe"
    assert not (root / "lib").exists()
    assert all(
        (root / relative).parent == root / "bin"
        for relative in (
            packaged_file.path
            for packaged_file in runtime_contract("windows-x64").libraries
        )
    )
    assert spec.env["PATH"] == r"C:\Windows\System32"
    assert dict(spec.env) == {
        "SystemRoot": r"C:\Windows",
        "PATH": r"C:\Windows\System32",
    }
    assert "NVIDIA GPU Computing Toolkit" not in spec.env["PATH"]
    assert source_environment["Path"].startswith(r"C:\Program Files")


def test_windows_bundle_requires_the_canonical_system_root(
    package_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_bundle(package_root, "windows-x64")
    _set_host(
        monkeypatch,
        system="win32",
        machine="AMD64",
        environment={"PATH": r"C:\Windows\System32"},
    )
    with pytest.raises(BundleError, match="SystemRoot"):
        resolve_runtime_bundle()


@pytest.mark.parametrize(
    "system_root",
    [
        r"Windows",
        r"C:/Windows",
        "C:\\Windows\\.",
        "C:\\Windows\\",
        r"C:\Windows\\System32\..\Windows",
        r"\\server\share\Windows",
    ],
)
def test_windows_bundle_rejects_noncanonical_system_root_spelling(
    package_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    system_root: str,
) -> None:
    _make_bundle(package_root, "windows-x64")
    _set_host(
        monkeypatch,
        system="win32",
        machine="AMD64",
        environment={"SystemRoot": system_root},
    )

    with pytest.raises(BundleError, match="canonical absolute path"):
        resolve_runtime_bundle()


def test_resolver_requires_the_package_local_runtime(package_root: Path) -> None:
    with pytest.raises(BundleError, match="bundled linux-x64 runtime is missing"):
        resolve_runtime_bundle()


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("worker", "../bin/audio2face_worker"),
        ("worker", "/bin/audio2face_worker"),
        ("worker", r"bin\audio2face_worker"),
        ("worker", "bin/../audio2face_worker"),
        ("trtexec", "lib/trtexec"),
    ],
)
def test_manifest_rejects_unsafe_or_misplaced_paths(
    package_root: Path,
    field: str,
    unsafe: str,
) -> None:
    root, manifest = _make_bundle(package_root)
    manifest[field] = unsafe
    _rewrite_manifest(root, manifest)
    with pytest.raises(BundleError, match=field):
        resolve_runtime_bundle()


def test_manifest_rejects_unknown_and_missing_fields(package_root: Path) -> None:
    root, manifest = _make_bundle(package_root)
    manifest["unexpected"] = True
    del manifest["licenses"]
    _rewrite_manifest(root, manifest)
    with pytest.raises(BundleError, match="missing fields: licenses") as error:
        resolve_runtime_bundle()
    assert "unknown fields: unexpected" in str(error.value)


def test_manifest_rejects_schema_and_host_platform_mismatch(
    package_root: Path,
) -> None:
    root, manifest = _make_bundle(package_root)
    manifest["schema"] = "audio2face-runtime/99"
    _rewrite_manifest(root, manifest)
    with pytest.raises(BundleError, match="unsupported bundle schema"):
        resolve_runtime_bundle()

    manifest["schema"] = RUNTIME_SCHEMA
    manifest["platform"] = "windows-x64"
    _rewrite_manifest(root, manifest)
    with pytest.raises(BundleError, match="does not match host"):
        resolve_runtime_bundle()


def test_manifest_rejects_duplicate_json_fields(package_root: Path) -> None:
    root, _manifest = _make_bundle(package_root)
    (root / "bundle.json").write_text(
        '{"schema":"audio2face-runtime/3","schema":"again"}', encoding="utf-8"
    )
    with pytest.raises(BundleError, match="duplicate field 'schema'"):
        resolve_runtime_bundle()


def test_linux_executables_restore_owner_x_bit_after_elf_validation(
    package_root: Path,
) -> None:
    root, _manifest = _make_bundle(package_root)
    worker = root / "bin" / "audio2face_worker"
    trtexec = root / "bin" / "trtexec"
    executables = (worker, trtexec)
    for executable in executables:
        executable.chmod(
            executable.stat().st_mode
            & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        )
    resolve_runtime_bundle()
    assert all(
        stat.S_IMODE(path.stat().st_mode) & stat.S_IXUSR
        for path in executables
    )

    _write_elf_x64(worker, executable=False)
    image = bytearray(worker.read_bytes())
    struct.pack_into("<H", image, 18, 183)
    worker.write_bytes(image)
    with pytest.raises(BundleError, match="ELF64 x86-64"):
        resolve_runtime_bundle()
    assert not (stat.S_IMODE(worker.stat().st_mode) & stat.S_IXUSR)


def test_linux_execute_bit_repair_reports_read_only_install(
    package_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _manifest = _make_bundle(package_root)
    worker = root / "bin" / "audio2face_worker"
    worker.chmod(worker.stat().st_mode & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))

    original_chmod = Path.chmod

    def reject_worker(path: Path, mode: int) -> None:
        if path == worker:
            raise PermissionError("read-only extension repository")
        original_chmod(path, mode)

    monkeypatch.setattr(Path, "chmod", reject_worker)
    with pytest.raises(BundleError, match="restore.*worker execute bit"):
        resolve_runtime_bundle()


def test_windows_executables_require_pe32_plus_amd64(
    package_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _manifest = _make_bundle(package_root, "windows-x64")
    _set_host(
        monkeypatch,
        system="win32",
        machine="AMD64",
        environment={"SystemRoot": r"C:\Windows"},
    )
    trtexec = root / "bin" / "trtexec.exe"
    image = bytearray(trtexec.read_bytes())
    struct.pack_into("<H", image, 128 + 4, 0xAA64)
    trtexec.write_bytes(image)
    with pytest.raises(BundleError, match=r"PE32\+ AMD64"):
        resolve_runtime_bundle()


@pytest.mark.parametrize(
    ("platform_id", "library"),
    [
        (platform_id, library)
        for platform_id in ("linux-x64", "windows-x64")
        for library in (
            packaged_file.path
            for packaged_file in runtime_contract(platform_id).libraries
        )
    ],
)
def test_every_contract_library_requires_its_native_binary_format(
    package_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform_id: str,
    library: str,
) -> None:
    root, _manifest = _make_bundle(package_root, platform_id)
    if platform_id == "windows-x64":
        _set_host(
            monkeypatch,
            system="win32",
            machine="AMD64",
            environment={"SystemRoot": r"C:\Windows"},
        )
    (root / library).write_bytes(b"corrupt native library")

    with pytest.raises(BundleError, match=re.escape(f"library {Path(library).name}")):
        resolve_runtime_bundle()


@pytest.mark.parametrize("platform_id", ["linux-x64", "windows-x64"])
def test_contract_libraries_reject_the_wrong_x64_architecture(
    package_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform_id: str,
) -> None:
    root, _manifest = _make_bundle(package_root, platform_id)
    if platform_id == "windows-x64":
        _set_host(
            monkeypatch,
            system="win32",
            machine="AMD64",
            environment={"SystemRoot": r"C:\Windows"},
        )
    library = root / runtime_contract(platform_id).libraries[0].path
    image = bytearray(library.read_bytes())
    if platform_id == "linux-x64":
        struct.pack_into("<H", image, 18, 183)
        expected = "ELF64 x86-64"
    else:
        pe_offset = struct.unpack_from("<I", image, 0x3C)[0]
        struct.pack_into("<H", image, pe_offset + 4, 0xAA64)
        expected = r"PE32\+ AMD64"
    library.write_bytes(image)

    with pytest.raises(BundleError, match=expected):
        resolve_runtime_bundle()


def test_linux_contract_libraries_reject_big_endian_elf(
    package_root: Path,
) -> None:
    root, _manifest = _make_bundle(package_root)
    library = root / runtime_contract("linux-x64").libraries[0].path
    image = bytearray(library.read_bytes())
    image[5] = 2
    library.write_bytes(image)

    with pytest.raises(BundleError, match="little-endian"):
        resolve_runtime_bundle()


def test_manifest_members_cannot_escape_through_symlinks(
    package_root: Path,
    tmp_path: Path,
) -> None:
    root, manifest = _make_bundle(package_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "license.txt").write_text("outside", encoding="utf-8")
    link = root / "licenses" / "audio2face-LICENSE.txt"
    link.unlink()
    try:
        link.symlink_to(outside / "license.txt")
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(BundleError, match="filesystem alias"):
        resolve_runtime_bundle()


@pytest.mark.parametrize("directory", ["bin", "lib", "licenses"])
def test_runtime_rejects_extra_files(
    package_root: Path,
    directory: str,
) -> None:
    root, _manifest = _make_bundle(package_root)
    (root / directory / "unexpected").write_bytes(b"not canonical")
    with pytest.raises(BundleError, match=f"bundle {directory} must contain exactly"):
        resolve_runtime_bundle()


def test_runtime_rejects_extra_root_entries(package_root: Path) -> None:
    root, _manifest = _make_bundle(package_root)
    (root / "unexpected").mkdir()
    with pytest.raises(BundleError, match="bundle root must contain exactly"):
        resolve_runtime_bundle()


def test_runtime_root_cannot_escape_through_a_symlink(
    package_root: Path,
    tmp_path: Path,
) -> None:
    outside_package = tmp_path / "outside-package"
    outside_package.mkdir()
    outside_runtime, _manifest = _make_bundle(outside_package)
    try:
        (package_root / "runtime").symlink_to(
            outside_runtime,
            target_is_directory=True,
        )
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(BundleError, match="filesystem alias"):
        resolve_runtime_bundle()


def test_runtime_rejects_an_aliased_package_ancestor(
    package_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_bundle(package_root)
    alias = tmp_path / "audio2face-alias"
    try:
        alias.symlink_to(package_root, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    monkeypatch.setattr(runtime_bundle, "__file__", str(alias / "runtime_bundle.py"))

    with pytest.raises(BundleError, match="filesystem alias"):
        resolve_runtime_bundle()


def test_linux_child_environment_does_not_consume_ambient_values(
    package_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_bundle(package_root)
    _set_host(
        monkeypatch,
        system="linux",
        machine="x86_64",
        environment={"PATH": 1},
    )
    spec = resolve_runtime_bundle()
    assert set(spec.env) == {"PATH", "LD_LIBRARY_PATH"}
