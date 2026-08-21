"""Resolve the native runtime bundled inside this Blender extension.

The installed extension contains exactly one platform runtime at ``runtime/``.
This module never searches writable extension data, ``PATH``, a CUDA toolkit,
or another Audio2Face installation.  Model repositories remain at the exact
paths selected in Add-on Preferences and are attached only after the bundled
runtime has passed validation.
"""

from __future__ import annotations

import json
import ntpath
import os
import platform as host_platform
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping

from .path_contract import require_unaliased_path
from .runtime_contract import (
    RUNTIME_MANIFEST_FIELDS,
    RUNTIME_SCHEMA,
    RuntimePlatformContract,
    runtime_contract,
)
from .strict_json import duplicate_key_hook, invalid_constant_hook


class BundleError(RuntimeError):
    """Raised when this extension's bundled runtime is absent or unsafe."""


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    """Immutable, fully validated package-local native runtime."""

    platform: str
    root: Path
    executable: Path
    trtexec: Path
    env: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))


@dataclass(frozen=True, slots=True)
class RuntimeModelSpec:
    """One validated bundled runtime paired with two selected NVIDIA models."""

    runtime: RuntimeBundle
    audio2face_model: Path
    audio2emotion_model: Path


def current_platform_id() -> str:
    """Return the one supported runtime identifier for the current host."""

    system = sys.platform
    machine = host_platform.machine()
    if system == "linux" and machine == "x86_64":
        return "linux-x64"
    if system == "win32" and machine == "AMD64":
        return "windows-x64"
    if system not in {"linux", "win32"}:
        raise BundleError(
            "Audio2Face bundled inference supports Linux and Windows only; "
            f"detected system {system!r}"
        )
    raise BundleError(
        "Audio2Face bundled inference supports x86-64 only; "
        f"detected machine {machine!r}"
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    path = require_unaliased_path(
        path,
        description="bundled runtime manifest",
        error_type=BundleError,
    )
    try:
        if not path.is_file():
            raise BundleError(f"bundle manifest is not a regular file: {path}")
        size = path.stat().st_size
        if size < 1 or size > 64 * 1024:
            raise BundleError(f"bundle manifest is unexpectedly large: {path}")
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=duplicate_key_hook(
                BundleError,
                "bundle manifest",
            ),
            parse_constant=invalid_constant_hook(
                BundleError,
                "bundle manifest",
            ),
        )
    except FileNotFoundError as exc:
        raise BundleError(f"bundled runtime manifest is missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"could not read bundled runtime manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError("bundle manifest root must be an object")
    keys = frozenset(value)
    missing = sorted(RUNTIME_MANIFEST_FIELDS - keys)
    unknown = sorted(keys - RUNTIME_MANIFEST_FIELDS)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise BundleError(f"invalid bundle manifest ({'; '.join(details)})")
    return value


def _directory_entries(path: Path, expected: frozenset[str], label: str) -> dict[str, Path]:
    path = require_unaliased_path(
        path,
        description=f"bundle {label}",
        error_type=BundleError,
    )
    try:
        if not path.is_dir():
            raise BundleError(f"bundle {label} is not a real directory: {path}")
        entries = {entry.name: entry for entry in path.iterdir()}
    except OSError as exc:
        raise BundleError(f"bundle {label} is inaccessible: {path}") from exc
    if frozenset(entries) != expected:
        raise BundleError(
            f"bundle {label} must contain exactly {sorted(expected)}; "
            f"found {sorted(entries)}"
        )
    for name, entry in entries.items():
        entry = require_unaliased_path(
            entry,
            description=f"bundle {label}/{name}",
            error_type=BundleError,
        )
        entries[name] = entry
        try:
            if not entry.is_file() or entry.stat().st_size < 1:
                raise BundleError(
                    f"bundle {label}/{name} is not a non-empty regular file"
                )
        except OSError as exc:
            raise BundleError(f"bundle {label}/{name} is inaccessible") from exc
    return entries


def _validate_runtime_tree(
    root: Path,
    contract: RuntimePlatformContract,
) -> dict[str, dict[str, Path]]:
    try:
        root_entries = {entry.name for entry in root.iterdir()}
    except OSError as exc:
        raise BundleError(f"bundled runtime root is inaccessible: {root}") from exc
    if root_entries != contract.root_entries:
        raise BundleError(
            f"bundle root must contain exactly {sorted(contract.root_entries)}; "
            f"found {sorted(root_entries)}"
        )
    directories = {
        "bin": _directory_entries(root / "bin", contract.bin_entries, "bin"),
        "licenses": _directory_entries(
            root / "licenses", contract.license_entries, "licenses"
        ),
    }
    if "lib" in contract.root_entries:
        directories["lib"] = _directory_entries(
            root / "lib", contract.library_entries, "lib"
        )
    return directories


def _validate_manifest(
    manifest: dict[str, Any],
    contract: RuntimePlatformContract,
) -> None:
    expected = contract.manifest()
    if manifest["schema"] != RUNTIME_SCHEMA:
        raise BundleError(
            f"unsupported bundle schema {manifest['schema']!r}; expected {RUNTIME_SCHEMA!r}"
        )
    if manifest["platform"] != contract.platform:
        raise BundleError(
            f"bundle platform {manifest['platform']!r} does not match host "
            f"{contract.platform!r}"
        )
    for field in ("worker", "trtexec", "library_directories", "licenses"):
        if manifest[field] != expected[field]:
            raise BundleError(
                f"bundle manifest {field} does not match the exact "
                f"{contract.platform} runtime contract"
            )


def _validate_elf_x64(path: Path, label: str) -> None:
    try:
        with path.open("rb") as stream:
            header = stream.read(20)
    except OSError as exc:
        raise BundleError(f"could not inspect bundle {label}: {path}") from exc
    if (
        len(header) < 20
        or header[:4] != b"\x7fELF"
        or header[4] != 2
        or header[5] != 1
        or header[6] != 1
        or struct.unpack_from("<H", header, 18)[0] != 62
    ):
        raise BundleError(
            f"bundle {label} is not a Linux ELF64 x86-64 little-endian binary"
        )


def _validate_pe_x64(path: Path, label: str) -> None:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            dos_header = stream.read(64)
            if len(dos_header) < 64 or dos_header[:2] != b"MZ":
                raise BundleError(
                    f"bundle {label} is not a Windows PE32+ AMD64 binary"
                )
            pe_offset = struct.unpack_from("<I", dos_header, 0x3C)[0]
            if pe_offset < 64 or pe_offset > size - 26:
                raise BundleError(
                    f"bundle {label} is not a Windows PE32+ AMD64 binary"
                )
            stream.seek(pe_offset)
            pe_header = stream.read(26)
    except OSError as exc:
        raise BundleError(f"could not inspect bundle {label}: {path}") from exc
    if (
        len(pe_header) != 26
        or pe_header[:4] != b"PE\x00\x00"
        or struct.unpack_from("<H", pe_header, 4)[0] != 0x8664
        or struct.unpack_from("<H", pe_header, 24)[0] != 0x20B
    ):
        raise BundleError(f"bundle {label} is not a Windows PE32+ AMD64 binary")


def _validate_native_binary(path: Path, platform_id: str, label: str) -> None:
    if platform_id == "linux-x64":
        _validate_elf_x64(path, label)
        return
    if platform_id == "windows-x64":
        _validate_pe_x64(path, label)
        return
    raise BundleError(f"cannot validate native binary for platform {platform_id!r}")


def _validate_executable(path: Path, platform_id: str, label: str) -> None:
    if platform_id == "linux-x64":
        if not os.access(path, os.X_OK):
            raise BundleError(f"bundle {label} is not executable: {path}")
    _validate_native_binary(path, platform_id, label)


def _require_windows_directory(value: str, description: str) -> Path:
    path = require_unaliased_path(
        value,
        description=description,
        error_type=BundleError,
    )
    if not path.is_dir():
        raise BundleError(f"{description} is not a directory: {path}")
    return path


def _windows_system_directories(source: Mapping[str, str]) -> tuple[str, str]:
    value = source.get("SystemRoot")
    if type(value) is not str or not value:
        raise BundleError("Windows SystemRoot is unavailable")
    lexical = PureWindowsPath(value)
    if (
        not lexical.is_absolute()
        or len(lexical.drive) != 2
        or lexical.drive[1] != ":"
        or not lexical.drive[0].isalpha()
        or value != ntpath.normpath(value)
    ):
        raise BundleError("Windows SystemRoot must be one canonical absolute path")
    system_root = _require_windows_directory(value, "Windows SystemRoot")
    system32 = _require_windows_directory(
        str(system_root / "System32"),
        "Windows System32",
    )
    return str(system_root), str(system32)


def _child_environment(
    executable_directory: Path,
    library_directory: Path,
    platform_id: str,
) -> Mapping[str, str]:
    """Return the exact package-local child environment for one platform."""

    source = os.environ
    if platform_id == "windows-x64":
        if library_directory != executable_directory:
            raise BundleError("Windows runtime libraries must be beside the executables")
        system_root, system32 = _windows_system_directories(source)
        return {
            "SystemRoot": system_root,
            "PATH": system32,
        }
    if platform_id == "linux-x64":
        if library_directory == executable_directory:
            raise BundleError("Linux runtime libraries must use the lib directory")
        return {
            "PATH": str(executable_directory),
            "LD_LIBRARY_PATH": str(library_directory),
        }
    raise BundleError(f"cannot build child environment for {platform_id!r}")


def resolve_runtime_bundle() -> RuntimeBundle:
    """Validate and return this installed extension's one bundled runtime."""

    platform_id = current_platform_id()
    contract = runtime_contract(platform_id)
    package_module = require_unaliased_path(
        __file__,
        description="Audio2Face runtime module",
        error_type=BundleError,
    )
    if not package_module.is_file():
        raise BundleError(
            f"Audio2Face runtime module is not a regular file: {package_module}"
        )
    package_root = package_module.parent
    runtime_path = package_root / "runtime"
    root = require_unaliased_path(
        runtime_path,
        description=f"bundled {platform_id} runtime",
        error_type=BundleError,
    )
    if not root.is_dir():
        raise BundleError(f"bundled runtime root is not a directory: {root}")

    directories = _validate_runtime_tree(root, contract)
    manifest = _read_manifest(root / "bundle.json")
    _validate_manifest(manifest, contract)

    executable = directories["bin"][Path(contract.worker).name]
    trtexec = directories["bin"][Path(contract.trtexec).name]
    _validate_executable(executable, platform_id, "worker")
    _validate_executable(trtexec, platform_id, "trtexec")
    for packaged_file in contract.libraries:
        name = Path(packaged_file.path).name
        directory_name = Path(packaged_file.path).parent.name
        path = directories[directory_name][name]
        _validate_native_binary(path, platform_id, f"library {name}")

    library_directory = root / contract.library_directories[0]
    environment = _child_environment(
        executable.parent,
        library_directory,
        platform_id,
    )
    return RuntimeBundle(
        platform=platform_id,
        root=root,
        executable=executable,
        trtexec=trtexec,
        env=environment,
    )
