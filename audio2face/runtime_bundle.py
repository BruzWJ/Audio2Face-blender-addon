"""Resolve an installed, self-contained Audio2Face native runtime bundle.

This module deliberately has no :mod:`bpy` dependency.  Blender-facing code is
responsible for supplying the extension's writable data root; this resolver
never searches the extension sources, ``PATH``, or any system installation.
"""

from __future__ import annotations

import json
import os
import platform as host_platform
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping


RUNTIME_SCHEMA = "audio2face-runtime/2"
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "platform",
        "worker",
        "trtexec",
        "audio2face_model",
        "audio2emotion_model",
        "library_directories",
        "licenses",
    }
)


class BundleError(RuntimeError):
    """Raised when an installed runtime bundle is absent or unsafe to launch."""


@dataclass(frozen=True, slots=True)
class BundleLaunchSpec:
    """Immutable, fully validated child-process launch configuration."""

    platform: str
    root: Path
    executable: Path
    trtexec: Path
    env: Mapping[str, str]
    audio2face_model: Path
    audio2emotion_model: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))


def current_platform_id(
    *, system: str | None = None, machine: str | None = None
) -> str:
    """Return the normalized runtime catalog platform identifier."""

    raw_system = system if system is not None else sys.platform
    raw_machine = (
        machine if machine is not None else host_platform.machine()
    ).lower()
    expected_machine = {"linux": "x86_64", "win32": "amd64"}.get(raw_system)
    if expected_machine is None:
        raise BundleError(
            "Audio2Face bundled inference supports Linux and Windows only; "
            f"detected system {raw_system!r}"
        )
    if raw_machine != expected_machine:
        raise BundleError(
            "Audio2Face bundled inference supports x86-64 only; "
            f"detected machine {raw_machine!r}"
        )
    if raw_system == "linux":
        return "linux-x64"
    return "windows-x64"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"bundle manifest contains duplicate field {key!r}")
        result[key] = value
    return result


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > 64 * 1024:
            raise BundleError(f"bundle manifest is unexpectedly large: {path}")
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                BundleError(f"bundle manifest contains invalid number {token}")
            ),
        )
    except BundleError:
        raise
    except FileNotFoundError as exc:
        raise BundleError(f"installed runtime manifest is missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"could not read installed runtime manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError("bundle manifest root must be an object")
    keys = frozenset(value)
    missing = sorted(_MANIFEST_FIELDS - keys)
    unknown = sorted(keys - _MANIFEST_FIELDS)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise BundleError(f"invalid bundle manifest ({'; '.join(details)})")
    return value


def _relative_path(value: Any, label: str, required_directory: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise BundleError(f"bundle manifest {label} must be a non-empty string")
    if "\x00" in value or "\\" in value:
        raise BundleError(f"bundle manifest {label} must be a portable relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BundleError(f"bundle manifest {label} must be a canonical relative path")
    if not path.parts or path.parts[0] != required_directory:
        raise BundleError(
            f"bundle manifest {label} must be inside {required_directory}/"
        )
    return path


def _resolve_member(
    root: Path,
    value: Any,
    label: str,
    required_directory: str,
    *,
    directory: bool,
) -> Path:
    relative = _relative_path(value, label, required_directory)
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BundleError(f"bundle {label} is missing or inaccessible: {candidate}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise BundleError(f"bundle {label} escapes the installed runtime root") from exc
    if directory:
        if not resolved.is_dir():
            raise BundleError(f"bundle {label} is not a directory: {resolved}")
    elif not resolved.is_file():
        raise BundleError(f"bundle {label} is not a file: {resolved}")
    return resolved


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
        raise BundleError(f"bundle {label} is not a Linux ELF64 x86-64 executable")


def _validate_pe_x64(path: Path, label: str) -> None:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            dos_header = stream.read(64)
            if len(dos_header) < 64 or dos_header[:2] != b"MZ":
                raise BundleError(
                    f"bundle {label} is not a Windows PE32+ AMD64 executable"
                )
            pe_offset = struct.unpack_from("<I", dos_header, 0x3C)[0]
            if pe_offset < 64 or pe_offset > size - 26:
                raise BundleError(
                    f"bundle {label} is not a Windows PE32+ AMD64 executable"
                )
            stream.seek(pe_offset)
            pe_header = stream.read(26)
    except BundleError:
        raise
    except OSError as exc:
        raise BundleError(f"could not inspect bundle {label}: {path}") from exc
    if (
        len(pe_header) != 26
        or pe_header[:4] != b"PE\x00\x00"
        or struct.unpack_from("<H", pe_header, 4)[0] != 0x8664
        or struct.unpack_from("<H", pe_header, 24)[0] != 0x20B
    ):
        raise BundleError(f"bundle {label} is not a Windows PE32+ AMD64 executable")


def _validate_executable(path: Path, platform_id: str, label: str) -> None:
    if platform_id == "linux-x64":
        if not os.access(path, os.X_OK):
            raise BundleError(f"bundle {label} is not executable: {path}")
        _validate_elf_x64(path, label)
    else:
        if path.suffix.lower() != ".exe":
            raise BundleError(f"bundle {label} must have an .exe suffix on Windows")
        _validate_pe_x64(path, label)


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise BundleError(f"bundle manifest {label} must be a non-empty array")
    if not all(isinstance(item, str) and item for item in value):
        raise BundleError(f"bundle manifest {label} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise BundleError(f"bundle manifest {label} contains duplicate paths")
    return value


def _prepend_environment(
    source: Mapping[str, str], directories: list[Path], platform_id: str
) -> Mapping[str, str]:
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in source.items()):
        raise BundleError("child environment keys and values must be strings")
    child = dict(source)
    separator = ";" if platform_id == "windows-x64" else ":"

    def update(name: str) -> None:
        key = name
        if platform_id == "windows-x64":
            key = next((item for item in child if item.upper() == name), name)
        prefix = separator.join(str(path) for path in directories)
        current = child.get(key, "")
        child[key] = f"{prefix}{separator}{current}" if current else prefix

    update("PATH")
    if platform_id == "linux-x64":
        update("LD_LIBRARY_PATH")
    return child


def _resolve_model(
    root: Path,
    manifest: Mapping[str, Any],
    field: str,
    expected_relative: str,
    *,
    require_engine: bool,
) -> Path:
    """Resolve and validate one canonical managed model payload."""

    if manifest[field] != expected_relative:
        raise BundleError(
            f"bundle manifest {field} must be exactly {expected_relative!r}"
        )
    expected_path = PurePosixPath(expected_relative)
    model = _resolve_member(
        root, manifest[field], field, "models", directory=False
    )
    for filename in ("network.onnx", "trt_info.json"):
        sibling = _resolve_member(
            root,
            (expected_path.parent / filename).as_posix(),
            f"{field}.{filename}",
            "models",
            directory=False,
        )
        if sibling.parent != model.parent:
            raise BundleError(
                f"bundled {field} {filename} is not beside model.json"
            )
    if require_engine:
        engine_path = (expected_path.parent / "network.trt").as_posix()
        engine_candidate = root.joinpath(*PurePosixPath(engine_path).parts)
        if not engine_candidate.is_file():
            raise BundleError(
                f"GPU-specific TensorRT engine has not been built for "
                f"{field}: {engine_candidate}"
            )
        engine = _resolve_member(
            root,
            engine_path,
            f"{field}.network.trt",
            "models",
            directory=False,
        )
        if engine.parent != model.parent:
            raise BundleError(
                f"bundled {field} network.trt is not beside model.json"
            )
    return model


def resolve_runtime_bundle(
    package_root: Path,
    *,
    platform: str | None = None,
    system: str | None = None,
    machine: str | None = None,
    environ: Mapping[str, str] | None = None,
    require_engine: bool = True,
) -> BundleLaunchSpec:
    """Validate the installed bundle beneath ``package_root/runtime``.

    ``package_root`` must be Blender's writable per-extension data directory.
    No implicit source-tree or system-path fallback is attempted.
    """

    if platform is None:
        platform_id = current_platform_id(system=system, machine=machine)
    else:
        if platform not in {"linux-x64", "windows-x64"}:
            raise BundleError(f"unsupported runtime platform {platform!r}")
        platform_id = platform
        if system is not None or machine is not None:
            detected = current_platform_id(system=system, machine=machine)
            if detected != platform_id:
                raise BundleError(
                    f"explicit runtime platform {platform_id!r} does not match {detected!r}"
                )
    try:
        base = Path(package_root).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BundleError(f"runtime data root is missing or inaccessible: {package_root}") from exc
    if not base.is_dir():
        raise BundleError(f"runtime data root is not a directory: {base}")
    runtime_root = base / "runtime" / platform_id
    try:
        root = runtime_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BundleError(f"installed {platform_id} runtime is missing: {runtime_root}") from exc
    try:
        root.relative_to(base)
    except ValueError as exc:
        raise BundleError(
            f"installed runtime root escapes the extension data root: {runtime_root}"
        ) from exc
    if not root.is_dir():
        raise BundleError(f"installed runtime root is not a directory: {root}")

    manifest = _read_manifest(root / "bundle.json")
    if manifest["schema"] != RUNTIME_SCHEMA:
        raise BundleError(
            f"unsupported bundle schema {manifest['schema']!r}; expected {RUNTIME_SCHEMA!r}"
        )
    if manifest["platform"] != platform_id:
        raise BundleError(
            f"bundle platform {manifest['platform']!r} does not match {platform_id!r}"
        )

    executable = _resolve_member(root, manifest["worker"], "worker", "bin", directory=False)
    trtexec = _resolve_member(root, manifest["trtexec"], "trtexec", "bin", directory=False)
    audio2face_model = _resolve_model(
        root,
        manifest,
        "audio2face_model",
        "models/audio2face/model.json",
        require_engine=require_engine,
    )
    audio2emotion_model = _resolve_model(
        root,
        manifest,
        "audio2emotion_model",
        "models/audio2emotion/model.json",
        require_engine=require_engine,
    )
    _validate_executable(executable, platform_id, "worker")
    _validate_executable(trtexec, platform_id, "trtexec")

    library_directories = [
        _resolve_member(root, item, f"library_directories[{index}]", "lib", directory=True)
        for index, item in enumerate(
            _string_list(manifest["library_directories"], "library_directories")
        )
    ]
    if len(library_directories) != len(set(library_directories)):
        raise BundleError("bundle library_directories resolve to duplicate paths")
    license_files = [
        _resolve_member(root, item, f"licenses[{index}]", "licenses", directory=False)
        for index, item in enumerate(_string_list(manifest["licenses"], "licenses"))
    ]
    if len(license_files) != len(set(license_files)):
        raise BundleError("bundle licenses resolve to duplicate paths")

    source_environment = os.environ if environ is None else environ
    environment = _prepend_environment(
        source_environment,
        list(dict.fromkeys([executable.parent, trtexec.parent, *library_directories])),
        platform_id,
    )
    return BundleLaunchSpec(
        platform=platform_id,
        root=root,
        executable=executable,
        trtexec=trtexec,
        env=environment,
        audio2face_model=audio2face_model,
        audio2emotion_model=audio2emotion_model,
    )


__all__ = [
    "BundleError",
    "BundleLaunchSpec",
    "RUNTIME_SCHEMA",
    "current_platform_id",
    "resolve_runtime_bundle",
]
