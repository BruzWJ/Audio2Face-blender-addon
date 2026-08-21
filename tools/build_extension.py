#!/usr/bin/env python3
"""Build one Blender 5.2 extension containing one native Audio2Face runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform as platform_module
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from audio2face.runtime_contract import (  # noqa: E402
    RUNTIME_CONTRACTS,
    RUNTIME_MANIFEST_FIELDS,
    runtime_contract,
)
from audio2face.path_contract import require_unaliased_path  # noqa: E402
from audio2face.strict_json import (  # noqa: E402
    duplicate_key_hook,
    invalid_constant_hook,
)


ADDON_SOURCE = REPOSITORY_ROOT / "audio2face"
DIST_DIRECTORY = REPOSITORY_ROOT / "dist"
SUPPORTED_PLATFORMS = tuple(RUNTIME_CONTRACTS)


class ExtensionBuildError(RuntimeError):
    """The platform extension cannot be built safely."""

def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=duplicate_key_hook(ExtensionBuildError, label),
            parse_constant=invalid_constant_hook(ExtensionBuildError, label),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExtensionBuildError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExtensionBuildError(f"{label} root must be an object")
    return value


def _flat_regular_files(directory: Path, expected: set[str], label: str) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise ExtensionBuildError(f"{label} is not a real directory: {directory}")
    entries = {entry.name: entry for entry in directory.iterdir()}
    if set(entries) != expected:
        raise ExtensionBuildError(
            f"{label} must contain {sorted(expected)}; got {sorted(entries)}"
        )
    for name, entry in entries.items():
        if not entry.is_file() or entry.is_symlink() or entry.stat().st_size < 1:
            raise ExtensionBuildError(f"{label} entry is not a non-empty regular file: {name}")


def _validate_elf_x64(path: Path, label: str) -> None:
    try:
        with path.open("rb") as stream:
            header = stream.read(20)
    except OSError as exc:
        raise ExtensionBuildError(f"cannot inspect {label}: {path}") from exc
    if (
        len(header) < 20
        or header[:7] != b"\x7fELF\x02\x01\x01"
        or header[18:20] != b">\x00"
    ):
        raise ExtensionBuildError(f"{label} is not Linux ELF64 x86-64: {path}")


def _validate_pe_x64(path: Path, label: str) -> None:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            dos = stream.read(64)
            if len(dos) != 64 or dos[:2] != b"MZ":
                raise ExtensionBuildError(f"{label} is not Windows PE32+ AMD64")
            offset = int.from_bytes(dos[60:64], "little")
            if offset < 64 or offset > size - 26:
                raise ExtensionBuildError(f"{label} has an invalid Windows PE offset")
            stream.seek(offset)
            pe = stream.read(26)
    except OSError as exc:
        raise ExtensionBuildError(f"cannot inspect {label}: {path}") from exc
    if (
        len(pe) != 26
        or pe[:4] != b"PE\x00\x00"
        or pe[4:6] != b"d\x86"
        or pe[24:26] != b"\x0b\x02"
    ):
        raise ExtensionBuildError(f"{label} is not Windows PE32+ AMD64: {path}")


def _validate_native_binary(path: Path, platform_id: str, label: str) -> None:
    if platform_id == "windows-x64":
        _validate_pe_x64(path, label)
        return
    if platform_id == "linux-x64":
        _validate_elf_x64(path, label)
        return
    raise ExtensionBuildError(
        f"cannot validate native binary for platform {platform_id!r}"
    )


def validate_runtime(runtime: Path, platform_id: str) -> dict[str, Any]:
    """Validate the exact native runtime tree embedded by this release."""

    try:
        contract = runtime_contract(platform_id)
    except ValueError as exc:
        raise ExtensionBuildError(str(exc)) from exc
    if not runtime.is_dir() or runtime.is_symlink():
        raise ExtensionBuildError(f"runtime handoff is missing: {runtime}")
    actual_root = {entry.name for entry in runtime.iterdir()}
    if actual_root != contract.root_entries:
        raise ExtensionBuildError(
            f"runtime root must contain {sorted(contract.root_entries)}; "
            f"got {sorted(actual_root)}"
        )
    for entry in runtime.rglob("*"):
        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            raise ExtensionBuildError(f"runtime contains a symlink or special file: {entry}")

    _flat_regular_files(runtime / "bin", contract.bin_entries, "runtime/bin")
    if "lib" in contract.root_entries:
        _flat_regular_files(runtime / "lib", contract.library_entries, "runtime/lib")
    _flat_regular_files(
        runtime / "licenses",
        contract.license_entries,
        "runtime/licenses",
    )

    bundle = _read_json(runtime / "bundle.json", "runtime manifest")
    if set(bundle) != RUNTIME_MANIFEST_FIELDS:
        raise ExtensionBuildError(
            f"runtime manifest fields must be {sorted(RUNTIME_MANIFEST_FIELDS)}; "
            f"got {sorted(bundle)}"
        )
    if bundle != contract.manifest():
        raise ExtensionBuildError("runtime manifest does not match the staged runtime shape")
    worker = runtime.joinpath(*PurePosixPath(contract.worker).parts)
    trtexec = runtime.joinpath(*PurePosixPath(contract.trtexec).parts)
    if platform_id == "linux-x64" and (
        not os.access(worker, os.X_OK) or not os.access(trtexec, os.X_OK)
    ):
        raise ExtensionBuildError("Linux worker and trtexec must be executable")
    _validate_native_binary(worker, platform_id, "Audio2Face worker")
    _validate_native_binary(trtexec, platform_id, "TensorRT trtexec")
    for packaged_file in contract.libraries:
        relative = packaged_file.path
        library = runtime.joinpath(*PurePosixPath(relative).parts)
        _validate_native_binary(library, platform_id, relative)
    return bundle


def _manifest_identity(manifest: Path) -> tuple[str, str]:
    try:
        text = manifest.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ExtensionBuildError(f"cannot read Blender manifest {manifest}: {exc}") from exc
    fields: dict[str, str] = {}
    for field in ("id", "version", "blender_version_min", "blender_version_max"):
        matches = re.findall(
            rf'^\s*{re.escape(field)}\s*=\s*"([^"]+)"\s*$',
            text,
            flags=re.MULTILINE,
        )
        if len(matches) != 1:
            raise ExtensionBuildError(f"Blender manifest needs exactly one {field}")
        fields[field] = matches[0]
    if fields["id"] != "audio2face":
        raise ExtensionBuildError(f"unexpected Blender extension id {fields['id']!r}")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", fields["version"]):
        raise ExtensionBuildError(f"invalid extension version {fields['version']!r}")
    if fields["blender_version_min"] != "5.2.0" or fields[
        "blender_version_max"
    ] != "5.3.0":
        raise ExtensionBuildError("extension manifest must target exactly Blender 5.2")
    return fields["id"], fields["version"]


def rewrite_manifest_platform(manifest: Path, platform_id: str) -> tuple[str, str]:
    extension_id, version = _manifest_identity(manifest)
    text = manifest.read_text(encoding="utf-8")
    pattern = re.compile(r"^\s*platforms\s*=\s*\[[^\r\n]*\]\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ExtensionBuildError("Blender manifest needs exactly one platforms declaration")
    rewritten = pattern.sub(f'platforms = ["{platform_id}"]', text)
    manifest.write_text(rewritten, encoding="utf-8", newline="\n")
    if f'platforms = ["{platform_id}"]' not in manifest.read_text(encoding="utf-8"):
        raise ExtensionBuildError("could not pin the staged Blender manifest platform")
    return extension_id, version


def _copy_addon_source(stage: Path) -> None:
    if (ADDON_SOURCE / "runtime").exists() or (ADDON_SOURCE / "runtime").is_symlink():
        raise ExtensionBuildError(
            "source audio2face/runtime must not exist; embed only build/runtime/<platform>"
        )
    for entry in ADDON_SOURCE.rglob("*"):
        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            raise ExtensionBuildError(f"add-on source contains a symlink or special file: {entry}")
    shutil.copytree(
        ADDON_SOURCE,
        stage,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def _run(command: Sequence[os.PathLike[str] | str], *, capture: bool = False) -> str:
    args = [os.fspath(item) for item in command]
    print("+ " + subprocess.list2cmdline(args), flush=True)
    try:
        result = subprocess.run(
            args,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        output = getattr(exc, "stdout", None)
        detail = f"\n{output}" if output else ""
        raise ExtensionBuildError(f"Blender command failed{detail}") from exc
    return result.stdout.strip() if capture and result.stdout else ""


def require_native_platform(platform_id: str) -> None:
    system = sys.platform
    machine = platform_module.machine()
    if system == "win32" and machine == "AMD64":
        host = "windows-x64"
    elif system == "linux" and machine == "x86_64":
        host = "linux-x64"
    else:
        raise ExtensionBuildError(f"unsupported extension release host {system}/{machine}")
    if host != platform_id:
        raise ExtensionBuildError(
            f"--platform {platform_id} does not match native extension host {host}"
        )


def validate_blender(blender: Path, platform_id: str) -> Path:
    executable = require_unaliased_path(
        blender,
        description="Blender executable",
        error_type=ExtensionBuildError,
    )
    if not executable.is_file():
        raise ExtensionBuildError(f"Blender path is not a file: {executable}")
    version_output = _run([executable, "--version"], capture=True)
    first_line = version_output.splitlines()[0] if version_output else ""
    if re.fullmatch(r"Blender 5\.2\.[0-9]+(?: LTS)?", first_line) is None:
        raise ExtensionBuildError(
            f"extension releases require Blender 5.2; got {first_line!r}"
        )
    expected_build_platform = "Windows" if platform_id == "windows-x64" else "Linux"
    build_platform_lines = re.findall(
        r"^\s*build platform:\s*(\S+)\s*$", version_output, flags=re.MULTILINE
    )
    if build_platform_lines != [expected_build_platform]:
        raise ExtensionBuildError(
            f"Blender build platform must be {expected_build_platform}; "
            f"got {build_platform_lines}"
        )
    return executable


def _safe_zip_member(name: str) -> PurePosixPath:
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise ExtensionBuildError(f"unsafe extension ZIP member {name!r}")
    path = PurePosixPath(name)
    if not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ExtensionBuildError(f"non-canonical extension ZIP member {name!r}")
    if re.match(r"^[A-Za-z]:", name):
        raise ExtensionBuildError(f"absolute extension ZIP member {name!r}")
    return path


def _file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _zip_member_digest(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo
) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(info, "r") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def validate_extension_archive(
    archive_path: Path,
    staged_addon: Path,
    platform_id: str,
) -> None:
    """Require Blender's verified package-files-at-ZIP-root layout byte-for-byte."""

    expected_files = {
        path.relative_to(staged_addon).as_posix(): path
        for path in staged_addon.rglob("*")
        if path.is_file()
    }
    expected_directories = {
        path.relative_to(staged_addon).as_posix()
        for path in staged_addon.rglob("*")
        if path.is_dir()
    }
    if not expected_files:
        raise ExtensionBuildError("staged extension is empty")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            actual_files: dict[str, zipfile.ZipInfo] = {}
            actual_directories: set[str] = set()
            seen: set[str] = set()
            folded: dict[str, str] = {}
            for info in archive.infolist():
                member = _safe_zip_member(info.filename)
                name = member.as_posix()
                if name in seen:
                    raise ExtensionBuildError(f"extension ZIP duplicates {name}")
                seen.add(name)
                previous = folded.get(name.casefold())
                if previous is not None and previous != name:
                    raise ExtensionBuildError(
                        f"extension ZIP case-collides {previous!r} and {name!r}"
                    )
                folded[name.casefold()] = name
                mode = info.external_attr >> 16
                if info.flag_bits & 0x1:
                    raise ExtensionBuildError(f"extension ZIP has an unsafe member: {name}")
                if info.is_dir():
                    if mode and not stat.S_ISDIR(mode):
                        raise ExtensionBuildError(
                            f"extension ZIP directory has non-directory mode: {name}"
                        )
                    actual_directories.add(name)
                else:
                    if mode and not stat.S_ISREG(mode):
                        raise ExtensionBuildError(
                            f"extension ZIP file has non-regular mode: {name}"
                        )
                    actual_files[name] = info
            extra_directories = sorted(actual_directories - expected_directories)
            if extra_directories:
                raise ExtensionBuildError(
                    "extension ZIP contains undeclared directories: "
                    f"{extra_directories}"
                )
            if set(actual_files) != set(expected_files):
                missing = sorted(set(expected_files) - set(actual_files))
                extra = sorted(set(actual_files) - set(expected_files))
                raise ExtensionBuildError(
                    f"extension ZIP layout differs from staged audio2face source; "
                    f"missing={missing}, extra={extra}"
                )
            if "audio2face/blender_manifest.toml" in actual_files:
                raise ExtensionBuildError(
                    "Blender 5.2 extension ZIP must place package files directly at ZIP root"
                )
            for name, source in expected_files.items():
                source_identity = _file_digest(source)
                info = actual_files[name]
                archive_identity = _zip_member_digest(archive, info)
                if archive_identity != source_identity:
                    raise ExtensionBuildError(f"extension ZIP bytes differ for {name}")
                if platform_id == "linux-x64":
                    source_mode = stat.S_IMODE(source.stat().st_mode)
                    archive_mode = stat.S_IMODE(info.external_attr >> 16)
                    if archive_mode != source_mode:
                        raise ExtensionBuildError(
                            f"extension ZIP mode differs for {name}: "
                            f"expected {source_mode:o}, got {archive_mode:o}"
                        )
    except (OSError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise ExtensionBuildError(f"cannot validate extension ZIP {archive_path}: {exc}") from exc


def build_extension(blender: Path, platform_id: str) -> Path:
    require_native_platform(platform_id)
    executable = validate_blender(blender, platform_id)
    runtime = REPOSITORY_ROOT / "build" / "runtime" / platform_id
    validate_runtime(runtime, platform_id)

    with tempfile.TemporaryDirectory(prefix="audio2face-extension-") as temporary:
        staged_addon = Path(temporary) / "audio2face"
        _copy_addon_source(staged_addon)
        shutil.copytree(runtime, staged_addon / "runtime", symlinks=False)
        extension_id, version = rewrite_manifest_platform(
            staged_addon / "blender_manifest.toml", platform_id
        )
        validate_runtime(staged_addon / "runtime", platform_id)

        DIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
        output = DIST_DIRECTORY / f"{extension_id}-{version}-{platform_id}.zip"
        partial = output.with_name(output.stem + ".partial.zip")
        if output.exists() or output.is_symlink():
            raise ExtensionBuildError(
                f"extension output already exists; use a clean dist tree: {output}"
            )
        if partial.exists() or partial.is_symlink():
            raise ExtensionBuildError(f"stale partial extension output exists: {partial}")
        _run([executable, "--command", "extension", "validate", staged_addon])
        try:
            _run(
                [
                    executable,
                    "--command",
                    "extension",
                    "build",
                    "--source-dir",
                    staged_addon,
                    "--output-filepath",
                    partial,
                ]
            )
            if not partial.is_file() or partial.stat().st_size < 1:
                raise ExtensionBuildError(
                    f"Blender did not produce extension ZIP: {partial}"
                )
            validate_extension_archive(partial, staged_addon, platform_id)
            if output.exists() or output.is_symlink():
                raise ExtensionBuildError(f"extension output appeared during build: {output}")
            partial.replace(output)
        except ExtensionBuildError:
            partial.unlink(missing_ok=True)
            raise
        except OSError as exc:
            partial.unlink(missing_ok=True)
            raise ExtensionBuildError(f"cannot publish extension output: {exc}") from exc
    return output


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one platform-specific Blender 5.2 Audio2Face extension."
    )
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--platform", required=True, choices=SUPPORTED_PLATFORMS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = create_argument_parser().parse_args(argv)
    try:
        output = build_extension(arguments.blender, arguments.platform)
    except ExtensionBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Built Blender 5.2 Audio2Face extension: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
