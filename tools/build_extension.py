#!/usr/bin/env python3
"""Build one Blender 5.2 extension containing one native Audio2Face runtime."""

from __future__ import annotations

import argparse
import hashlib
import lzma
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence


TOOLS_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = TOOLS_DIRECTORY.parent
sys.path.insert(0, os.fspath(TOOLS_DIRECTORY))
sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

import runtime_build_common as common  # noqa: E402
from audio2face.path_contract import require_unaliased_path  # noqa: E402


ADDON_SOURCE = REPOSITORY_ROOT / "audio2face"
DIST_DIRECTORY = REPOSITORY_ROOT / "dist"


def _validate_elf_x64(path: Path, label: str) -> None:
    try:
        with path.open("rb") as stream:
            header = stream.read(20)
    except OSError as exc:
        raise common.BuildError(f"cannot inspect {label}: {path}") from exc
    if (
        len(header) < 20
        or header[:7] != b"\x7fELF\x02\x01\x01"
        or header[18:20] != b">\x00"
    ):
        raise common.BuildError(f"{label} is not Linux ELF64 x86-64: {path}")


def _validate_pe_x64(path: Path, label: str) -> None:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            dos = stream.read(64)
            if len(dos) != 64 or dos[:2] != b"MZ":
                raise common.BuildError(f"{label} is not Windows PE32+ AMD64")
            offset = int.from_bytes(dos[60:64], "little")
            if offset < 64 or offset > size - 26:
                raise common.BuildError(f"{label} has an invalid Windows PE offset")
            stream.seek(offset)
            pe = stream.read(26)
    except OSError as exc:
        raise common.BuildError(f"cannot inspect {label}: {path}") from exc
    if (
        len(pe) != 26
        or pe[:4] != b"PE\x00\x00"
        or pe[4:6] != b"d\x86"
        or pe[24:26] != b"\x0b\x02"
    ):
        raise common.BuildError(f"{label} is not Windows PE32+ AMD64: {path}")


def _validate_native_binary(path: Path, platform_id: str, label: str) -> None:
    if platform_id == "windows-x64":
        _validate_pe_x64(path, label)
        return
    if platform_id == "linux-x64":
        _validate_elf_x64(path, label)
        return
    raise common.BuildError(
        f"cannot validate native binary for platform {platform_id!r}"
    )


def validate_runtime(runtime: Path, platform_id: str) -> None:
    """Validate the exact native runtime tree embedded by this release."""

    common.validate_runtime_package(runtime, platform_id)
    contract = common.runtime_contract(platform_id)
    for binary in common.native_runtime_files(runtime, contract):
        relative = binary.relative_to(runtime).as_posix()
        _validate_native_binary(binary, platform_id, relative)


def _manifest_identity(manifest: Path) -> tuple[str, str]:
    try:
        text = manifest.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise common.BuildError(
            f"cannot read Blender manifest {manifest}: {exc}"
        ) from exc
    fields: dict[str, str] = {}
    for field in ("id", "version", "blender_version_min", "blender_version_max"):
        matches = re.findall(
            rf'^\s*{re.escape(field)}\s*=\s*"([^"]+)"\s*$',
            text,
            flags=re.MULTILINE,
        )
        if len(matches) != 1:
            raise common.BuildError(f"Blender manifest needs exactly one {field}")
        fields[field] = matches[0]
    if fields["id"] != "audio2face":
        raise common.BuildError(f"unexpected Blender extension id {fields['id']!r}")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", fields["version"]):
        raise common.BuildError(f"invalid extension version {fields['version']!r}")
    if (
        fields["blender_version_min"] != "5.2.0"
        or fields["blender_version_max"] != "5.3.0"
    ):
        raise common.BuildError("extension manifest must target exactly Blender 5.2")
    return fields["id"], fields["version"]


def rewrite_manifest_platform(manifest: Path, platform_id: str) -> tuple[str, str]:
    extension_id, version = _manifest_identity(manifest)
    text = manifest.read_text(encoding="utf-8")
    pattern = re.compile(r"^\s*platforms\s*=\s*\[[^\r\n]*\]\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise common.BuildError(
            "Blender manifest needs exactly one platforms declaration"
        )
    rewritten = pattern.sub(f'platforms = ["{platform_id}"]', text)
    manifest.write_text(rewritten, encoding="utf-8", newline="\n")
    if f'platforms = ["{platform_id}"]' not in manifest.read_text(encoding="utf-8"):
        raise common.BuildError("could not pin the package manifest platform")
    return extension_id, version


def _copy_addon_source(package_root: Path) -> None:
    if (ADDON_SOURCE / "runtime").exists() or (ADDON_SOURCE / "runtime").is_symlink():
        raise common.BuildError(
            "source audio2face/runtime must not exist; embed only build/runtime/<platform>"
        )
    for entry in ADDON_SOURCE.rglob("*"):
        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            raise common.BuildError(
                f"add-on source contains a symlink or special file: {entry}"
            )
    shutil.copytree(
        ADDON_SOURCE,
        package_root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def validate_blender(
    blender: Path,
    platform_id: str,
    runner: common.CommandRunner,
) -> Path:
    executable = require_unaliased_path(
        blender,
        description="Blender executable",
        error_type=common.BuildError,
    )
    if not executable.is_file():
        raise common.BuildError(f"Blender path is not a file: {executable}")
    version_output = runner.run([executable, "--version"], env=os.environ, capture=True)
    first_line = version_output.splitlines()[0] if version_output else ""
    if re.fullmatch(r"Blender 5\.2\.[0-9]+(?: LTS)?", first_line) is None:
        raise common.BuildError(
            f"extension releases require Blender 5.2; got {first_line!r}"
        )
    try:
        expected_build_platform = {
            "windows-x64": "Windows",
            "linux-x64": "Linux",
        }[platform_id]
    except KeyError as exc:
        raise common.BuildError(
            f"unsupported Blender release platform {platform_id!r}"
        ) from exc
    build_platform_lines = re.findall(
        r"^\s*build platform:\s*(\S+)\s*$", version_output, flags=re.MULTILINE
    )
    if build_platform_lines != [expected_build_platform]:
        raise common.BuildError(
            f"Blender build platform must be {expected_build_platform}; "
            f"got {build_platform_lines}"
        )
    return executable


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
    package_root: Path,
    platform_id: str,
) -> None:
    """Require Blender's verified package-files-at-ZIP-root layout byte-for-byte."""

    expected_files = {
        path.relative_to(package_root).as_posix(): path
        for path in package_root.rglob("*")
        if path.is_file()
    }
    expected_directories = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_dir()
    }
    if not expected_files:
        raise common.BuildError("extension package root is empty")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            actual_files: dict[str, zipfile.ZipInfo] = {}
            actual_directories: set[str] = set()
            seen: set[str] = set()
            folded: dict[str, str] = {}
            for info in archive.infolist():
                member = common.safe_member_path(info.filename, "extension ZIP member")
                name = member.as_posix()
                if name in seen:
                    raise common.BuildError(f"extension ZIP duplicates {name}")
                seen.add(name)
                previous = folded.get(name.casefold())
                if previous is not None and previous != name:
                    raise common.BuildError(
                        f"extension ZIP case-collides {previous!r} and {name!r}"
                    )
                folded[name.casefold()] = name
                mode = info.external_attr >> 16
                if info.flag_bits & 0x1:
                    raise common.BuildError(
                        f"extension ZIP has an unsafe member: {name}"
                    )
                if info.is_dir():
                    if mode and not stat.S_ISDIR(mode):
                        raise common.BuildError(
                            f"extension ZIP directory has non-directory mode: {name}"
                        )
                    actual_directories.add(name)
                else:
                    if mode and not stat.S_ISREG(mode):
                        raise common.BuildError(
                            f"extension ZIP file has non-regular mode: {name}"
                        )
                    if info.compress_type != zipfile.ZIP_LZMA:
                        raise common.BuildError(
                            f"extension ZIP file is not ZIP-LZMA compressed: {name}"
                        )
                    actual_files[name] = info
            extra_directories = sorted(actual_directories - expected_directories)
            if extra_directories:
                raise common.BuildError(
                    "extension ZIP contains undeclared directories: "
                    f"{extra_directories}"
                )
            if set(actual_files) != set(expected_files):
                missing = sorted(set(expected_files) - set(actual_files))
                extra = sorted(set(actual_files) - set(expected_files))
                raise common.BuildError(
                    f"extension ZIP layout differs from the package root; "
                    f"missing={missing}, extra={extra}"
                )
            if "audio2face/blender_manifest.toml" in actual_files:
                raise common.BuildError(
                    "Blender 5.2 extension ZIP must place package files directly at ZIP root"
                )
            for name, source in expected_files.items():
                source_identity = _file_digest(source)
                info = actual_files[name]
                archive_identity = _zip_member_digest(archive, info)
                if archive_identity != source_identity:
                    raise common.BuildError(f"extension ZIP bytes differ for {name}")
                if platform_id == "linux-x64":
                    source_mode = stat.S_IMODE(source.stat().st_mode)
                    archive_mode = stat.S_IMODE(info.external_attr >> 16)
                    if archive_mode != source_mode:
                        raise common.BuildError(
                            f"extension ZIP mode differs for {name}: "
                            f"expected {source_mode:o}, got {archive_mode:o}"
                        )
    except (OSError, lzma.LZMAError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise common.BuildError(
            f"cannot validate extension ZIP {archive_path}: {exc}"
        ) from exc


def write_extension_archive(package_root: Path, archive_path: Path) -> None:
    """Write one all-file ZIP-LZMA archive accepted by Blender 5.2."""

    try:
        with zipfile.ZipFile(
            archive_path,
            "x",
            compression=zipfile.ZIP_LZMA,
            allowZip64=True,
        ) as archive:
            for source in sorted(package_root.rglob("*")):
                if source.is_symlink() or not (source.is_file() or source.is_dir()):
                    raise common.BuildError(
                        f"extension package contains a symlink or special file: {source}"
                    )
                if source.is_dir():
                    continue
                relative = source.relative_to(package_root).as_posix()
                print(
                    f"Packaging {relative} with ZIP-LZMA "
                    f"({source.stat().st_size} bytes)",
                    flush=True,
                )
                archive.write(source, relative, compress_type=zipfile.ZIP_LZMA)
    except common.BuildError:
        raise
    except (OSError, lzma.LZMAError, RuntimeError, zipfile.LargeZipFile) as exc:
        raise common.BuildError(
            f"cannot create extension ZIP {archive_path}: {exc}"
        ) from exc


def build_extension(blender: Path, platform_id: str) -> Path:
    common.require_native_target(platform_id)
    runner = common.CommandRunner()
    executable = validate_blender(blender, platform_id, runner)
    runtime = REPOSITORY_ROOT / "build" / "runtime" / platform_id
    validate_runtime(runtime, platform_id)

    work_directory = REPOSITORY_ROOT / "build"
    work_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="audio2face-extension-", dir=work_directory
    ) as temporary:
        package_root = Path(temporary) / "audio2face"
        _copy_addon_source(package_root)
        try:
            shutil.copytree(
                runtime,
                package_root / "runtime",
                symlinks=False,
                copy_function=os.link,
            )
        except OSError as exc:
            raise common.BuildError(
                f"cannot add the runtime to Blender's package root: {exc}"
            ) from exc
        extension_id, version = rewrite_manifest_platform(
            package_root / "blender_manifest.toml", platform_id
        )
        validate_runtime(package_root / "runtime", platform_id)

        DIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
        output = DIST_DIRECTORY / f"{extension_id}-{version}-{platform_id}.zip"
        partial = output.with_name(output.stem + ".partial.zip")
        if output.exists() or output.is_symlink():
            raise common.BuildError(
                f"extension output already exists; use a clean dist tree: {output}"
            )
        if partial.exists() or partial.is_symlink():
            raise common.BuildError(f"stale partial extension output exists: {partial}")
        runner.run(
            [executable, "--command", "extension", "validate", package_root],
            env=os.environ,
        )
        try:
            write_extension_archive(package_root, partial)
            if not partial.is_file() or partial.stat().st_size < 1:
                raise common.BuildError(f"extension ZIP was not produced: {partial}")
            validate_extension_archive(partial, package_root, platform_id)
            runner.run(
                [executable, "--command", "extension", "validate", partial],
                env=os.environ,
            )
            if output.exists() or output.is_symlink():
                raise common.BuildError(
                    f"extension output appeared during build: {output}"
                )
            partial.replace(output)
        except common.BuildError:
            partial.unlink(missing_ok=True)
            raise
        except OSError as exc:
            partial.unlink(missing_ok=True)
            raise common.BuildError(f"cannot publish extension output: {exc}") from exc
    return output


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one platform-specific Blender 5.2 Audio2Face extension."
    )
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--platform", required=True, choices=common.SUPPORTED_PLATFORMS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = create_argument_parser().parse_args(argv)
    try:
        output = build_extension(arguments.blender, arguments.platform)
    except common.BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Built Blender 5.2 Audio2Face extension: {output} "
        f"({output.stat().st_size} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
