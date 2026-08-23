#!/usr/bin/env python3
"""Build the pinned Linux x64 Audio2Face runtime for extension embedding."""

from __future__ import annotations

import contextlib
import hashlib
import json
import lzma
import os
import re
import shlex
import stat
import struct
import subprocess
import urllib.parse
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import runtime_build_common as common
from runtime_build_common import BuildError
from audio2face.strict_json import (
    duplicate_key_hook,
    invalid_constant_hook,
)

PLATFORM_ID = "linux-x64"
LINUX_EXTERNAL_LIBRARIES = frozenset(
    {
        "ld-linux-x86-64.so.2",
        "libc.so.6",
        "libcuda.so.1",
        "libdl.so.2",
        "libm.so.6",
        "libpthread.so.0",
        "librt.so.1",
    }
)


def finalize_linux_cuda_toolkit(toolkit: Path) -> None:
    """Put NVIDIA's Linux redistributables in nvcc's canonical toolkit layout."""

    archive_libraries = toolkit / "lib"
    compiler_libraries = toolkit / common.LINUX_CUDA_LIBRARY_DIRECTORY
    if not archive_libraries.is_dir() or archive_libraries.is_symlink():
        raise BuildError("Linux CUDA redistributables have no regular lib directory")
    if compiler_libraries.exists() or compiler_libraries.is_symlink():
        raise BuildError(
            "Linux CUDA redistributables unexpectedly contain both lib and lib64"
        )
    for filename in common.LINUX_CUDA_COMPILER_LIBRARIES:
        library = archive_libraries / filename
        if not library.is_file() or library.is_symlink():
            raise BuildError(
                f"Linux CUDA redistributables are missing compiler library {filename}"
            )
    archive_libraries.rename(compiler_libraries)


def _rpm_header(
    data: bytes, offset: int, label: str
) -> tuple[int, dict[int, tuple[int, int, int]], bytes]:
    if offset < 0 or offset + 16 > len(data):
        raise BuildError(f"{label} RPM header is truncated")
    if data[offset : offset + 4] != b"\x8e\xad\xe8\x01":
        raise BuildError(f"{label} RPM header magic is invalid")
    if data[offset + 4 : offset + 8] != b"\0\0\0\0":
        raise BuildError(f"{label} RPM header reserved bytes are nonzero")
    index_count, store_size = struct.unpack_from(">II", data, offset + 8)
    index_start = offset + 16
    store_start = index_start + index_count * 16
    end = store_start + store_size
    if index_count > 100_000 or store_size > len(data) or end > len(data):
        raise BuildError(f"{label} RPM header bounds are invalid")
    entries: dict[int, tuple[int, int, int]] = {}
    for index in range(index_count):
        tag, value_type, value_offset, count = struct.unpack_from(
            ">IIII", data, index_start + index * 16
        )
        if tag in entries:
            raise BuildError(f"{label} RPM header repeats tag {tag}")
        if value_offset >= store_size:
            raise BuildError(f"{label} RPM tag {tag} points outside its store")
        entries[tag] = (value_type, value_offset, count)
    return end, entries, data[store_start:end]


def _rpm_string(
    entries: Mapping[int, tuple[int, int, int]],
    store: bytes,
    tag: int,
    label: str,
) -> str:
    entry = entries.get(tag)
    if entry is None:
        raise BuildError(f"{label} RPM is missing string tag {tag}")
    value_type, offset, count = entry
    if value_type != 6 or count != 1:
        raise BuildError(f"{label} RPM tag {tag} is not one string")
    terminator = store.find(b"\0", offset)
    if terminator < 0:
        raise BuildError(f"{label} RPM string tag {tag} is unterminated")
    try:
        value = store[offset:terminator].decode("utf-8")
    except UnicodeError as exc:
        raise BuildError(f"{label} RPM string tag {tag} is not UTF-8") from exc
    if not value or value.strip() != value:
        raise BuildError(f"{label} RPM tag {tag} must be a non-empty, trimmed string")
    return value


def _rpm_payload(archive_path: Path, artifact_url: str, source_rpm_url: str) -> bytes:
    label = archive_path.name
    try:
        data = archive_path.read_bytes()
    except OSError as exc:
        raise BuildError(f"cannot read locked RPM {archive_path}: {exc}") from exc
    if len(data) < 112 or data[:4] != b"\xed\xab\xee\xdb":
        raise BuildError(f"{label} is not an RPM package")
    signature_end, _signature_entries, _signature_store = _rpm_header(
        data, 96, f"{label} signature"
    )
    main_offset = (signature_end + 7) & ~7
    payload_offset, entries, store = _rpm_header(data, main_offset, label)
    name = _rpm_string(entries, store, 1000, label)
    version = _rpm_string(entries, store, 1001, label)
    release = _rpm_string(entries, store, 1002, label)
    architecture = _rpm_string(entries, store, 1022, label)
    nevra = f"{name}-{version}-{release}.{architecture}"
    artifact_name = PurePosixPath(urllib.parse.urlsplit(artifact_url).path).name
    if artifact_name != f"{nevra}.rpm":
        raise BuildError(
            f"locked RPM identity mismatch: expected {artifact_name}, got {nevra}.rpm"
        )
    source_name = _rpm_string(entries, store, 1044, label)
    expected_source = PurePosixPath(urllib.parse.urlsplit(source_rpm_url).path).name
    if source_name != expected_source:
        raise BuildError(
            f"locked RPM source mismatch: expected {expected_source}, got {source_name}"
        )
    if _rpm_string(entries, store, 1124, label) != "cpio":
        raise BuildError(f"{label} payload format is not cpio")
    if _rpm_string(entries, store, 1125, label) != "xz":
        raise BuildError(f"{label} payload compressor is not xz")
    try:
        payload = lzma.decompress(data[payload_offset:], format=lzma.FORMAT_XZ)
    except lzma.LZMAError as exc:
        raise BuildError(
            f"cannot decompress locked RPM payload {label}: {exc}"
        ) from exc
    if len(payload) > 16 * 1024 * 1024:
        raise BuildError(f"locked RPM payload is unexpectedly large: {label}")
    return payload


def _cpio_locked_members(
    payload: bytes, wanted: set[str], label: str
) -> dict[str, bytes]:
    position = 0
    seen: set[str] = set()
    found: dict[str, bytes] = {}
    while True:
        if position + 110 > len(payload):
            raise BuildError(f"{label} cpio header is truncated")
        header = payload[position : position + 110]
        position += 110
        if header[:6] != b"070701":
            raise BuildError(f"{label} cpio member does not use newc format")
        try:
            fields = [
                int(header[6 + index * 8 : 14 + index * 8], 16) for index in range(13)
            ]
        except ValueError as exc:
            raise BuildError(
                f"{label} cpio header has invalid hexadecimal fields"
            ) from exc
        mode = fields[1]
        file_size = fields[6]
        name_size = fields[11]
        if name_size < 2 or position + name_size > len(payload):
            raise BuildError(f"{label} cpio member name has invalid bounds")
        name_bytes = payload[position : position + name_size]
        position += name_size
        position = (position + 3) & ~3
        if name_bytes[-1:] != b"\0":
            raise BuildError(f"{label} cpio member name is not terminated")
        try:
            archive_name = name_bytes[:-1].decode("utf-8")
        except UnicodeError as exc:
            raise BuildError(f"{label} cpio member name is not UTF-8") from exc
        if archive_name == "TRAILER!!!":
            if file_size != 0 or position != len(payload):
                raise BuildError(f"{label} cpio trailer is not canonical")
            break
        if not archive_name.startswith("./"):
            raise BuildError(f"{label} cpio member lacks the canonical ./ prefix")
        member = archive_name[2:]
        common.safe_member_path(member, f"{label} cpio member")
        if member in seen:
            raise BuildError(f"{label} cpio repeats member {member}")
        seen.add(member)
        if position + file_size > len(payload):
            raise BuildError(f"{label} cpio member data is truncated: {member}")
        content = payload[position : position + file_size]
        position += file_size
        position = (position + 3) & ~3
        if member in wanted:
            if not stat.S_ISREG(mode):
                raise BuildError(f"locked RPM member is not a regular file: {member}")
            found[member] = content
    if set(found) != wanted:
        raise BuildError(
            f"{label} RPM members differ from lock: "
            f"missing={sorted(wanted - set(found))}"
        )
    return found


def _verify_locked_bytes(content: bytes, entry: Mapping[str, Any], label: str) -> None:
    if len(content) != entry["size"]:
        raise BuildError(
            f"{label} size mismatch: expected {entry['size']}, got {len(content)}"
        )
    digest = hashlib.sha256(content).hexdigest()
    if digest != entry["sha256"]:
        raise BuildError(
            f"{label} SHA-256 mismatch: expected {entry['sha256']}, got {digest}"
        )


def _read_exact(source: BinaryIO, size: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = source.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise BuildError(f"{label} is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _rpm_stream_header(
    source: BinaryIO, label: str
) -> tuple[dict[int, tuple[int, int, int]], bytes]:
    header = _read_exact(source, 16, f"{label} RPM header")
    if header[:4] != b"\x8e\xad\xe8\x01" or header[4:8] != b"\0\0\0\0":
        raise BuildError(f"{label} RPM header is invalid")
    index_count, store_size = struct.unpack_from(">II", header, 8)
    if index_count > 100_000 or store_size > 64 * 1024 * 1024:
        raise BuildError(f"{label} RPM header bounds are invalid")
    index = _read_exact(source, index_count * 16, f"{label} RPM index")
    store = _read_exact(source, store_size, f"{label} RPM store")
    entries: dict[int, tuple[int, int, int]] = {}
    for position in range(index_count):
        tag, value_type, value_offset, count = struct.unpack_from(
            ">IIII", index, position * 16
        )
        if tag in entries:
            raise BuildError(f"{label} RPM header repeats tag {tag}")
        if value_offset >= store_size:
            raise BuildError(f"{label} RPM tag {tag} points outside its store")
        entries[tag] = (value_type, value_offset, count)
    return entries, store


@contextlib.contextmanager
def _rpm_xz_payload(
    archive_path: Path,
    artifact_url: str,
    source_rpm: str,
) -> Iterator[BinaryIO]:
    label = archive_path.name
    try:
        with archive_path.open("rb") as source:
            lead = _read_exact(source, 96, f"{label} RPM lead")
            if lead[:4] != b"\xed\xab\xee\xdb" or lead[4:6] != b"\x03\x00":
                raise BuildError(f"{label} is not a version 3 RPM package")
            _rpm_stream_header(source, f"{label} signature")
            alignment = (-source.tell()) & 7
            if alignment and any(
                _read_exact(source, alignment, f"{label} RPM alignment")
            ):
                raise BuildError(f"{label} RPM alignment bytes are nonzero")
            entries, store = _rpm_stream_header(source, label)
            name = _rpm_string(entries, store, 1000, label)
            version = _rpm_string(entries, store, 1001, label)
            release = _rpm_string(entries, store, 1002, label)
            architecture = _rpm_string(entries, store, 1022, label)
            nevra = f"{name}-{version}-{release}.{architecture}.rpm"
            expected_filename = PurePosixPath(
                urllib.parse.urlsplit(artifact_url).path
            ).name
            if nevra != expected_filename:
                raise BuildError(
                    f"locked RPM identity mismatch: expected {expected_filename}, "
                    f"got {nevra}"
                )
            if _rpm_string(entries, store, 1044, label) != source_rpm:
                raise BuildError(f"locked RPM source tag mismatch: {label}")
            if _rpm_string(entries, store, 1124, label) != "cpio":
                raise BuildError(f"{label} payload format is not cpio")
            if _rpm_string(entries, store, 1125, label) != "xz":
                raise BuildError(f"{label} payload compressor is not xz")
            with lzma.LZMAFile(source, mode="rb", format=lzma.FORMAT_XZ) as payload:
                yield payload
            if source.read(1):
                raise BuildError(f"{label} has bytes after its XZ payload")
    except BuildError:
        raise
    except (OSError, EOFError, lzma.LZMAError) as exc:
        raise BuildError(f"cannot read locked TensorRT RPM {label}: {exc}") from exc


def _consume_stream(
    source: BinaryIO,
    size: int,
    label: str,
    output: BinaryIO | None = None,
) -> str | None:
    digest = hashlib.sha256() if output is not None else None
    remaining = size
    while remaining:
        chunk = source.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise BuildError(f"{label} is truncated")
        remaining -= len(chunk)
        if output is not None:
            output.write(chunk)
            assert digest is not None
            digest.update(chunk)
    return digest.hexdigest() if digest is not None else None


def _consume_zero_padding(source: BinaryIO, size: int, label: str) -> None:
    if size and any(_read_exact(source, size, label)):
        raise BuildError(f"{label} contains nonzero bytes")


def _extract_linux_tensorrt_rpm(
    archive_path: Path,
    artifact_url: str,
    source_rpm: str,
    files: Mapping[str, Mapping[str, Any]],
    destination: Path,
) -> None:
    wanted: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for output, entry in files.items():
        member = str(entry["member"])
        if member in wanted:
            raise BuildError(f"TensorRT RPM selection repeats member {member}")
        wanted[member] = (output, entry)
    found: set[str] = set()
    seen: set[str] = set()
    position = 0
    with _rpm_xz_payload(archive_path, artifact_url, source_rpm) as payload:
        while True:
            header = _read_exact(payload, 110, f"{archive_path.name} cpio header")
            position += 110
            if header[:6] != b"070701":
                raise BuildError(
                    f"{archive_path.name} cpio member does not use newc format"
                )
            try:
                fields = [
                    int(header[6 + index * 8 : 14 + index * 8], 16)
                    for index in range(13)
                ]
            except ValueError as exc:
                raise BuildError(
                    f"{archive_path.name} cpio header has invalid fields"
                ) from exc
            mode = fields[1]
            file_size = fields[6]
            name_size = fields[11]
            if name_size < 2 or name_size > 4096 or file_size > 16 * 1024**3:
                raise BuildError(f"{archive_path.name} cpio member bounds are invalid")
            name_bytes = _read_exact(
                payload, name_size, f"{archive_path.name} cpio member name"
            )
            position += name_size
            _consume_zero_padding(
                payload,
                (-position) & 3,
                f"{archive_path.name} cpio name padding",
            )
            position = (position + 3) & ~3
            if name_bytes[-1:] != b"\0":
                raise BuildError(f"{archive_path.name} cpio name is not terminated")
            try:
                archive_name = name_bytes[:-1].decode("utf-8")
            except UnicodeError as exc:
                raise BuildError(f"{archive_path.name} cpio name is not UTF-8") from exc
            if archive_name == "TRAILER!!!":
                if file_size != 0 or mode != 0:
                    raise BuildError(f"{archive_path.name} cpio trailer is invalid")
                if payload.read(1):
                    raise BuildError(
                        f"{archive_path.name} has data after its cpio trailer"
                    )
                break
            if not archive_name.startswith("./"):
                raise BuildError(
                    f"{archive_path.name} cpio member lacks the canonical ./ prefix"
                )
            member = archive_name[2:]
            member_path = common.safe_member_path(
                member, f"{archive_path.name} cpio member"
            )
            if member in seen:
                raise BuildError(f"{archive_path.name} repeats cpio member {member}")
            seen.add(member)
            file_type = stat.S_IFMT(mode)
            if file_type == stat.S_IFDIR:
                if file_size:
                    raise BuildError(
                        f"{archive_path.name} cpio directory has content: {member}"
                    )
            elif file_type == stat.S_IFLNK:
                if file_size > 4096:
                    raise BuildError(
                        f"{archive_path.name} cpio symlink is too large: {member}"
                    )
                target_bytes = _read_exact(
                    payload,
                    file_size,
                    f"{archive_path.name} cpio symlink {member}",
                )
                try:
                    link_target = target_bytes.decode("utf-8")
                except UnicodeError as exc:
                    raise BuildError(
                        f"{archive_path.name} cpio symlink is not UTF-8: {member}"
                    ) from exc
                common._normalized_link_target(member_path, link_target)
            elif file_type == stat.S_IFREG:
                selection = wanted.get(member)
                if selection is None:
                    _consume_stream(
                        payload,
                        file_size,
                        f"{archive_path.name} cpio member {member}",
                    )
                else:
                    output_name, entry = selection
                    if file_size != entry["size"]:
                        raise BuildError(
                            f"locked TensorRT member size drifted: {member}"
                        )
                    if mode & 0o777 != entry["mode"]:
                        raise BuildError(
                            f"locked TensorRT member mode drifted: {member}"
                        )
                    target = common._member_destination(
                        destination,
                        common.safe_member_path(output_name, "TensorRT output"),
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with target.open("xb") as output:
                            digest = _consume_stream(
                                payload,
                                file_size,
                                f"{archive_path.name} cpio member {member}",
                                output,
                            )
                    except BaseException:
                        target.unlink(missing_ok=True)
                        raise
                    if digest != entry["sha256"]:
                        target.unlink(missing_ok=True)
                        raise BuildError(
                            f"locked TensorRT member SHA-256 drifted: {member}"
                        )
                    target.chmod(entry["mode"])
                    found.add(member)
            else:
                raise BuildError(
                    f"{archive_path.name} contains a special cpio member: {member}"
                )
            position += file_size
            _consume_zero_padding(
                payload,
                (-position) & 3,
                f"{archive_path.name} cpio data padding",
            )
            position = (position + 3) & ~3
    if found != set(wanted):
        raise BuildError(
            f"{archive_path.name} lacks locked TensorRT members: "
            f"{sorted(set(wanted) - found)}"
        )


def materialize_linux_tensorrt(lock: Mapping[str, Any], work_root: Path) -> Path:
    locked = lock["tensorrt"]["linux_packages"]
    root = work_root / "inputs" / "tensorrt"
    root.mkdir(parents=True, exist_ok=False)
    selected_outputs: set[str] = set()
    for role in common.LINUX_TENSORRT_PACKAGE_ROLES:
        package = locked["packages"][role]
        relative_artifact = package["artifact"]
        artifact = {
            "url": locked["base_url"] + relative_artifact["relative_path"],
            "size": relative_artifact["size"],
            "sha256": relative_artifact["sha256"],
        }
        archive = common.download_artifact(
            artifact,
            work_root / "downloads" / "tensorrt",
            f"NVIDIA TensorRT Linux {role} RPM",
        )
        try:
            _extract_linux_tensorrt_rpm(
                archive,
                artifact["url"],
                locked["source_rpm"],
                package["files"],
                root,
            )
        finally:
            try:
                archive.unlink()
            except OSError as exc:
                raise BuildError(
                    f"cannot delete consumed TensorRT RPM {archive}: {exc}"
                ) from exc
        selected_outputs.update(package["files"])
    for link_name, target_name in common.LINUX_TENSORRT_LINKER_HARDLINKS.items():
        link = common._member_destination(
            root, common.safe_member_path(link_name, "linker input")
        )
        target = common._member_destination(
            root,
            common.safe_member_path(target_name, "linker input target"),
        )
        if not target.is_file() or target.is_symlink():
            raise BuildError(f"TensorRT linker input target is not regular: {target}")
        os.link(target, link)
    expected_files = selected_outputs | set(common.LINUX_TENSORRT_LINKER_HARDLINKS)
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for entry in root.rglob("*"):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise BuildError(f"TensorRT root contains a filesystem alias: {relative}")
        if entry.is_dir():
            actual_directories.add(relative)
        elif entry.is_file():
            actual_files.add(relative)
        else:
            raise BuildError(f"TensorRT root contains a special file: {relative}")
    expected_directories = {
        PurePosixPath(path).parent.as_posix() for path in expected_files
    }
    if actual_files != expected_files or actual_directories != expected_directories:
        raise BuildError("materialized TensorRT root does not match the exact closure")
    return root


def materialize_linux_runtime(
    lock: Mapping[str, Any], work_root: Path
) -> tuple[Path, Path]:
    locked = lock["linux_runtime"]
    runtime = work_root / "inputs" / "linux-gnu-runtime"
    notices = work_root / "notices" / "gcc-runtime"
    runtime.mkdir(parents=True, exist_ok=False)
    notices.mkdir(parents=True, exist_ok=False)
    source_url = locked["source_rpm"]["url"]
    source_archive = common.download_artifact(
        locked["source_rpm"],
        work_root / "downloads" / "linux-runtime",
        "Rocky Linux GNU runtime corresponding source RPM",
    )
    for package_name, package in locked["packages"].items():
        archive = common.download_artifact(
            package["artifact"],
            work_root / "downloads" / "linux-runtime",
            f"Rocky Linux {package_name} runtime",
        )
        package_licenses = {
            output: entry
            for output, entry in locked["licenses"].items()
            if entry["package"] == package_name
        }
        wanted = {package["member"]} | {
            entry["member"] for entry in package_licenses.values()
        }
        members = _cpio_locked_members(
            _rpm_payload(
                archive,
                package["artifact"]["url"],
                source_url,
            ),
            wanted,
            archive.name,
        )
        runtime_content = members[package["member"]]
        _verify_locked_bytes(runtime_content, package, package["output"])
        runtime_output = runtime / PurePosixPath(package["output"]).name
        runtime_output.write_bytes(runtime_content)
        runtime_output.chmod(0o755)
        for output, license_entry in package_licenses.items():
            content = members[license_entry["member"]]
            _verify_locked_bytes(content, license_entry, output)
            (notices / PurePosixPath(output).name).write_bytes(content)
        archive.unlink()
    source_archive.unlink()

    provenance = notices / "gcc-runtime-PROVENANCE.txt"
    provenance_record = {
        "schema": "audio2face-gcc-runtime-provenance/1",
        "producer_image": lock["linux_toolchain"]["producer_image"],
        "binary_rpms": locked["packages"],
        "license_members": locked["licenses"],
        "source_rpm": locked["source_rpm"],
        "release_gates": [
            "Publication requires legal review of the locked GNU runtime terms.",
            "Publication requires continued distribution of the locked corresponding "
            "source RPM under its applicable terms.",
            "Publication requires the complete packaged ELF closure and driver-only "
            "clean-machine inference validation.",
        ],
    }
    provenance.write_text(
        json.dumps(provenance_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    contract = common.runtime_contract(PLATFORM_ID)
    expected_runtime = {
        PurePosixPath(entry.path).name
        for entry in contract.files_for_source("platform_runtime")
    }
    if {entry.name for entry in runtime.iterdir()} != expected_runtime:
        raise BuildError("materialized GNU runtime does not match the exact closure")
    expected_notices = {
        PurePosixPath(entry.path).name
        for source in ("platform_runtime_notice", "platform_runtime_provenance")
        for entry in contract.files_for_source(source)
    }
    if {entry.name for entry in notices.iterdir()} != expected_notices:
        raise BuildError("materialized GNU runtime notices do not match the contract")
    return runtime, notices


def materialize_linux_producer_packages(
    lock: Mapping[str, Any], work_root: Path
) -> tuple[Path, ...]:
    """Download every RPM added to the pinned Rocky producer container."""

    destination = work_root / "downloads" / "linux-producer"
    packages: list[Path] = []
    names: set[str] = set()
    for package_key in common.LINUX_PRODUCER_PACKAGES:
        package = lock["linux_toolchain"]["packages"][package_key]
        archive = common.download_artifact(
            package["artifact"],
            destination,
            f"Rocky Linux producer package {package['name']}",
        )
        if archive.name in names:
            raise BuildError(f"Linux producer RPM filenames collide: {archive.name}")
        names.add(archive.name)
        packages.append(archive)
    return tuple(packages)


class LinuxProducerRunner(common.CommandRunner):
    """Execute native build commands only in one verified Rocky OCI container."""

    def __init__(
        self,
        work_root: Path,
        docker: Path,
        container_id: str,
        host_environment: Mapping[str, str],
    ) -> None:
        self.work_root = work_root
        self.docker = docker
        self.container_id = container_id
        self.host_environment = dict(host_environment)
        self.user = f"{os.getuid()}:{os.getgid()}"

    def run(
        self,
        command: Sequence[os.PathLike[str] | str],
        *,
        env: Mapping[str, str],
        cwd: Path | None = None,
        capture: bool = False,
    ) -> str:
        args = [os.fspath(value) for value in command]
        print(f"+ [pinned Rocky producer] {shlex.join(args)}", flush=True)
        command_directory = self.work_root if cwd is None else cwd
        directory = command_directory.resolve(strict=False)
        if not (
            common._inside(directory, self.work_root.resolve())
            or common._inside(directory, common.REPOSITORY_ROOT.resolve())
        ):
            raise BuildError(
                f"Linux producer command uses an unmounted directory: {directory}"
            )
        container_environment = {
            key: value
            for key, value in env.items()
            if key
            in {
                "HOME",
                "LANG",
                "LC_ALL",
                "LD_LIBRARY_PATH",
                "PATH",
                "PM_PACKAGES_ROOT",
                "TMPDIR",
            }
        }
        if "PATH" not in container_environment:
            raise BuildError("Linux producer command environment has no PATH")
        docker_args: list[os.PathLike[str] | str] = [
            self.docker,
            "exec",
            "--user",
            self.user,
            "--workdir",
            directory,
        ]
        for key, value in sorted(container_environment.items()):
            if "\0" in value or "\n" in value or "\r" in value:
                raise BuildError(f"Linux producer environment has unsafe {key} value")
            docker_args.extend(("--env", f"{key}={value}"))
        docker_args.append(self.container_id)
        docker_args.extend(args)
        try:
            result = subprocess.run(
                [os.fspath(value) for value in docker_args],
                env=self.host_environment,
                check=True,
                text=True,
                stdout=subprocess.PIPE if capture else None,
                stderr=subprocess.STDOUT if capture else None,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            output = getattr(exc, "stdout", None)
            detail = f"\n{output}" if output else ""
            raise BuildError(
                f"pinned Rocky producer command failed: {shlex.join(args)}{detail}"
            ) from exc
        return result.stdout.rstrip("\r\n") if capture and result.stdout else ""


def _parse_linux_image_identity(output: str, lock: Mapping[str, Any]) -> None:
    lines = output.splitlines()
    if len(lines) != 4:
        raise BuildError(f"Docker returned invalid producer image identity: {output!r}")
    image_id, architecture, operating_system, raw_digests = lines
    producer = lock["linux_toolchain"]["producer_image"]
    if image_id != f"sha256:{producer['config_sha256']}":
        raise BuildError(f"Rocky producer config digest drifted: {image_id!r}")
    if architecture != producer["architecture"] or operating_system != "linux":
        raise BuildError(
            f"Rocky producer image platform drifted: {operating_system}/{architecture}"
        )
    try:
        repo_digests = json.loads(
            raw_digests,
            object_pairs_hook=duplicate_key_hook(
                BuildError,
                "Docker RepoDigests",
            ),
            parse_constant=invalid_constant_hook(
                BuildError,
                "Docker RepoDigests",
            ),
        )
    except json.JSONDecodeError as exc:
        raise BuildError("Docker returned invalid producer RepoDigests JSON") from exc
    if not isinstance(repo_digests, list) or producer["reference"] not in repo_digests:
        raise BuildError(
            "Docker did not resolve the exact locked Rocky producer digest"
        )


@contextlib.contextmanager
def linux_producer_runner(
    host_runner: common.CommandRunner,
    lock: Mapping[str, Any],
    work_root: Path,
    environment: Mapping[str, str],
    packages: Sequence[Path],
) -> Iterator[LinuxProducerRunner]:
    """Create the sole Linux compiler boundary from the locked OCI digest."""

    docker = common.require_host_program("docker", environment)
    producer = lock["linux_toolchain"]["producer_image"]
    reference = producer["reference"]
    host_runner.run(
        [docker, "pull", "--platform", "linux/amd64", reference],
        env=environment,
    )
    identity = host_runner.run(
        [
            docker,
            "image",
            "inspect",
            "--format",
            "{{.Id}}\n{{.Architecture}}\n{{.Os}}\n{{json .RepoDigests}}",
            reference,
        ],
        env=environment,
        capture=True,
    )
    _parse_linux_image_identity(identity, lock)
    for path in (common.REPOSITORY_ROOT.resolve(), work_root.resolve()):
        if "," in os.fspath(path) or "\0" in os.fspath(path):
            raise BuildError(f"Linux producer mount path is unsafe: {path}")
    container_id = host_runner.run(
        [
            docker,
            "run",
            "--detach",
            "--rm",
            "--platform",
            "linux/amd64",
            "--network",
            "none",
            "--mount",
            (
                f"type=bind,source={common.REPOSITORY_ROOT.resolve()},"
                f"target={common.REPOSITORY_ROOT.resolve()},readonly"
            ),
            "--mount",
            (f"type=bind,source={work_root.resolve()},target={work_root.resolve()}"),
            "--entrypoint",
            "/bin/bash",
            reference,
            "-lc",
            "while :; do /usr/bin/sleep 3600; done",
        ],
        env=environment,
        capture=True,
    )
    if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
        raise BuildError(
            f"Docker returned invalid producer container ID {container_id!r}"
        )
    runner = LinuxProducerRunner(
        work_root,
        docker,
        container_id,
        environment,
    )
    try:
        runner.user = "0:0"
        runner.run(
            [
                "/usr/bin/rpm",
                "-Uvh",
                "--nodeps",
                "--noscripts",
                *packages,
            ],
            env=environment,
        )
        package_names = [
            lock["linux_toolchain"]["packages"][key]["name"]
            for key in common.LINUX_PRODUCER_PACKAGES
        ]
        verification = runner.run(
            ["/usr/bin/rpm", "-V", "--nodeps", *package_names],
            env=environment,
            capture=True,
        )
        if verification:
            raise BuildError(
                "installed Rocky producer files differ from locked RPMs: "
                f"{verification}"
            )
        runner.user = f"{os.getuid()}:{os.getgid()}"
        yield runner
    finally:
        subprocess.run(
            [docker, "rm", "--force", container_id],
            env=dict(environment),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def release_environment(work_root: Path) -> dict[str, str]:
    """Create the exact host-command environment for a Linux release build."""

    home = work_root / "producer-home"
    temporary = work_root / "producer-tmp"
    home.mkdir(parents=True, exist_ok=False)
    temporary.mkdir(parents=True, exist_ok=False)
    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": os.fspath(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PM_PACKAGES_ROOT": os.fspath(work_root / "packman-cache"),
        "TMPDIR": os.fspath(temporary),
    }


def private_build_environment(
    base: Mapping[str, str],
    cuda_root: Path,
    tensorrt_root: Path,
    cmake_root: Path,
    ninja: Path,
    compiler: Path,
) -> dict[str, str]:
    environment = dict(base)
    cuda_library = cuda_root / common.LINUX_CUDA_LIBRARY_DIRECTORY
    tensorrt_library = tensorrt_root / "lib"
    existing_path = environment.get("PATH")
    if not isinstance(existing_path, str) or not existing_path:
        raise BuildError("Linux private build environment is missing PATH")
    path_entries = [
        cmake_root / "bin",
        ninja.parent,
        cuda_root / "bin",
        tensorrt_library,
        compiler.parent,
        *(Path(path) for path in existing_path.split(os.pathsep)),
    ]
    environment["PATH"] = os.pathsep.join(str(path) for path in path_entries)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        (str(cuda_library), str(tensorrt_library))
    )
    return environment


def _linux_distribution_identity(
    runner: common.CommandRunner, environment: Mapping[str, str]
) -> tuple[str, str]:
    output = runner.run(
        ["/usr/bin/cat", "/etc/os-release"],
        env=environment,
        capture=True,
    )
    values: dict[str, str] = {}
    pattern = re.compile(r'^(ID|VERSION_ID)=(?:"([^"\\]*)"|([A-Za-z0-9._-]+))$')
    for line in output.splitlines():
        match = pattern.fullmatch(line)
        if match is None:
            continue
        key = match.group(1)
        if key in values:
            raise BuildError(f"Linux producer identity repeats {key}")
        values[key] = match.group(2) if match.group(2) is not None else match.group(3)
    if set(values) != {"ID", "VERSION_ID"}:
        raise BuildError("Linux producer identity needs exact ID and VERSION_ID fields")
    return values["ID"], values["VERSION_ID"]


def _rpm_nevra_from_artifact(artifact: Mapping[str, Any]) -> str:
    filename = PurePosixPath(urllib.parse.urlsplit(artifact["url"]).path).name
    if not filename.endswith(".rpm"):
        raise BuildError(f"locked RPM artifact has an invalid filename: {filename}")
    return filename[:-4]


def _validate_linux_producer_packages(
    runner: common.CommandRunner,
    lock: Mapping[str, Any],
    environment: Mapping[str, str],
) -> None:
    rpm = Path("/usr/bin/rpm")
    toolchain = lock["linux_toolchain"]
    expected = {
        package["name"]: package["nevra"] for package in toolchain["packages"].values()
    }
    expected["glibc"] = toolchain["glibc_nevra"]
    runtime_names = {"libstdcxx": "libstdc++", "libgcc": "libgcc"}
    for key, name in runtime_names.items():
        expected[name] = _rpm_nevra_from_artifact(
            lock["linux_runtime"]["packages"][key]["artifact"]
        )
    query_format = "%{NAME}-%{VERSION}-%{RELEASE}.%{ARCH}"
    for name, expected_nevra in expected.items():
        actual = runner.run(
            [rpm, "-q", "--qf", query_format, name],
            env=environment,
            capture=True,
        )
        if actual != expected_nevra:
            raise BuildError(
                f"Rocky producer package {name} must be {expected_nevra}; "
                f"got {actual!r}"
            )


def validate_native_compiler(
    runner: common.CommandRunner,
    lock: Mapping[str, Any],
    environment: Mapping[str, str],
) -> Path:
    toolchain = lock["linux_toolchain"]
    distribution_id, distribution_version = _linux_distribution_identity(
        runner, environment
    )
    if (
        distribution_id != toolchain["distribution_id"]
        or distribution_version != toolchain["distribution_version"]
    ):
        raise BuildError(
            "linux-x64 releases require the pinned producer distribution; "
            f"got {distribution_id} {distribution_version}"
        )
    _validate_linux_producer_packages(runner, lock, environment)
    compiler = Path(toolchain["gxx_path"])
    version = runner.run([compiler, "-dumpfullversion"], env=environment, capture=True)
    if version != toolchain["gxx_version"]:
        raise BuildError(
            f"Linux producer g++ version must be {toolchain['gxx_version']}; "
            f"got {version!r}"
        )
    target = runner.run([compiler, "-dumpmachine"], env=environment, capture=True)
    if target != toolchain["gxx_target"]:
        raise BuildError(
            f"Linux producer g++ target must be {toolchain['gxx_target']}; "
            f"got {target!r}"
        )
    readelf = Path(toolchain["readelf_path"])
    readelf_output = runner.run([readelf, "--version"], env=environment, capture=True)
    first_line = readelf_output.splitlines()[0] if readelf_output else ""
    if first_line != toolchain["readelf_version"]:
        raise BuildError(
            f"Linux producer readelf must be {toolchain['readelf_version']!r}; "
            f"got {first_line!r}"
        )
    return compiler


def fetch_sdk_dependencies(
    runner: common.CommandRunner,
    sdk_source: Path,
    environment: Mapping[str, str],
) -> Path:
    script = sdk_source / "fetch_deps.sh"
    if not script.is_file() or not os.access(script, os.X_OK):
        raise BuildError(f"pinned SDK fetch script is not executable: {script}")
    runner.run([script, "release"], cwd=sdk_source, env=environment)
    ninja = sdk_source / "_deps" / "build-deps" / "ninja" / "ninja"
    if not ninja.is_file():
        raise BuildError(f"Packman did not materialize pinned Ninja: {ninja}")
    return ninja


def linux_compile_flags(lock: Mapping[str, Any]) -> tuple[str, str]:
    toolchain = lock["linux_toolchain"]
    abi_definition = f"-D_GLIBCXX_USE_CXX11_ABI={toolchain['cxx11_abi']}"
    architecture_flags = list(toolchain["architecture_flags"])
    cxx_flags = " ".join((abi_definition, *architecture_flags))
    cuda_flags = " ".join(
        (
            abi_definition,
            f"-Xcompiler={','.join(architecture_flags)}",
        )
    )
    return cxx_flags, cuda_flags


def write_provenance(
    lock: Mapping[str, Any],
    trtexec: Path,
    work_root: Path,
) -> Path:
    notices = work_root / "notices"
    notices.mkdir(parents=True, exist_ok=True)
    lock_digest = common.file_sha256(common.LOCK_PATH)
    provenance = notices / "trtexec-PROVENANCE.txt"
    linux_packages = lock["tensorrt"]["linux_packages"]
    trtexec_package = linux_packages["packages"]["trtexec"]
    trtexec_entry = trtexec_package["files"][
        common.runtime_contract(PLATFORM_ID).trtexec
    ]
    record: dict[str, Any] = {
        "schema": "audio2face-trtexec-provenance/1",
        "platform": PLATFORM_ID,
        "runtime_lock_sha256": lock_digest,
        "tensorrt_binary": {
            "version": lock["tensorrt"]["version"],
            "cuda": lock["tensorrt"]["cuda"],
            "input": {
                "base_url": linux_packages["base_url"],
                "source_rpm": linux_packages["source_rpm"],
                "packages": linux_packages["packages"],
            },
        },
        "trtexec": {
            "rpm": trtexec_package["artifact"]["relative_path"],
            "rpm_member": trtexec_entry["member"],
            "size": trtexec.stat().st_size,
            "sha256": common.file_sha256(trtexec),
        },
    }
    provenance.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


def _validate_elf64_x86_64(path: Path) -> None:
    try:
        with path.open("rb") as stream:
            header = stream.read(64)
    except OSError as exc:
        raise BuildError(f"cannot inspect packaged ELF file {path}: {exc}") from exc
    if (
        len(header) != 64
        or header[:7] != b"\x7fELF\x02\x01\x01"
        or struct.unpack_from("<H", header, 16)[0] not in {2, 3}
        or struct.unpack_from("<H", header, 18)[0] != 62
        or struct.unpack_from("<I", header, 20)[0] != 1
        or struct.unpack_from("<H", header, 52)[0] != 64
    ):
        raise BuildError(f"packaged native file is not Linux ELF64 x86-64: {path}")


def _readelf_dynamic_entries(
    output: str, path: Path
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if "Dynamic section" not in output or "There is no dynamic section" in output:
        raise BuildError(f"readelf did not report a dynamic section for {path}")
    dependencies = tuple(
        match.group(1)
        for match in re.finditer(
            r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]",
            output,
        )
    )
    if len(dependencies) != len(set(dependencies)):
        raise BuildError(f"ELF file repeats a DT_NEEDED entry: {path}")
    for dependency in dependencies:
        if (
            not dependency
            or "/" in dependency
            or "\\" in dependency
            or "\0" in dependency
        ):
            raise BuildError(f"ELF file has an unsafe DT_NEEDED entry: {path}")
    sonames = tuple(
        match.group(1)
        for match in re.finditer(
            r"\(SONAME\)\s+Library soname: \[([^\]]+)\]",
            output,
        )
    )
    rpaths = tuple(
        match.group(1)
        for match in re.finditer(
            r"\(RPATH\)\s+Library rpath: \[([^\]]*)\]",
            output,
        )
    )
    runpaths = tuple(
        match.group(1)
        for match in re.finditer(
            r"\(RUNPATH\)\s+Library runpath: \[([^\]]*)\]",
            output,
        )
    )
    return dependencies, sonames, rpaths, runpaths


def _audit_elf_dynamic_identity(
    relative: PurePosixPath,
    sonames: tuple[str, ...],
    rpaths: tuple[str, ...],
    runpaths: tuple[str, ...],
) -> None:
    library = next(
        (
            entry
            for entry in common.runtime_contract(PLATFORM_ID).libraries
            if entry.path == relative.as_posix()
        ),
        None,
    )
    if library is None:
        if relative.parts[0] == "lib":
            raise BuildError(
                f"packaged library is absent from the locked contract: {relative}"
            )
        if sonames:
            raise BuildError(f"packaged executable declares DT_SONAME: {relative}")
    else:
        expected_sonames = (library.elf_soname or relative.name,)
        if sonames != expected_sonames:
            raise BuildError(
                "packaged library DT_SONAME differs from the locked contract: "
                f"{relative}: expected {expected_sonames}, got {sonames}"
            )
    if rpaths:
        raise BuildError(f"packaged ELF file declares forbidden DT_RPATH: {relative}")
    expected_runpaths = library.elf_runpaths if library is not None else ()
    if runpaths != expected_runpaths:
        raise BuildError(
            "packaged ELF file DT_RUNPATH differs from the locked contract: "
            f"{relative}: expected {expected_runpaths}, got {runpaths}"
        )


def _glibc_version_tuple(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)(?:\.([0-9]+))?", value)
    if match is None:
        raise BuildError(f"invalid GLIBC version {value!r}")
    return tuple(int(part or 0) for part in match.groups())


def _readelf_version_sets(
    output: str,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    definitions = {name: set() for name in ("GLIBC", "GLIBCXX", "CXXABI", "GCC")}
    requirements = {name: set() for name in definitions}
    current: dict[str, set[str]] | None = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("Version definition section"):
            current = definitions
            continue
        if line.startswith("Version needs section"):
            current = requirements
            continue
        if line.startswith("Version symbols section"):
            current = None
            continue
        if current is None:
            continue
        for match in re.finditer(
            r"\bName:\s+((GLIBCXX|CXXABI|GLIBC|GCC)_[A-Za-z0-9_.]+)\b",
            line,
        ):
            current[match.group(2)].add(match.group(1).split("_", 1)[1])
    return definitions, requirements


def _audit_glibc_requirements(requirements: set[str], path: Path, maximum: str) -> None:
    if "PRIVATE" in requirements:
        raise BuildError(f"ELF file requires GLIBC_PRIVATE: {path}")
    limit = _glibc_version_tuple(maximum)
    numeric: set[str] = set()
    for version in requirements:
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,2}", version) is None:
            raise BuildError(
                f"ELF file requires invalid GLIBC version {version}: {path}"
            )
        numeric.add(version)
    excessive = sorted(
        version for version in numeric if _glibc_version_tuple(version) > limit
    )
    if excessive:
        raise BuildError(
            f"ELF file requires GLIBC newer than {maximum}: {path}: {excessive}"
        )


def _symbol_version_tuple(value: str, namespace: str) -> tuple[int, int, int]:
    if namespace == "GLIBCXX" and value.startswith("3.4"):
        return _glibc_version_tuple(value)
    if namespace in {"CXXABI", "GCC"}:
        return _glibc_version_tuple(value)
    raise BuildError(f"invalid {namespace} symbol version {value!r}")


def audit_linux_dependencies(
    runner: common.CommandRunner,
    runtime: Path,
    lock: Mapping[str, Any],
    environment: Mapping[str, str],
) -> None:
    contract = common.runtime_contract(PLATFORM_ID)
    readelf = Path(lock["linux_toolchain"]["readelf_path"])
    packaged = frozenset(
        path.name for path in common.native_runtime_files(runtime, contract)
    )
    unresolved: dict[str, list[str]] = {}
    version_definitions = {name: set() for name in ("GLIBCXX", "CXXABI", "GCC")}
    version_requirements = {name: set() for name in version_definitions}
    native_files = tuple(
        sorted(
            (
                entry
                for directory in (runtime / "bin", runtime / "lib")
                for entry in directory.iterdir()
                if entry.is_file()
            ),
            key=lambda path: path.relative_to(runtime).as_posix(),
        )
    )
    if native_files != tuple(
        sorted(
            common.native_runtime_files(runtime, contract),
            key=lambda path: path.relative_to(runtime).as_posix(),
        )
    ):
        raise BuildError("Linux ELF audit input differs from the runtime contract")
    for path in native_files:
        relative = PurePosixPath(path.relative_to(runtime).as_posix())
        _validate_elf64_x86_64(path)
        dynamic = runner.run(
            [readelf, "--wide", "--dynamic", path],
            env=environment,
            capture=True,
        )
        dependencies, sonames, rpaths, runpaths = _readelf_dynamic_entries(
            dynamic, path
        )
        _audit_elf_dynamic_identity(relative, sonames, rpaths, runpaths)
        if "libnvJitLink.so.12" in dependencies and (
            "libnvJitLink.so.12" not in packaged
        ):
            raise BuildError(
                f"{relative} requires unreviewed libnvJitLink.so.12; "
                "do not publish until the exact runtime contract is reviewed"
            )
        missing = sorted(
            name
            for name in dependencies
            if name not in packaged and name not in LINUX_EXTERNAL_LIBRARIES
        )
        if missing:
            unresolved[relative.as_posix()] = missing
        versions = runner.run(
            [readelf, "--wide", "--version-info", path],
            env=environment,
            capture=True,
        )
        definitions, requirements = _readelf_version_sets(versions)
        _audit_glibc_requirements(
            requirements["GLIBC"],
            path,
            lock["linux_toolchain"]["glibc_version"],
        )
        if path.name in {"libstdc++.so.6", "libgcc_s.so.1"}:
            for namespace in version_definitions:
                version_definitions[namespace].update(definitions[namespace])
        for namespace in version_requirements:
            version_requirements[namespace].update(requirements[namespace])
    if unresolved:
        raise BuildError(
            "Linux runtime has undeclared non-system dependencies: "
            + json.dumps(unresolved, sort_keys=True)
        )
    expected_definition_maxima = {
        "GLIBCXX": "3.4.25",
        "CXXABI": "1.3.11",
        "GCC": "7.0.0",
    }
    for namespace, expected_maximum in expected_definition_maxima.items():
        numeric_definitions = {
            version
            for version in version_definitions[namespace]
            if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,2}", version)
        }
        if not numeric_definitions:
            raise BuildError(
                f"packaged GNU runtime defines no numeric {namespace} symbol versions"
            )
        actual_maximum = max(
            numeric_definitions,
            key=lambda version: _symbol_version_tuple(version, namespace),
        )
        if actual_maximum != expected_maximum:
            raise BuildError(
                f"packaged GNU runtime {namespace} definition ceiling must be "
                f"{expected_maximum}; got {actual_maximum}"
            )
        unsupported = sorted(
            version_requirements[namespace] - version_definitions[namespace]
        )
        if unsupported:
            raise BuildError(
                f"packaged ELFs require unsupported {namespace} versions: {unsupported}"
            )


def build_linux_runtime(work_root: Path) -> Path:
    """Build one complete Linux x64 runtime in an isolated work tree."""

    common.require_native_target(PLATFORM_ID)
    lock = common.load_lock()
    host_runner = common.CommandRunner()
    environment = release_environment(work_root)
    git = common.require_host_program("git", environment)

    sdk_source = work_root / "source" / "audio2face-sdk"
    common.checkout_exact(
        host_runner,
        git,
        lock["audio2face_sdk"]["repository"],
        lock["audio2face_sdk"]["commit"],
        sdk_source,
        env=environment,
    )
    cmake_root = common.materialize_archive_root(
        lock["cmake"]["artifacts"][PLATFORM_ID],
        "cmake",
        PLATFORM_ID,
        work_root,
    )
    cuda_root = common.materialize_cuda(lock, PLATFORM_ID, work_root)
    finalize_linux_cuda_toolkit(cuda_root)
    tensorrt_root = materialize_linux_tensorrt(lock, work_root)
    trtexec = common.pinned_trtexec(tensorrt_root, PLATFORM_ID)
    linux_runtime, linux_notices = materialize_linux_runtime(lock, work_root)
    producer_packages = materialize_linux_producer_packages(lock, work_root)
    ninja = fetch_sdk_dependencies(
        host_runner,
        sdk_source,
        environment,
    )

    with linux_producer_runner(
        host_runner,
        lock,
        work_root,
        environment,
        producer_packages,
    ) as runner:
        compiler = validate_native_compiler(runner, lock, environment)
        cmake = common.validate_cmake(
            runner,
            cmake_root,
            PLATFORM_ID,
            lock["cmake"]["version"],
            environment,
        )
        build_environment = private_build_environment(
            environment,
            cuda_root,
            tensorrt_root,
            cmake_root,
            ninja,
            compiler,
        )
        trtexec_provenance = write_provenance(
            lock,
            trtexec,
            work_root,
        )
        runtime = work_root / "runtime" / PLATFORM_ID
        if runtime.exists() or runtime.is_symlink():
            raise BuildError(f"runtime package output already exists: {runtime}")
        contract = common.runtime_contract(PLATFORM_ID)
        bundle_manifest = work_root / "notices" / "bundle.json"
        bundle_manifest.write_text(
            json.dumps(contract.manifest(), indent=2) + "\n", encoding="utf-8"
        )
        external_files = common.runtime_package_map(
            contract,
            bundle_manifest=bundle_manifest,
            sdk_source=sdk_source,
            cuda_runtime=(cuda_root / common.LINUX_CUDA_LIBRARY_DIRECTORY),
            tensorrt_runtime=tensorrt_root / "lib",
            platform_runtime=linux_runtime,
            platform_notices=linux_notices,
            platform_metadata=None,
            platform_provenance=linux_notices / "gcc-runtime-PROVENANCE.txt",
            trtexec=trtexec,
            trtexec_provenance=trtexec_provenance,
        )
        cxx_flags, cuda_flags = linux_compile_flags(lock)
        common.configure_and_package_worker(
            runner,
            cmake,
            ninja,
            compiler,
            cuda_root / "bin" / "nvcc",
            sdk_source,
            cuda_root,
            tensorrt_root,
            runtime,
            contract,
            external_files,
            work_root,
            build_environment,
            (
                f"-DCMAKE_CXX_FLAGS:STRING={cxx_flags}",
                f"-DCMAKE_CUDA_FLAGS:STRING={cuda_flags}",
            ),
            (
                ("CMAKE_CXX_FLAGS", cxx_flags),
                ("CMAKE_CUDA_FLAGS", cuda_flags),
            ),
        )
        common.validate_runtime_package(runtime, PLATFORM_ID)
        audit_linux_dependencies(
            runner,
            runtime,
            lock,
            build_environment,
        )
    return runtime
