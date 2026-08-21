#!/usr/bin/env python3
"""Build one native Audio2Face GPU runtime for extension embedding.

All CUDA, TensorRT, CMake, SDK, and Windows CRT release inputs come from
worker/runtime-lock.json. Windows contributes its pinned Visual C++ toolset,
Windows SDK, Git, and Python. Linux contributes Docker, Git, and Python; its
compiler and system headers live in the locked Rocky producer. Installed
CUDA/TensorRT software is never searched or consumed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import lzma
import os
import platform as platform_module
import re
import shlex
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterator, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from audio2face.runtime_contract import (  # noqa: E402
    RUNTIME_CONTRACTS,
    RUNTIME_MANIFEST_FIELDS,
    RuntimePackagedFile,
    RuntimePlatformContract,
    runtime_contract,
)
from audio2face.strict_json import (  # noqa: E402
    duplicate_key_hook,
    invalid_constant_hook,
)


LOCK_PATH = REPOSITORY_ROOT / "worker" / "runtime-lock.json"
LOCK_SCHEMA = "audio2face-runtime-inputs/1"
SUPPORTED_PLATFORMS = tuple(RUNTIME_CONTRACTS)
CUDA_COMPONENTS = (
    "cuda_nvcc",
    "cuda_cudart",
    "cuda_cccl",
    "cuda_nvtx",
    "cuda_nvrtc",
    "cuda_profiler_api",
    "libcublas",
    "libcurand",
)
CUDA_MANIFEST_PLATFORMS = {
    "windows-x64": "windows-x86_64",
    "linux-x64": "linux-x86_64",
}
MSVC_RUNTIME_FILES = tuple(
    PurePosixPath(entry.path).name
    for entry in runtime_contract("windows-x64").files_for_source(
        "platform_runtime"
    )
)
LINUX_PRODUCER_PACKAGES = (
    "gcc_toolset_runtime",
    "binutils",
    "gcc",
    "gxx",
    "libstdcxx_devel",
    "glibc_devel",
    "glibc_headers",
    "kernel_headers",
    "libmpc",
)
WINDOWS_VCVARS_ENVIRONMENT_KEYS = (
    "COMSPEC",
    "INCLUDE",
    "LIB",
    "LIBPATH",
    "PATH",
    "PATHEXT",
    "SystemRoot",
    "UCRTVersion",
    "UniversalCRTSdkDir",
    "VCINSTALLDIR",
    "VCToolsInstallDir",
    "VCToolsRedistDir",
    "VCToolsVersion",
    "VisualStudioVersion",
    "VSCMD_ARG_HOST_ARCH",
    "VSCMD_ARG_TGT_ARCH",
    "VSINSTALLDIR",
    "WindowsLibPath",
    "WindowsSdkBinPath",
    "WindowsSdkDir",
    "WindowsSDKLibVersion",
    "WindowsSDKVersion",
)
WINDOWS_REQUIRED_ENVIRONMENT_KEYS = frozenset(
    {
        "COMSPEC",
        "INCLUDE",
        "LIB",
        "LIBPATH",
        "PATH",
        "SystemRoot",
        "UCRTVersion",
        "UniversalCRTSdkDir",
        "VCINSTALLDIR",
        "VCToolsInstallDir",
        "VCToolsVersion",
        "VisualStudioVersion",
        "VSCMD_ARG_HOST_ARCH",
        "VSCMD_ARG_TGT_ARCH",
        "VSINSTALLDIR",
        "WindowsSdkBinPath",
        "WindowsSdkDir",
        "WindowsSDKLibVersion",
        "WindowsSDKVersion",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
GPU_PATH_MARKERS = (
    "cuda",
    "tensorrt",
    "nvinfer",
    "nvonnx",
    "cublas",
    "curand",
    "nvrtc",
    "nvcc",
)
WINDOWS_SYSTEM_DLLS = frozenset(
    {
        "advapi32.dll",
        "bcrypt.dll",
        "cfgmgr32.dll",
        "combase.dll",
        "crypt32.dll",
        "cryptbase.dll",
        "d3d11.dll",
        "d3d12.dll",
        "dbghelp.dll",
        "devobj.dll",
        "dnsapi.dll",
        "dxcore.dll",
        "dxgi.dll",
        "gdi32.dll",
        "gdi32full.dll",
        "imagehlp.dll",
        "imm32.dll",
        "iphlpapi.dll",
        "kernel32.dll",
        "kernelbase.dll",
        "msvcp_win.dll",
        "msvcrt.dll",
        "netapi32.dll",
        "normaliz.dll",
        "ntdll.dll",
        "ole32.dll",
        "oleaut32.dll",
        "powrprof.dll",
        "profapi.dll",
        "psapi.dll",
        "rpcrt4.dll",
        "sechost.dll",
        "setupapi.dll",
        "shell32.dll",
        "shlwapi.dll",
        "sspicli.dll",
        "ucrtbase.dll",
        "user32.dll",
        "userenv.dll",
        "version.dll",
        "winmm.dll",
        "wintrust.dll",
        "wldap32.dll",
        "ws2_32.dll",
    }
)
WINDOWS_DRIVER_DLLS = frozenset({"nvcuda.dll"})
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


class BuildError(RuntimeError):
    """A release invariant was not satisfied."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BuildError(f"{label} must be a JSON object")
    return value


def _field(value: Mapping[str, Any], name: str, label: str) -> Any:
    if name not in value:
        raise BuildError(f"{label} is missing required field {name!r}")
    return value[name]


def _keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise BuildError(
            f"{label} keys must be {sorted(expected)}; got {sorted(actual)}"
        )


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise BuildError(f"{label} must be a non-empty, trimmed string")
    return value


def _sha256(value: Any, label: str) -> str:
    digest = _string(value, label)
    if not SHA256_RE.fullmatch(digest):
        raise BuildError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _size(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BuildError(f"{label} must be a positive integer")
    return value


def _https_url(value: Any, label: str, *, directory: bool = False) -> str:
    url = _string(value, label)
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise BuildError(f"{label} must be a direct credential-free HTTPS URL")
    if directory != url.endswith("/"):
        suffix = "end in /" if directory else "name a file"
        raise BuildError(f"{label} must {suffix}")
    return url


def _archive_path(value: Any, label: str) -> str:
    path = _string(value, label)
    _safe_member_path(path, label)
    if path.endswith("/"):
        raise BuildError(f"{label} must name a file")
    return path


def _validate_artifact(value: Any, label: str) -> dict[str, Any]:
    artifact = _object(value, label)
    _keys(artifact, {"url", "size", "sha256"}, label)
    _https_url(artifact["url"], f"{label}.url")
    _size(artifact["size"], f"{label}.size")
    _sha256(artifact["sha256"], f"{label}.sha256")
    return artifact


def _validate_platform_artifacts(value: Any, label: str) -> dict[str, Any]:
    artifacts = _object(value, label)
    _keys(artifacts, set(SUPPORTED_PLATFORMS), label)
    for platform_id in SUPPORTED_PLATFORMS:
        _validate_artifact(artifacts[platform_id], f"{label}.{platform_id}")
    return artifacts


def load_lock() -> dict[str, Any]:
    """Load and strictly validate the only supported release-input schema."""

    path = LOCK_PATH
    try:
        raw = path.read_text(encoding="utf-8")
        data = _object(
            json.loads(
                raw,
                object_pairs_hook=duplicate_key_hook(BuildError, "JSON"),
                parse_constant=invalid_constant_hook(BuildError, "JSON"),
            ),
            str(path),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot read runtime lock {path}: {exc}") from exc

    _keys(
        data,
        {
            "schema",
            "audio2face_sdk",
            "tensorrt_source",
            "cmake",
            "cuda",
            "tensorrt",
            "msvc_runtime",
            "windows_toolchain",
            "linux_toolchain",
            "linux_runtime",
        },
        "runtime lock",
    )
    if data["schema"] != LOCK_SCHEMA:
        raise BuildError(f"unsupported runtime lock schema {data['schema']!r}")

    sdk = _object(data["audio2face_sdk"], "audio2face_sdk")
    _keys(sdk, {"version", "repository", "commit"}, "audio2face_sdk")
    _string(sdk["version"], "audio2face_sdk.version")
    _https_url(sdk["repository"], "audio2face_sdk.repository")
    if not COMMIT_RE.fullmatch(_string(sdk["commit"], "audio2face_sdk.commit")):
        raise BuildError("audio2face_sdk.commit must be a lowercase full commit ID")

    source = _object(data["tensorrt_source"], "tensorrt_source")
    _keys(
        source,
        {"version", "tag", "repository", "commit", "submodules"},
        "tensorrt_source",
    )
    for field in ("version", "tag"):
        _string(source[field], f"tensorrt_source.{field}")
    _https_url(source["repository"], "tensorrt_source.repository")
    if not COMMIT_RE.fullmatch(_string(source["commit"], "tensorrt_source.commit")):
        raise BuildError("tensorrt_source.commit must be a lowercase full commit ID")
    submodules = _object(source["submodules"], "tensorrt_source.submodules")
    if not submodules:
        raise BuildError("tensorrt_source.submodules cannot be empty")
    for submodule_path, commit in submodules.items():
        _archive_path(submodule_path, "TensorRT submodule path")
        if not COMMIT_RE.fullmatch(_string(commit, f"submodule {submodule_path}")):
            raise BuildError(f"submodule {submodule_path} needs a full commit ID")

    cmake = _object(data["cmake"], "cmake")
    _keys(cmake, {"version", "artifacts"}, "cmake")
    _string(cmake["version"], "cmake.version")
    _validate_platform_artifacts(cmake["artifacts"], "cmake.artifacts")

    cuda = _object(data["cuda"], "cuda")
    _keys(cuda, {"version", "base_url", "manifest", "components"}, "cuda")
    _string(cuda["version"], "cuda.version")
    _https_url(cuda["base_url"], "cuda.base_url", directory=True)
    _validate_artifact(cuda["manifest"], "cuda.manifest")
    components = _object(cuda["components"], "cuda.components")
    _keys(components, set(CUDA_COMPONENTS), "cuda.components")
    for component_name in CUDA_COMPONENTS:
        label = f"cuda.components.{component_name}"
        component = _object(components[component_name], label)
        _keys(component, {"version", "artifacts"}, label)
        _string(component["version"], f"{label}.version")
        artifacts = _object(component["artifacts"], f"{label}.artifacts")
        _keys(artifacts, set(SUPPORTED_PLATFORMS), f"{label}.artifacts")
        for platform_id in SUPPORTED_PLATFORMS:
            artifact_label = f"{label}.artifacts.{platform_id}"
            artifact = _object(artifacts[platform_id], artifact_label)
            _keys(artifact, {"relative_path", "size", "sha256"}, artifact_label)
            _archive_path(artifact["relative_path"], f"{artifact_label}.relative_path")
            _size(artifact["size"], f"{artifact_label}.size")
            _sha256(artifact["sha256"], f"{artifact_label}.sha256")

    tensorrt = _object(data["tensorrt"], "tensorrt")
    _keys(tensorrt, {"version", "cuda", "artifacts"}, "tensorrt")
    _string(tensorrt["version"], "tensorrt.version")
    _string(tensorrt["cuda"], "tensorrt.cuda")
    _validate_platform_artifacts(tensorrt["artifacts"], "tensorrt.artifacts")

    msvc = _object(data["msvc_runtime"], "msvc_runtime")
    _keys(
        msvc,
        {
            "product_version",
            "package_id",
            "package_version",
            "artifact",
            "manifest_member",
            "files",
        },
        "msvc_runtime",
    )
    for field in ("product_version", "package_id", "package_version"):
        _string(msvc[field], f"msvc_runtime.{field}")
    _validate_artifact(msvc["artifact"], "msvc_runtime.artifact")
    _archive_path(msvc["manifest_member"], "msvc_runtime.manifest_member")
    files = _object(msvc["files"], "msvc_runtime.files")
    _keys(files, set(MSVC_RUNTIME_FILES), "msvc_runtime.files")
    for filename in MSVC_RUNTIME_FILES:
        label = f"msvc_runtime.files.{filename}"
        file_entry = _object(files[filename], label)
        _keys(file_entry, {"member", "size", "sha256"}, label)
        _archive_path(file_entry["member"], f"{label}.member")
        _size(file_entry["size"], f"{label}.size")
        _sha256(file_entry["sha256"], f"{label}.sha256")

    toolchain = _object(data["windows_toolchain"], "windows_toolchain")
    _keys(
        toolchain,
        {"vctools_version", "cl_version", "windows_sdk_version"},
        "windows_toolchain",
    )
    for field in ("vctools_version", "cl_version"):
        version = _string(toolchain[field], f"windows_toolchain.{field}")
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+){2,3}", version) is None:
            raise BuildError(f"windows_toolchain.{field} is not an exact version")
    windows_sdk_version = _string(
        toolchain["windows_sdk_version"],
        "windows_toolchain.windows_sdk_version",
    )
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+){3}\\", windows_sdk_version) is None:
        raise BuildError(
            "windows_toolchain.windows_sdk_version must be the exact vcvars value"
        )

    linux_toolchain = _object(data["linux_toolchain"], "linux_toolchain")
    _keys(
        linux_toolchain,
        {
            "distribution_id",
            "distribution_version",
            "glibc_version",
            "glibc_nevra",
            "producer_image",
            "packages",
            "gxx_path",
            "gxx_version",
            "gxx_target",
            "readelf_path",
            "readelf_version",
            "cxx11_abi",
            "architecture_flags",
        },
        "linux_toolchain",
    )
    for field in (
        "distribution_id",
        "distribution_version",
        "glibc_version",
        "glibc_nevra",
        "gxx_path",
        "gxx_version",
        "gxx_target",
        "readelf_path",
        "readelf_version",
    ):
        _string(linux_toolchain[field], f"linux_toolchain.{field}")
    if linux_toolchain["gxx_path"] != (
        "/opt/rh/gcc-toolset-11/root/usr/bin/g++"
    ):
        raise BuildError("linux_toolchain.gxx_path is not the pinned Toolset 11 path")
    if linux_toolchain["readelf_path"] != (
        "/opt/rh/gcc-toolset-11/root/usr/bin/readelf"
    ):
        raise BuildError(
            "linux_toolchain.readelf_path is not the pinned Toolset 11 path"
        )
    producer = _object(
        linux_toolchain["producer_image"], "linux_toolchain.producer_image"
    )
    _keys(
        producer,
        {
            "reference",
            "architecture",
            "config_sha256",
        },
        "linux_toolchain.producer_image",
    )
    reference = _string(producer["reference"], "producer_image.reference")
    reference_match = re.fullmatch(
        r"quay\.io/rockylinux/rockylinux@sha256:([0-9a-f]{64})", reference
    )
    if reference_match is None:
        raise BuildError("producer_image.reference must pin the Rocky OCI digest")
    if producer["architecture"] != "amd64":
        raise BuildError("producer_image.architecture must be amd64")
    _sha256(producer["config_sha256"], "producer_image.config_sha256")

    producer_packages = _object(
        linux_toolchain["packages"], "linux_toolchain.packages"
    )
    _keys(
        producer_packages,
        set(LINUX_PRODUCER_PACKAGES),
        "linux_toolchain.packages",
    )
    for package_name, package_value in producer_packages.items():
        label = f"linux_toolchain.packages.{package_name}"
        package = _object(package_value, label)
        _keys(package, {"name", "nevra", "artifact"}, label)
        name = _string(package["name"], f"{label}.name")
        nevra = _string(package["nevra"], f"{label}.nevra")
        if not nevra.startswith(f"{name}-"):
            raise BuildError(f"{label}.nevra does not match its package name")
        artifact = _validate_artifact(package["artifact"], f"{label}.artifact")
        if PurePosixPath(urllib.parse.urlsplit(artifact["url"]).path).name != (
            f"{nevra}.rpm"
        ):
            raise BuildError(f"{label}.artifact URL does not match its NEVRA")
    if isinstance(linux_toolchain["cxx11_abi"], bool) or (
        linux_toolchain["cxx11_abi"] != 0
    ):
        raise BuildError("linux_toolchain.cxx11_abi must be exactly 0")
    architecture_flags = linux_toolchain["architecture_flags"]
    if architecture_flags != ["-march=x86-64", "-mtune=generic"]:
        raise BuildError(
            "linux_toolchain.architecture_flags must pin x86-64/generic"
        )

    linux_runtime = _object(data["linux_runtime"], "linux_runtime")
    _keys(
        linux_runtime,
        {"source_rpm", "packages", "licenses"},
        "linux_runtime",
    )
    source_rpm = _validate_artifact(
        linux_runtime["source_rpm"], "linux_runtime.source_rpm"
    )
    if not source_rpm["url"].endswith(".src.rpm"):
        raise BuildError("linux_runtime.source_rpm must name a source RPM")
    runtime_packages = _object(
        linux_runtime["packages"], "linux_runtime.packages"
    )
    _keys(runtime_packages, {"libstdcxx", "libgcc"}, "linux_runtime.packages")
    for package_name, package_value in runtime_packages.items():
        label = f"linux_runtime.packages.{package_name}"
        package = _object(package_value, label)
        _keys(
            package,
            {"artifact", "member", "output", "size", "sha256"},
            label,
        )
        artifact = _validate_artifact(package["artifact"], f"{label}.artifact")
        if not artifact["url"].endswith(".x86_64.rpm"):
            raise BuildError(f"{label}.artifact must name an x86_64 RPM")
        _archive_path(package["member"], f"{label}.member")
        output = _archive_path(package["output"], f"{label}.output")
        if not output.startswith("lib/") or len(PurePosixPath(output).parts) != 2:
            raise BuildError(f"{label}.output must be a flat lib/ path")
        _size(package["size"], f"{label}.size")
        _sha256(package["sha256"], f"{label}.sha256")
    locked_runtime_outputs = {
        package["output"] for package in runtime_packages.values()
    }
    expected_runtime_outputs = {
        entry.path
        for entry in runtime_contract("linux-x64").files_for_source(
            "platform_runtime"
        )
    }
    if locked_runtime_outputs != expected_runtime_outputs:
        raise BuildError(
            "linux_runtime package outputs must be the exact GNU runtime pair"
        )

    runtime_licenses = _object(
        linux_runtime["licenses"], "linux_runtime.licenses"
    )
    expected_license_outputs = {
        entry.path
        for entry in runtime_contract("linux-x64").files_for_source(
            "platform_runtime_notice"
        )
    }
    _keys(runtime_licenses, expected_license_outputs, "linux_runtime.licenses")
    for output, license_value in runtime_licenses.items():
        label = f"linux_runtime.licenses.{output}"
        license_entry = _object(license_value, label)
        _keys(
            license_entry,
            {"package", "member", "size", "sha256"},
            label,
        )
        package_name = _string(license_entry["package"], f"{label}.package")
        if package_name not in runtime_packages:
            raise BuildError(f"{label}.package does not name a locked runtime RPM")
        _archive_path(license_entry["member"], f"{label}.member")
        _size(license_entry["size"], f"{label}.size")
        _sha256(license_entry["sha256"], f"{label}.sha256")

    return data


def detect_host_platform() -> str:
    """Return the only native release platform matching this process."""

    system = sys.platform
    machine = platform_module.machine()
    if system == "win32" and machine == "AMD64":
        return "windows-x64"
    if system == "linux" and machine == "x86_64":
        return "linux-x64"
    raise BuildError(f"unsupported release host {system}/{machine}")


def require_native_target(platform_id: str) -> None:
    host = detect_host_platform()
    if host != platform_id:
        raise BuildError(
            f"--platform {platform_id} does not match native host {host}; "
            "cross-compilation is not supported"
        )
    if sys.maxsize <= 2**32:
        raise BuildError("the release builder requires a native 64-bit Python")


def _safe_member_path(name: str, label: str = "archive member") -> PurePosixPath:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise BuildError(f"unsafe {label}: {name!r}")
    if name.startswith("/") or name.startswith("//") or WINDOWS_DRIVE_RE.match(name):
        raise BuildError(f"absolute {label} is forbidden: {name!r}")
    path = PurePosixPath(name)
    if not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise BuildError(f"non-canonical {label}: {name!r}")
    return path


def _normalized_link_target(member: PurePosixPath, linkname: str) -> PurePosixPath:
    if (
        not linkname
        or "\x00" in linkname
        or "\\" in linkname
        or linkname.startswith("/")
        or WINDOWS_DRIVE_RE.match(linkname)
    ):
        raise BuildError(f"unsafe archive link target {linkname!r}")
    parts: list[str] = list(member.parent.parts)
    for part in PurePosixPath(linkname).parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                raise BuildError(f"archive link escapes extraction root: {linkname!r}")
            parts.pop()
        else:
            parts.append(part)
    if not parts:
        raise BuildError(f"archive link targets extraction root: {linkname!r}")
    return PurePosixPath(*parts)


def _member_destination(root: Path, member: PurePosixPath) -> Path:
    destination = root.joinpath(*member.parts)
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise BuildError(f"archive member escapes extraction root: {member}") from exc
    return destination


def _register_member(
    member: PurePosixPath,
    seen: set[str],
    folded: dict[str, str],
    *,
    case_insensitive: bool,
) -> None:
    name = member.as_posix()
    if name in seen:
        raise BuildError(f"archive contains duplicate member {name!r}")
    seen.add(name)
    if case_insensitive:
        folded_name = name.casefold()
        previous = folded.get(folded_name)
        if previous is not None and previous != name:
            raise BuildError(
                f"archive paths collide on Windows: {previous!r} and {name!r}"
            )
        folded[folded_name] = name


def _validated_zip_infos(
    archive: zipfile.ZipFile, *, case_insensitive: bool
) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    result: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    seen: set[str] = set()
    folded: dict[str, str] = {}
    for info in archive.infolist():
        member = _safe_member_path(info.filename)
        _register_member(member, seen, folded, case_insensitive=case_insensitive)
        if info.flag_bits & 0x1:
            raise BuildError(f"encrypted ZIP member is forbidden: {info.filename}")
        mode = info.external_attr >> 16
        if mode and stat.S_ISLNK(mode):
            raise BuildError(f"ZIP symlink is forbidden: {info.filename}")
        if mode and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise BuildError(f"special ZIP member is forbidden: {info.filename}")
        result.append((info, member))
    if not result:
        raise BuildError("archive is empty")
    return result


def safe_extract_zip(archive_path: Path, destination: Path, platform_id: str) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = _validated_zip_infos(
                archive, case_insensitive=platform_id == "windows-x64"
            )
            for info, member in infos:
                target = _member_destination(destination, member)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                mode = info.external_attr >> 16
                if mode:
                    target.chmod(mode & 0o777)
    except (OSError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise BuildError(f"cannot safely extract {archive_path}: {exc}") from exc


def safe_extract_tar(archive_path: Path, destination: Path, platform_id: str) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            entries: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
            seen: set[str] = set()
            folded: dict[str, str] = {}
            regular_members: set[str] = set()
            for info in archive.getmembers():
                member = _safe_member_path(info.name)
                _register_member(
                    member,
                    seen,
                    folded,
                    case_insensitive=platform_id == "windows-x64",
                )
                if not (
                    info.isdir()
                    or info.isreg()
                    or info.issym()
                    or info.islnk()
                ):
                    raise BuildError(f"special TAR member is forbidden: {info.name}")
                if info.issym():
                    _normalized_link_target(member, info.linkname)
                elif info.islnk():
                    target = _safe_member_path(info.linkname, "hard-link target")
                    if target.as_posix() == member.as_posix():
                        raise BuildError(f"self-referential hard link: {info.name}")
                elif info.isreg():
                    regular_members.add(member.as_posix())
                entries.append((info, member))
            if not entries:
                raise BuildError("archive is empty")

            for info, member in entries:
                target = _member_destination(destination, member)
                if info.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif info.isreg():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(info)
                    if source is None:
                        raise BuildError(f"cannot read TAR member {info.name}")
                    with source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    target.chmod(info.mode & 0o777)

            for info, member in entries:
                target = _member_destination(destination, member)
                if info.islnk():
                    linked = _safe_member_path(info.linkname, "hard-link target")
                    if linked.as_posix() not in regular_members:
                        raise BuildError(
                            f"hard link does not target a regular member: {info.linkname}"
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(_member_destination(destination, linked), target)

            for info, member in entries:
                if not info.issym():
                    continue
                _normalized_link_target(member, info.linkname)
                target = _member_destination(destination, member)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(info.linkname, target)
    except (OSError, tarfile.TarError) as exc:
        raise BuildError(f"cannot safely extract {archive_path}: {exc}") from exc


def safe_extract(archive_path: Path, destination: Path, platform_id: str) -> None:
    name = archive_path.name.lower()
    if name.endswith((".zip", ".vsix")):
        safe_extract_zip(archive_path, destination, platform_id)
        return
    if name.endswith((".tar.gz", ".tar.xz")):
        safe_extract_tar(archive_path, destination, platform_id)
        return
    raise BuildError(f"unsupported locked archive format: {archive_path.name}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _url_filename(url: str) -> str:
    name = PurePosixPath(urllib.parse.urlsplit(url).path).name
    _safe_member_path(name, "download filename")
    return name


def download_artifact(
    artifact: Mapping[str, Any], destination_directory: Path, label: str
) -> Path:
    """Download one direct HTTPS artifact and verify bytes before use."""

    url = str(artifact["url"])
    expected_size = int(artifact["size"])
    expected_sha256 = str(artifact["sha256"])
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / _url_filename(url)
    if destination.exists() or destination.is_symlink():
        raise BuildError(f"download destination already exists: {destination}")

    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise BuildError(f"stale partial download exists: {temporary}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Audio2Face-runtime-release-builder/1"},
        method="GET",
    )
    digest = hashlib.sha256()
    count = 0
    print(f"Downloading {label}: {url}", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            final_url = response.geturl()
            if urllib.parse.urlsplit(final_url).scheme != "https":
                raise BuildError(f"{label} redirected away from HTTPS: {final_url}")
            with temporary.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    if count > expected_size:
                        raise BuildError(
                            f"{label} exceeded locked size {expected_size}"
                        )
                    digest.update(chunk)
                    output.write(chunk)
    except BuildError:
        temporary.unlink(missing_ok=True)
        raise
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise BuildError(f"cannot download {label}: {exc}") from exc
    if count != expected_size:
        temporary.unlink(missing_ok=True)
        raise BuildError(f"{label} size mismatch: expected {expected_size}, got {count}")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise BuildError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    temporary.replace(destination)
    return destination


def _archive_root_name(archive: Path) -> str:
    for suffix in (".tar.gz", ".tar.xz", ".zip"):
        if archive.name.endswith(suffix):
            root = archive.name[: -len(suffix)]
            if root:
                return root
    raise BuildError(f"locked archive has no canonical root rule: {archive.name}")


def exact_archive_root(extracted: Path, archive: Path, label: str) -> Path:
    children = list(extracted.iterdir())
    expected = _archive_root_name(archive)
    if (
        len(children) != 1
        or children[0].name != expected
        or not children[0].is_dir()
        or children[0].is_symlink()
    ):
        names = [child.name for child in children]
        raise BuildError(
            f"{label} archive root must be {expected!r}; got {names}"
        )
    return children[0]


def merge_component_tree(source: Path, destination: Path, label: str) -> None:
    """Merge a CUDA component into one private toolkit without overwrites."""

    destination.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.rglob("*"), key=lambda path: path.as_posix()):
        relative = item.relative_to(source)
        if relative == Path("LICENSE"):
            continue
        target = destination / relative
        if item.is_symlink():
            link_target = os.readlink(item)
            if target.exists() or target.is_symlink():
                raise BuildError(f"{label} collides at {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(link_target, target)
        elif item.is_dir():
            if target.exists() and not target.is_dir():
                raise BuildError(f"{label} collides at {relative}")
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            if target.exists() or target.is_symlink():
                raise BuildError(f"{label} collides at {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target, follow_symlinks=False)
        else:
            raise BuildError(f"{label} contains unsupported file {relative}")


def validate_cuda_manifest(
    manifest_path: Path, lock: Mapping[str, Any], platform_id: str
) -> None:
    try:
        manifest = _object(
            json.loads(
                manifest_path.read_text(encoding="utf-8"),
                object_pairs_hook=duplicate_key_hook(BuildError, "CUDA manifest"),
                parse_constant=invalid_constant_hook(BuildError, "CUDA manifest"),
            ),
            "CUDA manifest",
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot parse pinned CUDA manifest: {exc}") from exc
    cuda = lock["cuda"]
    if _field(manifest, "release_product", "CUDA manifest") != "cuda":
        raise BuildError("pinned CUDA manifest is not a CUDA product manifest")
    if _field(manifest, "release_label", "CUDA manifest") != cuda["version"]:
        raise BuildError(
            "CUDA manifest release label does not match runtime-lock.json"
        )
    manifest_platform = CUDA_MANIFEST_PLATFORMS[platform_id]
    for component_name in CUDA_COMPONENTS:
        locked_component = cuda["components"][component_name]
        actual_component = _object(
            _field(manifest, component_name, "CUDA manifest"),
            f"CUDA manifest {component_name}",
        )
        if _field(
            actual_component,
            "version",
            f"CUDA manifest {component_name}",
        ) != locked_component["version"]:
            raise BuildError(f"CUDA manifest version drift for {component_name}")
        actual_artifact = _object(
            _field(
                actual_component,
                manifest_platform,
                f"CUDA manifest {component_name}",
            ),
            f"CUDA manifest {component_name}.{manifest_platform}",
        )
        locked_artifact = locked_component["artifacts"][platform_id]
        expected = {
            "relative_path": locked_artifact["relative_path"],
            "size": str(locked_artifact["size"]),
            "sha256": locked_artifact["sha256"],
        }
        for field, expected_value in expected.items():
            actual_value = _field(
                actual_artifact,
                field,
                f"CUDA manifest {component_name}.{manifest_platform}",
            )
            if actual_value != expected_value:
                raise BuildError(
                    f"CUDA manifest drift for {component_name}.{field}: "
                    f"expected {expected_value!r}, got {actual_value!r}"
                )


def materialize_cuda(
    lock: Mapping[str, Any], platform_id: str, work_root: Path
) -> Path:
    cuda = lock["cuda"]
    downloads = work_root / "downloads" / "cuda"
    manifest_path = download_artifact(cuda["manifest"], downloads, "CUDA manifest")
    validate_cuda_manifest(manifest_path, lock, platform_id)

    toolkit = work_root / "inputs" / "cuda"
    for component_name in CUDA_COMPONENTS:
        locked = cuda["components"][component_name]["artifacts"][platform_id]
        artifact = {
            "url": urllib.parse.urljoin(cuda["base_url"], locked["relative_path"]),
            "size": locked["size"],
            "sha256": locked["sha256"],
        }
        archive = download_artifact(artifact, downloads, f"CUDA {component_name}")
        extracted = work_root / "extract" / "cuda" / component_name
        safe_extract(archive, extracted, platform_id)
        component_root = exact_archive_root(
            extracted, archive, f"CUDA {component_name}"
        )
        license_file = component_root / "LICENSE"
        if not license_file.is_file() or license_file.is_symlink():
            raise BuildError(f"CUDA {component_name} archive has no regular LICENSE")
        merge_component_tree(component_root, toolkit, f"CUDA {component_name}")

    nvcc_name = "nvcc.exe" if platform_id == "windows-x64" else "nvcc"
    nvcc = toolkit / "bin" / nvcc_name
    if not nvcc.is_file():
        raise BuildError(f"private CUDA compiler is missing: {nvcc}")
    return toolkit


def materialize_archive_root(
    artifact: Mapping[str, Any],
    label: str,
    platform_id: str,
    work_root: Path,
) -> Path:
    archive = download_artifact(artifact, work_root / "downloads" / label, label)
    extracted = work_root / "extract" / label
    safe_extract(archive, extracted, platform_id)
    return exact_archive_root(extracted, archive, label)


def materialize_msvc_runtime(
    lock: Mapping[str, Any], work_root: Path
) -> tuple[Path, Path]:
    """Extract only the three locked x64 CRT DLLs and signed package metadata."""

    msvc = lock["msvc_runtime"]
    archive_path = download_artifact(
        msvc["artifact"], work_root / "downloads" / "msvc", "MSVC x64 CRT"
    )
    runtime = work_root / "inputs" / "msvc-runtime"
    runtime.mkdir(parents=True, exist_ok=False)
    manifest_output = work_root / "notices" / "msvc-redist-MANIFEST.json"
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = _validated_zip_infos(archive, case_insensitive=True)
            by_name = {member.as_posix(): info for info, member in infos}
            for filename in MSVC_RUNTIME_FILES:
                entry = msvc["files"][filename]
                member = entry["member"]
                info = by_name.get(member)
                if info is None or info.is_dir():
                    raise BuildError(f"MSVC archive is missing {member}")
                output = runtime / filename
                digest = hashlib.sha256()
                count = 0
                with archive.open(info, "r") as source, output.open("xb") as target:
                    while chunk := source.read(1024 * 1024):
                        count += len(chunk)
                        digest.update(chunk)
                        target.write(chunk)
                if count != entry["size"] or digest.hexdigest() != entry["sha256"]:
                    raise BuildError(f"MSVC payload bytes drifted for {filename}")

            manifest_member = msvc["manifest_member"]
            manifest_info = by_name.get(manifest_member)
            if manifest_info is None or manifest_info.is_dir():
                raise BuildError(f"MSVC package manifest is missing {manifest_member}")
            manifest_bytes = archive.read(manifest_info)
            try:
                manifest = _object(
                    json.loads(
                        manifest_bytes.decode("utf-8"),
                        object_pairs_hook=duplicate_key_hook(
                            BuildError,
                            "MSVC package manifest",
                        ),
                        parse_constant=invalid_constant_hook(
                            BuildError,
                            "MSVC package manifest",
                        ),
                    ),
                    "MSVC package manifest",
                )
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise BuildError(f"cannot parse MSVC package manifest: {exc}") from exc
            if (
                _field(manifest, "id", "MSVC package manifest")
                != msvc["package_id"]
                or _field(manifest, "version", "MSVC package manifest")
                != msvc["package_version"]
            ):
                raise BuildError("MSVC package manifest identity drifted")
            manifest_output.write_bytes(manifest_bytes)
    except (OSError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise BuildError(f"cannot extract pinned MSVC CRT archive: {exc}") from exc
    return runtime, manifest_output


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
    return _string(value, f"{label} RPM tag {tag}")


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
    expected_source = PurePosixPath(
        urllib.parse.urlsplit(source_rpm_url).path
    ).name
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
        raise BuildError(f"cannot decompress locked RPM payload {label}: {exc}") from exc
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
                int(header[6 + index * 8 : 14 + index * 8], 16)
                for index in range(13)
            ]
        except ValueError as exc:
            raise BuildError(f"{label} cpio header has invalid hexadecimal fields") from exc
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
        _safe_member_path(member, f"{label} cpio member")
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


def materialize_linux_runtime(
    lock: Mapping[str, Any], work_root: Path
) -> tuple[Path, Path]:
    locked = lock["linux_runtime"]
    runtime = work_root / "inputs" / "linux-gnu-runtime"
    notices = work_root / "notices" / "gcc-runtime"
    runtime.mkdir(parents=True, exist_ok=False)
    notices.mkdir(parents=True, exist_ok=False)
    source_url = locked["source_rpm"]["url"]
    download_artifact(
        locked["source_rpm"],
        work_root / "downloads" / "linux-runtime",
        "Rocky Linux GNU runtime corresponding source RPM",
    )
    for package_name, package in locked["packages"].items():
        archive = download_artifact(
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
            "Publication requires the complete staged ELF closure and driver-only "
            "clean-machine inference validation.",
        ],
    }
    provenance.write_text(
        json.dumps(provenance_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    contract = runtime_contract("linux-x64")
    expected_runtime = {
        PurePosixPath(entry.path).name
        for entry in contract.files_for_source("platform_runtime")
    }
    if {entry.name for entry in runtime.iterdir()} != expected_runtime:
        raise BuildError("materialized GNU runtime does not match the exact closure")
    expected_notices = {
        PurePosixPath(entry.path).name
        for source in ("platform_runtime_notice", "platform_runtime_provenance")
        for entry in contract.files_for_source(
            source
        )
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
    for package_key in LINUX_PRODUCER_PACKAGES:
        package = lock["linux_toolchain"]["packages"][package_key]
        archive = download_artifact(
            package["artifact"],
            destination,
            f"Rocky Linux producer package {package['name']}",
        )
        if archive.name in names:
            raise BuildError(
                f"Linux producer RPM filenames collide: {archive.name}"
            )
        names.add(archive.name)
        packages.append(archive)
    return tuple(packages)


class CommandRunner:
    def __init__(self, work_root: Path) -> None:
        self.work_root = work_root
        self.commands: list[list[str]] = []

    def run(
        self,
        command: Sequence[os.PathLike[str] | str],
        *,
        env: Mapping[str, str],
        cwd: Path | None = None,
        capture: bool = False,
    ) -> str:
        args = [os.fspath(value) for value in command]
        self.commands.append(args)
        print(f"+ {shlex.join(args)}", flush=True)
        try:
            result = subprocess.run(
                args,
                cwd=cwd,
                env=dict(env),
                check=True,
                text=True,
                stdout=subprocess.PIPE if capture else None,
                stderr=subprocess.STDOUT if capture else None,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            output = getattr(exc, "stdout", None)
            detail = f"\n{output}" if output else ""
            raise BuildError(f"command failed: {shlex.join(args)}{detail}") from exc
        return result.stdout.strip() if capture and result.stdout else ""


class LinuxProducerRunner(CommandRunner):
    """Execute native build commands only in one verified Rocky OCI container."""

    def __init__(
        self,
        work_root: Path,
        docker: Path,
        container_id: str,
        host_environment: Mapping[str, str],
    ) -> None:
        super().__init__(work_root)
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
        self.commands.append(args)
        print(f"+ [pinned Rocky producer] {shlex.join(args)}", flush=True)
        command_directory = self.work_root if cwd is None else cwd
        directory = command_directory.resolve(strict=False)
        if not (
            _inside(directory, self.work_root.resolve())
            or _inside(directory, REPOSITORY_ROOT.resolve())
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
        docker_args = [
            self.docker,
            "exec",
            "--user",
            self.user,
            "--workdir",
            directory,
        ]
        for key, value in sorted(container_environment.items()):
            if "\0" in value or "\n" in value or "\r" in value:
                raise BuildError(
                    f"Linux producer environment has unsafe {key} value"
                )
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
        return result.stdout.strip() if capture and result.stdout else ""


def _parse_linux_image_identity(
    output: str, lock: Mapping[str, Any]
) -> None:
    lines = output.splitlines()
    if len(lines) != 4:
        raise BuildError(f"Docker returned invalid producer image identity: {output!r}")
    image_id, architecture, operating_system, raw_digests = lines
    producer = lock["linux_toolchain"]["producer_image"]
    if image_id != f"sha256:{producer['config_sha256']}":
        raise BuildError(
            f"Rocky producer config digest drifted: {image_id!r}"
        )
    if architecture != producer["architecture"] or operating_system != "linux":
        raise BuildError(
            "Rocky producer image platform drifted: "
            f"{operating_system}/{architecture}"
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
    host_runner: CommandRunner,
    lock: Mapping[str, Any],
    work_root: Path,
    environment: Mapping[str, str],
    packages: Sequence[Path],
) -> Iterator[LinuxProducerRunner]:
    """Create the sole Linux compiler boundary from the locked OCI digest."""

    docker = require_host_program("docker", environment)
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
    for path in (REPOSITORY_ROOT.resolve(), work_root.resolve()):
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
                f"type=bind,source={REPOSITORY_ROOT.resolve()},"
                f"target={REPOSITORY_ROOT.resolve()},readonly"
            ),
            "--mount",
            (
                f"type=bind,source={work_root.resolve()},"
                f"target={work_root.resolve()}"
            ),
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
        raise BuildError(f"Docker returned invalid producer container ID {container_id!r}")
    runner = LinuxProducerRunner(
        work_root,
        docker,
        container_id,
        environment,
    )
    runner.commands.extend(host_runner.commands)
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
            for key in LINUX_PRODUCER_PACKAGES
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


def require_host_program(name: str, environment: Mapping[str, str]) -> Path:
    """Resolve one declared host tool through the release environment only."""

    if not name or Path(name).name != name:
        raise BuildError(f"host tool name must be one filename: {name!r}")
    search_path = environment.get("PATH")
    if not isinstance(search_path, str) or not search_path:
        raise BuildError("release environment has no canonical PATH")
    resolved: set[Path] = set()
    for raw_directory in search_path.split(os.pathsep):
        if not raw_directory:
            raise BuildError("release PATH contains an empty search directory")
        directory = Path(raw_directory)
        if not directory.is_absolute():
            raise BuildError(
                f"release PATH contains a non-absolute directory: {raw_directory!r}"
            )
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            resolved.add(candidate.resolve())
    if not resolved:
        raise BuildError(f"required native host tool is not on PATH: {name}")
    if len(resolved) != 1:
        raise BuildError(
            f"required native host tool is ambiguous on release PATH: "
            f"{name}: {sorted(os.fspath(path) for path in resolved)}"
        )
    return next(iter(resolved))


def checkout_exact(
    runner: CommandRunner,
    git: Path,
    repository: str,
    commit: str,
    destination: Path,
    *,
    env: Mapping[str, str],
    submodules: Mapping[str, str],
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    runner.run([git, "init", destination], env=env)
    runner.run([git, "-C", destination, "config", "core.autocrlf", "false"], env=env)
    runner.run([git, "-C", destination, "remote", "add", "origin", repository], env=env)
    runner.run(
        [git, "-C", destination, "fetch", "--depth", "1", "--no-tags", "origin", commit],
        env=env,
    )
    runner.run([git, "-C", destination, "checkout", "--detach", "FETCH_HEAD"], env=env)
    actual = runner.run([git, "-C", destination, "rev-parse", "HEAD"], env=env, capture=True)
    if actual != commit:
        raise BuildError(f"source checkout drift: expected {commit}, got {actual}")

    if not submodules:
        return
    runner.run(
        [
            git,
            "-C",
            destination,
            "submodule",
            "update",
            "--init",
            "--recursive",
            "--depth",
            "1",
        ],
        env=env,
    )
    status = runner.run(
        [git, "-C", destination, "submodule", "status", "--recursive"],
        env=env,
        capture=True,
    )
    actual_submodules: dict[str, str] = {}
    for line in status.splitlines():
        match = re.fullmatch(r" ([0-9a-f]{40}) ([^ ]+)(?: \(.+\))?", line)
        if match is None:
            raise BuildError(f"unexpected Git submodule status: {line!r}")
        actual_submodules[match.group(2)] = match.group(1)
    if actual_submodules != dict(submodules):
        raise BuildError(
            "TensorRT submodule commits differ from runtime-lock.json: "
            f"expected {dict(submodules)}, got {actual_submodules}"
        )


def _windows_environment_value(
    environment: Mapping[str, str], canonical: str
) -> str | None:
    matches = [key for key in environment if key.casefold() == canonical.casefold()]
    if len(matches) > 1:
        raise BuildError(f"environment contains duplicate case variants of {canonical}")
    if not matches:
        return None
    value = environment[matches[0]]
    if not isinstance(value, str) or not value:
        raise BuildError(f"Windows release environment has empty {canonical}")
    return value


def _windows_release_environment(
    source: Mapping[str, str], work_root: Path
) -> dict[str, str]:
    """Copy only the declared native-build values emitted by vcvars64."""

    environment: dict[str, str] = {}
    for canonical in WINDOWS_VCVARS_ENVIRONMENT_KEYS:
        value = _windows_environment_value(source, canonical)
        if value is not None:
            environment[canonical] = value
    missing = sorted(WINDOWS_REQUIRED_ENVIRONMENT_KEYS - set(environment))
    if missing:
        raise BuildError(
            "Windows release requires these vcvars64 environment values: "
            + ", ".join(missing)
        )
    root_keys = (
        "SystemRoot",
        "UniversalCRTSdkDir",
        "VCINSTALLDIR",
        "VCToolsInstallDir",
        "VSINSTALLDIR",
        "WindowsSdkBinPath",
        "WindowsSdkDir",
    )
    roots = {key: PureWindowsPath(environment[key]) for key in root_keys}
    non_absolute = sorted(key for key, path in roots.items() if not path.is_absolute())
    if non_absolute:
        raise BuildError(
            "Windows release requires absolute vcvars64 roots: "
            + ", ".join(non_absolute)
        )
    comspec = PureWindowsPath(environment["COMSPEC"])
    expected_comspec = roots["SystemRoot"] / "System32" / "cmd.exe"
    if comspec != expected_comspec:
        raise BuildError(
            f"COMSPEC must be the SystemRoot command processor: {expected_comspec}"
        )

    home = work_root / "producer-home"
    temporary = work_root / "producer-tmp"
    home.mkdir(parents=True, exist_ok=False)
    temporary.mkdir(parents=True, exist_ok=False)
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": "NUL",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": os.fspath(home),
            "PM_PACKAGES_ROOT": os.fspath(work_root / "packman-cache"),
            "TEMP": os.fspath(temporary),
            "TMP": os.fspath(temporary),
            "USERPROFILE": os.fspath(home),
        }
    )
    return environment


def release_environment(work_root: Path) -> dict[str, str]:
    """Create the exact host-command environment for one release build."""

    if os.name == "nt":
        return _windows_release_environment(os.environ, work_root)

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
    platform_id: str,
    cuda_root: Path,
    tensorrt_root: Path,
    cmake_root: Path,
    ninja: Path,
    compiler: Path,
) -> dict[str, str]:
    if platform_id not in RUNTIME_CONTRACTS:
        raise BuildError(f"unsupported private build platform {platform_id!r}")
    environment = dict(base)
    cuda_library = cuda_root / ("bin" if platform_id == "windows-x64" else "lib")
    tensorrt_library = tensorrt_root / "lib"
    path_entries = [
        cmake_root / "bin",
        ninja.parent,
        cuda_root / "bin",
        tensorrt_library,
        compiler.parent,
    ]
    if platform_id == "windows-x64":
        required_paths: dict[str, Path] = {}
        for key in (
            "COMSPEC",
            "SystemRoot",
            "UniversalCRTSdkDir",
            "VCToolsInstallDir",
            "VSINSTALLDIR",
            "WindowsSdkBinPath",
            "WindowsSdkDir",
        ):
            value = environment.get(key)
            if not isinstance(value, str) or not value:
                raise BuildError(f"Windows private build environment is missing {key}")
            path = Path(value)
            if not path.is_absolute():
                raise BuildError(
                    f"Windows private build environment has non-absolute {key}"
                )
            required_paths[key] = path.resolve(strict=False)
        sdk_version = environment.get("WindowsSDKVersion")
        if (
            not isinstance(sdk_version, str)
            or not sdk_version.endswith("\\")
            or sdk_version.count("\\") != 1
        ):
            raise BuildError(
                "Windows private build environment has non-canonical "
                "WindowsSDKVersion"
            )
        path_entries.extend(
            (
                required_paths["COMSPEC"].parent,
                required_paths["WindowsSdkBinPath"] / sdk_version[:-1] / "x64",
            )
        )
        allowed_search_roots = tuple(
            required_paths[key]
            for key in (
                "SystemRoot",
                "UniversalCRTSdkDir",
                "VCToolsInstallDir",
                "VSINSTALLDIR",
                "WindowsSdkDir",
            )
        )
        for key in ("INCLUDE", "LIB", "LIBPATH"):
            value = environment.get(key)
            if not isinstance(value, str) or not value:
                raise BuildError(f"Windows private build environment is missing {key}")
            for item in value.split(os.pathsep):
                path = Path(item)
                if not item or not path.is_absolute():
                    raise BuildError(
                        f"Windows private build environment has unsafe {key} path: "
                        f"{item!r}"
                    )
                resolved = path.resolve(strict=False)
                if not any(_inside(resolved, root) for root in allowed_search_roots):
                    raise BuildError(
                        f"Windows private build environment has external {key} path: "
                        f"{resolved}"
                    )
    else:
        existing_path = environment.get("PATH")
        if not isinstance(existing_path, str) or not existing_path:
            raise BuildError("Linux private build environment is missing PATH")
        path_entries.extend(Path(path) for path in existing_path.split(os.pathsep))
    private_path = os.pathsep.join(str(path) for path in path_entries)
    environment["PATH"] = private_path
    if platform_id == "linux-x64":
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            (str(cuda_library), str(tensorrt_library))
        )
    return environment


def _linux_distribution_identity(
    runner: CommandRunner, environment: Mapping[str, str]
) -> tuple[str, str]:
    output = runner.run(
        ["/usr/bin/cat", "/etc/os-release"],
        env=environment,
        capture=True,
    )
    lines = output.splitlines()
    values: dict[str, str] = {}
    pattern = re.compile(
        r'^(ID|VERSION_ID)=(?:"([^"\\]*)"|([A-Za-z0-9._-]+))$'
    )
    for line in lines:
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
    runner: CommandRunner,
    lock: Mapping[str, Any],
    environment: Mapping[str, str],
) -> None:
    rpm = Path("/usr/bin/rpm")
    toolchain = lock["linux_toolchain"]
    expected = {
        package["name"]: package["nevra"]
        for package in toolchain["packages"].values()
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
    runner: CommandRunner,
    platform_id: str,
    lock: Mapping[str, Any],
    environment: Mapping[str, str],
) -> Path:
    if platform_id == "windows-x64":
        def required_environment_value(name: str) -> str:
            value = environment.get(name)
            if value is None or not value:
                raise BuildError(f"windows-x64 releases require {name}")
            return value

        if required_environment_value("VisualStudioVersion") != "17.0":
            raise BuildError(
                "windows-x64 releases require a native x64 Visual Studio 2022 "
                "developer environment (VisualStudioVersion=17.0)"
            )
        if required_environment_value("VSCMD_ARG_HOST_ARCH") != "x64" or (
            required_environment_value("VSCMD_ARG_TGT_ARCH") != "x64"
        ):
            raise BuildError(
                "windows-x64 releases require VSCMD_ARG_HOST_ARCH=x64 and "
                "VSCMD_ARG_TGT_ARCH=x64"
            )
        toolchain = lock["windows_toolchain"]
        if required_environment_value("VCToolsVersion") != toolchain[
            "vctools_version"
        ]:
            raise BuildError(
                "VCToolsVersion does not match the pinned Windows producer"
            )
        if required_environment_value("WindowsSDKVersion") != toolchain[
            "windows_sdk_version"
        ]:
            raise BuildError(
                "WindowsSDKVersion does not match the pinned Windows producer"
            )
        compiler = require_host_program("cl.exe", environment)
        compiler_identity = PureWindowsPath(os.fspath(compiler))
        vctools_root = PureWindowsPath(
            required_environment_value("VCToolsInstallDir")
        )
        expected_compiler = vctools_root / "bin" / "Hostx64" / "x64" / "cl.exe"
        if compiler_identity != expected_compiler:
            raise BuildError(
                "cl.exe is not the compiler declared by VCToolsInstallDir: "
                f"expected {expected_compiler}, got {compiler_identity}"
            )
        compiler_help = runner.run(
            [compiler, "/?"], env=environment, capture=True
        )
        versions = re.findall(
            r"Compiler Version ([0-9.]+) for x64", compiler_help
        )
        if versions != [toolchain["cl_version"]]:
            raise BuildError(
                f"cl.exe version does not match the pinned Windows producer: {versions}"
            )
        return compiler
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
    version = runner.run(
        [compiler, "-dumpfullversion"], env=environment, capture=True
    )
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
    readelf_output = runner.run(
        [readelf, "--version"], env=environment, capture=True
    )
    first_line = readelf_output.splitlines()[0] if readelf_output else ""
    if first_line != toolchain["readelf_version"]:
        raise BuildError(
            f"Linux producer readelf must be {toolchain['readelf_version']!r}; "
            f"got {first_line!r}"
        )
    return compiler


def fetch_sdk_dependencies(
    runner: CommandRunner,
    sdk_source: Path,
    platform_id: str,
    environment: Mapping[str, str],
) -> Path:
    if platform_id == "windows-x64":
        comspec_value = environment.get("COMSPEC")
        if not comspec_value:
            raise BuildError("COMSPEC is required to run the pinned Packman batch file")
        comspec_input = Path(comspec_value)
        if not comspec_input.is_absolute():
            raise BuildError("COMSPEC must be an absolute path to cmd.exe")
        comspec = comspec_input.resolve()
        if comspec.name.lower() != "cmd.exe" or not comspec.is_file():
            raise BuildError(f"COMSPEC does not name cmd.exe: {comspec}")
        script = sdk_source / "fetch_deps.bat"
        if not script.is_file():
            raise BuildError(f"pinned SDK fetch script is missing: {script}")
        command_text = f'call "{script}" release'
        runner.run([comspec, "/d", "/s", "/c", command_text], cwd=sdk_source, env=environment)
        ninja = sdk_source / "_deps" / "build-deps" / "ninja" / "ninja.exe"
    else:
        script = sdk_source / "fetch_deps.sh"
        if not script.is_file() or not os.access(script, os.X_OK):
            raise BuildError(f"pinned SDK fetch script is not executable: {script}")
        runner.run([script, "release"], cwd=sdk_source, env=environment)
        ninja = sdk_source / "_deps" / "build-deps" / "ninja" / "ninja"
    if not ninja.is_file():
        raise BuildError(f"Packman did not materialize pinned Ninja: {ninja}")
    return ninja


def validate_cmake(
    runner: CommandRunner,
    cmake_root: Path,
    platform_id: str,
    expected_version: str,
    environment: Mapping[str, str],
) -> Path:
    executable = cmake_root / "bin" / (
        "cmake.exe" if platform_id == "windows-x64" else "cmake"
    )
    if not executable.is_file():
        raise BuildError(f"pinned CMake executable is missing: {executable}")
    version = runner.run([executable, "--version"], env=environment, capture=True)
    first_line = version.splitlines()[0] if version else ""
    if first_line != f"cmake version {expected_version}":
        raise BuildError(
            f"pinned CMake version mismatch: expected {expected_version}, got {first_line!r}"
        )
    return executable


def _parse_cmake_cache(cache_path: Path) -> dict[str, str]:
    try:
        lines = cache_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BuildError(f"cannot read CMake cache {cache_path}: {exc}") from exc
    values: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        declaration, value = line.split("=", 1)
        key, separator, _cache_type = declaration.partition(":")
        if not separator or not key:
            continue
        if key in values:
            raise BuildError(f"CMake cache {cache_path} repeats {key}")
        values[key] = value
    return values


def _absolute_path(value: str) -> Path | None:
    if not value or value.endswith("-NOTFOUND"):
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        return None
    return candidate.resolve(strict=False)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def audit_cmake_paths(
    cache_path: Path,
    work_root: Path,
    required_private_keys: Sequence[str],
) -> None:
    """Reject any resolved CUDA/TensorRT path outside the isolated build."""

    root = work_root.resolve()
    values = _parse_cmake_cache(cache_path)
    for key in required_private_keys:
        if key not in values:
            raise BuildError(f"CMake cache {cache_path} does not resolve required {key}")
        path = _absolute_path(values[key])
        if path is None or not _inside(path, root):
            raise BuildError(
                f"CMake cache {cache_path} resolved {key} outside work root: "
                f"{values[key]!r}"
            )
    violations: list[str] = []
    for key, value in values.items():
        for segment in value.split(";"):
            path = _absolute_path(segment)
            if path is None:
                continue
            lowered = str(path).lower()
            if any(marker in lowered for marker in GPU_PATH_MARKERS) and not _inside(
                path, root
            ):
                violations.append(f"{key}={segment}")
    if violations:
        raise BuildError(
            f"CMake cache {cache_path} contains external GPU paths: "
            + ", ".join(violations)
        )


def audit_ninja_paths(build_ninja: Path, work_root: Path) -> None:
    try:
        lines = build_ninja.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BuildError(f"cannot audit generated Ninja file {build_ninja}: {exc}") from exc
    root = work_root.resolve()
    violations: set[str] = set()
    posix_paths = re.compile(r"(?<![A-Za-z0-9_.-])(/[^	\r\n ;\"']+)")
    windows_paths = re.compile(r"([A-Za-z]:[/\\][^	\r\n ;\"']+)")
    for line in lines:
        lowered_line = line.lower()
        if not any(marker in lowered_line for marker in GPU_PATH_MARKERS):
            continue
        for expression in (posix_paths, windows_paths):
            for match in expression.finditer(line):
                path = Path(match.group(1).replace("$ ", " ")).resolve(strict=False)
                lowered_path = str(path).lower()
                if any(marker in lowered_path for marker in GPU_PATH_MARKERS) and not _inside(
                    path, root
                ):
                    violations.add(str(path))
    if violations:
        raise BuildError(
            f"generated Ninja build contains external GPU paths: {sorted(violations)}"
        )


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


def audit_linux_compile_flags(cache_path: Path, lock: Mapping[str, Any]) -> None:
    values = _parse_cmake_cache(cache_path)
    expected_cxx, expected_cuda = linux_compile_flags(lock)
    expected = {
        "CMAKE_CXX_FLAGS": expected_cxx,
        "CMAKE_CUDA_FLAGS": expected_cuda,
    }
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise BuildError(
                f"CMake cache {cache_path} does not preserve pinned {key}: "
                f"expected {expected_value!r}, got {values.get(key)!r}"
            )


def audit_rpath_disabled(cache_path: Path) -> None:
    values = _parse_cmake_cache(cache_path)
    if values.get("CMAKE_SKIP_RPATH") != "ON":
        raise BuildError(f"CMake cache {cache_path} does not disable all RPATHs")


def configure_and_build_trtexec(
    runner: CommandRunner,
    cmake: Path,
    ninja: Path,
    compiler: Path,
    cuda_root: Path,
    tensorrt_root: Path,
    source: Path,
    lock: Mapping[str, Any],
    platform_id: str,
    work_root: Path,
    environment: Mapping[str, str],
) -> Path:
    build = work_root / "build" / "trtexec"
    output = work_root / "build" / "trtexec-output"
    nvcc = cuda_root / "bin" / (
        "nvcc.exe" if platform_id == "windows-x64" else "nvcc"
    )
    command = [
        cmake,
        "-S",
        source,
        "-B",
        build,
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE:STRING=Release",
        "-DCMAKE_SKIP_RPATH:BOOL=ON",
        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja}",
        f"-DCMAKE_CXX_COMPILER:FILEPATH={compiler}",
        f"-DCMAKE_CUDA_HOST_COMPILER:FILEPATH={compiler}",
        f"-DCMAKE_CUDA_COMPILER:FILEPATH={nvcc}",
        f"-DCUDA_TOOLKIT_ROOT_DIR:PATH={cuda_root}",
        f"-DCUDA_VERSION:STRING={lock['tensorrt']['cuda']}",
        f"-DTRT_LIB_DIR:PATH={tensorrt_root / 'lib'}",
        f"-DTRT_OUT_DIR:PATH={output}",
        "-DBUILD_PLUGINS:BOOL=OFF",
        "-DBUILD_PARSERS:BOOL=OFF",
        "-DBUILD_SAMPLES:BOOL=ON",
    ]
    if platform_id == "linux-x64":
        cxx_flags, cuda_flags = linux_compile_flags(lock)
        command.extend(
            (
                f"-DCMAKE_CXX_FLAGS:STRING={cxx_flags}",
                f"-DCMAKE_CUDA_FLAGS:STRING={cuda_flags}",
            )
        )
    runner.run(command, env=environment)
    audit_cmake_paths(
        build / "CMakeCache.txt",
        work_root,
        (
            "CMAKE_CUDA_COMPILER",
            "CUDA_TOOLKIT_ROOT_DIR",
            "TRT_LIB_DIR",
            "TRT_OUT_DIR",
        ),
    )
    audit_ninja_paths(build / "build.ninja", work_root)
    audit_rpath_disabled(build / "CMakeCache.txt")
    if platform_id == "linux-x64":
        audit_linux_compile_flags(build / "CMakeCache.txt", lock)
    runner.run([cmake, "--build", build, "--target", "trtexec", "--parallel"], env=environment)
    trtexec = output / ("trtexec.exe" if platform_id == "windows-x64" else "trtexec")
    if not trtexec.is_file():
        raise BuildError(f"source build did not produce canonical trtexec: {trtexec}")
    return trtexec


def _redacted_commands(runner: CommandRunner) -> list[list[str]]:
    work = str(runner.work_root)
    repository = str(REPOSITORY_ROOT)
    return [
        [
            argument.replace(work, "<work>").replace(repository, "<repository>")
            for argument in command
        ]
        for command in runner.commands
    ]


def write_provenance(
    lock: Mapping[str, Any],
    platform_id: str,
    runner: CommandRunner,
    work_root: Path,
    msvc_manifest: Path | None,
) -> tuple[Path, Path | None]:
    notices = work_root / "notices"
    notices.mkdir(parents=True, exist_ok=True)
    lock_digest = file_sha256(LOCK_PATH)
    trtexec_provenance = notices / "trtexec-PROVENANCE.txt"
    record: dict[str, Any] = {
        "schema": "audio2face-build-provenance/1",
        "platform": platform_id,
        "runtime_lock_sha256": lock_digest,
        "audio2face_sdk": lock["audio2face_sdk"],
        "tensorrt_source": lock["tensorrt_source"],
        "tensorrt_binary": {
            "version": lock["tensorrt"]["version"],
            "cuda": lock["tensorrt"]["cuda"],
            "artifact": lock["tensorrt"]["artifacts"][platform_id],
        },
        "cuda": {
            "version": lock["cuda"]["version"],
            "manifest": lock["cuda"]["manifest"],
            "components": {
                name: {
                    "version": lock["cuda"]["components"][name]["version"],
                    "artifact": lock["cuda"]["components"][name]["artifacts"][platform_id],
                }
                for name in CUDA_COMPONENTS
            },
        },
        "cmake": {
            "version": lock["cmake"]["version"],
            "artifact": lock["cmake"]["artifacts"][platform_id],
        },
        "commands_through_trtexec_build": _redacted_commands(runner),
    }
    if platform_id == "windows-x64":
        record["producer_toolchain"] = lock["windows_toolchain"]
    else:
        record["producer_toolchain"] = lock["linux_toolchain"]
        record["reproducibility"] = (
            "The pinned native producer does not imply bit-for-bit reproducible binaries."
        )
    trtexec_provenance.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    msvc_provenance: Path | None = None
    if platform_id == "windows-x64":
        if msvc_manifest is None:
            raise BuildError("Windows provenance requires the pinned MSVC manifest")
        msvc_provenance = notices / "msvc-runtime-PROVENANCE.txt"
        msvc_record = {
            "schema": "audio2face-msvc-runtime-provenance/1",
            "runtime_lock_sha256": lock_digest,
            "package": lock["msvc_runtime"],
            "preserved_manifest_sha256": file_sha256(msvc_manifest),
            "release_gate": (
                "Publication requires legal review of Microsoft redistribution terms; "
                "this record does not create or replace license terms."
            ),
        }
        msvc_provenance.write_text(
            json.dumps(msvc_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return trtexec_provenance, msvc_provenance


def _cmake_list(values: Sequence[os.PathLike[str] | str], label: str) -> str:
    encoded: list[str] = []
    for value in values:
        item = os.fspath(value)
        if (
            not item
            or ";" in item
            or "\0" in item
            or "\n" in item
            or "\r" in item
            or "]==]" in item
        ):
            raise BuildError(f"{label} contains a value unsafe for a CMake list: {item!r}")
        encoded.append(item)
    if not encoded:
        raise BuildError(f"{label} cannot be empty")
    return ";".join(encoded)


def _stage_source_for_file(
    entry: RuntimePackagedFile,
    *,
    sdk_source: Path,
    cuda_runtime: Path,
    tensorrt_runtime: Path,
    platform_runtime: Path,
    platform_notices: Path | None,
    platform_metadata: Path | None,
    platform_provenance: Path | None,
    trtexec_source: Path,
    trtexec_provenance: Path,
) -> Path:
    name = PurePosixPath(entry.path).name
    directory_sources = {
        "cuda_runtime": cuda_runtime,
        "tensorrt_runtime": tensorrt_runtime,
        "platform_runtime": platform_runtime,
    }
    if platform_notices is not None:
        directory_sources["platform_runtime_notice"] = platform_notices
    direct_sources = {
        "project_license": REPOSITORY_ROOT / "LICENSE",
        "project_notices": REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md",
        "sdk_license": sdk_source / "LICENSE.txt",
        "sdk_cuda_license": sdk_source / "licenses" / "cuda-LICENSE.txt",
        "sdk_tensorrt_license": sdk_source / "licenses" / "TensorRT-Readme.txt",
        "sdk_tensorrt_acknowledgements": (
            sdk_source / "licenses" / "TensorRT-Acknowledgements.txt"
        ),
        "trtexec_source_license": trtexec_source / "LICENSE",
        "trtexec_provenance": trtexec_provenance,
    }
    if platform_metadata is not None:
        direct_sources["platform_runtime_metadata"] = platform_metadata
    if platform_provenance is not None:
        direct_sources["platform_runtime_provenance"] = platform_provenance
    if entry.source in directory_sources:
        return directory_sources[entry.source] / name
    if entry.source in direct_sources:
        return direct_sources[entry.source]
    raise BuildError(
        f"runtime contract source role {entry.source!r} has no release input"
    )


def runtime_stage_map(
    contract: RuntimePlatformContract,
    *,
    bundle_manifest: Path,
    sdk_source: Path,
    cuda_runtime: Path,
    tensorrt_runtime: Path,
    platform_runtime: Path,
    platform_notices: Path | None,
    platform_metadata: Path | None,
    platform_provenance: Path | None,
    trtexec: Path,
    trtexec_source: Path,
    trtexec_provenance: Path,
) -> tuple[str, tuple[Path, ...], tuple[str, ...]]:
    """Resolve the shared package contract to one CMake staging map."""

    audio2x_entries = contract.files_for_source("audio2x")
    if len(audio2x_entries) != 1:
        raise BuildError("runtime contract must declare exactly one audio2x path")
    external: list[tuple[Path, str]] = [
        (trtexec, contract.trtexec),
        (bundle_manifest, "bundle.json"),
    ]
    for entry in (*contract.libraries, *contract.licenses):
        if entry.source == "audio2x":
            continue
        external.append(
            (
                _stage_source_for_file(
                    entry,
                    sdk_source=sdk_source,
                    cuda_runtime=cuda_runtime,
                    tensorrt_runtime=tensorrt_runtime,
                    platform_runtime=platform_runtime,
                    platform_notices=platform_notices,
                    platform_metadata=platform_metadata,
                    platform_provenance=platform_provenance,
                    trtexec_source=trtexec_source,
                    trtexec_provenance=trtexec_provenance,
                ),
                entry.path,
            )
        )
    expected_paths = {
        "bundle.json",
        contract.worker,
        contract.trtexec,
        *(entry.path for entry in contract.libraries),
        *(entry.path for entry in contract.licenses),
    }
    actual_paths = {
        contract.worker,
        audio2x_entries[0].path,
        *(destination for _source, destination in external),
    }
    if actual_paths != expected_paths or len(actual_paths) != len(external) + 2:
        raise BuildError("runtime staging map does not exactly cover the package contract")
    for source, destination in external:
        if not source.is_file() or source.stat().st_size < 1:
            raise BuildError(
                f"runtime staging source for {destination} is not a non-empty file: "
                f"{source}"
            )
    return (
        audio2x_entries[0].path,
        tuple(source for source, _destination in external),
        tuple(destination for _source, destination in external),
    )


def configure_and_stage_worker(
    runner: CommandRunner,
    cmake: Path,
    ninja: Path,
    compiler: Path,
    sdk_source: Path,
    cuda_root: Path,
    tensorrt_root: Path,
    msvc_runtime: Path | None,
    msvc_manifest: Path | None,
    msvc_provenance: Path | None,
    linux_runtime: Path | None,
    linux_notices: Path | None,
    trtexec: Path,
    trtexec_provenance: Path,
    tensorrt_source: Path,
    lock: Mapping[str, Any],
    platform_id: str,
    work_root: Path,
    environment: Mapping[str, str],
) -> Path:
    build = work_root / "build" / "worker"
    contract = runtime_contract(platform_id)
    bundle_manifest = work_root / "notices" / "bundle.json"
    bundle_manifest.write_text(
        json.dumps(contract.manifest(), indent=2) + "\n",
        encoding="utf-8",
    )
    cuda_runtime = cuda_root / ("bin" if platform_id == "windows-x64" else "lib")
    if platform_id == "windows-x64":
        if msvc_runtime is None or msvc_manifest is None or msvc_provenance is None:
            raise BuildError("Windows worker staging requires pinned MSVC inputs")
        platform_runtime = msvc_runtime
        platform_notices = None
        platform_metadata = msvc_manifest
        platform_provenance = msvc_provenance
    else:
        if linux_runtime is None or linux_notices is None:
            raise BuildError("Linux worker staging requires pinned GNU runtime inputs")
        platform_runtime = linux_runtime
        platform_notices = linux_notices
        platform_metadata = None
        platform_provenance = linux_notices / "gcc-runtime-PROVENANCE.txt"
    audio2x_path, external_sources, external_paths = runtime_stage_map(
        contract,
        bundle_manifest=bundle_manifest,
        sdk_source=sdk_source,
        cuda_runtime=cuda_runtime,
        tensorrt_runtime=tensorrt_root / "lib",
        platform_runtime=platform_runtime,
        platform_notices=platform_notices,
        platform_metadata=platform_metadata,
        platform_provenance=platform_provenance,
        trtexec=trtexec,
        trtexec_source=tensorrt_source,
        trtexec_provenance=trtexec_provenance,
    )
    nvcc = cuda_root / "bin" / (
        "nvcc.exe" if platform_id == "windows-x64" else "nvcc"
    )
    command: list[os.PathLike[str] | str] = [
        cmake,
        "-S",
        REPOSITORY_ROOT / "worker",
        "-B",
        build,
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE:STRING=Release",
        "-DCMAKE_SKIP_RPATH:BOOL=ON",
        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja}",
        f"-DCMAKE_CXX_COMPILER:FILEPATH={compiler}",
        f"-DCMAKE_CUDA_HOST_COMPILER:FILEPATH={compiler}",
        f"-DCMAKE_CUDA_COMPILER:FILEPATH={nvcc}",
        f"-DCUDAToolkit_ROOT:PATH={cuda_root}",
        f"-DTENSORRT_ROOT_DIR:PATH={tensorrt_root}",
        f"-DA2F_SDK_SOURCE_DIR:PATH={sdk_source}",
        f"-DA2F_STAGE_WORKER_PATH:STRING={contract.worker}",
        f"-DA2F_STAGE_AUDIO2X_PATH:STRING={audio2x_path}",
        f"-DA2F_STAGE_TRTEXEC_PATH:STRING={contract.trtexec}",
        "-DA2F_STAGE_EXTERNAL_SOURCES:STRING="
        + _cmake_list(external_sources, "runtime staging sources"),
        "-DA2F_STAGE_EXTERNAL_PATHS:STRING="
        + _cmake_list(external_paths, "runtime package paths"),
    ]
    required_cache_keys = [
        "CMAKE_CUDA_COMPILER",
        "CUDAToolkit_ROOT",
        "TENSORRT_ROOT_DIR",
        "A2F_SDK_SOURCE_DIR",
    ]
    if platform_id == "linux-x64":
        cxx_flags, cuda_flags = linux_compile_flags(lock)
        command.extend(
            (
                f"-DCMAKE_CXX_FLAGS:STRING={cxx_flags}",
                f"-DCMAKE_CUDA_FLAGS:STRING={cuda_flags}",
            )
        )
    runner.run(command, env=environment)
    audit_cmake_paths(build / "CMakeCache.txt", work_root, required_cache_keys)
    audit_ninja_paths(build / "build.ninja", work_root)
    audit_rpath_disabled(build / "CMakeCache.txt")
    if platform_id == "linux-x64":
        audit_linux_compile_flags(build / "CMakeCache.txt", lock)
    runner.run(
        [cmake, "--build", build, "--target", "audio2face_runtime_stage", "--parallel"],
        env=environment,
    )
    staged = build / "runtime" / platform_id
    if not staged.is_dir() or staged.is_symlink():
        raise BuildError(f"worker did not produce staged runtime: {staged}")
    return staged


def validate_staged_runtime(staged: Path, platform_id: str) -> None:
    try:
        contract = runtime_contract(platform_id)
    except ValueError as exc:
        raise BuildError(str(exc)) from exc
    if not staged.is_dir() or staged.is_symlink():
        raise BuildError(f"staged runtime root is not a real directory: {staged}")
    actual_root = {entry.name for entry in staged.iterdir()}
    if actual_root != contract.root_entries:
        raise BuildError(
            f"staged runtime root must be {sorted(contract.root_entries)}; "
            f"got {sorted(actual_root)}"
        )
    expected_directories = {
        "bin": contract.bin_entries,
        "licenses": contract.license_entries,
    }
    if contract.library_entries:
        expected_directories["lib"] = contract.library_entries
    for directory, expected_entries in expected_directories.items():
        path = staged / directory
        if not path.is_dir() or path.is_symlink():
            raise BuildError(f"staged runtime directory is invalid: {path}")
        entries = {entry.name: entry for entry in path.iterdir()}
        if set(entries) != expected_entries:
            raise BuildError(
                f"staged runtime/{directory} must contain "
                f"{sorted(expected_entries)}; got {sorted(entries)}"
            )
        for name, entry in entries.items():
            if entry.is_symlink() or not entry.is_file() or entry.stat().st_size < 1:
                raise BuildError(
                    f"staged runtime/{directory}/{name} is not a non-empty "
                    "regular file"
                )
    for entry in staged.rglob("*"):
        if entry.is_symlink():
            raise BuildError(f"staged runtime contains a symlink: {entry}")
        if not (entry.is_dir() or entry.is_file()):
            raise BuildError(f"staged runtime contains a special file: {entry}")
    if platform_id == "linux-x64":
        for relative in (contract.worker, contract.trtexec):
            executable = staged.joinpath(*PurePosixPath(relative).parts)
            if not os.access(executable, os.X_OK):
                raise BuildError(f"staged Linux executable lacks execute mode: {relative}")
    try:
        bundle = _object(
            json.loads(
                (staged / "bundle.json").read_text(encoding="utf-8"),
                object_pairs_hook=duplicate_key_hook(BuildError, "JSON"),
                parse_constant=invalid_constant_hook(BuildError, "JSON"),
            ),
            "staged bundle.json",
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot parse staged bundle.json: {exc}") from exc
    if set(bundle) != RUNTIME_MANIFEST_FIELDS:
        raise BuildError(
            "staged bundle.json fields must be "
            f"{sorted(RUNTIME_MANIFEST_FIELDS)}; got {sorted(bundle)}"
        )
    if bundle != contract.manifest():
        raise BuildError(
            f"staged bundle.json does not match the exact {platform_id} contract"
        )


def _native_runtime_files(
    staged: Path, contract: RuntimePlatformContract
) -> tuple[Path, ...]:
    paths = (
        contract.worker,
        contract.trtexec,
        *(entry.path for entry in contract.libraries),
    )
    return tuple(staged.joinpath(*PurePosixPath(path).parts) for path in paths)


def _dumpbin_dependencies(output: str, path: Path) -> frozenset[str]:
    marker = "Image has the following dependencies:"
    if output.count(marker) != 1:
        raise BuildError(f"dumpbin did not report one dependency table for {path}")
    dependencies = frozenset(
        match.group(1).casefold()
        for match in re.finditer(
            r"^\s+([A-Za-z0-9_.+-]+\.dll)\s*$",
            output,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    )
    if not dependencies:
        raise BuildError(f"dumpbin reported no dependencies for {path}")
    return dependencies


def _windows_system_dependency(name: str) -> bool:
    return (
        name in WINDOWS_SYSTEM_DLLS
        or name.startswith("api-ms-win-")
        or name.startswith("ext-ms-win-")
    )


def audit_windows_dependencies(
    runner: CommandRunner,
    staged: Path,
    compiler: Path,
    environment: Mapping[str, str],
) -> None:
    contract = runtime_contract("windows-x64")
    dumpbin = compiler.parent / "dumpbin.exe"
    if not dumpbin.is_file():
        raise BuildError(f"pinned Windows producer dumpbin is unavailable: {dumpbin}")
    packaged = frozenset(
        path.name.casefold() for path in _native_runtime_files(staged, contract)
    )
    unresolved: dict[str, list[str]] = {}
    for path in _native_runtime_files(staged, contract):
        output = runner.run(
            [dumpbin, "/NOLOGO", "/DEPENDENTS", path],
            env=environment,
            capture=True,
        )
        dependencies = _dumpbin_dependencies(output, path)
        missing = sorted(
            name
            for name in dependencies
            if name not in packaged
            and name not in WINDOWS_DRIVER_DLLS
            and not _windows_system_dependency(name)
        )
        if missing:
            unresolved[path.relative_to(staged).as_posix()] = missing
    if unresolved:
        raise BuildError(
            "Windows runtime has undeclared non-system dependencies: "
            + json.dumps(unresolved, sort_keys=True)
        )


def _validate_elf64_x86_64(path: Path) -> None:
    try:
        with path.open("rb") as stream:
            header = stream.read(64)
    except OSError as exc:
        raise BuildError(f"cannot inspect staged ELF file {path}: {exc}") from exc
    if (
        len(header) != 64
        or header[:7] != b"\x7fELF\x02\x01\x01"
        or struct.unpack_from("<H", header, 16)[0] not in {2, 3}
        or struct.unpack_from("<H", header, 18)[0] != 62
        or struct.unpack_from("<I", header, 20)[0] != 1
        or struct.unpack_from("<H", header, 52)[0] != 64
    ):
        raise BuildError(f"staged native file is not Linux ELF64 x86-64: {path}")


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
    expected_soname = relative.name if relative.parts[0] == "lib" else None
    if expected_soname is None:
        if sonames:
            raise BuildError(f"staged executable declares DT_SONAME: {relative}")
    elif sonames != (expected_soname,):
        raise BuildError(
            f"staged library filename and DT_SONAME differ: {relative}: {sonames}"
        )
    if rpaths:
        raise BuildError(f"staged ELF file declares forbidden DT_RPATH: {relative}")
    if runpaths:
        raise BuildError(f"staged ELF file declares forbidden DT_RUNPATH: {relative}")


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


def _audit_glibc_requirements(
    requirements: set[str], path: Path, maximum: str
) -> None:
    if "PRIVATE" in requirements:
        raise BuildError(f"ELF file requires GLIBC_PRIVATE: {path}")
    limit = _glibc_version_tuple(maximum)
    numeric: set[str] = set()
    for version in requirements:
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,2}", version) is None:
            raise BuildError(f"ELF file requires invalid GLIBC version {version}: {path}")
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
    runner: CommandRunner,
    staged: Path,
    lock: Mapping[str, Any],
    environment: Mapping[str, str],
) -> None:
    contract = runtime_contract("linux-x64")
    readelf = Path(lock["linux_toolchain"]["readelf_path"])
    packaged = frozenset(
        path.name for path in _native_runtime_files(staged, contract)
    )
    unresolved: dict[str, list[str]] = {}
    version_definitions = {name: set() for name in ("GLIBCXX", "CXXABI", "GCC")}
    version_requirements = {name: set() for name in version_definitions}
    native_files = tuple(
        sorted(
            (
                entry
                for directory in (staged / "bin", staged / "lib")
                for entry in directory.iterdir()
                if entry.is_file()
            ),
            key=lambda path: path.relative_to(staged).as_posix(),
        )
    )
    if native_files != tuple(
        sorted(
            _native_runtime_files(staged, contract),
            key=lambda path: path.relative_to(staged).as_posix(),
        )
    ):
        raise BuildError("Linux ELF audit input differs from the runtime contract")
    for path in native_files:
        relative = PurePosixPath(path.relative_to(staged).as_posix())
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
            if name not in packaged
            and name not in LINUX_EXTERNAL_LIBRARIES
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
                f"staged GNU runtime defines no numeric {namespace} symbol versions"
            )
        actual_maximum = max(
            numeric_definitions,
            key=lambda version: _symbol_version_tuple(version, namespace),
        )
        if actual_maximum != expected_maximum:
            raise BuildError(
                f"staged GNU runtime {namespace} definition ceiling must be "
                f"{expected_maximum}; got {actual_maximum}"
            )
        unsupported = sorted(
            version_requirements[namespace] - version_definitions[namespace]
        )
        if unsupported:
            raise BuildError(
                f"staged ELFs require unsupported {namespace} versions: {unsupported}"
            )


def audit_native_dependencies(
    runner: CommandRunner,
    staged: Path,
    platform_id: str,
    compiler: Path,
    lock: Mapping[str, Any],
    environment: Mapping[str, str],
) -> None:
    if platform_id == "windows-x64":
        audit_windows_dependencies(runner, staged, compiler, environment)
    else:
        audit_linux_dependencies(runner, staged, lock, environment)


def publish_stage(staged: Path, platform_id: str) -> Path:
    output = REPOSITORY_ROOT / "build" / "runtime" / platform_id
    if output.exists() or output.is_symlink():
        raise BuildError(f"runtime handoff already exists; use a clean build tree: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise BuildError(f"stale runtime handoff exists: {partial}")
    shutil.copytree(staged, partial, symlinks=False)
    validate_staged_runtime(partial, platform_id)
    partial.replace(output)
    return output


def build_runtime(platform_id: str, work_root: Path) -> Path:
    require_native_target(platform_id)
    lock = load_lock()
    host_runner = CommandRunner(work_root)
    environment = release_environment(work_root)
    git = require_host_program(
        "git.exe" if platform_id == "windows-x64" else "git",
        environment,
    )

    source_root = work_root / "source"
    sdk_source = source_root / "audio2face-sdk"
    checkout_exact(
        host_runner,
        git,
        lock["audio2face_sdk"]["repository"],
        lock["audio2face_sdk"]["commit"],
        sdk_source,
        env=environment,
        submodules={},
    )
    tensorrt_source = source_root / "tensorrt"
    checkout_exact(
        host_runner,
        git,
        lock["tensorrt_source"]["repository"],
        lock["tensorrt_source"]["commit"],
        tensorrt_source,
        submodules=lock["tensorrt_source"]["submodules"],
        env=environment,
    )

    cmake_root = materialize_archive_root(
        lock["cmake"]["artifacts"][platform_id],
        "cmake",
        platform_id,
        work_root,
    )
    cuda_root = materialize_cuda(lock, platform_id, work_root)
    tensorrt_root = materialize_archive_root(
        lock["tensorrt"]["artifacts"][platform_id],
        "tensorrt",
        platform_id,
        work_root,
    )
    msvc_runtime: Path | None = None
    msvc_manifest: Path | None = None
    linux_runtime: Path | None = None
    linux_notices: Path | None = None
    linux_producer_packages: tuple[Path, ...] = ()
    if platform_id == "windows-x64":
        msvc_runtime, msvc_manifest = materialize_msvc_runtime(lock, work_root)
    else:
        linux_runtime, linux_notices = materialize_linux_runtime(lock, work_root)
        linux_producer_packages = materialize_linux_producer_packages(
            lock, work_root
        )

    ninja = fetch_sdk_dependencies(
        host_runner, sdk_source, platform_id, environment
    )
    runner_context: contextlib.AbstractContextManager[CommandRunner]
    if platform_id == "linux-x64":
        runner_context = linux_producer_runner(
            host_runner,
            lock,
            work_root,
            environment,
            linux_producer_packages,
        )
    else:
        runner_context = contextlib.nullcontext(host_runner)
    with runner_context as runner:
        compiler = validate_native_compiler(
            runner, platform_id, lock, environment
        )
        cmake = validate_cmake(
            runner,
            cmake_root,
            platform_id,
            lock["cmake"]["version"],
            environment,
        )
        build_environment = private_build_environment(
            environment,
            platform_id,
            cuda_root,
            tensorrt_root,
            cmake_root,
            ninja,
            compiler,
        )
        trtexec = configure_and_build_trtexec(
            runner,
            cmake,
            ninja,
            compiler,
            cuda_root,
            tensorrt_root,
            tensorrt_source,
            lock,
            platform_id,
            work_root,
            build_environment,
        )
        trtexec_provenance, msvc_provenance = write_provenance(
            lock, platform_id, runner, work_root, msvc_manifest
        )
        staged = configure_and_stage_worker(
            runner,
            cmake,
            ninja,
            compiler,
            sdk_source,
            cuda_root,
            tensorrt_root,
            msvc_runtime,
            msvc_manifest,
            msvc_provenance,
            linux_runtime,
            linux_notices,
            trtexec,
            trtexec_provenance,
            tensorrt_source,
            lock,
            platform_id,
            work_root,
            build_environment,
        )
        validate_staged_runtime(staged, platform_id)
        audit_native_dependencies(
            runner,
            staged,
            platform_id,
            compiler,
            lock,
            build_environment,
        )
    return staged


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one pinned native Audio2Face runtime at "
            "build/runtime/<platform> for extension embedding."
        )
    )
    parser.add_argument("--platform", required=True, choices=SUPPORTED_PLATFORMS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = create_argument_parser().parse_args(argv)
    try:
        with tempfile.TemporaryDirectory(
            prefix="audio2face-runtime-"
        ) as temporary:
            work_root = Path(temporary).resolve()
            staged = build_runtime(arguments.platform, work_root)
            output = publish_stage(staged, arguments.platform)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Staged {arguments.platform} runtime for extension embedding: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
