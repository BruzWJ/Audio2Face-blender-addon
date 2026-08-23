#!/usr/bin/env python3
"""Platform-neutral mechanics for native Audio2Face runtime builders.

The Windows and Linux modules own their toolchains, environments, provenance,
and native dependency audits. This module owns only shared acquisition,
extraction, CMake validation, package assembly, and publication behavior.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform as platform_module
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

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
CUDA_EXCLUDED_BUILD_INPUTS = frozenset(
    {
        "libnvptxcompiler_static.a",
        "libnvrtc-builtins_static.a",
        "libnvrtc_static.a",
        "nvptxcompiler_static.lib",
        "nvrtc-builtins_static.lib",
        "nvrtc64_120_0.alt.dll",
        "nvrtc_static.lib",
    }
)
CUDA_MANIFEST_PLATFORMS = {
    "windows-x64": "windows-x86_64",
    "linux-x64": "linux-x86_64",
}
LINUX_CUDA_LIBRARY_DIRECTORY = "lib64"
LINUX_CUDA_COMPILER_LIBRARIES = (
    "libcudadevrt.a",
    "libcudart_static.a",
)
NVIDIA_TENSORRT_RHEL8_BASE_URL = (
    "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/"
)
LINUX_TENSORRT_PACKAGE_ROLES = (
    "headers",
    "plugin_headers",
    "runtime",
    "plugin_runtime",
    "parser_runtime",
    "trtexec",
)
LINUX_TENSORRT_PACKAGE_NAMES = {
    "headers": "libnvinfer-headers-devel",
    "plugin_headers": "libnvinfer-headers-plugin-devel",
    "runtime": "libnvinfer10",
    "plugin_runtime": "libnvinfer-plugin10",
    "parser_runtime": "libnvonnxparsers10",
    "trtexec": "libnvinfer-bin",
}
LINUX_TENSORRT_LINKER_HARDLINKS = {
    "lib/libnvinfer.so": "lib/libnvinfer.so.10",
    "lib/libnvinfer_plugin.so": "lib/libnvinfer_plugin.so.10",
    "lib/libnvonnxparser.so": "lib/libnvonnxparser.so.10",
}
MSVC_RUNTIME_FILES = tuple(
    PurePosixPath(entry.path).name
    for entry in runtime_contract("windows-x64").files_for_source("platform_runtime")
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
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


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


def safe_member_path(name: str, label: str = "archive member") -> PurePosixPath:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise BuildError(f"unsafe {label}: {name!r}")
    if name.startswith("/") or WINDOWS_DRIVE_RE.match(name):
        raise BuildError(f"absolute {label} is forbidden: {name!r}")
    path = PurePosixPath(name)
    if not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise BuildError(f"non-canonical {label}: {name!r}")
    return path


def _archive_path(value: Any, label: str) -> str:
    path = _string(value, label)
    safe_member_path(path, label)
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


def _validate_rooted_artifact(value: Any, label: str) -> None:
    artifact = _object(value, label)
    _keys(artifact, {"url", "size", "sha256", "archive_root"}, label)
    _https_url(artifact["url"], f"{label}.url")
    _size(artifact["size"], f"{label}.size")
    _sha256(artifact["sha256"], f"{label}.sha256")
    root = safe_member_path(
        _string(artifact["archive_root"], f"{label}.archive_root"),
        f"{label}.archive_root",
    )
    if len(root.parts) != 1:
        raise BuildError(f"{label}.archive_root must be one directory name")


def _validate_relative_artifact(value: Any, label: str) -> dict[str, Any]:
    artifact = _object(value, label)
    _keys(artifact, {"relative_path", "size", "sha256"}, label)
    relative_path = _archive_path(artifact["relative_path"], f"{label}.relative_path")
    if len(PurePosixPath(relative_path).parts) != 1:
        raise BuildError(f"{label}.relative_path must be one filename")
    _size(artifact["size"], f"{label}.size")
    _sha256(artifact["sha256"], f"{label}.sha256")
    return artifact


def _validate_platform_artifacts(value: Any, label: str) -> None:
    artifacts = _object(value, label)
    _keys(artifacts, set(SUPPORTED_PLATFORMS), label)
    for platform_id in SUPPORTED_PLATFORMS:
        _validate_rooted_artifact(artifacts[platform_id], f"{label}.{platform_id}")


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
    _keys(
        tensorrt,
        {"version", "cuda", "windows_artifact", "linux_packages"},
        "tensorrt",
    )
    tensorrt_version = _string(tensorrt["version"], "tensorrt.version")
    tensorrt_cuda = _string(tensorrt["cuda"], "tensorrt.cuda")
    _validate_rooted_artifact(tensorrt["windows_artifact"], "tensorrt.windows_artifact")
    linux_tensorrt = _object(tensorrt["linux_packages"], "tensorrt.linux_packages")
    _keys(
        linux_tensorrt,
        {"base_url", "source_rpm", "packages"},
        "tensorrt.linux_packages",
    )
    if (
        _https_url(
            linux_tensorrt["base_url"],
            "tensorrt.linux_packages.base_url",
            directory=True,
        )
        != NVIDIA_TENSORRT_RHEL8_BASE_URL
    ):
        raise BuildError(
            "tensorrt.linux_packages.base_url must be NVIDIA's RHEL8 x86_64 repository"
        )
    source_rpm = _string(
        linux_tensorrt["source_rpm"], "tensorrt.linux_packages.source_rpm"
    )
    if source_rpm != f"tensorrt-{tensorrt_version}-1.cuda{tensorrt_cuda}.src.rpm":
        raise BuildError(
            "tensorrt.linux_packages.source_rpm does not match the locked release"
        )
    packages = _object(linux_tensorrt["packages"], "tensorrt.linux_packages.packages")
    _keys(
        packages,
        set(LINUX_TENSORRT_PACKAGE_ROLES),
        "tensorrt.linux_packages.packages",
    )
    runtime_outputs_by_role = {
        "runtime": {
            "lib/libnvinfer.so.10",
            "lib/libnvinfer_builder_resource.so.10.13.3",
        },
        "plugin_runtime": {"lib/libnvinfer_plugin.so.10"},
        "parser_runtime": {"lib/libnvonnxparser.so.10"},
        "trtexec": {runtime_contract("linux-x64").trtexec},
    }
    selected_outputs: set[str] = set()
    for role in LINUX_TENSORRT_PACKAGE_ROLES:
        label = f"tensorrt.linux_packages.packages.{role}"
        package = _object(packages[role], label)
        _keys(package, {"artifact", "files"}, label)
        artifact = _validate_relative_artifact(package["artifact"], f"{label}.artifact")
        expected_filename = (
            f"{LINUX_TENSORRT_PACKAGE_NAMES[role]}-{tensorrt_version}-"
            f"1.cuda{tensorrt_cuda}.x86_64.rpm"
        )
        if artifact["relative_path"] != expected_filename:
            raise BuildError(f"{label}.artifact does not name the exact package")
        files = _object(package["files"], f"{label}.files")
        if not files:
            raise BuildError(f"{label}.files must select at least one RPM member")
        role_outputs = set(files)
        if role in ("headers", "plugin_headers"):
            for output in role_outputs:
                output_path = PurePosixPath(_archive_path(output, f"{label}.files key"))
                if (
                    len(output_path.parts) != 2
                    or output_path.parts[0] != "include"
                    or output_path.suffix != ".h"
                ):
                    raise BuildError(
                        f"{label}.files must contain flat include/*.h paths"
                    )
        elif role_outputs != runtime_outputs_by_role[role]:
            raise BuildError(f"{label}.files do not match the exact runtime contract")
        repeated = selected_outputs & role_outputs
        if repeated:
            raise BuildError(f"TensorRT RPM outputs are repeated: {sorted(repeated)}")
        selected_outputs.update(role_outputs)
        for output, file_value in files.items():
            file_label = f"{label}.files.{output}"
            entry = _object(file_value, file_label)
            _keys(entry, {"member", "size", "sha256", "mode"}, file_label)
            member = _archive_path(entry["member"], f"{file_label}.member")
            if role in ("headers", "plugin_headers") and member != (
                f"usr/include/{PurePosixPath(output).name}"
            ):
                raise BuildError(
                    f"{file_label}.member is not its canonical include path"
                )
            _size(entry["size"], f"{file_label}.size")
            _sha256(entry["sha256"], f"{file_label}.sha256")
            expected_mode = 0o644 if role in ("headers", "plugin_headers") else 0o755
            if entry["mode"] != expected_mode:
                raise BuildError(
                    f"{file_label}.mode must be the exact regular-file mode "
                    f"{expected_mode:o}"
                )
    expected_tensorrt_runtime = {
        entry.path
        for entry in runtime_contract("linux-x64").files_for_source("tensorrt_runtime")
    }
    selected_runtime = {
        output for output in selected_outputs if output.startswith(("lib/", "bin/"))
    }
    if selected_runtime != expected_tensorrt_runtime | {
        runtime_contract("linux-x64").trtexec
    }:
        raise BuildError("TensorRT RPM files do not match the Linux runtime contract")

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
            "windows_toolchain.windows_sdk_version must be the exact vcvarsall value"
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
    if linux_toolchain["gxx_path"] != ("/opt/rh/gcc-toolset-11/root/usr/bin/g++"):
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

    producer_packages = _object(linux_toolchain["packages"], "linux_toolchain.packages")
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
        raise BuildError("linux_toolchain.architecture_flags must pin x86-64/generic")

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
    runtime_packages = _object(linux_runtime["packages"], "linux_runtime.packages")
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
        for entry in runtime_contract("linux-x64").files_for_source("platform_runtime")
    }
    if locked_runtime_outputs != expected_runtime_outputs:
        raise BuildError(
            "linux_runtime package outputs must be the exact GNU runtime pair"
        )

    runtime_licenses = _object(linux_runtime["licenses"], "linux_runtime.licenses")
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
        member = safe_member_path(info.filename)
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
                member = safe_member_path(info.name)
                _register_member(
                    member,
                    seen,
                    folded,
                    case_insensitive=platform_id == "windows-x64",
                )
                if not (info.isdir() or info.isreg() or info.issym() or info.islnk()):
                    raise BuildError(f"special TAR member is forbidden: {info.name}")
                if info.issym():
                    _normalized_link_target(member, info.linkname)
                elif info.islnk():
                    target = safe_member_path(info.linkname, "hard-link target")
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
                    linked = safe_member_path(info.linkname, "hard-link target")
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
    if name.endswith(".zip"):
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
    safe_member_path(name, "download filename")
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
        raise BuildError(
            f"{label} size mismatch: expected {expected_size}, got {count}"
        )
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise BuildError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
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


def exact_archive_root(extracted: Path, expected: str, label: str) -> Path:
    children = list(extracted.iterdir())
    if (
        len(children) != 1
        or children[0].name != expected
        or not children[0].is_dir()
        or children[0].is_symlink()
    ):
        names = [child.name for child in children]
        raise BuildError(f"{label} archive root must be {expected!r}; got {names}")
    return children[0]


def merge_component_tree(source: Path, destination: Path, label: str) -> None:
    """Move one CUDA component into the private single-copy toolkit."""

    destination.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.rglob("*"), key=lambda path: path.as_posix()):
        relative = item.relative_to(source)
        if relative == Path("LICENSE") or item.name in CUDA_EXCLUDED_BUILD_INPUTS:
            continue
        target = destination / relative
        if item.is_symlink():
            if target.exists() or target.is_symlink():
                raise BuildError(f"{label} collides at {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            item.replace(target)
        elif item.is_dir():
            if target.exists() and not target.is_dir():
                raise BuildError(f"{label} collides at {relative}")
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            if target.exists() or target.is_symlink():
                raise BuildError(f"{label} collides at {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            item.replace(target)
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
        raise BuildError("CUDA manifest release label does not match runtime-lock.json")
    manifest_platform = CUDA_MANIFEST_PLATFORMS[platform_id]
    for component_name in CUDA_COMPONENTS:
        locked_component = cuda["components"][component_name]
        actual_component = _object(
            _field(manifest, component_name, "CUDA manifest"),
            f"CUDA manifest {component_name}",
        )
        if (
            _field(
                actual_component,
                "version",
                f"CUDA manifest {component_name}",
            )
            != locked_component["version"]
        ):
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
    manifest_path.unlink()

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
        archive.unlink()
        component_root = exact_archive_root(
            extracted, _archive_root_name(archive), f"CUDA {component_name}"
        )
        license_file = component_root / "LICENSE"
        if not license_file.is_file() or license_file.is_symlink():
            raise BuildError(f"CUDA {component_name} archive has no regular LICENSE")
        merge_component_tree(component_root, toolkit, f"CUDA {component_name}")
        shutil.rmtree(extracted)

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
    archive.unlink()
    return exact_archive_root(extracted, str(artifact["archive_root"]), label)


class CommandRunner:
    def run(
        self,
        command: Sequence[os.PathLike[str] | str],
        *,
        env: Mapping[str, str],
        cwd: Path | None = None,
        capture: bool = False,
    ) -> str:
        args = [os.fspath(value) for value in command]
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
        return result.stdout.rstrip("\r\n") if capture and result.stdout else ""


def require_host_program(name: str, environment: Mapping[str, str]) -> Path:
    """Resolve one declared host tool through the release environment only."""

    if not name or Path(name).name != name:
        raise BuildError(f"host tool name must be one filename: {name!r}")
    search_path = environment.get("PATH")
    if not isinstance(search_path, str) or not search_path:
        raise BuildError("release environment has no canonical PATH")
    first_resolved: Path | None = None
    for raw_directory in search_path.split(os.pathsep):
        if not raw_directory:
            raise BuildError("release PATH contains an empty search directory")
        directory = Path(raw_directory)
        if not directory.is_absolute():
            raise BuildError(
                f"release PATH contains a non-absolute directory: {raw_directory!r}"
            )
        candidate = directory / name
        if (
            first_resolved is None
            and candidate.is_file()
            and os.access(candidate, os.X_OK)
        ):
            first_resolved = candidate.resolve()
    if first_resolved is None:
        raise BuildError(f"required native host tool is not on PATH: {name}")
    return first_resolved


def checkout_exact(
    runner: CommandRunner,
    git: Path,
    repository: str,
    commit: str,
    destination: Path,
    *,
    env: Mapping[str, str],
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    runner.run([git, "init", destination], env=env)
    runner.run([git, "-C", destination, "config", "core.autocrlf", "false"], env=env)
    runner.run([git, "-C", destination, "remote", "add", "origin", repository], env=env)
    runner.run(
        [
            git,
            "-C",
            destination,
            "fetch",
            "--depth",
            "1",
            "--no-tags",
            "origin",
            commit,
        ],
        env=env,
    )
    runner.run([git, "-C", destination, "checkout", "--detach", "FETCH_HEAD"], env=env)
    actual = runner.run(
        [git, "-C", destination, "rev-parse", "HEAD"], env=env, capture=True
    )
    if actual != commit:
        raise BuildError(f"source checkout drift: expected {commit}, got {actual}")


def validate_cmake(
    runner: CommandRunner,
    cmake_root: Path,
    platform_id: str,
    expected_version: str,
    environment: Mapping[str, str],
) -> Path:
    executable = (
        cmake_root / "bin" / ("cmake.exe" if platform_id == "windows-x64" else "cmake")
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
            raise BuildError(
                f"CMake cache {cache_path} does not resolve required {key}"
            )
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
        raise BuildError(
            f"cannot audit generated Ninja file {build_ninja}: {exc}"
        ) from exc
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
                if any(
                    marker in lowered_path for marker in GPU_PATH_MARKERS
                ) and not _inside(path, root):
                    violations.add(str(path))
    if violations:
        raise BuildError(
            f"generated Ninja build contains external GPU paths: {sorted(violations)}"
        )


def audit_rpath_disabled(cache_path: Path) -> None:
    values = _parse_cmake_cache(cache_path)
    if values.get("CMAKE_SKIP_RPATH") != "ON":
        raise BuildError(f"CMake cache {cache_path} does not disable all RPATHs")


def configure_and_package_worker(
    runner: CommandRunner,
    cmake: Path,
    ninja: Path,
    compiler: Path,
    nvcc: Path,
    sdk_source: Path,
    cuda_root: Path,
    tensorrt_root: Path,
    runtime: Path,
    contract: RuntimePlatformContract,
    external_files: Sequence[tuple[Path, str]],
    work_root: Path,
    environment: Mapping[str, str],
    extra_arguments: Sequence[str] = (),
    expected_cache_values: Sequence[tuple[str, str]] = (),
) -> None:
    """Configure, build, and package one concrete platform worker."""

    build = work_root / "build" / "worker"
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
        f"-DA2F_RUNTIME_OUTPUT_DIR:PATH={runtime}",
        *extra_arguments,
    ]
    runner.run(command, env=environment)
    audit_cmake_paths(
        build / "CMakeCache.txt",
        work_root,
        (
            "CMAKE_CUDA_COMPILER",
            "CUDAToolkit_ROOT",
            "TENSORRT_ROOT_DIR",
            "A2F_SDK_SOURCE_DIR",
            "A2F_RUNTIME_OUTPUT_DIR",
        ),
    )
    audit_ninja_paths(build / "build.ninja", work_root)
    audit_rpath_disabled(build / "CMakeCache.txt")
    cache_values = _parse_cmake_cache(build / "CMakeCache.txt")
    for key, expected in expected_cache_values:
        if cache_values.get(key) != expected:
            raise BuildError(
                f"CMake cache does not preserve pinned {key}: "
                f"expected {expected!r}, got {cache_values.get(key)!r}"
            )
    runner.run(
        [
            cmake,
            "--build",
            build,
            "--target",
            "audio2face_worker",
            "--parallel",
            "2",
        ],
        env=environment,
    )
    assemble_runtime_package(runtime, contract, external_files)


def pinned_trtexec(tensorrt_root: Path, platform_id: str) -> Path:
    """Return the one trtexec shipped by the pinned TensorRT input."""

    contract = runtime_contract(platform_id)
    relative = PurePosixPath(contract.trtexec)
    executable = tensorrt_root.joinpath(*relative.parts)
    if (
        not executable.is_file()
        or executable.is_symlink()
        or executable.stat().st_size < 1
    ):
        raise BuildError(
            f"pinned TensorRT input is missing its regular {relative.as_posix()}: "
            f"{executable}"
        )
    if platform_id == "linux-x64" and not os.access(executable, os.X_OK):
        raise BuildError(f"pinned TensorRT trtexec is not executable: {executable}")
    return executable


def _runtime_source_for_file(
    entry: RuntimePackagedFile,
    *,
    sdk_source: Path,
    cuda_runtime: Path,
    tensorrt_runtime: Path,
    platform_runtime: Path,
    platform_notices: Path | None,
    platform_metadata: Path | None,
    platform_provenance: Path | None,
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
        "trtexec_provenance": trtexec_provenance,
    }
    if platform_metadata is not None:
        direct_sources["platform_runtime_metadata"] = platform_metadata
    if platform_provenance is not None:
        direct_sources["platform_runtime_provenance"] = platform_provenance
    if entry.source in directory_sources:
        source_root = directory_sources[entry.source]
        source = source_root / name
        if entry.source != "cuda_runtime":
            return source
        try:
            resolved_root = source_root.resolve(strict=True)
            resolved_source = source.resolve(strict=True)
            resolved_source.relative_to(resolved_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise BuildError(
                f"CUDA runtime source {source} does not resolve inside "
                f"{source_root}: {exc}"
            ) from exc
        if resolved_source.parent != resolved_root:
            raise BuildError(
                f"CUDA runtime source {source} does not resolve to a flat library"
            )
        return resolved_source
    if entry.source in direct_sources:
        return direct_sources[entry.source]
    raise BuildError(
        f"runtime contract source role {entry.source!r} has no release input"
    )


def runtime_package_map(
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
    trtexec_provenance: Path,
) -> tuple[tuple[Path, str], ...]:
    """Resolve every non-generated runtime file from the package contract."""

    external: list[tuple[Path, str]] = [
        (trtexec, contract.trtexec),
        (bundle_manifest, "bundle.json"),
    ]
    for entry in (*contract.libraries, *contract.licenses):
        if entry.source == "audio2x":
            continue
        external.append(
            (
                _runtime_source_for_file(
                    entry,
                    sdk_source=sdk_source,
                    cuda_runtime=cuda_runtime,
                    tensorrt_runtime=tensorrt_runtime,
                    platform_runtime=platform_runtime,
                    platform_notices=platform_notices,
                    platform_metadata=platform_metadata,
                    platform_provenance=platform_provenance,
                    trtexec_provenance=trtexec_provenance,
                ),
                entry.path,
            )
        )
    seen_sources: set[Path] = set()
    for source, destination in external:
        if source.is_symlink() or not source.is_file() or source.stat().st_size < 1:
            raise BuildError(
                f"runtime package source for {destination} is not a non-empty "
                "regular file: "
                f"{source}"
            )
        resolved = source.resolve(strict=True)
        if resolved in seen_sources:
            raise BuildError(f"runtime package map repeats source {resolved}")
        seen_sources.add(resolved)
    return tuple(external)


def assemble_runtime_package(
    runtime: Path,
    contract: RuntimePlatformContract,
    external_files: Sequence[tuple[Path, str]],
) -> None:
    """Add declared runtime inputs to the native target output directory."""

    audio2x_entries = contract.files_for_source("audio2x")
    generated_paths = {contract.worker, audio2x_entries[0].path}
    expected_external_paths = {
        "bundle.json",
        contract.trtexec,
        *(entry.path for entry in contract.libraries if entry.source != "audio2x"),
        *(entry.path for entry in contract.licenses),
    }
    actual_external_paths = [relative for _source, relative in external_files]
    if set(actual_external_paths) != expected_external_paths or len(
        actual_external_paths
    ) != len(expected_external_paths):
        raise BuildError(
            "external runtime files do not exactly match the package contract"
        )
    if not runtime.is_dir() or runtime.is_symlink():
        raise BuildError(f"native build did not produce runtime directory: {runtime}")
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for entry in runtime.rglob("*"):
        relative = entry.relative_to(runtime).as_posix()
        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            raise BuildError(f"native runtime output contains a special file: {entry}")
        if entry.is_dir():
            actual_directories.add(relative)
            continue
        if entry.stat().st_size < 1:
            raise BuildError(f"native runtime output is empty: {entry}")
        actual_files.add(relative)
    expected_directories = {PurePosixPath(path).parts[0] for path in generated_paths}
    if actual_files != generated_paths or actual_directories != expected_directories:
        raise BuildError(
            "native runtime outputs must be exactly "
            f"{sorted(generated_paths)}; got files={sorted(actual_files)}, "
            f"directories={sorted(actual_directories)}"
        )

    for source, relative in external_files:
        destination = runtime.joinpath(*PurePosixPath(relative).parts)
        if destination.exists() or destination.is_symlink():
            raise BuildError(
                f"runtime package destination already exists: {destination}"
            )
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source.resolve(strict=True), destination)
        except OSError as exc:
            raise BuildError(
                f"cannot add runtime package file {relative}: {exc}"
            ) from exc


def validate_runtime_package(runtime: Path, platform_id: str) -> None:
    try:
        contract = runtime_contract(platform_id)
    except ValueError as exc:
        raise BuildError(str(exc)) from exc
    if not runtime.is_dir() or runtime.is_symlink():
        raise BuildError(f"runtime package root is not a real directory: {runtime}")
    actual_root = {entry.name for entry in runtime.iterdir()}
    if actual_root != contract.root_entries:
        raise BuildError(
            f"runtime package root must be {sorted(contract.root_entries)}; "
            f"got {sorted(actual_root)}"
        )
    expected_directories = {
        "bin": contract.bin_entries,
        "licenses": contract.license_entries,
    }
    if contract.library_entries:
        expected_directories["lib"] = contract.library_entries
    for directory, expected_entries in expected_directories.items():
        path = runtime / directory
        if not path.is_dir() or path.is_symlink():
            raise BuildError(f"runtime package directory is invalid: {path}")
        entries = {entry.name: entry for entry in path.iterdir()}
        if set(entries) != expected_entries:
            raise BuildError(
                f"runtime package/{directory} must contain "
                f"{sorted(expected_entries)}; got {sorted(entries)}"
            )
        for name, entry in entries.items():
            if entry.is_symlink() or not entry.is_file() or entry.stat().st_size < 1:
                raise BuildError(
                    f"runtime package/{directory}/{name} is not a non-empty "
                    "regular file"
                )
    for entry in runtime.rglob("*"):
        if entry.is_symlink():
            raise BuildError(f"runtime package contains a symlink: {entry}")
        if not (entry.is_dir() or entry.is_file()):
            raise BuildError(f"runtime package contains a special file: {entry}")
    if platform_id == "linux-x64":
        for relative in (contract.worker, contract.trtexec):
            executable = runtime.joinpath(*PurePosixPath(relative).parts)
            if not os.access(executable, os.X_OK):
                raise BuildError(
                    f"packaged Linux executable lacks execute mode: {relative}"
                )
    try:
        bundle = _object(
            json.loads(
                (runtime / "bundle.json").read_text(encoding="utf-8"),
                object_pairs_hook=duplicate_key_hook(BuildError, "JSON"),
                parse_constant=invalid_constant_hook(BuildError, "JSON"),
            ),
            "packaged bundle.json",
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot parse packaged bundle.json: {exc}") from exc
    if set(bundle) != RUNTIME_MANIFEST_FIELDS:
        raise BuildError(
            "packaged bundle.json fields must be "
            f"{sorted(RUNTIME_MANIFEST_FIELDS)}; got {sorted(bundle)}"
        )
    if bundle != contract.manifest():
        raise BuildError(
            f"packaged bundle.json does not match the exact {platform_id} contract"
        )


def native_runtime_files(
    runtime: Path, contract: RuntimePlatformContract
) -> tuple[Path, ...]:
    """Return every executable or library whose machine format is audited."""

    paths = (
        contract.worker,
        contract.trtexec,
        *(entry.path for entry in contract.libraries),
    )
    return tuple(runtime.joinpath(*PurePosixPath(path).parts) for path in paths)


def publish_runtime(runtime: Path, platform_id: str) -> Path:
    output = REPOSITORY_ROOT / "build" / "runtime" / platform_id
    if output.exists() or output.is_symlink():
        raise BuildError(
            f"runtime handoff already exists; use a clean build tree: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise BuildError(f"stale runtime handoff exists: {partial}")
    validate_runtime_package(runtime, platform_id)
    try:
        runtime.replace(partial)
        validate_runtime_package(partial, platform_id)
        if output.exists() or output.is_symlink():
            raise BuildError(f"runtime handoff appeared during build: {output}")
        partial.replace(output)
    except BuildError:
        if partial.is_dir() and not partial.is_symlink():
            shutil.rmtree(partial)
        raise
    except OSError as exc:
        if partial.is_dir() and not partial.is_symlink():
            shutil.rmtree(partial)
        raise BuildError(f"cannot publish runtime handoff: {exc}") from exc
    return output
