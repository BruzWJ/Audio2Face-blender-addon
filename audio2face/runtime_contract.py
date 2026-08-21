"""Exact native-runtime package contract shared by build and launch code."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


RUNTIME_SCHEMA = "audio2face-runtime/3"
RUNTIME_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "platform",
        "worker",
        "trtexec",
        "library_directories",
        "licenses",
    }
)
PACKAGE_SOURCE_ROLES = frozenset(
    {
        "audio2x",
        "cuda_runtime",
        "tensorrt_runtime",
        "platform_runtime",
        "project_license",
        "project_notices",
        "sdk_license",
        "sdk_cuda_license",
        "sdk_tensorrt_license",
        "sdk_tensorrt_acknowledgements",
        "trtexec_source_license",
        "trtexec_provenance",
        "platform_runtime_metadata",
        "platform_runtime_provenance",
        "platform_runtime_notice",
    }
)


@dataclass(frozen=True, slots=True)
class RuntimePackagedFile:
    """One exact package path and the one build input role that supplies it."""

    path: str
    source: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, str)
            or not self.path
            or self.path.strip() != self.path
            or "\\" in self.path
            or "\0" in self.path
        ):
            raise ValueError(f"runtime package path is not canonical: {self.path!r}")
        if self.source not in PACKAGE_SOURCE_ROLES:
            raise ValueError(f"unknown runtime package source role {self.source!r}")


_COMMON_NOTICE_FILES = (
    RuntimePackagedFile("licenses/audio2face-LICENSE.txt", "project_license"),
    RuntimePackagedFile("licenses/THIRD_PARTY_NOTICES.md", "project_notices"),
    RuntimePackagedFile("licenses/audio2face-sdk-LICENSE.txt", "sdk_license"),
    RuntimePackagedFile("licenses/cuda-LICENSE.txt", "sdk_cuda_license"),
    RuntimePackagedFile("licenses/tensorrt-LICENSE.txt", "sdk_tensorrt_license"),
    RuntimePackagedFile(
        "licenses/tensorrt-ACKNOWLEDGEMENTS.txt",
        "sdk_tensorrt_acknowledgements",
    ),
    RuntimePackagedFile(
        "licenses/trtexec-source-LICENSE.txt",
        "trtexec_source_license",
    ),
    RuntimePackagedFile(
        "licenses/trtexec-PROVENANCE.txt",
        "trtexec_provenance",
    ),
)


@dataclass(frozen=True, slots=True)
class RuntimePlatformContract:
    """Complete file and manifest contract for one native release target."""

    platform: str
    worker: str
    trtexec: str
    library_directories: tuple[str, ...]
    libraries: tuple[RuntimePackagedFile, ...]
    licenses: tuple[RuntimePackagedFile, ...]

    def __post_init__(self) -> None:
        if not self.platform or self.platform.strip() != self.platform:
            raise ValueError("runtime platform must be a non-empty canonical string")
        if (
            len(self.library_directories) != 1
            or self.library_directories[0] not in {"bin", "lib"}
        ):
            raise ValueError("runtime must use exactly one bin or lib library directory")
        bin_paths = (self.worker, self.trtexec)
        libraries = tuple(entry.path for entry in self.libraries)
        licenses = tuple(entry.path for entry in self.licenses)
        self._validate_flat_paths(bin_paths, "bin", "executables")
        self._validate_flat_paths(
            libraries,
            self.library_directories[0],
            "libraries",
        )
        self._validate_flat_paths(licenses, "licenses", "licenses")
        if set(bin_paths) & set(libraries):
            raise ValueError("runtime executables and libraries must not share a path")
        if len(self.files_for_source("audio2x")) != 1:
            raise ValueError("runtime contract must contain exactly one audio2x library")

    def files_for_source(self, source: str) -> tuple[RuntimePackagedFile, ...]:
        """Return the exact package files supplied by one declared source role."""

        if source not in PACKAGE_SOURCE_ROLES:
            raise ValueError(f"unknown runtime package source role {source!r}")
        return tuple(
            entry
            for entry in (*self.libraries, *self.licenses)
            if entry.source == source
        )

    @staticmethod
    def _validate_flat_paths(
        paths: tuple[str, ...], directory: str, label: str
    ) -> None:
        if not paths or len(paths) != len(set(paths)):
            raise ValueError(f"runtime {label} must be non-empty and unique")
        for path in paths:
            parts = path.split("/") if isinstance(path, str) else []
            if (
                len(parts) != 2
                or parts[0] != directory
                or not parts[1]
                or parts[1] in {".", ".."}
                or "\\" in parts[1]
                or "\0" in parts[1]
            ):
                raise ValueError(
                    f"runtime {label} path must be exactly {directory}/<name>: {path!r}"
                )

    @property
    def bin_entries(self) -> frozenset[str]:
        paths = (self.worker, self.trtexec) + tuple(
            entry.path for entry in self.libraries if entry.path.startswith("bin/")
        )
        return frozenset(path.split("/", 1)[1] for path in paths)

    @property
    def library_entries(self) -> frozenset[str]:
        return frozenset(
            path.split("/", 1)[1]
            for path in (entry.path for entry in self.libraries)
            if path.startswith("lib/")
        )

    @property
    def license_entries(self) -> frozenset[str]:
        return frozenset(entry.path.split("/", 1)[1] for entry in self.licenses)

    @property
    def root_entries(self) -> frozenset[str]:
        return frozenset(
            {"bundle.json", "bin", "licenses", *self.library_directories}
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema": RUNTIME_SCHEMA,
            "platform": self.platform,
            "worker": self.worker,
            "trtexec": self.trtexec,
            "library_directories": list(self.library_directories),
            "licenses": [entry.path for entry in self.licenses],
        }


RUNTIME_CONTRACTS: Mapping[str, RuntimePlatformContract] = MappingProxyType(
    {
        "windows-x64": RuntimePlatformContract(
            platform="windows-x64",
            worker="bin/audio2face_worker.exe",
            trtexec="bin/trtexec.exe",
            library_directories=("bin",),
            libraries=(
                RuntimePackagedFile("bin/audio2x.dll", "audio2x"),
                RuntimePackagedFile("bin/cudart64_12.dll", "cuda_runtime"),
                RuntimePackagedFile("bin/cublas64_12.dll", "cuda_runtime"),
                RuntimePackagedFile("bin/cublasLt64_12.dll", "cuda_runtime"),
                RuntimePackagedFile("bin/curand64_10.dll", "cuda_runtime"),
                RuntimePackagedFile("bin/nvrtc64_120_0.dll", "cuda_runtime"),
                RuntimePackagedFile(
                    "bin/nvrtc-builtins64_129.dll", "cuda_runtime"
                ),
                RuntimePackagedFile("bin/nvinfer_10.dll", "tensorrt_runtime"),
                RuntimePackagedFile(
                    "bin/nvinfer_plugin_10.dll", "tensorrt_runtime"
                ),
                RuntimePackagedFile(
                    "bin/nvonnxparser_10.dll", "tensorrt_runtime"
                ),
                RuntimePackagedFile(
                    "bin/nvinfer_builder_resource_10.dll", "tensorrt_runtime"
                ),
                RuntimePackagedFile("bin/msvcp140.dll", "platform_runtime"),
                RuntimePackagedFile("bin/vcruntime140.dll", "platform_runtime"),
                RuntimePackagedFile(
                    "bin/vcruntime140_1.dll", "platform_runtime"
                ),
            ),
            licenses=_COMMON_NOTICE_FILES
            + (
                RuntimePackagedFile(
                    "licenses/msvc-runtime-MANIFEST.json",
                    "platform_runtime_metadata",
                ),
                RuntimePackagedFile(
                    "licenses/msvc-runtime-PROVENANCE.txt",
                    "platform_runtime_provenance",
                ),
            ),
        ),
        "linux-x64": RuntimePlatformContract(
            platform="linux-x64",
            worker="bin/audio2face_worker",
            trtexec="bin/trtexec",
            library_directories=("lib",),
            libraries=(
                RuntimePackagedFile("lib/libaudio2x.so", "audio2x"),
                RuntimePackagedFile("lib/libcudart.so.12", "cuda_runtime"),
                RuntimePackagedFile("lib/libcublas.so.12", "cuda_runtime"),
                RuntimePackagedFile("lib/libcublasLt.so.12", "cuda_runtime"),
                RuntimePackagedFile("lib/libcurand.so.10", "cuda_runtime"),
                RuntimePackagedFile("lib/libnvrtc.so.12", "cuda_runtime"),
                RuntimePackagedFile(
                    "lib/libnvrtc-builtins.so.12.9", "cuda_runtime"
                ),
                RuntimePackagedFile(
                    "lib/libnvinfer.so.10", "tensorrt_runtime"
                ),
                RuntimePackagedFile(
                    "lib/libnvinfer_plugin.so.10", "tensorrt_runtime"
                ),
                RuntimePackagedFile(
                    "lib/libnvonnxparser.so.10", "tensorrt_runtime"
                ),
                RuntimePackagedFile(
                    "lib/libnvinfer_builder_resource.so.10", "tensorrt_runtime"
                ),
                RuntimePackagedFile("lib/libstdc++.so.6", "platform_runtime"),
                RuntimePackagedFile("lib/libgcc_s.so.1", "platform_runtime"),
            ),
            licenses=_COMMON_NOTICE_FILES
            + (
                RuntimePackagedFile(
                    "licenses/gcc-runtime-COPYING.txt", "platform_runtime_notice"
                ),
                RuntimePackagedFile(
                    "licenses/gcc-runtime-COPYING.LIB.txt",
                    "platform_runtime_notice",
                ),
                RuntimePackagedFile(
                    "licenses/gcc-runtime-COPYING.RUNTIME.txt",
                    "platform_runtime_notice",
                ),
                RuntimePackagedFile(
                    "licenses/gcc-runtime-COPYING3.txt",
                    "platform_runtime_notice",
                ),
                RuntimePackagedFile(
                    "licenses/gcc-runtime-COPYING3.LIB.txt",
                    "platform_runtime_notice",
                ),
                RuntimePackagedFile(
                    "licenses/gcc-runtime-PROVENANCE.txt",
                    "platform_runtime_provenance",
                ),
            ),
        ),
    }
)

for _platform_key, _platform_contract in RUNTIME_CONTRACTS.items():
    if _platform_key != _platform_contract.platform:
        raise ValueError("runtime contract mapping key must equal contract.platform")
if {
    entry.source
    for _platform_contract in RUNTIME_CONTRACTS.values()
    for entry in (*_platform_contract.libraries, *_platform_contract.licenses)
} != PACKAGE_SOURCE_ROLES:
    raise ValueError("runtime package source roles must all be used by a contract")


def runtime_contract(platform_id: str) -> RuntimePlatformContract:
    """Return the exact contract for a supported platform identifier."""

    try:
        return RUNTIME_CONTRACTS[platform_id]
    except KeyError as exc:
        raise ValueError(f"unsupported runtime platform {platform_id!r}") from exc
