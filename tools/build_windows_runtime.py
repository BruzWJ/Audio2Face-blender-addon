#!/usr/bin/env python3
"""Build the pinned Windows x64 Audio2Face runtime for extension embedding."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import runtime_build_common as common
from runtime_build_common import BuildError
from audio2face.strict_json import (
    duplicate_key_hook,
    invalid_constant_hook,
)

PLATFORM_ID = "windows-x64"
WINDOWS_VCVARSALL_ENVIRONMENT_KEYS = (
    "COMSPEC",
    "INCLUDE",
    "LIB",
    "LIBPATH",
    "PATH",
    "PROCESSOR_ARCHITECTURE",
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
)
WINDOWS_VS_STATE_LOCAL = 1
WINDOWS_VS_STATE_REGISTERED = 2
WINDOWS_VS_STATE_NO_ERRORS = 8
WINDOWS_VS_STATE_COMPLETE = (1 << 32) - 1
WINDOWS_VS_REQUIRED_STATE = (
    WINDOWS_VS_STATE_LOCAL | WINDOWS_VS_STATE_REGISTERED | WINDOWS_VS_STATE_NO_ERRORS
)
WINDOWS_VCVARSALL_PATH_VARIABLE = "A2F_VCVARSALL"
WINDOWS_VCVARSALL_NETFXSDK_VARIABLE = "NETFXSDKDir"
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


def _windows_tensorrt_files(lock: Mapping[str, Any]) -> dict[str, str]:
    """Map exact Windows archive members to the private build layout."""

    artifact = lock["tensorrt"]["windows_artifact"]
    archive_root = PurePosixPath(artifact["archive_root"])
    contract = common.runtime_contract(PLATFORM_ID)
    files: dict[str, str] = {
        (archive_root / PurePosixPath(contract.trtexec)).as_posix(): contract.trtexec
    }
    runtime_names = {
        PurePosixPath(entry.path).name
        for entry in contract.files_for_source("tensorrt_runtime")
    }
    for name in runtime_names:
        files[(archive_root / "lib" / name).as_posix()] = f"lib/{name}"
    for name in runtime_names:
        if "builder_resource" in name:
            continue
        import_name = PurePosixPath(name).with_suffix(".lib").name
        files[(archive_root / "lib" / import_name).as_posix()] = f"lib/{import_name}"
    linux_packages = lock["tensorrt"]["linux_packages"]["packages"]
    header_outputs = {
        output
        for role in ("headers", "plugin_headers")
        for output in linux_packages[role]["files"]
    }
    for output in header_outputs:
        name = PurePosixPath(output).name
        files[(archive_root / "include" / name).as_posix()] = f"include/{name}"
    return files


def materialize_windows_tensorrt(lock: Mapping[str, Any], work_root: Path) -> Path:
    """Extract only the exact Windows TensorRT compile/runtime closure."""

    artifact = lock["tensorrt"]["windows_artifact"]
    archive_path = common.download_artifact(
        artifact,
        work_root / "downloads" / "tensorrt",
        "NVIDIA TensorRT Windows archive",
    )
    root = work_root / "inputs" / "tensorrt"
    root.mkdir(parents=True, exist_ok=False)
    selection = _windows_tensorrt_files(lock)
    expected_archive_root = str(artifact["archive_root"])
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = common._validated_zip_infos(archive, case_insensitive=True)
            by_name: dict[str, zipfile.ZipInfo] = {}
            for info, member in infos:
                if member.parts[0] != expected_archive_root:
                    raise BuildError(
                        "TensorRT Windows archive contains a file outside its "
                        f"locked root: {member.as_posix()}"
                    )
                by_name[member.as_posix()] = info
            for member, output_name in selection.items():
                info = by_name.get(member)
                if info is None or info.is_dir() or info.file_size < 1:
                    raise BuildError(
                        f"TensorRT Windows archive is missing regular member {member}"
                    )
                target = common._member_destination(
                    root,
                    common.safe_member_path(output_name, "TensorRT Windows output"),
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    except (OSError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise BuildError(
            f"cannot extract pinned TensorRT Windows archive: {exc}"
        ) from exc
    finally:
        try:
            archive_path.unlink()
        except OSError as exc:
            raise BuildError(
                f"cannot delete consumed TensorRT archive {archive_path}: {exc}"
            ) from exc
    actual_files = {
        entry.relative_to(root).as_posix()
        for entry in root.rglob("*")
        if entry.is_file()
    }
    actual_directories = {
        entry.relative_to(root).as_posix()
        for entry in root.rglob("*")
        if entry.is_dir()
    }
    expected_files = set(selection.values())
    expected_directories = {
        PurePosixPath(path).parent.as_posix() for path in expected_files
    }
    if actual_files != expected_files or actual_directories != expected_directories:
        raise BuildError(
            "materialized TensorRT Windows root does not match the exact closure"
        )
    return root


def materialize_msvc_runtime(
    lock: Mapping[str, Any], work_root: Path
) -> tuple[Path, Path]:
    """Extract only the three locked x64 CRT DLLs and signed package metadata."""

    msvc = lock["msvc_runtime"]
    archive_path = common.download_artifact(
        msvc["artifact"], work_root / "downloads" / "msvc", "MSVC x64 CRT"
    )
    runtime = work_root / "inputs" / "msvc-runtime"
    runtime.mkdir(parents=True, exist_ok=False)
    manifest_output = work_root / "notices" / "msvc-redist-MANIFEST.json"
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = common._validated_zip_infos(archive, case_insensitive=True)
            by_name = {member.as_posix(): info for info, member in infos}
            for filename in common.MSVC_RUNTIME_FILES:
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
                manifest = common._object(
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
                common._field(manifest, "id", "MSVC package manifest")
                != msvc["package_id"]
                or common._field(manifest, "version", "MSVC package manifest")
                != msvc["package_version"]
            ):
                raise BuildError("MSVC package manifest identity drifted")
            manifest_output.write_bytes(manifest_bytes)
    except (OSError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise BuildError(f"cannot extract pinned MSVC CRT archive: {exc}") from exc
    archive_path.unlink()
    return runtime, manifest_output


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


def _discover_windows_vcvarsall(
    source: Mapping[str, str], toolchain: Mapping[str, Any]
) -> Path:
    """Select the newest error-free VS 2022 instance with the locked compiler."""

    program_files = _windows_environment_value(source, "ProgramFiles(x86)")
    if program_files is None:
        raise BuildError(
            "Windows release cannot locate Visual Studio because "
            "ProgramFiles(x86) is missing"
        )
    program_files_root = Path(program_files)
    if not program_files_root.is_absolute():
        raise BuildError("ProgramFiles(x86) must be an absolute path")
    vswhere = (
        program_files_root / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    )
    if not vswhere.is_file():
        raise BuildError(f"Visual Studio Installer is missing vswhere.exe: {vswhere}")

    command = [
        os.fspath(vswhere),
        "-all",
        "-products",
        "*",
        "-version",
        "[17.0,18.0)",
        "-sort",
        "-format",
        "json",
        "-utf8",
    ]
    try:
        result = subprocess.run(
            command,
            env=dict(source),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", b"")
        if isinstance(stderr, bytes):
            detail = stderr.decode("utf-8", errors="replace").strip()
        else:
            detail = str(stderr).strip()
        suffix = f": {detail}" if detail else ""
        raise BuildError(f"Visual Studio discovery failed{suffix}") from exc
    try:
        output = result.stdout.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BuildError("vswhere returned non-UTF-8 output") from exc
    try:
        installations = json.loads(
            output,
            object_pairs_hook=duplicate_key_hook(
                BuildError,
                "vswhere installation",
            ),
            parse_constant=invalid_constant_hook(
                BuildError,
                "vswhere installation",
            ),
        )
    except json.JSONDecodeError as exc:
        raise BuildError("vswhere returned invalid JSON") from exc
    if not isinstance(installations, list):
        raise BuildError("vswhere result must be a JSON array")

    vctools_version = common._string(
        common._field(toolchain, "vctools_version", "windows_toolchain"),
        "windows_toolchain.vctools_version",
    )
    for index, raw_installation in enumerate(installations):
        label = f"vswhere installation {index}"
        installation = common._object(raw_installation, label)
        state = common._field(installation, "state", label)
        if (
            isinstance(state, bool)
            or not isinstance(state, int)
            or state < 0
            or state > WINDOWS_VS_STATE_COMPLETE
        ):
            raise BuildError(f"{label}.state must be a Windows Setup state integer")
        if (state & WINDOWS_VS_REQUIRED_STATE) != WINDOWS_VS_REQUIRED_STATE:
            continue
        raw_path = common._field(installation, "installationPath", label)
        installation_path = Path(common._string(raw_path, f"{label}.installationPath"))
        if not installation_path.is_absolute():
            raise BuildError(f"{label}.installationPath must be absolute")
        vcvarsall = installation_path / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        compiler = (
            installation_path
            / "VC"
            / "Tools"
            / "MSVC"
            / vctools_version
            / "bin"
            / "Hostx64"
            / "x64"
            / "cl.exe"
        )
        if vcvarsall.is_file() and compiler.is_file():
            return vcvarsall
    raise BuildError(
        "No registered Visual Studio 2022 instance contains both vcvarsall.bat "
        f"and the locked MSVC toolset {vctools_version}"
    )


def _decode_windows_command_output(value: bytes, label: str) -> str:
    if len(value) % 2:
        raise BuildError(f"{label} returned truncated UTF-16 output")
    try:
        return value.decode("utf-16-le")
    except UnicodeDecodeError as exc:
        raise BuildError(f"{label} returned invalid UTF-16 output") from exc


def _capture_windows_vcvarsall_environment(
    source: Mapping[str, str], toolchain: Mapping[str, Any]
) -> dict[str, str]:
    """Run vcvarsall with the locked compiler and SDK, then capture its environment."""

    vcvarsall = _discover_windows_vcvarsall(source, toolchain)
    comspec_value = _windows_environment_value(source, "COMSPEC")
    system_root_value = _windows_environment_value(source, "SystemRoot")
    if comspec_value is None or system_root_value is None:
        raise BuildError("Windows release requires COMSPEC and SystemRoot")
    comspec = PureWindowsPath(comspec_value)
    expected_comspec = PureWindowsPath(system_root_value) / "System32" / "cmd.exe"
    if not comspec.is_absolute() or comspec != expected_comspec:
        raise BuildError(
            f"COMSPEC must be the SystemRoot command processor: {expected_comspec}"
        )

    source_path = _windows_environment_value(source, "PATH")
    if source_path is None:
        raise BuildError("Windows release requires PATH")
    search_paths = [entry for entry in source_path.split(os.pathsep) if entry]
    if not search_paths:
        raise BuildError("Windows release PATH has no search directories")
    relative_search_paths = [
        entry for entry in search_paths if not PureWindowsPath(entry).is_absolute()
    ]
    if relative_search_paths:
        raise BuildError(
            "Windows release PATH contains non-absolute directories: "
            + ", ".join(relative_search_paths)
        )
    canonical_source_path = os.pathsep.join(search_paths)

    preserved_vcvarsall_keys = {
        "comspec",
        "path",
        "processor_architecture",
        "systemroot",
    }
    generated_vcvarsall_keys = {
        key.casefold()
        for key in WINDOWS_VCVARSALL_ENVIRONMENT_KEYS
        if key.casefold() not in preserved_vcvarsall_keys
    }
    generated_vcvarsall_keys.add(WINDOWS_VCVARSALL_NETFXSDK_VARIABLE.casefold())
    capture_environment = {
        key: value
        for key, value in source.items()
        if key.casefold() not in generated_vcvarsall_keys
        and key.casefold() != WINDOWS_VCVARSALL_PATH_VARIABLE.casefold()
    }
    capture_path_key = next(
        key for key in capture_environment if key.casefold() == "path"
    )
    capture_environment[capture_path_key] = canonical_source_path
    capture_environment[WINDOWS_VCVARSALL_PATH_VARIABLE] = os.fspath(vcvarsall)

    vctools_version = common._string(
        common._field(toolchain, "vctools_version", "windows_toolchain"),
        "windows_toolchain.vctools_version",
    )
    windows_sdk_version = common._string(
        common._field(toolchain, "windows_sdk_version", "windows_toolchain"),
        "windows_toolchain.windows_sdk_version",
    )
    if not windows_sdk_version.endswith("\\"):
        raise BuildError(
            "windows_toolchain.windows_sdk_version must end in a backslash"
        )
    sdk_argument = windows_sdk_version[:-1]
    command_body = (
        f'call "%{WINDOWS_VCVARSALL_PATH_VARIABLE}%" amd64 {sdk_argument} '
        f"-vcvars_ver={vctools_version} >nul && set"
    )
    command_line = f'"{comspec_value}" /d /u /s /c "{command_body}"'
    try:
        result = subprocess.run(
            command_line,
            env=capture_environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", b"")
        if isinstance(stderr, bytes):
            detail = _decode_windows_command_output(stderr, "vcvarsall").strip()
        else:
            detail = str(stderr).strip()
        suffix = f": {detail}" if detail else ""
        raise BuildError(f"vcvarsall initialization failed{suffix}") from exc
    output = _decode_windows_command_output(result.stdout, "vcvarsall")
    environment: dict[str, str] = {}
    casefolded_names: set[str] = set()
    selected_names = {
        name.casefold(): name for name in WINDOWS_VCVARSALL_ENVIRONMENT_KEYS
    }
    selected_names[WINDOWS_VCVARSALL_NETFXSDK_VARIABLE.casefold()] = (
        WINDOWS_VCVARSALL_NETFXSDK_VARIABLE
    )
    for line in output.splitlines():
        if not line or line.startswith("="):
            continue
        if "=" not in line:
            raise BuildError(f"vcvarsall returned malformed environment line: {line!r}")
        name, value = line.split("=", 1)
        casefolded = name.casefold()
        canonical = selected_names.get(casefolded)
        if canonical is None:
            continue
        if not value:
            raise BuildError(f"vcvarsall returned an empty environment value: {line!r}")
        if casefolded in casefolded_names:
            raise BuildError(f"vcvarsall returned duplicate environment name {name!r}")
        casefolded_names.add(casefolded)
        environment[canonical] = value

    netfxsdk_value = environment.pop(WINDOWS_VCVARSALL_NETFXSDK_VARIABLE, None)
    if netfxsdk_value is not None:
        netfxsdk_root = PureWindowsPath(netfxsdk_value)
        if not netfxsdk_root.is_absolute():
            raise BuildError("vcvarsall returned a non-absolute NETFXSDKDir")
        netfxsdk_search_paths = {
            "INCLUDE": netfxsdk_root / "Include" / "um",
            "LIB": netfxsdk_root / "Lib" / "um" / "x64",
        }
        for name, netfxsdk_path in netfxsdk_search_paths.items():
            value = environment.get(name)
            if value is None:
                continue
            retained = [
                item
                for item in value.split(os.pathsep)
                if PureWindowsPath(item) != netfxsdk_path
            ]
            if not retained:
                raise BuildError(
                    f"vcvarsall {name} contains only the optional .NET Framework SDK"
                )
            environment[name] = os.pathsep.join(retained)
    return environment


def _windows_release_environment(
    source: Mapping[str, str], work_root: Path
) -> dict[str, str]:
    """Copy only the declared native-build values emitted by vcvarsall."""

    environment: dict[str, str] = {}
    for canonical in WINDOWS_VCVARSALL_ENVIRONMENT_KEYS:
        value = _windows_environment_value(source, canonical)
        if value is not None:
            environment[canonical] = value
    missing = sorted(set(WINDOWS_VCVARSALL_ENVIRONMENT_KEYS) - set(environment))
    if missing:
        raise BuildError(
            "Windows release requires these vcvarsall environment values: "
            + ", ".join(missing)
        )
    if environment["PROCESSOR_ARCHITECTURE"] != "AMD64":
        raise BuildError("Windows release requires PROCESSOR_ARCHITECTURE=AMD64")
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
            "Windows release requires absolute vcvarsall roots: "
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


def release_environment(work_root: Path, lock: Mapping[str, Any]) -> dict[str, str]:
    vcvarsall_environment = _capture_windows_vcvarsall_environment(
        os.environ,
        lock["windows_toolchain"],
    )
    return _windows_release_environment(vcvarsall_environment, work_root)


def private_build_environment(
    base: Mapping[str, str],
    cuda_root: Path,
    tensorrt_root: Path,
    cmake_root: Path,
    ninja: Path,
    compiler: Path,
) -> dict[str, str]:
    environment = dict(base)
    path_entries = [
        cmake_root / "bin",
        ninja.parent,
        cuda_root / "bin",
        tensorrt_root / "lib",
        compiler.parent,
    ]
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
            "Windows private build environment has non-canonical WindowsSDKVersion"
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
                    f"Windows private build environment has unsafe {key} path: {item!r}"
                )
            resolved = path.resolve(strict=False)
            if not any(common._inside(resolved, root) for root in allowed_search_roots):
                raise BuildError(
                    f"Windows private build environment has external {key} path: "
                    f"{resolved}"
                )
    environment["PATH"] = os.pathsep.join(str(path) for path in path_entries)
    return environment


def validate_native_compiler(
    runner: common.CommandRunner,
    lock: Mapping[str, Any],
    environment: Mapping[str, str],
) -> Path:
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
    if required_environment_value("VCToolsVersion") != toolchain["vctools_version"]:
        raise BuildError("VCToolsVersion does not match the pinned Windows producer")
    if (
        required_environment_value("WindowsSDKVersion")
        != toolchain["windows_sdk_version"]
    ):
        raise BuildError("WindowsSDKVersion does not match the pinned Windows producer")
    compiler = common.require_host_program("cl.exe", environment)
    compiler_identity = PureWindowsPath(os.fspath(compiler))
    vctools_root = PureWindowsPath(required_environment_value("VCToolsInstallDir"))
    expected_compiler = vctools_root / "bin" / "Hostx64" / "x64" / "cl.exe"
    if compiler_identity != expected_compiler:
        raise BuildError(
            "cl.exe is not the compiler declared by VCToolsInstallDir: "
            f"expected {expected_compiler}, got {compiler_identity}"
        )
    compiler_help = runner.run([compiler, "/?"], env=environment, capture=True)
    versions = re.findall(r"Compiler Version ([0-9.]+) for x64", compiler_help)
    if versions != [toolchain["cl_version"]]:
        raise BuildError(
            f"cl.exe version does not match the pinned Windows producer: {versions}"
        )
    return compiler


def fetch_sdk_dependencies(
    runner: common.CommandRunner,
    sdk_source: Path,
    environment: Mapping[str, str],
) -> Path:
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
    batch_environment = dict(environment)
    batch_environment["A2F_FETCH_DEPS"] = f'"{script}"'
    runner.run(
        [comspec, "/d", "/s", "/c", "call %A2F_FETCH_DEPS% release"],
        cwd=sdk_source,
        env=batch_environment,
    )
    ninja = sdk_source / "_deps" / "build-deps" / "ninja" / "ninja.exe"
    if not ninja.is_file():
        raise BuildError(f"Packman did not materialize pinned Ninja: {ninja}")
    return ninja


def write_provenance(
    lock: Mapping[str, Any],
    trtexec: Path,
    work_root: Path,
    msvc_manifest: Path,
) -> tuple[Path, Path]:
    notices = work_root / "notices"
    notices.mkdir(parents=True, exist_ok=True)
    lock_digest = common.file_sha256(common.LOCK_PATH)
    artifact = lock["tensorrt"]["windows_artifact"]
    trtexec_provenance = notices / "trtexec-PROVENANCE.txt"
    record: dict[str, Any] = {
        "schema": "audio2face-trtexec-provenance/1",
        "platform": PLATFORM_ID,
        "runtime_lock_sha256": lock_digest,
        "tensorrt_binary": {
            "version": lock["tensorrt"]["version"],
            "cuda": lock["tensorrt"]["cuda"],
            "input": {"archive": artifact},
        },
        "trtexec": {
            "archive_member": (
                PurePosixPath(artifact["archive_root"])
                / PurePosixPath(common.runtime_contract(PLATFORM_ID).trtexec)
            ).as_posix(),
            "size": trtexec.stat().st_size,
            "sha256": common.file_sha256(trtexec),
        },
    }
    trtexec_provenance.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    msvc_provenance = notices / "msvc-runtime-PROVENANCE.txt"
    msvc_record = {
        "schema": "audio2face-msvc-runtime-provenance/1",
        "runtime_lock_sha256": lock_digest,
        "package": lock["msvc_runtime"],
        "preserved_manifest_sha256": common.file_sha256(msvc_manifest),
        "release_gate": (
            "Publication requires legal review of Microsoft redistribution terms; "
            "this record does not create or replace license terms."
        ),
    }
    msvc_provenance.write_text(
        json.dumps(msvc_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return trtexec_provenance, msvc_provenance


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
    runner: common.CommandRunner,
    runtime: Path,
    compiler: Path,
    environment: Mapping[str, str],
) -> None:
    dumpbin = compiler.parent / "dumpbin.exe"
    if not dumpbin.is_file():
        raise BuildError(f"pinned Windows producer dumpbin is unavailable: {dumpbin}")
    contract = common.runtime_contract(PLATFORM_ID)
    native_files = common.native_runtime_files(runtime, contract)
    packaged = frozenset(path.name.casefold() for path in native_files)
    unresolved: dict[str, list[str]] = {}
    for path in native_files:
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
            unresolved[path.relative_to(runtime).as_posix()] = missing
    if unresolved:
        raise BuildError(
            "Windows runtime has undeclared non-system dependencies: "
            + json.dumps(unresolved, sort_keys=True)
        )


def build_windows_runtime(work_root: Path) -> Path:
    """Build one complete Windows x64 runtime in an isolated work tree."""

    common.require_native_target(PLATFORM_ID)
    lock = common.load_lock()
    runner = common.CommandRunner()
    environment = release_environment(work_root, lock)
    git = common.require_host_program("git.exe", environment)

    sdk_source = work_root / "source" / "audio2face-sdk"
    common.checkout_exact(
        runner,
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
    tensorrt_root = materialize_windows_tensorrt(lock, work_root)
    trtexec = common.pinned_trtexec(tensorrt_root, PLATFORM_ID)
    msvc_runtime, msvc_manifest = materialize_msvc_runtime(lock, work_root)
    ninja = fetch_sdk_dependencies(runner, sdk_source, environment)
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
    trtexec_provenance, msvc_provenance = write_provenance(
        lock,
        trtexec,
        work_root,
        msvc_manifest,
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
        cuda_runtime=cuda_root / "bin",
        tensorrt_runtime=tensorrt_root / "lib",
        platform_runtime=msvc_runtime,
        platform_notices=None,
        platform_metadata=msvc_manifest,
        platform_provenance=msvc_provenance,
        trtexec=trtexec,
        trtexec_provenance=trtexec_provenance,
    )
    common.configure_and_package_worker(
        runner,
        cmake,
        ninja,
        compiler,
        cuda_root / "bin" / "nvcc.exe",
        sdk_source,
        cuda_root,
        tensorrt_root,
        runtime,
        contract,
        external_files,
        work_root,
        build_environment,
    )
    common.validate_runtime_package(runtime, PLATFORM_ID)
    audit_windows_dependencies(
        runner,
        runtime,
        compiler,
        build_environment,
    )
    return runtime
