from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath
from types import ModuleType

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str) -> ModuleType:
    path = REPOSITORY_ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime_tool = _load_tool("build_runtime")
extension_tool = _load_tool("build_extension")


def _write_lock(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_runtime_lock_has_exact_release_inputs() -> None:
    lock = runtime_tool.load_lock()

    assert lock["schema"] == runtime_tool.LOCK_SCHEMA
    assert lock["audio2face_sdk"]["commit"] == (
        "1ca0f02535ed774f5dbcd724a31cd486368dc783"
    )
    assert lock["tensorrt_source"]["commit"] == (
        "94e2b9ef6d2cce74c76cdad499cca36cc4949197"
    )
    assert lock["tensorrt_source"]["submodules"] == {
        "parsers/onnx": "9a9f7883dd7b8cb0a718395bac2075fab6f97da8",
        "parsers/onnx/third_party/onnx": (
            "e709452ef2bbc1d113faf678c24e6d3467696e83"
        ),
        "parsers/onnx/third_party/onnx/third_party/pybind11": (
            "a2e59f0e7065404b44dfe92a28aca47ba1378dc4"
        ),
        "third_party/cub": "c3cceac115c072fb63df1836ff46d8c60d9eb304",
        "third_party/protobuf": "aea4a275e28329f648e046469c095eef74254bb2",
        "third_party/protobuf/third_party/benchmark": (
            "5b7683f49e1e9223cf9927b24f6fd3d6bd82e3f8"
        ),
        "third_party/protobuf/third_party/googletest": (
            "5ec7f0c4a113e2f18ac2c6cc7df51ad6afc24081"
        ),
    }
    assert set(lock["cuda"]["components"]) == set(runtime_tool.CUDA_COMPONENTS)
    assert "cuda_profiler_api" in lock["cuda"]["components"]
    assert lock["msvc_runtime"]["product_version"] == "14.44.35211.0"
    assert set(lock["msvc_runtime"]["files"]) == set(
        runtime_tool.MSVC_RUNTIME_FILES
    )
    assert lock["windows_toolchain"] == {
        "vctools_version": "14.43.34808",
        "cl_version": "19.43.34810",
        "windows_sdk_version": "10.0.22621.0\\",
    }
    linux_toolchain = lock["linux_toolchain"]
    assert linux_toolchain["distribution_id"] == "rocky"
    assert linux_toolchain["distribution_version"] == "8.9"
    assert linux_toolchain["glibc_version"] == "2.28"
    assert linux_toolchain["glibc_nevra"] == (
        "glibc-2.28-236.el8_9.7.x86_64"
    )
    assert linux_toolchain["producer_image"] == {
        "reference": (
            "quay.io/rockylinux/rockylinux@sha256:"
            "2fefe8993465ffa179682aeb5f2104fae221330ffbb3acadbc7aa218921fa647"
        ),
        "architecture": "amd64",
        "config_sha256": (
            "20917bc29576fedbc00e9c5d0df20bee45b9952dae4936a26a0d623e6b023f4e"
        ),
    }
    assert linux_toolchain["gxx_path"] == (
        "/opt/rh/gcc-toolset-11/root/usr/bin/g++"
    )
    assert linux_toolchain["gxx_version"] == "11.2.1"
    assert linux_toolchain["gxx_target"] == "x86_64-redhat-linux"
    assert linux_toolchain["readelf_path"] == (
        "/opt/rh/gcc-toolset-11/root/usr/bin/readelf"
    )
    assert linux_toolchain["readelf_version"] == (
        "GNU readelf version 2.36.1-4.el8_9"
    )
    assert linux_toolchain["cxx11_abi"] == 0
    assert linux_toolchain["architecture_flags"] == [
        "-march=x86-64",
        "-mtune=generic",
    ]
    assert set(linux_toolchain["packages"]) == {
        "gcc_toolset_runtime",
        "binutils",
        "gcc",
        "gxx",
        "libstdcxx_devel",
        "glibc_devel",
        "glibc_headers",
        "kernel_headers",
        "libmpc",
    }

    linux_runtime = lock["linux_runtime"]
    assert linux_runtime["source_rpm"]["sha256"] == (
        "b135158c8b66a7dac26f50abae8e99056979e5dd03032d2a604aacd565ab89eb"
    )
    assert {
        name: entry["output"]
        for name, entry in linux_runtime["packages"].items()
    } == {
        "libstdcxx": "lib/libstdc++.so.6",
        "libgcc": "lib/libgcc_s.so.1",
    }
    assert set(linux_runtime["licenses"]) == {
        "licenses/gcc-runtime-COPYING.txt",
        "licenses/gcc-runtime-COPYING.LIB.txt",
        "licenses/gcc-runtime-COPYING.RUNTIME.txt",
        "licenses/gcc-runtime-COPYING3.txt",
        "licenses/gcc-runtime-COPYING3.LIB.txt",
    }


def test_linux_runtime_materialization_uses_contract_notice_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"locked"
    identity = {
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    package_names = ("libstdcxx", "libgcc")
    packages = {
        "libstdcxx": {
            "artifact": {"url": "https://example.invalid/libstdcxx.rpm"},
            "member": "usr/lib64/libstdc++.so.6",
            "output": "lib/libstdc++.so.6",
            **identity,
        },
        "libgcc": {
            "artifact": {"url": "https://example.invalid/libgcc.rpm"},
            "member": "usr/lib64/libgcc_s.so.1",
            "output": "lib/libgcc_s.so.1",
            **identity,
        },
    }
    notice_paths = (
        "licenses/gcc-runtime-COPYING.txt",
        "licenses/gcc-runtime-COPYING.LIB.txt",
        "licenses/gcc-runtime-COPYING.RUNTIME.txt",
        "licenses/gcc-runtime-COPYING3.txt",
        "licenses/gcc-runtime-COPYING3.LIB.txt",
    )
    licenses = {
        path: {
            "package": package_names[index % len(package_names)],
            "member": f"usr/share/licenses/{PurePosixPath(path).name}",
            **identity,
        }
        for index, path in enumerate(notice_paths)
    }
    lock = {
        "linux_runtime": {
            "source_rpm": {"url": "https://example.invalid/source.rpm"},
            "packages": packages,
            "licenses": licenses,
        },
        "linux_toolchain": {"producer_image": {"reference": "locked-image"}},
    }
    downloads = iter(tmp_path / f"{name}.rpm" for name in ("source", *package_names))
    for path in list(downloads):
        path.write_bytes(payload)
    downloads = iter(tmp_path / f"{name}.rpm" for name in ("source", *package_names))
    monkeypatch.setattr(
        runtime_tool,
        "download_artifact",
        lambda *_args, **_kwargs: next(downloads),
    )
    monkeypatch.setattr(runtime_tool, "_rpm_payload", lambda *_args: payload)
    monkeypatch.setattr(
        runtime_tool,
        "_cpio_locked_members",
        lambda _payload, wanted, _label: {name: payload for name in wanted},
    )

    runtime, notices = runtime_tool.materialize_linux_runtime(lock, tmp_path / "work")

    assert {entry.name for entry in runtime.iterdir()} == {
        "libstdc++.so.6",
        "libgcc_s.so.1",
    }
    contract = runtime_tool.runtime_contract("linux-x64")
    notice_files = (
        *contract.files_for_source("platform_runtime_notice"),
        *contract.files_for_source("platform_runtime_provenance"),
    )
    assert {entry.name for entry in notices.iterdir()} == {
        PurePosixPath(entry.path).name for entry in notice_files
    }


def test_runtime_lock_rejects_unknown_or_missing_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = runtime_tool.load_lock()
    unknown = copy.deepcopy(original)
    unknown["unexpected"] = {}
    unknown_path = tmp_path / "unknown.json"
    _write_lock(unknown_path, unknown)
    monkeypatch.setattr(runtime_tool, "LOCK_PATH", unknown_path)
    with pytest.raises(runtime_tool.BuildError, match="runtime lock keys"):
        runtime_tool.load_lock()

    missing = copy.deepcopy(original)
    del missing["cuda"]["components"]["cuda_profiler_api"]
    missing_path = tmp_path / "missing.json"
    _write_lock(missing_path, missing)
    monkeypatch.setattr(runtime_tool, "LOCK_PATH", missing_path)
    with pytest.raises(runtime_tool.BuildError, match="cuda.components keys"):
        runtime_tool.load_lock()


def test_release_json_readers_reject_duplicates_and_nonfinite_numbers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"runtime-build-lock/3","schema":"runtime-build-lock/3"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_tool, "LOCK_PATH", duplicate)
    with pytest.raises(runtime_tool.BuildError, match="duplicate field 'schema'"):
        runtime_tool.load_lock()

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(
        extension_tool.ExtensionBuildError,
        match="invalid number NaN",
    ):
        extension_tool._read_json(nonfinite, "release JSON")


def test_host_platform_matching_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_tool.sys, "platform", "linux")
    monkeypatch.setattr(runtime_tool.platform_module, "machine", lambda: "x86_64")
    assert runtime_tool.detect_host_platform() == "linux-x64"

    monkeypatch.setattr(runtime_tool.sys, "platform", "linux-gnu")
    with pytest.raises(runtime_tool.BuildError, match="unsupported release host"):
        runtime_tool.detect_host_platform()

    monkeypatch.setattr(runtime_tool.sys, "platform", "win32")
    monkeypatch.setattr(runtime_tool.platform_module, "machine", lambda: "amd64")
    with pytest.raises(runtime_tool.BuildError, match="unsupported release host"):
        runtime_tool.detect_host_platform()


def test_command_runner_preserves_leading_capture_protocol_whitespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40

    def run(command: list[str], **_kwargs: object) -> object:
        return runtime_tool.subprocess.CompletedProcess(
            command,
            0,
            stdout=f" {commit} third_party/example (heads/main)\r\n",
        )

    monkeypatch.setattr(runtime_tool.subprocess, "run", run)

    output = runtime_tool.CommandRunner(tmp_path).run(
        ["git", "submodule", "status"],
        env={},
        capture=True,
    )

    assert output == f" {commit} third_party/example (heads/main)"


@pytest.mark.parametrize("prefix", ["-", "+", "U"])
def test_checkout_exact_rejects_nonclean_submodule_status(
    tmp_path: Path,
    prefix: str,
) -> None:
    source_commit = "a" * 40
    submodule_commit = "b" * 40

    class Runner:
        def run(
            self,
            command: list[Path | str],
            *,
            env: dict[str, str],
            cwd: Path | None = None,
            capture: bool = False,
        ) -> str:
            del env, cwd, capture
            if command[-2:] == ["rev-parse", "HEAD"]:
                return source_commit
            if command[-3:] == ["submodule", "status", "--recursive"]:
                return (
                    f"{prefix}{submodule_commit} third_party/example "
                    "(heads/main)"
                )
            return ""

    with pytest.raises(
        runtime_tool.BuildError,
        match="unexpected Git submodule status",
    ):
        runtime_tool.checkout_exact(
            Runner(),
            Path("git"),
            "https://example.invalid/source.git",
            source_commit,
            tmp_path / "source",
            env={},
            submodules={"third_party/example": submodule_commit},
        )


def test_extension_release_requires_native_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(extension_tool.sys, "platform", "linux")
    monkeypatch.setattr(extension_tool.platform_module, "machine", lambda: "x86_64")
    extension_tool.require_native_platform("linux-x64")
    with pytest.raises(extension_tool.ExtensionBuildError, match="does not match"):
        extension_tool.require_native_platform("windows-x64")

    monkeypatch.setattr(extension_tool.sys, "platform", "linux-gnu")
    with pytest.raises(extension_tool.ExtensionBuildError, match="unsupported"):
        extension_tool.require_native_platform("linux-x64")


def test_extension_builder_rejects_an_aliased_blender_executable(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "blender"
    executable.write_bytes(b"binary")
    executable.chmod(0o755)
    alias = tmp_path / "blender-alias"
    try:
        alias.symlink_to(executable)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable")

    with pytest.raises(extension_tool.ExtensionBuildError, match="filesystem alias"):
        extension_tool.validate_blender(alias, "linux-x64")


def test_windows_compiler_requires_exact_pinned_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler = Path(
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/"
        "14.43.34808/bin/Hostx64/x64/cl.exe"
    )
    calls: list[tuple[list[Path | str], dict[str, str], bool]] = []

    class Runner:
        def run(
            self,
            command: list[Path | str],
            *,
            env: dict[str, str],
            capture: bool,
        ) -> str:
            calls.append((command, env, capture))
            return (
                "Microsoft (R) C/C++ Optimizing Compiler Version "
                "19.43.34810 for x64"
            )

    environment = {
        "VisualStudioVersion": "17.0",
        "VSCMD_ARG_HOST_ARCH": "x64",
        "VSCMD_ARG_TGT_ARCH": "x64",
        "VCToolsInstallDir": (
            "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/"
            "MSVC/14.43.34808"
        ),
        "VCToolsVersion": "14.43.34808",
        "WindowsSDKVersion": "10.0.22621.0\\",
    }
    monkeypatch.setattr(
        runtime_tool,
        "require_host_program",
        lambda name, environment: compiler,
    )

    lock = runtime_tool.load_lock()
    assert runtime_tool.validate_native_compiler(
        Runner(), "windows-x64", lock, environment
    ) == compiler
    assert calls == [([compiler, "/?"], environment, True)]

    environment["WindowsSDKVersion"] = "10.0.22621.0\\\\"
    with pytest.raises(runtime_tool.BuildError, match="WindowsSDKVersion"):
        runtime_tool.validate_native_compiler(
            Runner(), "windows-x64", lock, environment
        )


def test_release_environment_removes_ambient_gpu_search_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    separator = os.pathsep
    monkeypatch.setenv(
        "PATH",
        separator.join(("/host/compiler/bin", "/opt/cuda/bin", "/opt/TensorRT/bin")),
    )
    monkeypatch.setenv(
        "LIB",
        separator.join(("/host/compiler/lib", "/opt/cuda/lib")),
    )
    monkeypatch.setenv("CUDA_HOME", "/ambient/cuda")
    monkeypatch.setenv("CUDA_PATH_V12_9", "/ambient/cuda")
    monkeypatch.setenv("TENSORRT_ROOT_DIR", "/ambient/tensorrt")

    environment = runtime_tool.release_environment(
        tmp_path,
        runtime_tool.load_lock(),
    )

    assert set(environment) == {
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_TERMINAL_PROMPT",
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PM_PACKAGES_ROOT",
        "TMPDIR",
    }
    assert environment["PATH"] == (
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    assert "LIB" not in environment
    assert "CUDA_HOME" not in environment
    assert "CUDA_PATH_V12_9" not in environment
    assert "TENSORRT_ROOT_DIR" not in environment
    assert environment["PM_PACKAGES_ROOT"] == str(tmp_path / "packman-cache")


def test_windows_private_build_environment_excludes_ambient_path(
    tmp_path: Path,
) -> None:
    visual_studio = tmp_path / "VisualStudio"
    vc_tools = visual_studio / "VC" / "Tools" / "MSVC" / "14.43.34808"
    windows_sdk = tmp_path / "WindowsKits" / "10"
    system_root = tmp_path / "Windows"
    compiler = vc_tools / "bin" / "Hostx64" / "x64" / "cl.exe"
    base = {
        "COMSPEC": str(system_root / "System32" / "cmd.exe"),
        "INCLUDE": os.pathsep.join(
            (
                str(vc_tools / "include"),
                str(windows_sdk / "Include" / "10.0.22621.0" / "ucrt"),
            )
        ),
        "LIB": str(vc_tools / "lib" / "x64"),
        "LIBPATH": str(system_root / "Microsoft.NET" / "Framework64"),
        "PATH": str(tmp_path / "ambient-cuda" / "bin"),
        "SystemRoot": str(system_root),
        "UniversalCRTSdkDir": str(windows_sdk),
        "VCToolsInstallDir": str(vc_tools),
        "VSINSTALLDIR": str(visual_studio),
        "WindowsSdkBinPath": str(windows_sdk / "bin"),
        "WindowsSdkDir": str(windows_sdk),
        "WindowsSDKVersion": "10.0.22621.0\\",
    }
    cuda_root = tmp_path / "inputs" / "cuda"
    tensorrt_root = tmp_path / "inputs" / "tensorrt"
    cmake_root = tmp_path / "inputs" / "cmake"
    ninja = tmp_path / "inputs" / "ninja" / "ninja.exe"

    environment = runtime_tool.private_build_environment(
        base,
        "windows-x64",
        cuda_root,
        tensorrt_root,
        cmake_root,
        ninja,
        compiler,
    )

    path_entries = environment["PATH"].split(os.pathsep)
    assert str(tmp_path / "ambient-cuda" / "bin") not in path_entries
    assert path_entries == [
        str(cmake_root / "bin"),
        str(ninja.parent),
        str(cuda_root / "bin"),
        str(tensorrt_root / "lib"),
        str(compiler.parent),
        str(system_root / "System32"),
        str(windows_sdk / "bin" / "10.0.22621.0" / "x64"),
    ]

    base["LIB"] = str(tmp_path / "ambient-cuda" / "lib")
    with pytest.raises(runtime_tool.BuildError, match="external LIB path"):
        runtime_tool.private_build_environment(
            base,
            "windows-x64",
            cuda_root,
            tensorrt_root,
            cmake_root,
            ninja,
            compiler,
        )


def test_windows_release_environment_is_an_exact_vcvars_allowlist(
    tmp_path: Path,
) -> None:
    source = {
        name: f"declared-{name}"
        for name in runtime_tool.WINDOWS_VCVARS_ENVIRONMENT_KEYS
    }
    source.update(
        {
            "COMSPEC": "C:/Windows/System32/cmd.exe",
            "SystemRoot": "C:/Windows",
            "UniversalCRTSdkDir": "C:/Program Files (x86)/Windows Kits/10",
            "VCINSTALLDIR": "C:/Visual Studio/VC",
            "VCToolsInstallDir": "C:/Visual Studio/VC/Tools/MSVC/14.43.34808",
            "VSINSTALLDIR": "C:/Visual Studio",
            "WindowsSdkBinPath": "C:/Program Files (x86)/Windows Kits/10/bin",
            "WindowsSdkDir": "C:/Program Files (x86)/Windows Kits/10",
        }
    )
    source.update(
        {
            "CL": "/D AMBIENT_CL_OPTION",
            "_CL_": "/FIambient.h",
            "LINK": "/LIBPATH:C:/ambient",
            "CFLAGS": "-march=native",
            "CXXFLAGS": "-march=native",
            "NVCC_PREPEND_FLAGS": "--compiler-bindir C:/ambient",
            "CMAKE_TOOLCHAIN_FILE": "C:/ambient/toolchain.cmake",
            "CUDA_PATH": "C:/ambient/cuda",
            "TENSORRT_ROOT_DIR": "C:/ambient/tensorrt",
        }
    )

    environment = runtime_tool._windows_release_environment(source, tmp_path)

    owned = {
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_TERMINAL_PROMPT",
        "HOME",
        "PM_PACKAGES_ROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
    }
    assert set(environment) == set(runtime_tool.WINDOWS_VCVARS_ENVIRONMENT_KEYS) | owned
    for name in runtime_tool.WINDOWS_VCVARS_ENVIRONMENT_KEYS:
        assert environment[name] == source[name]
    assert environment["HOME"] == str(tmp_path / "producer-home")
    assert environment["USERPROFILE"] == environment["HOME"]
    assert environment["TEMP"] == str(tmp_path / "producer-tmp")
    assert environment["TMP"] == environment["TEMP"]


def test_windows_vcvars_discovery_selects_newest_exact_pinned_toolchain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program_files = tmp_path / "Program Files (x86)"
    vswhere = (
        program_files
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    newer_without_pin = tmp_path / "Visual Studio" / "2022" / "Newer"
    newest_with_pin = tmp_path / "Visual Studio" / "2022" / "Community"
    older_with_pin = tmp_path / "Visual Studio" / "2022" / "BuildTools"
    toolchain = runtime_tool.load_lock()["windows_toolchain"]
    vcvars_relative = Path("VC/Auxiliary/Build/vcvars64.bat")
    compiler_relative = (
        Path("VC/Tools/MSVC")
        / toolchain["vctools_version"]
        / "bin"
        / "Hostx64"
        / "x64"
        / "cl.exe"
    )
    files = (
        vswhere,
        newer_without_pin / vcvars_relative,
        newest_with_pin / vcvars_relative,
        newest_with_pin / compiler_relative,
        older_with_pin / vcvars_relative,
        older_with_pin / compiler_relative,
    )
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> object:
        calls.append((command, kwargs))
        installations = [
            {
                "installationPath": str(newer_without_pin),
                "installationVersion": "17.14.10000.1",
            },
            {
                "installationPath": str(newest_with_pin),
                "installationVersion": "17.13.35931.197",
            },
            {
                "installationPath": str(older_with_pin),
                "installationVersion": "17.11.35327.3",
            },
        ]
        return runtime_tool.subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(installations).encode("utf-8"),
        )

    monkeypatch.setattr(runtime_tool.subprocess, "run", run)
    source = {
        "ProgramFiles(x86)": str(program_files),
    }

    vcvars = runtime_tool._discover_windows_vcvars(source, toolchain)

    assert vcvars == newest_with_pin / vcvars_relative
    assert len(calls) == 1
    command, options = calls[0]
    assert Path(command[0]) == vswhere
    assert command[1:] == [
        "-products",
        "*",
        "-version",
        "[17.0,18.0)",
        "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-sort",
        "-format",
        "json",
        "-utf8",
    ]
    assert options["check"] is True
    assert options["stdout"] is runtime_tool.subprocess.PIPE
    assert options["stderr"] is runtime_tool.subprocess.PIPE


def test_windows_vcvars_discovery_rejects_missing_pinned_toolset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program_files = tmp_path / "Program Files (x86)"
    vswhere = (
        program_files
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    vswhere.parent.mkdir(parents=True)
    vswhere.write_bytes(b"test")
    installation = tmp_path / "Visual Studio" / "2022" / "Community"
    vcvars = installation / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    vcvars.parent.mkdir(parents=True)
    vcvars.write_bytes(b"test")
    output = json.dumps([{"installationPath": str(installation)}]).encode()
    monkeypatch.setattr(
        runtime_tool.subprocess,
        "run",
        lambda *_args, **_kwargs: runtime_tool.subprocess.CompletedProcess(
            [],
            0,
            stdout=output,
        ),
    )

    with pytest.raises(runtime_tool.BuildError, match="locked MSVC toolset"):
        runtime_tool._discover_windows_vcvars(
            {"ProgramFiles(x86)": str(program_files)},
            runtime_tool.load_lock()["windows_toolchain"],
        )


def test_windows_vcvars_capture_uses_pinned_versions_and_sanitized_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_tool.os, "pathsep", ";")
    toolchain = runtime_tool.load_lock()["windows_toolchain"]
    vcvars = Path(
        "C:/Program Files/Microsoft Visual Studio/2022/Community/"
        "VC/Auxiliary/Build/vcvars64.bat"
    )
    source = {
        "A2F_VCVARS64": "C:/hostile.bat",
        "COMSPEC": "C:/Windows/System32/cmd.exe",
        "CUDA_PATH": "C:/ambient/cuda",
        "INCLUDE": "C:/ambient/include",
        "LIB": "C:/ambient/lib",
        "PATH": "C:/Windows/System32;;C:/host/git/bin;",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "ProgramFiles(x86)": "C:/Program Files (x86)",
        "SystemRoot": "C:/Windows",
        "VCToolsInstallDir": "C:/ambient/vctools",
    }
    monkeypatch.setattr(
        runtime_tool,
        "_discover_windows_vcvars",
        lambda discovered_source, discovered_toolchain: (
            vcvars
            if discovered_source is source and discovered_toolchain is toolchain
            else pytest.fail("capture changed its discovery inputs")
        ),
    )
    captured_text = (
        "COMSPEC=C:\\Windows\\System32\\cmd.exe\r\n"
        "INCLUDE=C:\\Visual Studio\\include;C:\\Windows Kits\\Include\r\n"
        "LIB=C:\\Visual Studio\\lib\r\n"
        "PATH=C:\\Visual Studio\\bin;C:\\Windows\\System32\r\n"
        "A2F_TEST_VALUE=left=right\r\n"
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> object:
        calls.append((command, kwargs))
        return runtime_tool.subprocess.CompletedProcess(
            command,
            0,
            stdout=captured_text.encode("utf-16le"),
        )

    monkeypatch.setattr(runtime_tool.subprocess, "run", run)

    environment = runtime_tool._capture_windows_vcvars_environment(
        source,
        toolchain,
    )

    assert environment["A2F_TEST_VALUE"] == "left=right"
    assert environment["INCLUDE"] == (
        "C:\\Visual Studio\\include;C:\\Windows Kits\\Include"
    )
    assert len(calls) == 1
    command, options = calls[0]
    assert command == (
        '"C:/Windows/System32/cmd.exe" /d /u /s /c '
        '"call "%A2F_VCVARS64%" 10.0.22621.0 '
        '-vcvars_ver=14.43.34808 >nul && set"'
    )
    assert options["env"] == {
        "A2F_VCVARS64": str(vcvars),
        "COMSPEC": source["COMSPEC"],
        "CUDA_PATH": source["CUDA_PATH"],
        "PATH": "C:/Windows/System32;C:/host/git/bin",
        "PATHEXT": source["PATHEXT"],
        "ProgramFiles(x86)": source["ProgramFiles(x86)"],
        "SystemRoot": source["SystemRoot"],
    }
    assert options["check"] is True
    assert options["stdout"] is runtime_tool.subprocess.PIPE
    assert options["stderr"] is runtime_tool.subprocess.PIPE


def test_windows_command_output_rejects_invalid_utf16() -> None:
    with pytest.raises(runtime_tool.BuildError, match="truncated UTF-16"):
        runtime_tool._decode_windows_command_output(b"x", "vcvars64")
    with pytest.raises(runtime_tool.BuildError, match="invalid UTF-16"):
        runtime_tool._decode_windows_command_output(b"\x00\xd8", "vcvars64")


def test_windows_vcvars_capture_rejects_duplicate_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_tool.os, "pathsep", ";")
    monkeypatch.setattr(
        runtime_tool,
        "_discover_windows_vcvars",
        lambda *_args: Path("C:/Visual Studio/vcvars64.bat"),
    )
    output = "PATH=C:\\Windows\r\nPath=C:\\Visual Studio\r\n".encode("utf-16le")
    monkeypatch.setattr(
        runtime_tool.subprocess,
        "run",
        lambda *_args, **_kwargs: runtime_tool.subprocess.CompletedProcess(
            [],
            0,
            stdout=output,
        ),
    )
    source = {
        "COMSPEC": "C:/Windows/System32/cmd.exe",
        "PATH": "C:/Windows/System32",
        "PATHEXT": ".EXE;.BAT",
        "SystemRoot": "C:/Windows",
    }

    with pytest.raises(runtime_tool.BuildError, match="duplicate environment name"):
        runtime_tool._capture_windows_vcvars_environment(
            source,
            runtime_tool.load_lock()["windows_toolchain"],
        )


def test_windows_release_environment_rejects_case_colliding_keys(
    tmp_path: Path,
) -> None:
    source = {
        name: f"declared-{name}"
        for name in runtime_tool.WINDOWS_VCVARS_ENVIRONMENT_KEYS
    }
    source.update(
        {
            "COMSPEC": "C:/Windows/System32/cmd.exe",
            "SystemRoot": "C:/Windows",
            "UniversalCRTSdkDir": "C:/Program Files (x86)/Windows Kits/10",
            "VCINSTALLDIR": "C:/Visual Studio/VC",
            "VCToolsInstallDir": "C:/Visual Studio/VC/Tools/MSVC/14.43.34808",
            "VSINSTALLDIR": "C:/Visual Studio",
            "WindowsSdkBinPath": "C:/Program Files (x86)/Windows Kits/10/bin",
            "WindowsSdkDir": "C:/Program Files (x86)/Windows Kits/10",
        }
    )
    source["Path"] = "C:/case-collision"

    with pytest.raises(runtime_tool.BuildError, match="duplicate case variants of PATH"):
        runtime_tool._windows_release_environment(source, tmp_path)


def test_host_program_resolution_uses_only_declared_path(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "declared-bin" / "release-tool"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)

    assert runtime_tool.require_host_program(
        "release-tool",
        {"PATH": str(executable.parent)},
    ) == executable.resolve()

    with pytest.raises(runtime_tool.BuildError, match="not on PATH"):
        runtime_tool.require_host_program("release-tool", {"PATH": "/usr/bin"})

    second = tmp_path / "second-bin" / "release-tool"
    second.parent.mkdir()
    second.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    second.chmod(0o755)
    with pytest.raises(runtime_tool.BuildError, match="ambiguous"):
        runtime_tool.require_host_program(
            "release-tool",
            {"PATH": os.pathsep.join((str(executable.parent), str(second.parent)))},
        )


def test_cmake_cache_paths_are_not_quote_or_whitespace_normalized() -> None:
    assert runtime_tool._absolute_path('/tmp/release-input') == Path(
        "/tmp/release-input"
    )
    assert runtime_tool._absolute_path('"/tmp/release-input"') is None
    assert runtime_tool._absolute_path(" /tmp/release-input") is None

    with pytest.raises(runtime_tool.BuildError, match="unsafe for a CMake list"):
        runtime_tool._cmake_list(
            ("/tmp/close]==]bracket",),
            "test values",
        )


def test_cmake_cache_rejects_duplicate_declarations(tmp_path: Path) -> None:
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(
        "CUDAToolkit_ROOT:PATH=/private/cuda\n"
        "CUDAToolkit_ROOT:FILEPATH=/ambient/cuda\n",
        encoding="utf-8",
    )

    with pytest.raises(runtime_tool.BuildError, match="repeats CUDAToolkit_ROOT"):
        runtime_tool._parse_cmake_cache(cache)


def test_safe_zip_extraction_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "escape")

    with pytest.raises(runtime_tool.BuildError, match="non-canonical archive member"):
        runtime_tool.safe_extract_zip(archive, tmp_path / "extract", "linux-x64")
    assert not (tmp_path / "escape.txt").exists()


def test_archive_root_and_cuda_merge_require_exact_members(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "component-linux-x86_64-1.0-archive.tar.xz"
    archive.write_bytes(b"archive")
    extracted = tmp_path / "extracted"
    wrong = extracted / "renamed-root"
    wrong.mkdir(parents=True)
    with pytest.raises(runtime_tool.BuildError, match="archive root must be"):
        runtime_tool.exact_archive_root(extracted, archive, "component")

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "same.h").write_bytes(b"identical")
    (destination / "same.h").write_bytes(b"identical")
    with pytest.raises(runtime_tool.BuildError, match="collides"):
        runtime_tool.merge_component_tree(source, destination, "component")


def test_linux_producer_image_identity_is_cryptographically_exact() -> None:
    lock = runtime_tool.load_lock()
    producer = lock["linux_toolchain"]["producer_image"]
    identity = "\n".join(
        (
            f"sha256:{producer['config_sha256']}",
            "amd64",
            "linux",
            json.dumps([producer["reference"]]),
        )
    )
    runtime_tool._parse_linux_image_identity(identity, lock)

    with pytest.raises(runtime_tool.BuildError, match="config digest drifted"):
        runtime_tool._parse_linux_image_identity(
            identity.replace(producer["config_sha256"], "0" * 64),
            lock,
        )


def test_cmake_audit_rejects_external_cuda_path(tmp_path: Path) -> None:
    work = tmp_path / "work"
    private_cuda = work / "inputs" / "cuda"
    private_cuda.mkdir(parents=True)
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(
        "CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda/bin/nvcc\n"
        f"CUDAToolkit_ROOT:PATH={private_cuda}\n",
        encoding="utf-8",
    )

    with pytest.raises(runtime_tool.BuildError, match="outside work root"):
        runtime_tool.audit_cmake_paths(
            cache, work, ("CMAKE_CUDA_COMPILER", "CUDAToolkit_ROOT")
        )


def _minimal_staged_runtime(root: Path, platform_id: str) -> None:
    contract = runtime_tool.runtime_contract(platform_id)
    native_paths = (
        contract.worker,
        contract.trtexec,
        *(entry.path for entry in contract.libraries),
    )
    for relative in native_paths:
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        if platform_id == "linux-x64":
            header = bytearray(64)
            header[:7] = b"\x7fELF\x02\x01\x01"
            header[16:18] = (3).to_bytes(2, "little")
            header[18:20] = (62).to_bytes(2, "little")
            header[20:24] = (1).to_bytes(4, "little")
            header[52:54] = (64).to_bytes(2, "little")
        else:
            header = bytearray(90)
            header[:2] = b"MZ"
            header[60:64] = (64).to_bytes(4, "little")
            header[64:68] = b"PE\x00\x00"
            header[68:70] = (0x8664).to_bytes(2, "little")
            header[88:90] = (0x020B).to_bytes(2, "little")
        path.write_bytes(bytes(header) + relative.encode("utf-8"))
    for packaged_file in contract.licenses:
        relative = packaged_file.path
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    if platform_id == "linux-x64":
        for relative in (contract.worker, contract.trtexec):
            root.joinpath(*relative.split("/")).chmod(0o755)
    (root / "bundle.json").write_text(
        json.dumps(contract.manifest()),
        encoding="utf-8",
    )


def test_staged_runtime_rejects_extra_and_renamed_files(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    _minimal_staged_runtime(staged, "linux-x64")
    extra = staged / "lib" / "unexpected.so"
    extra.write_bytes(b"unexpected")
    with pytest.raises(runtime_tool.BuildError, match="runtime/lib must contain"):
        runtime_tool.validate_staged_runtime(staged, "linux-x64")

    extra.unlink()
    worker = staged / "bin" / "audio2face_worker"
    worker.rename(staged / "bin" / "worker")
    with pytest.raises(runtime_tool.BuildError, match="runtime/bin must contain"):
        runtime_tool.validate_staged_runtime(staged, "linux-x64")


def test_python_contract_is_the_only_staged_package_filename_authority() -> None:
    cmake_sources = (
        REPOSITORY_ROOT.joinpath("worker", "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        + REPOSITORY_ROOT.joinpath(
            "worker", "cmake", "StageRuntime.cmake"
        ).read_text(encoding="utf-8")
    )
    packaged_files = {
        Path(relative).name
        for contract in runtime_tool.RUNTIME_CONTRACTS.values()
        for relative in (
            *(entry.path for entry in contract.libraries),
            *(entry.path for entry in contract.licenses),
        )
    }

    assert packaged_files
    assert all(filename not in cmake_sources for filename in packaged_files)

    for contract in runtime_tool.RUNTIME_CONTRACTS.values():
        assert not hasattr(contract, "library_files")
        assert not hasattr(contract, "notice_files")
        assert all(
            isinstance(entry, runtime_tool.RuntimePackagedFile)
            for entry in (*contract.libraries, *contract.licenses)
        )


def test_linux_dependency_audit_fails_on_unbundled_native_library(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "staged"
    _minimal_staged_runtime(staged, "linux-x64")
    readelf = tmp_path / "readelf"
    readelf.write_text("pinned test readelf\n", encoding="utf-8")
    readelf.chmod(0o755)
    lock = copy.deepcopy(runtime_tool.load_lock())
    lock["linux_toolchain"]["readelf_path"] = str(readelf)

    class Runner:
        def run(
            self,
            command: list[Path | str],
            *,
            env: dict[str, str],
            capture: bool,
        ) -> str:
            del env, capture
            if "--dynamic" in command:
                path = Path(command[-1])
                entries = [
                    " 0x0000000000000001 (NEEDED)             "
                    "Shared library: [libc.so.6]"
                ]
                if path.name == "audio2face_worker":
                    entries.append(
                        " 0x0000000000000001 (NEEDED)             "
                        "Shared library: [libnotpackaged.so.1]"
                    )
                if path.parent.name == "lib":
                    entries.append(
                        " 0x000000000000000e (SONAME)             "
                        f"Library soname: [{path.name}]"
                    )
                return "Dynamic section at offset 0x100:\n" + "\n".join(entries)
            return (
                "Version needs section '.gnu.version_r':\n"
                "  0x0010:   Name: GLIBC_2.28  Flags: none  Version: 2\n"
            )

    with pytest.raises(
        runtime_tool.BuildError,
        match="undeclared non-system dependencies",
    ):
        runtime_tool.audit_linux_dependencies(
            Runner(), staged, lock, {}
        )


def test_native_dependency_parsers_reject_unsafe_abi_inputs() -> None:
    with pytest.raises(runtime_tool.BuildError, match="newer than 2.28"):
        runtime_tool._audit_glibc_requirements(
            {"2.29"},
            Path("libaudio2x.so"),
            "2.28",
        )
    with pytest.raises(runtime_tool.BuildError, match="forbidden DT_RPATH"):
        runtime_tool._audit_elf_dynamic_identity(
            PurePosixPath("bin/audio2face_worker"),
            (),
            ("$ORIGIN/../lib",),
            (),
        )
    with pytest.raises(runtime_tool.BuildError, match="forbidden DT_RUNPATH"):
        runtime_tool._audit_elf_dynamic_identity(
            PurePosixPath("bin/audio2face_worker"),
            (),
            (),
            ("$ORIGIN/../lib",),
        )


def test_publish_stage_uses_only_fixed_build_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    staged = tmp_path / "staged"
    _minimal_staged_runtime(staged, "linux-x64")
    monkeypatch.setattr(runtime_tool, "REPOSITORY_ROOT", repository)
    output = repository / "build" / "runtime" / "linux-x64"

    assert runtime_tool.publish_stage(staged, "linux-x64") == output

    assert (output / "bundle.json").is_file()
    assert set(path.name for path in output.iterdir()) == {
        "bundle.json",
        "bin",
        "lib",
        "licenses",
    }
    with pytest.raises(runtime_tool.BuildError, match="already exists"):
        runtime_tool.publish_stage(staged, "linux-x64")


def test_extension_runtime_rejects_wrong_architecture_and_mode(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "staged"
    _minimal_staged_runtime(staged, "linux-x64")
    library = staged / "lib" / "libaudio2x.so"
    content = bytearray(library.read_bytes())
    content[18:20] = (183).to_bytes(2, "little")
    library.write_bytes(content)
    with pytest.raises(
        extension_tool.ExtensionBuildError,
        match="not Linux ELF64 x86-64",
    ):
        extension_tool.validate_runtime(staged, "linux-x64")

    _minimal_staged_runtime(tmp_path / "mode", "linux-x64")
    mode_runtime = tmp_path / "mode"
    (mode_runtime / "bin" / "audio2face_worker").chmod(0o644)
    with pytest.raises(
        extension_tool.ExtensionBuildError,
        match="must be executable",
    ):
        extension_tool.validate_runtime(mode_runtime, "linux-x64")


def test_manifest_rewrite_pins_exactly_one_platform(tmp_path: Path) -> None:
    manifest = tmp_path / "blender_manifest.toml"
    manifest.write_text(
        'id = "audio2face"\n'
        'version = "0.1.0"\n'
        'blender_version_min = "5.2.0"\n'
        'blender_version_max = "5.3.0"\n'
        'platforms = ["windows-x64", "linux-x64"]\n',
        encoding="utf-8",
    )

    assert extension_tool.rewrite_manifest_platform(manifest, "linux-x64") == (
        "audio2face",
        "0.1.0",
    )
    text = manifest.read_text(encoding="utf-8")
    assert 'platforms = ["linux-x64"]' in text
    assert "windows-x64" not in text


def test_extension_zip_uses_package_files_at_root(tmp_path: Path) -> None:
    staged = tmp_path / "audio2face"
    (staged / "runtime").mkdir(parents=True)
    (staged / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (staged / "blender_manifest.toml").write_text(
        'id = "audio2face"\nplatforms = ["linux-x64"]\n', encoding="utf-8"
    )
    (staged / "runtime" / "bundle.json").write_text(
        '{"platform":"linux-x64"}\n', encoding="utf-8"
    )
    archive = tmp_path / "audio2face.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for source in staged.rglob("*"):
            if source.is_file():
                output.write(source, source.relative_to(staged).as_posix())

    extension_tool.validate_extension_archive(archive, staged, "linux-x64")

    nested = tmp_path / "nested.zip"
    with zipfile.ZipFile(nested, "w") as output:
        for source in staged.rglob("*"):
            if source.is_file():
                relative = source.relative_to(staged).as_posix()
                output.write(source, f"audio2face/{relative}")
    with pytest.raises(extension_tool.ExtensionBuildError, match="layout differs"):
        extension_tool.validate_extension_archive(nested, staged, "linux-x64")

    extra_directory = tmp_path / "extra-directory.zip"
    with zipfile.ZipFile(extra_directory, "w") as output:
        for source in staged.rglob("*"):
            if source.is_file():
                output.write(source, source.relative_to(staged).as_posix())
        output.writestr("undeclared/", b"")
    with pytest.raises(
        extension_tool.ExtensionBuildError,
        match="undeclared directories",
    ):
        extension_tool.validate_extension_archive(
            extra_directory,
            staged,
            "linux-x64",
        )

    special_mode = tmp_path / "special-mode.zip"
    with zipfile.ZipFile(special_mode, "w") as output:
        for source in staged.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(staged).as_posix()
            if relative != "__init__.py":
                output.write(source, relative)
                continue
            info = zipfile.ZipInfo(relative)
            info.create_system = 3
            info.external_attr = (stat.S_IFIFO | 0o644) << 16
            output.writestr(info, source.read_bytes())
    with pytest.raises(
        extension_tool.ExtensionBuildError,
        match="non-regular mode",
    ):
        extension_tool.validate_extension_archive(
            special_mode,
            staged,
            "linux-x64",
        )
