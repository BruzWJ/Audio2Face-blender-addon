"""Secure one-time installation of the local Audio2Face runtime and model.

This module performs blocking work and deliberately has no :mod:`bpy` import.
Blender runs :func:`install_managed_runtime` on a background thread and consumes
its progress messages from the main-thread timer.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable
from urllib.parse import urlparse

from .runtime_catalog import RuntimeArtifact
from .runtime_bundle import BundleError, BundleLaunchSpec, resolve_runtime_bundle


DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MAX_ZIP_MEMBERS = 100_000
RUNTIME_RECEIPT_FILENAME = ".a2f-archive-sha256"
INSTALL_LOCK_FILENAME = ".a2f-runtime-install.lock"
INSTALL_LOCK_TIMEOUT_SECONDS = 300.0
INSTALL_LOCK_POLL_SECONDS = 0.10


class RuntimeInstallError(RuntimeError):
    """Raised when download, verification, extraction, or TRT setup fails."""


class RuntimeInstallCancelled(RuntimeInstallError):
    """Raised after the user cancels managed-runtime installation."""


@dataclass(frozen=True, slots=True)
class InstallProgress:
    stage: str
    progress: float
    message: str


ProgressCallback = Callable[[InstallProgress], None]
OpenUrl = Callable[..., BinaryIO]


def _path_exists(path: Path) -> bool:
    """Return true for ordinary paths and broken symbolic links."""

    return os.path.lexists(path)


def _try_lock_file(handle: BinaryIO) -> bool:
    """Try to take one OS-held byte-range/file lock without blocking."""

    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            busy_errnos = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
            if exc.errno in busy_errnos or getattr(exc, "winerror", None) in {32, 33}:
                return False
            raise
        return True

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            return False
        raise
    return True


def _unlock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class _InterprocessInstallLock:
    """Cancellable OS-held lock for one extension data directory.

    The lock file is intentionally persistent.  Removing it would allow two
    processes to lock different inodes at the same pathname.  The lock itself
    belongs to the open file handle, so the operating system releases it when
    Blender exits or crashes.
    """

    def __init__(
        self,
        path: Path,
        *,
        canceled: threading.Event,
        timeout: float,
        poll_interval: float = INSTALL_LOCK_POLL_SECONDS,
    ) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise RuntimeInstallError("managed-runtime install lock timeout is invalid")
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or not math.isfinite(poll_interval)
            or poll_interval <= 0
        ):
            raise RuntimeInstallError("managed-runtime install lock poll interval is invalid")
        self.path = path
        self.canceled = canceled
        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)
        self._handle: BinaryIO | None = None
        self._held = False

    def __enter__(self) -> _InterprocessInstallLock:
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            descriptor = os.open(self.path, flags, 0o600)
            try:
                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                handle = os.fdopen(descriptor, "r+b", buffering=0)
            except Exception:
                os.close(descriptor)
                raise
        except OSError as exc:
            raise RuntimeInstallError(
                f"cannot open managed-runtime install lock {self.path}: {exc}"
            ) from exc

        self._handle = handle
        deadline = time.monotonic() + self.timeout
        try:
            while True:
                _check_cancelled(self.canceled)
                try:
                    if _try_lock_file(handle):
                        self._held = True
                        return self
                except OSError as exc:
                    raise RuntimeInstallError(
                        f"cannot lock managed-runtime install directory: {exc}"
                    ) from exc

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeInstallError(
                        "another Blender instance is installing the managed runtime; "
                        "timed out waiting for its install lock"
                    )
                wait_time = min(self.poll_interval, remaining)
                if self.canceled.wait(wait_time):
                    _check_cancelled(self.canceled)
        except Exception:
            try:
                try:
                    handle.close()
                except OSError:
                    pass
            finally:
                self._handle = None
            raise

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if self._held:
                try:
                    _unlock_file(handle)
                except OSError:
                    # Closing the handle also releases an OS-held lock.  Never
                    # turn an otherwise successful activation into a failure.
                    pass
        finally:
            self._held = False
            try:
                handle.close()
            except OSError:
                pass


def _emit(callback: ProgressCallback, stage: str, progress: float, message: str) -> None:
    value = float(progress)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise RuntimeInstallError("installer progress is outside [0, 1]")
    callback(InstallProgress(stage, value, message))


def _check_cancelled(canceled: threading.Event) -> None:
    if canceled.is_set():
        raise RuntimeInstallCancelled("managed-runtime installation was canceled")


def _download_archive(
    artifact: RuntimeArtifact,
    destination: Path,
    *,
    progress: ProgressCallback,
    canceled: threading.Event,
    open_url: OpenUrl,
) -> None:
    request = urllib.request.Request(
        artifact.url,
        headers={
            "Accept": "application/zip, application/octet-stream",
            "Accept-Encoding": "identity",
            "User-Agent": "Audio2Face/0.1",
        },
        method="GET",
    )
    digest = hashlib.sha256()
    downloaded = 0
    last_reported = -1
    try:
        response_context = open_url(request, timeout=30)
        with response_context as response, destination.open("wb") as output:
            final_url = response.geturl()
            parsed_final_url = urlparse(final_url)
            if (
                parsed_final_url.scheme != "https"
                or not parsed_final_url.netloc
                or parsed_final_url.username
                or parsed_final_url.password
            ):
                raise RuntimeInstallError(
                    "runtime download redirected to a non-HTTPS or credentialed URL"
                )
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    declared_length = int(content_length)
                except ValueError as exc:
                    raise RuntimeInstallError("runtime server returned an invalid Content-Length") from exc
                if declared_length != artifact.size:
                    raise RuntimeInstallError(
                        "runtime download size does not match the pinned release catalog"
                    )
            while True:
                _check_cancelled(canceled)
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > artifact.size:
                    raise RuntimeInstallError("runtime download exceeded its catalog size")
                digest.update(chunk)
                output.write(chunk)
                percent = int(downloaded * 1000 / artifact.size)
                if percent != last_reported:
                    last_reported = percent
                    _emit(
                        progress,
                        "downloading",
                        0.75 * downloaded / artifact.size,
                        f"Downloading managed runtime ({downloaded / (1024**2):.1f} MiB)",
                    )
            output.flush()
            os.fsync(output.fileno())
    except RuntimeInstallError:
        raise
    except Exception as exc:
        raise RuntimeInstallError(f"managed-runtime download failed: {exc}") from exc

    if downloaded != artifact.size:
        raise RuntimeInstallError(
            f"runtime download has {downloaded} bytes; expected {artifact.size}"
        )
    if digest.hexdigest() != artifact.sha256:
        raise RuntimeInstallError("runtime download failed its SHA-256 verification")


def _safe_member_path(name: str) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name:
        raise RuntimeInstallError(f"runtime archive contains an unsafe path {name!r}")
    path = PurePosixPath(name)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() != name
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise RuntimeInstallError(f"runtime archive contains an unsafe path {name!r}")
    return path


def _extract_archive(
    archive: Path,
    destination: Path,
    artifact: RuntimeArtifact,
    *,
    progress: ProgressCallback,
    canceled: threading.Event,
) -> None:
    try:
        with zipfile.ZipFile(archive, "r") as package:
            members = package.infolist()
            if not members or len(members) > MAX_ZIP_MEMBERS:
                raise RuntimeInstallError("runtime archive has an invalid member count")
            total_size = sum(member.file_size for member in members if not member.is_dir())
            if total_size != artifact.unpacked_size:
                raise RuntimeInstallError(
                    "runtime archive unpacked size does not match the pinned release catalog"
                )
            seen: set[str] = set()
            extracted = 0
            root = destination.resolve(strict=False)
            for member in members:
                _check_cancelled(canceled)
                member_name = member.filename[:-1] if member.is_dir() else member.filename
                relative = _safe_member_path(member_name)
                comparison_path = relative.as_posix()
                if artifact.platform == "windows-x64":
                    comparison_path = comparison_path.casefold()
                if comparison_path in seen:
                    raise RuntimeInstallError(
                        f"runtime archive contains duplicate path {member.filename!r}"
                    )
                seen.add(comparison_path)
                mode = (member.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise RuntimeInstallError("runtime archive may not contain symbolic links")
                if member.flag_bits & 0x1:
                    raise RuntimeInstallError("runtime archive may not contain encrypted files")
                output = destination.joinpath(*relative.parts)
                resolved_parent = output.parent.resolve(strict=False)
                try:
                    resolved_parent.relative_to(root)
                except ValueError as exc:
                    raise RuntimeInstallError("runtime archive path escapes its install root") from exc
                if member.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                with package.open(member, "r") as source, output.open("wb") as target:
                    while True:
                        _check_cancelled(canceled)
                        chunk = source.read(DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        extracted += len(chunk)
                        if extracted > artifact.unpacked_size:
                            raise RuntimeInstallError("runtime archive exceeded its unpacked size")
                        target.write(chunk)
                if os.name != "nt":
                    permissions = stat.S_IMODE(mode) or 0o644
                    output.chmod(permissions)
                _emit(
                    progress,
                    "extracting",
                    0.75 + 0.10 * extracted / artifact.unpacked_size,
                    "Installing verified runtime files",
                )
            if extracted != artifact.unpacked_size:
                raise RuntimeInstallError("runtime archive extraction was incomplete")
    except RuntimeInstallError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise RuntimeInstallError(f"cannot extract managed runtime: {exc}") from exc


def _trt_build_plan(
    spec: BundleLaunchSpec,
    model: Path,
    model_label: str,
) -> tuple[list[str], str]:
    """Load model metadata once and derive the command and progress message."""

    model_directory = model.parent
    onnx_path = model_directory / "network.onnx"
    info_path = model_directory / "trt_info.json"
    output_path = model_directory / "network.trt"
    if not onnx_path.is_file() or not info_path.is_file():
        raise RuntimeInstallError("managed model is missing network.onnx or trt_info.json")
    try:
        with info_path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeInstallError(f"cannot read managed model TRT settings: {exc}") from exc
    if not isinstance(document, dict):
        raise RuntimeInstallError("managed model trt_info.json must be an object")
    try:
        build_parameters = document["trt_build_param"]
        defaults = document["defaults"]
    except KeyError as exc:
        raise RuntimeInstallError(
            "managed model TRT settings must define trt_build_param and defaults"
        ) from exc
    if not isinstance(build_parameters, dict) or not isinstance(defaults, dict):
        raise RuntimeInstallError("managed model TRT settings have an invalid structure")
    trt_parameters: list[str] = []
    for group, parameters in build_parameters.items():
        if not isinstance(group, str) or not group:
            raise RuntimeInstallError(
                "managed model TRT parameter group names must be non-empty strings"
            )
        if not isinstance(parameters, list) or not all(
            isinstance(item, str)
            and item.startswith("--")
            and item != "--"
            and "\x00" not in item
            and "\r" not in item
            and "\n" not in item
            for item in parameters
        ):
            raise RuntimeInstallError(
                f"managed model TRT settings group {group!r} is invalid"
            )
        trt_parameters.extend(parameters)
    installer_owned_options = {"--onnx", "--saveengine"}
    for parameter in trt_parameters:
        option = parameter.partition("=")[0].casefold()
        if option in installer_owned_options:
            raise RuntimeInstallError(
                f"managed model TRT settings may not override installer option {option}"
            )
    format_values: dict[str, Any] = {}
    for name, value in defaults.items():
        if not isinstance(name, str) or isinstance(value, bool) or not isinstance(
            value, (int, float, str)
        ):
            raise RuntimeInstallError("managed model TRT defaults are invalid")
        format_values[name] = value
    # Blender previews one character stream, so optimize every conventional
    # batch placeholder for exactly one track.
    format_values["BATCH_SIZE"] = 1
    format_values["MIN_BATCH_SIZE"] = 1
    format_values["MAX_BATCH_SIZE"] = 1
    format_values["OPT_BATCH_SIZE"] = 1
    try:
        formatted = [parameter.format(**format_values) for parameter in trt_parameters]
    except (KeyError, ValueError) as exc:
        raise RuntimeInstallError(f"managed model TRT settings cannot be formatted: {exc}") from exc
    command = [
        str(spec.trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        *formatted,
    ]
    message = f"Optimizing the {model_label} model for this GPU"
    estimate = document.get("estimated_trt_builder_time")
    if (
        isinstance(estimate, (int, float))
        and not isinstance(estimate, bool)
        and math.isfinite(estimate)
        and estimate > 0
    ):
        message += f" (upstream estimate: about {int(round(estimate))} seconds)"
    return command, message


def _build_trt_engine(
    spec: BundleLaunchSpec,
    model: Path,
    model_id: str,
    model_label: str,
    *,
    progress_value: float,
    progress: ProgressCallback,
    canceled: threading.Event,
) -> None:
    output = model.parent / "network.trt"
    if output.exists():
        raise RuntimeInstallError(
            f"managed runtime archive may not contain a prebuilt {model_label} "
            "network.trt; "
            "the engine must be optimized locally for this GPU"
        )
    command, build_message = _trt_build_plan(spec, model, model_label)
    log_path = spec.root / f"trtexec-{model_id}-install.log"
    _emit(
        progress,
        f"building_{model_id}_model",
        progress_value,
        build_message,
    )
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=str(spec.root),
                env=dict(spec.env),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                shell=False,
            )
            while process.poll() is None:
                if not canceled.wait(0.10):
                    continue
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
                raise RuntimeInstallCancelled("managed-runtime installation was canceled")
            returncode = process.returncode
    except OSError as exc:
        raise RuntimeInstallError(f"could not run bundled trtexec: {exc}") from exc
    if returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            tail = ""
        detail = f"\n{tail}" if tail else ""
        raise RuntimeInstallError(
            f"TensorRT {model_label} model optimization failed with exit code "
            f"{returncode}{detail}"
        )


@dataclass(frozen=True, slots=True)
class _ActivationPlan:
    staged: Path
    destination: Path
    backup: Path


def _remove_activation_path(path: Path) -> None:
    """Remove an inactive activation path without following a symlink."""

    if not _path_exists(path):
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _prepare_activation(
    staging_root: Path,
    data_root: Path,
    platform: str,
) -> _ActivationPlan:
    """Validate paths and remove a stale backup before entering the gate."""

    staged = staging_root / "runtime" / platform
    destination_parent = data_root / "runtime"
    destination = destination_parent / platform
    if not staged.is_dir():
        raise RuntimeInstallError(
            f"runtime archive does not contain runtime/{platform}"
        )
    destination_parent.mkdir(parents=True, exist_ok=True)
    backup = destination_parent / f".{platform}.previous"
    _remove_activation_path(backup)
    return _ActivationPlan(staged=staged, destination=destination, backup=backup)


def _atomic_activate(plan: _ActivationPlan) -> Path | None:
    """Perform only the rename transaction and return its inactive backup."""

    if _path_exists(plan.backup):
        raise RuntimeInstallError("managed-runtime activation backup was not prepared")
    replaced = False
    try:
        if _path_exists(plan.destination):
            os.replace(plan.destination, plan.backup)
            replaced = True
        os.replace(plan.staged, plan.destination)
    except Exception:
        if (
            replaced
            and _path_exists(plan.backup)
            and not _path_exists(plan.destination)
        ):
            os.replace(plan.backup, plan.destination)
        raise
    return plan.backup if replaced else None


def _cleanup_activation_backup(backup: Path | None) -> None:
    """Best-effort cleanup after the new runtime is already active."""

    if backup is None:
        return
    try:
        _remove_activation_path(backup)
    except Exception:
        # Activation has committed.  A stale inactive backup is harmless and a
        # later install will remove it before its own rename transaction.
        pass


def validate_install_receipt(
    spec: BundleLaunchSpec,
    artifact: RuntimeArtifact,
) -> None:
    """Require the active runtime to match this add-on's pinned archive exactly."""

    if spec.platform != artifact.platform:
        raise RuntimeInstallError("managed-runtime receipt platform does not match the catalog")
    receipt = spec.root / RUNTIME_RECEIPT_FILENAME
    try:
        if receipt.stat().st_size != 65:
            raise RuntimeInstallError("managed-runtime archive receipt is invalid")
        checksum = receipt.read_text(encoding="ascii")
    except RuntimeInstallError:
        raise
    except (OSError, UnicodeError) as exc:
        raise RuntimeInstallError(f"managed-runtime archive receipt is missing: {receipt}") from exc
    if checksum != f"{artifact.sha256}\n":
        raise RuntimeInstallError(
            "installed runtime does not match this add-on release; install the pinned runtime again"
        )


def install_managed_runtime(
    artifact: RuntimeArtifact,
    data_root: str | Path,
    *,
    progress: ProgressCallback,
    canceled: threading.Event,
    activation_lock: Any,
    open_url: OpenUrl = urllib.request.urlopen,
    interprocess_lock_timeout: float = INSTALL_LOCK_TIMEOUT_SECONDS,
) -> BundleLaunchSpec:
    """Download, verify, optimize, and atomically activate one runtime release."""

    root = Path(data_root).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        _emit(
            progress,
            "waiting_for_install_lock",
            0.0,
            "Waiting for exclusive managed-runtime installer access",
        )
        with _InterprocessInstallLock(
            root / INSTALL_LOCK_FILENAME,
            canceled=canceled,
            timeout=interprocess_lock_timeout,
        ):
            temporary = Path(tempfile.mkdtemp(prefix=".a2f-install-", dir=root))
            archive = temporary / "runtime.zip"
            extracted = temporary / "extracted"
            extracted.mkdir()
            _emit(progress, "downloading", 0.0, "Starting managed-runtime download")
            _download_archive(
                artifact,
                archive,
                progress=progress,
                canceled=canceled,
                open_url=open_url,
            )
            _emit(progress, "verifying", 0.75, "Runtime archive verified")
            _extract_archive(
                archive,
                extracted,
                artifact,
                progress=progress,
                canceled=canceled,
            )
            try:
                staged_spec = resolve_runtime_bundle(
                    extracted,
                    platform=artifact.platform,
                    require_engine=False,
                )
            except BundleError as exc:
                raise RuntimeInstallError(f"downloaded runtime bundle is invalid: {exc}") from exc
            receipt = staged_spec.root / RUNTIME_RECEIPT_FILENAME
            if receipt.exists():
                raise RuntimeInstallError(
                    f"runtime archive may not provide the installer-owned {RUNTIME_RECEIPT_FILENAME}"
                )
            receipt.write_text(f"{artifact.sha256}\n", encoding="ascii")
            _build_trt_engine(
                staged_spec,
                staged_spec.audio2face_model,
                "audio2face",
                "Audio2Face",
                progress_value=0.87,
                progress=progress,
                canceled=canceled,
            )
            _build_trt_engine(
                staged_spec,
                staged_spec.audio2emotion_model,
                "audio2emotion",
                "Audio2Emotion",
                progress_value=0.93,
                progress=progress,
                canceled=canceled,
            )
            try:
                resolve_runtime_bundle(
                    extracted,
                    platform=artifact.platform,
                    require_engine=True,
                )
            except BundleError as exc:
                raise RuntimeInstallError(f"optimized runtime bundle is invalid: {exc}") from exc

            # Stale-backup deletion can be large, so it deliberately happens
            # before acquiring Blender's process-local cancellation gate.
            activation = _prepare_activation(extracted, root, artifact.platform)
            _emit(progress, "activating", 0.99, "Activating managed runtime")
            with activation_lock:
                _check_cancelled(canceled)
                backup = _atomic_activate(activation)

            # The new runtime is active.  Backup deletion is intentionally
            # outside the cancellation gate and can never fail the install.
            _cleanup_activation_backup(backup)
            result = resolve_runtime_bundle(
                root,
                platform=artifact.platform,
                require_engine=True,
            )
            validate_install_receipt(result, artifact)
            _emit(progress, "complete", 1.0, "Managed Audio2Face runtime is ready")
            return result
    except RuntimeInstallError:
        raise
    except Exception as exc:
        raise RuntimeInstallError(f"managed-runtime installation failed: {exc}") from exc
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


__all__ = [
    "InstallProgress",
    "RuntimeInstallCancelled",
    "RuntimeInstallError",
    "INSTALL_LOCK_FILENAME",
    "INSTALL_LOCK_TIMEOUT_SECONDS",
    "RUNTIME_RECEIPT_FILENAME",
    "install_managed_runtime",
    "validate_install_receipt",
]
