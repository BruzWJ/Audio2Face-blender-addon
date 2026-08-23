"""Build the two user-selected NVIDIA models for the local GPU.

This module performs blocking work without importing :mod:`bpy`.  The Blender
controller runs :func:`optimize_models` on one background thread and consumes
its progress records on the main thread.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import string
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .path_contract import require_unaliased_path
from .runtime_bundle import RuntimeModelSpec
from .strict_json import duplicate_key_hook, invalid_constant_hook


class ModelOptimizationError(RuntimeError):
    """Raised when TensorRT cannot build or activate the selected models."""


class ModelOptimizationCancelled(ModelOptimizationError):
    """Raised after the user cancels model optimization."""


@dataclass(frozen=True, slots=True)
class OptimizationProgress:
    progress: float
    message: str


ProgressCallback = Callable[[OptimizationProgress], None]


@dataclass(frozen=True, slots=True)
class _EngineCandidate:
    temporary: Path
    destination: Path


def _emit(callback: ProgressCallback, progress: float, message: str) -> None:
    value = float(progress)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ModelOptimizationError("optimization progress is outside [0, 1]")
    callback(OptimizationProgress(value, message))


def _check_cancelled(canceled: threading.Event) -> None:
    if canceled.is_set():
        raise ModelOptimizationCancelled("model optimization was canceled")


def _trt_build_plan(
    spec: RuntimeModelSpec,
    model: Path,
    model_label: str,
    *,
    onnx_path: Path,
    output_path: Path,
) -> tuple[list[str], str]:
    """Derive one TensorRT command from the selected model's own metadata."""

    model_directory = model.parent
    source_onnx_path = model_directory / "network.onnx"
    info_path = model_directory / "trt_info.json"
    if not source_onnx_path.is_file() or not info_path.is_file():
        raise ModelOptimizationError(
            f"selected {model_label} model is missing network.onnx or trt_info.json"
        )
    try:
        text = info_path.read_text(encoding="utf-8")
        document = json.loads(
            text,
            object_pairs_hook=duplicate_key_hook(
                ModelOptimizationError,
                "selected model trt_info.json",
            ),
            parse_constant=invalid_constant_hook(
                ModelOptimizationError,
                "selected model trt_info.json",
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelOptimizationError(
            f"cannot read selected {model_label} TRT settings: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise ModelOptimizationError(
            f"selected {model_label} trt_info.json must be an object"
        )
    expected_fields = {"estimated_trt_builder_time", "trt_build_param", "defaults"}
    if set(document) != expected_fields:
        raise ModelOptimizationError(
            f"selected {model_label} TRT settings must contain exactly "
            f"{sorted(expected_fields)}"
        )
    build_parameters = document["trt_build_param"]
    defaults = document["defaults"]
    if not isinstance(build_parameters, dict) or not isinstance(defaults, dict):
        raise ModelOptimizationError(
            f"selected {model_label} TRT settings have an invalid structure"
        )

    parameter_templates: list[str] = []
    for group, parameters in build_parameters.items():
        if not isinstance(group, str) or not group:
            raise ModelOptimizationError(
                f"selected {model_label} TRT parameter group names must be non-empty strings"
            )
        if not isinstance(parameters, list) or not parameters or not all(
            isinstance(item, str)
            and item.startswith("--")
            and item != "--"
            and "\x00" not in item
            and "\r" not in item
            and "\n" not in item
            for item in parameters
        ):
            raise ModelOptimizationError(
                f"selected {model_label} TRT settings group {group!r} is invalid"
            )
        parameter_templates.extend(parameters)
    if not parameter_templates or len(set(parameter_templates)) != len(parameter_templates):
        raise ModelOptimizationError(
            f"selected {model_label} TRT parameter templates must be non-empty and unique"
        )

    format_values: dict[str, int | float | str] = {}
    for name, value in defaults.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float, str))
        ):
            raise ModelOptimizationError(
                f"selected {model_label} TRT defaults are invalid"
            )
        format_values[name] = value

    # The worker owns one audio track; every non-batch dimension remains model-owned.
    for name in ("OPT_BATCH_SIZE", "MAX_BATCH_SIZE"):
        if name in format_values:
            format_values[name] = 1

    formatter = string.Formatter()
    placeholders: set[str] = set()
    for template in parameter_templates:
        try:
            parsed_template = tuple(formatter.parse(template))
        except ValueError as exc:
            raise ModelOptimizationError(
                f"selected {model_label} TRT parameter template is invalid: {exc}"
            ) from exc
        for _literal, field_name, format_spec, conversion in parsed_template:
            if field_name is None:
                continue
            if (
                re.fullmatch(r"[A-Z][A-Z0-9_]*", field_name) is None
                or format_spec
                or conversion is not None
            ):
                raise ModelOptimizationError(
                    f"selected {model_label} TRT placeholder {field_name!r} is invalid"
                )
            placeholders.add(field_name)
    if placeholders != set(format_values):
        raise ModelOptimizationError(
            f"selected {model_label} TRT placeholders and defaults must match exactly"
        )

    try:
        model_parameters = [
            template.format(**format_values) for template in parameter_templates
        ]
    except (KeyError, ValueError) as exc:
        raise ModelOptimizationError(
            f"selected {model_label} TRT settings cannot be formatted: {exc}"
        ) from exc

    runner_options = {"--onnx", "--saveengine", "--device", "--skipinference"}
    for parameter in model_parameters:
        option = parameter.split("=", 1)[0].split(maxsplit=1)[0].casefold()
        if option in runner_options:
            raise ModelOptimizationError(
                f"selected {model_label} TRT settings define runner-owned option {option}"
            )

    command = [
        str(spec.runtime.trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        "--device=0",
        "--skipInference",
        *model_parameters,
    ]
    estimate = document["estimated_trt_builder_time"]
    if (
        isinstance(estimate, bool)
        or not isinstance(estimate, (int, float))
        or not math.isfinite(estimate)
        or estimate <= 0
    ):
        raise ModelOptimizationError(
            f"selected {model_label} TRT build estimate must be a positive number"
        )
    message = (
        f"Optimizing the {model_label} model for CUDA device 0 "
        f"(NVIDIA estimate: about {int(round(estimate))} seconds)"
    )
    return command, message


def _trtexec_failure_message(
    model_label: str,
    returncode: int,
    log_path: Path,
    *,
    output_ready: bool,
) -> str:
    """Point Blender at the complete log without exposing raw process bytes."""

    if returncode != 0:
        summary = (
            f"TensorRT {model_label} optimization failed "
            f"(exit code {returncode})."
        )
    elif not output_ready:
        summary = (
            f"TensorRT {model_label} optimization did not produce a usable engine."
        )
    else:
        summary = f"TensorRT {model_label} optimization failed."
    return f"{summary}\nComplete TensorRT log: {log_path.name}"


def _windows_staging_directory(system_root: str) -> Path:
    """Create the one Windows TensorRT staging path used by this integration."""

    base = Path(system_root) / "Temp"
    if not str(base).isascii():
        raise ModelOptimizationError(
            "TensorRT model optimization requires an ASCII Windows system temp path"
        )
    try:
        directory = Path(tempfile.mkdtemp(prefix="audio2face-trt-", dir=base))
    except OSError as exc:
        raise ModelOptimizationError(
            f"cannot create the TensorRT staging directory {base}: {exc}"
        ) from exc
    return directory


def _build_engine(
    spec: RuntimeModelSpec,
    model: Path,
    model_id: str,
    model_label: str,
    *,
    progress_value: float,
    progress: ProgressCallback,
    canceled: threading.Event,
    log_directory: Path,
) -> _EngineCandidate:
    destination = model.parent / "network.trt"
    try:
        descriptor, candidate_name = tempfile.mkstemp(
            prefix=".audio2face-",
            suffix=".network.trt",
            dir=model.parent,
        )
        os.close(descriptor)
        candidate = Path(candidate_name)
        candidate.unlink()
    except OSError as exc:
        raise ModelOptimizationError(
            f"cannot create the {model_label} engine beside model.json: {exc}"
        ) from exc

    log_path = log_directory / f"trtexec-{model_id}.log"
    windows_staging_directory: Path | None = None
    candidate_ready = False
    try:
        command_onnx = model.parent / "network.onnx"
        command_output = candidate
        if spec.runtime.platform == "windows-x64":
            system_root = spec.runtime.env["SystemRoot"]
            windows_staging_directory = _windows_staging_directory(system_root)
            command_onnx = windows_staging_directory / "network.onnx"
            command_output = windows_staging_directory / "network.trt"
            shutil.copyfile(model.parent / "network.onnx", command_onnx)

        command, message = _trt_build_plan(
            spec,
            model,
            model_label,
            onnx_path=command_onnx,
            output_path=command_output,
        )
        _emit(progress, progress_value, message)
        child_environment = dict(spec.runtime.env)
        if windows_staging_directory is not None:
            child_environment["TEMP"] = str(windows_staging_directory)
            child_environment["TMP"] = str(windows_staging_directory)
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=str(spec.runtime.root),
                env=child_environment,
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
                raise ModelOptimizationCancelled("model optimization was canceled")
            returncode = process.returncode

        output_ready = (
            command_output.is_file() and command_output.stat().st_size > 0
        )
        if returncode != 0 or not output_ready:
            raise ModelOptimizationError(
                _trtexec_failure_message(
                    model_label,
                    returncode,
                    log_path,
                    output_ready=output_ready,
                )
            )
        if command_output != candidate:
            shutil.copyfile(command_output, candidate)
        candidate_ready = True
        return _EngineCandidate(candidate, destination)
    finally:
        if not candidate_ready:
            candidate.unlink(missing_ok=True)
        if windows_staging_directory is not None:
            shutil.rmtree(windows_staging_directory)


def _backup_path(destination: Path) -> Path:
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=".audio2face-backup-",
            suffix=".network.trt",
            dir=destination.parent,
        )
        os.close(descriptor)
        backup = Path(name)
        backup.unlink()
    except OSError as exc:
        raise ModelOptimizationError(
            f"cannot prepare an engine rollback path beside {destination}: {exc}"
        ) from exc
    return backup


def _engine_destination_exists(destination: Path) -> bool:
    require_unaliased_path(
        destination.parent,
        description="TensorRT engine directory",
        error_type=ModelOptimizationError,
    )
    try:
        details = destination.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ModelOptimizationError(
            f"TensorRT engine destination is inaccessible: {destination}"
        ) from exc
    is_reparse_point = bool(
        os.name == "nt"
        and details.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )
    if stat.S_ISLNK(details.st_mode) or is_reparse_point:
        raise ModelOptimizationError(
            f"TensorRT engine destination must not use a filesystem alias: {destination}"
        )
    if not stat.S_ISREG(details.st_mode):
        raise ModelOptimizationError(
            f"TensorRT engine destination must be a regular file: {destination}"
        )
    return True


def _activate_engines(candidates: tuple[_EngineCandidate, _EngineCandidate]) -> None:
    if candidates[0].destination == candidates[1].destination:
        raise ModelOptimizationError("Audio2Face and Audio2Emotion engine paths collide")
    temporary_paths = {candidate.temporary for candidate in candidates}
    destination_paths = {candidate.destination for candidate in candidates}
    if len(temporary_paths) != len(candidates):
        raise ModelOptimizationError("TensorRT engine candidate paths collide")
    if temporary_paths & destination_paths:
        raise ModelOptimizationError(
            "TensorRT engine candidates collide with active engine paths"
        )
    existing_destinations: set[Path] = set()
    for candidate in candidates:
        temporary = require_unaliased_path(
            candidate.temporary,
            description="TensorRT engine candidate",
            error_type=ModelOptimizationError,
        )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise ModelOptimizationError(
                f"TensorRT engine candidate is missing or empty: {temporary}"
            )
        if _engine_destination_exists(candidate.destination):
            existing_destinations.add(candidate.destination)
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for candidate in candidates:
            if candidate.destination in existing_destinations:
                backup = _backup_path(candidate.destination)
                os.replace(candidate.destination, backup)
                backups.append((candidate.destination, backup))
        for candidate in candidates:
            os.replace(candidate.temporary, candidate.destination)
            installed.append(candidate.destination)
    except Exception:
        for destination in reversed(installed):
            destination.unlink(missing_ok=True)
        for destination, backup in reversed(backups):
            if destination.exists():
                destination.unlink()
            os.replace(backup, destination)
        raise
    for _destination, backup in backups:
        backup.unlink(missing_ok=True)


def _cleanup_candidates(candidates: tuple[_EngineCandidate, ...]) -> None:
    for candidate in candidates:
        candidate.temporary.unlink(missing_ok=True)


def optimize_models(
    spec: RuntimeModelSpec,
    *,
    log_directory: str | Path,
    progress: ProgressCallback,
    canceled: threading.Event,
    commit_lock: Any,
) -> None:
    """Build and atomically activate both selected TensorRT engines."""

    logs = require_unaliased_path(
        log_directory,
        description="Audio2Face log directory",
        error_type=ModelOptimizationError,
    )
    if not logs.is_dir():
        raise ModelOptimizationError(
            f"Audio2Face log directory is not a directory: {logs}"
        )

    candidates: tuple[_EngineCandidate, ...] = ()
    try:
        _check_cancelled(canceled)
        face = _build_engine(
            spec,
            spec.audio2face_model,
            "audio2face",
            "Audio2Face",
            progress_value=0.0,
            progress=progress,
            canceled=canceled,
            log_directory=logs,
        )
        try:
            emotion = _build_engine(
                spec,
                spec.audio2emotion_model,
                "audio2emotion",
                "Audio2Emotion",
                progress_value=0.5,
                progress=progress,
                canceled=canceled,
                log_directory=logs,
            )
        except Exception:
            _cleanup_candidates((face,))
            raise
        candidates = (face, emotion)
        _emit(progress, 0.99, "Activating both optimized models")
        with commit_lock:
            _check_cancelled(canceled)
            _activate_engines((face, emotion))
        _emit(progress, 1.0, "Both NVIDIA models are optimized")
    except OSError as exc:
        raise ModelOptimizationError(f"model optimization failed: {exc}") from exc
    finally:
        _cleanup_candidates(candidates)
