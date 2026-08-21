from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from audio2face import model_optimize
from audio2face.model_optimize import (
    ModelOptimizationCancelled,
    ModelOptimizationError,
)
from audio2face.runtime_bundle import RuntimeModelSpec, RuntimeBundle


def _make_spec(
    root: Path,
    trt_info: dict[str, object] | None = None,
    *,
    platform: str = "linux-x64",
) -> RuntimeModelSpec:
    runtime_root = root / "runtime"
    binary_directory = runtime_root / "bin"
    binary_directory.mkdir(parents=True)
    executable = binary_directory / (
        "audio2face_worker.exe"
        if platform == "windows-x64"
        else "audio2face_worker"
    )
    trtexec = binary_directory / (
        "trtexec.exe" if platform == "windows-x64" else "trtexec"
    )
    executable.write_bytes(b"worker")
    trtexec.write_bytes(b"trtexec")
    environment = {"PATH": str(binary_directory)}
    if platform == "linux-x64":
        library_directory = runtime_root / "lib"
        library_directory.mkdir()
        environment["LD_LIBRARY_PATH"] = str(library_directory)
    else:
        environment["SystemRoot"] = str(root / "Windows")
    bundle = RuntimeBundle(
        platform=platform,
        root=runtime_root,
        executable=executable,
        trtexec=trtexec,
        env=environment,
    )
    document = trt_info
    if document is None:
        document = {
            "estimated_trt_builder_time": 150,
            "trt_build_param": {
                "precision": ["--fp16"],
                "shapes": [
                    "--minShapes=input:{MIN_BATCH_SIZE}x4",
                    "--optShapes=input:{OPT_BATCH_SIZE}x4",
                    "--maxShapes=input:{MAX_BATCH_SIZE}x4",
                ],
            },
            "defaults": {
                "MIN_BATCH_SIZE": 2,
                "OPT_BATCH_SIZE": 4,
                "MAX_BATCH_SIZE": 8,
            },
        }
    models: list[Path] = []
    for name in ("face", "emotion"):
        directory = root / name
        directory.mkdir()
        model = directory / "model.json"
        model.write_text('{"networkPath":"network.trt"}', encoding="utf-8")
        (directory / "network.onnx").write_bytes(b"onnx")
        (directory / "trt_info.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        models.append(model)
    return RuntimeModelSpec(bundle, models[0], models[1])


def test_trt_command_comes_from_model_metadata_and_one_owned_device(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path)
    onnx = spec.audio2face_model.with_name("network.onnx")
    output = spec.audio2face_model.with_name("candidate.trt")

    command, message = model_optimize._trt_build_plan(
        spec,
        spec.audio2face_model,
        "Audio2Face",
        onnx_path=onnx,
        output_path=output,
    )

    assert command == [
        str(spec.runtime.trtexec),
        f"--onnx={onnx}",
        f"--saveEngine={output}",
        "--device=0",
        "--fp16",
        "--minShapes=input:2x4",
        "--optShapes=input:4x4",
        "--maxShapes=input:8x4",
    ]
    assert message == (
        "Optimizing the Audio2Face model for CUDA device 0 "
        "(NVIDIA estimate: about 150 seconds)"
    )


@pytest.mark.parametrize("owned", ["--onnx=x", "--saveEngine=x", "--DEVICE=1"])
def test_model_metadata_cannot_define_runner_owned_options(
    tmp_path: Path,
    owned: str,
) -> None:
    spec = _make_spec(
        tmp_path,
        {
            "estimated_trt_builder_time": 1,
            "trt_build_param": {"invalid": [owned]},
            "defaults": {},
        },
    )
    with pytest.raises(ModelOptimizationError, match="runner-owned"):
        model_optimize._trt_build_plan(
            spec,
            spec.audio2face_model,
            "Audio2Face",
            onnx_path=spec.audio2face_model.with_name("network.onnx"),
            output_path=spec.audio2face_model.with_name("candidate.trt"),
        )


def test_model_metadata_rejects_duplicate_fields(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    spec.audio2face_model.with_name("trt_info.json").write_text(
        '{"estimated_trt_builder_time":1,"trt_build_param":{"x":["--fp16"]},'
        '"defaults":{},"defaults":{}}',
        encoding="utf-8",
    )
    with pytest.raises(ModelOptimizationError, match="duplicate field 'defaults'"):
        model_optimize._trt_build_plan(
            spec,
            spec.audio2face_model,
            "Audio2Face",
            onnx_path=spec.audio2face_model.with_name("network.onnx"),
            output_path=spec.audio2face_model.with_name("candidate.trt"),
        )


def test_empty_model_metadata_is_not_replaced_with_test_defaults(
    tmp_path: Path,
) -> None:
    spec = _make_spec(tmp_path, {})
    with pytest.raises(ModelOptimizationError, match="must contain exactly"):
        model_optimize._trt_build_plan(
            spec,
            spec.audio2face_model,
            "Audio2Face",
            onnx_path=spec.audio2face_model.with_name("network.onnx"),
            output_path=spec.audio2face_model.with_name("candidate.trt"),
        )


def test_build_engine_uses_bundled_trtexec_environment_and_writable_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    existing = spec.audio2face_model.with_name("network.trt")
    existing.write_bytes(b"old")
    calls: list[tuple[list[str], dict[str, object]]] = []

    class Finished:
        returncode = 0

        def poll(self) -> int:
            return 0

    def run(command: list[str], **kwargs: object) -> Finished:
        calls.append((command, kwargs))
        output = next(
            item.partition("=")[2]
            for item in command
            if item.startswith("--saveEngine=")
        )
        Path(output).write_bytes(b"new")
        kwargs["stdout"].write(b"complete\n")  # type: ignore[union-attr]
        return Finished()

    monkeypatch.setattr(model_optimize.subprocess, "Popen", run)
    candidate = model_optimize._build_engine(
        spec,
        spec.audio2face_model,
        "audio2face",
        "Audio2Face",
        progress_value=0.0,
        progress=lambda _event: None,
        canceled=threading.Event(),
        log_directory=logs,
    )

    assert candidate.temporary.read_bytes() == b"new"
    assert candidate.destination == existing
    assert existing.read_bytes() == b"old"
    assert calls[0][1]["cwd"] == str(spec.runtime.root)
    assert calls[0][1]["env"] == dict(spec.runtime.env)
    assert (logs / "trtexec-audio2face.log").read_text() == "complete\n"
    model_optimize._cleanup_candidates((candidate,))


def test_windows_build_engine_always_uses_the_system_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path, platform="windows-x64")
    logs = tmp_path / "logs"
    logs.mkdir()
    system_root = tmp_path / "Windows"
    (system_root / "Temp").mkdir(parents=True)
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    class Finished:
        returncode = 0

        def poll(self) -> int:
            return 0

    def run(command: list[str], **kwargs: object) -> Finished:
        commands.append(command)
        environments.append(kwargs["env"])  # type: ignore[arg-type]
        output = next(
            item.partition("=")[2]
            for item in command
            if item.startswith("--saveEngine=")
        )
        Path(output).write_bytes(b"engine")
        return Finished()

    monkeypatch.setattr(model_optimize.subprocess, "Popen", run)
    candidate = model_optimize._build_engine(
        spec,
        spec.audio2face_model,
        "audio2face",
        "Audio2Face",
        progress_value=0.0,
        progress=lambda _event: None,
        canceled=threading.Event(),
        log_directory=logs,
    )

    onnx = Path(
        next(
            item.partition("=")[2]
            for item in commands[0]
            if item.startswith("--onnx=")
        )
    )
    engine = Path(
        next(
            item.partition("=")[2]
            for item in commands[0]
            if item.startswith("--saveEngine=")
        )
    )
    assert onnx.parent.parent == system_root / "Temp"
    assert engine.parent == onnx.parent
    assert candidate.temporary.read_bytes() == b"engine"
    assert not onnx.parent.exists()
    assert environments == [
        {
            "PATH": str(spec.runtime.executable.parent),
            "SystemRoot": str(system_root),
            "TEMP": str(onnx.parent),
            "TMP": str(onnx.parent),
        }
    ]
    model_optimize._cleanup_candidates((candidate,))


def test_build_engine_cancellation_terminates_trtexec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    canceled = threading.Event()
    canceled.set()

    class Running:
        returncode = -15
        terminated = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float) -> int:
            assert timeout == 2.0
            return self.returncode

    process = Running()
    monkeypatch.setattr(model_optimize.subprocess, "Popen", lambda *_a, **_k: process)

    with pytest.raises(ModelOptimizationCancelled):
        model_optimize._build_engine(
            spec,
            spec.audio2face_model,
            "audio2face",
            "Audio2Face",
            progress_value=0.0,
            progress=lambda _event: None,
            canceled=canceled,
            log_directory=logs,
        )
    assert process.terminated
    assert not list(spec.audio2face_model.parent.glob(".audio2face-*.network.trt"))


def test_optimize_models_activates_both_engines_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    for model in (spec.audio2face_model, spec.audio2emotion_model):
        model.with_name("network.trt").write_bytes(b"old")

    def build(
        _spec: RuntimeModelSpec,
        model: Path,
        model_id: str,
        _label: str,
        **_kwargs: object,
    ) -> model_optimize._EngineCandidate:
        candidate = model.parent / f".{model_id}.candidate.trt"
        candidate.write_bytes(model_id.encode())
        return model_optimize._EngineCandidate(candidate, model.with_name("network.trt"))

    monkeypatch.setattr(model_optimize, "_build_engine", build)
    progress: list[model_optimize.OptimizationProgress] = []
    logs = tmp_path / "logs"
    logs.mkdir()
    model_optimize.optimize_models(
        spec,
        log_directory=logs,
        progress=progress.append,
        canceled=threading.Event(),
        commit_lock=threading.Lock(),
    )

    assert spec.audio2face_model.with_name("network.trt").read_bytes() == b"audio2face"
    assert spec.audio2emotion_model.with_name("network.trt").read_bytes() == b"audio2emotion"
    assert progress[-1].progress == 1.0
    assert progress[-1].message == "Both NVIDIA models are optimized"
    assert not list(tmp_path.rglob(".audio2face-backup-*"))


def test_engine_activation_rolls_back_both_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _make_spec(tmp_path)
    candidates: list[model_optimize._EngineCandidate] = []
    for index, model in enumerate((spec.audio2face_model, spec.audio2emotion_model)):
        destination = model.with_name("network.trt")
        destination.write_bytes(f"old-{index}".encode())
        temporary = model.parent / f"candidate-{index}.trt"
        temporary.write_bytes(f"new-{index}".encode())
        candidates.append(model_optimize._EngineCandidate(temporary, destination))

    real_replace = os.replace

    def fail_second_candidate(source: object, destination: object) -> None:
        if Path(source) == candidates[1].temporary:
            raise OSError("second activation failed")
        real_replace(source, destination)

    monkeypatch.setattr(model_optimize.os, "replace", fail_second_candidate)
    with pytest.raises(OSError, match="second activation failed"):
        model_optimize._activate_engines((candidates[0], candidates[1]))

    assert candidates[0].destination.read_bytes() == b"old-0"
    assert candidates[1].destination.read_bytes() == b"old-1"
    assert not list(tmp_path.rglob(".audio2face-backup-*"))


def test_optimizer_rejects_an_aliased_log_directory(tmp_path: Path) -> None:
    spec = _make_spec(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    alias = tmp_path / "logs-alias"
    try:
        alias.symlink_to(logs, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(model_optimize.ModelOptimizationError, match="filesystem alias"):
        model_optimize.optimize_models(
            spec,
            log_directory=alias,
            progress=lambda _event: None,
            canceled=threading.Event(),
            commit_lock=threading.Lock(),
        )


@pytest.mark.parametrize("target_exists", [False, True])
def test_engine_activation_rejects_an_aliased_destination(
    tmp_path: Path,
    target_exists: bool,
) -> None:
    candidates: list[model_optimize._EngineCandidate] = []
    for index in range(2):
        directory = tmp_path / f"model-{index}"
        directory.mkdir()
        temporary = directory / f"candidate-{index}.trt"
        temporary.write_bytes(b"candidate")
        candidates.append(
            model_optimize._EngineCandidate(
                temporary,
                directory / "network.trt",
            )
        )
    outside = tmp_path / "outside.trt"
    if target_exists:
        outside.write_bytes(b"outside")
    try:
        candidates[0].destination.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable")

    with pytest.raises(model_optimize.ModelOptimizationError, match="filesystem alias"):
        model_optimize._activate_engines((candidates[0], candidates[1]))

    assert candidates[0].destination.is_symlink()
    if target_exists:
        assert outside.read_bytes() == b"outside"
    assert all(candidate.temporary.is_file() for candidate in candidates)


def test_windows_staging_uses_only_system_temp(
    tmp_path: Path,
) -> None:
    system_root = tmp_path / "Windows"
    system_temp = system_root / "Temp"
    system_temp.mkdir(parents=True)
    result = model_optimize._windows_staging_directory(str(system_root))

    assert result.parent == system_temp
    result.rmdir()


def test_windows_staging_requires_an_ascii_system_root(
    tmp_path: Path,
) -> None:
    system_temp = tmp_path / "Wíndows" / "Temp"
    system_temp.mkdir(parents=True)
    with pytest.raises(ModelOptimizationError, match="ASCII"):
        model_optimize._windows_staging_directory(str(system_temp.parent))


@pytest.mark.parametrize(
    ("document", "message"),
    (
        (
            {
                "estimated_trt_builder_time": 1,
                "trt_build_param": {"x": ["--shape={BATCH}"]},
                "defaults": {},
            },
            "placeholders and defaults",
        ),
        (
            {
                "estimated_trt_builder_time": 1,
                "trt_build_param": {"x": ["--fp16"]},
                "defaults": {"UNUSED": 1},
            },
            "placeholders and defaults",
        ),
        (
            {
                "estimated_trt_builder_time": 0,
                "trt_build_param": {"x": ["--fp16"]},
                "defaults": {},
            },
            "positive number",
        ),
        (
            {
                "estimated_trt_builder_time": 1,
                "trt_build_param": {"x": ["--shape={BATCH!r}"]},
                "defaults": {"BATCH": 1},
            },
            "placeholder",
        ),
    ),
)
def test_trt_metadata_rejects_unused_or_noncanonical_fields(
    tmp_path: Path,
    document: dict[str, object],
    message: str,
) -> None:
    spec = _make_spec(tmp_path, document)
    with pytest.raises(ModelOptimizationError, match=message):
        model_optimize._trt_build_plan(
            spec,
            spec.audio2face_model,
            "Audio2Face",
            onnx_path=spec.audio2face_model.with_name("network.onnx"),
            output_path=spec.audio2face_model.with_name("candidate.trt"),
        )
