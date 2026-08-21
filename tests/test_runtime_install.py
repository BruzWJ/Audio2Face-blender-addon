from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import struct
import subprocess
import sys
import threading
import time
import warnings
import zipfile
from pathlib import Path

import pytest

from audio2face.runtime_bundle import BundleLaunchSpec, RUNTIME_SCHEMA
from audio2face.runtime_catalog import RuntimeArtifact
from audio2face import runtime_install
from audio2face.runtime_install import (
    RUNTIME_RECEIPT_FILENAME,
    RuntimeInstallError,
    validate_install_receipt,
)


class _Response(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        content_length: str | None = None,
        final_url: str = "https://downloads.example.test/runtime.zip",
    ) -> None:
        super().__init__(payload)
        self.headers = (
            {} if content_length is None else {"Content-Length": content_length}
        )
        self._final_url = final_url

    def geturl(self) -> str:
        return self._final_url


def _artifact(
    payload: bytes,
    *,
    size: int | None = None,
    unpacked_size: int = 1,
    sha256: str | None = None,
    platform: str = "linux-x64",
) -> RuntimeArtifact:
    return RuntimeArtifact(
        platform=platform,
        url="https://downloads.example.test/runtime.zip",
        sha256=sha256 or hashlib.sha256(payload).hexdigest(),
        size=len(payload) if size is None else size,
        unpacked_size=unpacked_size,
    )


def _download(
    tmp_path: Path,
    artifact: RuntimeArtifact,
    response: _Response,
) -> Path:
    destination = tmp_path / "runtime.zip"
    runtime_install._download_archive(
        artifact,
        destination,
        progress=lambda _event: None,
        canceled=threading.Event(),
        open_url=lambda _request, timeout: response,
    )
    return destination


def test_download_requires_exact_size_and_checksum(tmp_path: Path) -> None:
    payload = b"pinned runtime archive"
    artifact = _artifact(payload)

    destination = _download(
        tmp_path,
        artifact,
        _Response(payload, content_length=str(len(payload))),
    )

    assert destination.read_bytes() == payload


@pytest.mark.parametrize(
    ("artifact_factory", "response_factory", "match"),
    [
        (
            lambda payload: _artifact(payload, size=len(payload) + 1),
            lambda payload: _Response(payload),
            "expected",
        ),
        (
            lambda payload: _artifact(payload, size=len(payload) - 1),
            lambda payload: _Response(payload),
            "exceeded",
        ),
        (
            lambda payload: _artifact(payload),
            lambda payload: _Response(payload, content_length=str(len(payload) + 1)),
            "pinned release catalog",
        ),
        (
            lambda payload: _artifact(payload),
            lambda payload: _Response(payload, content_length="not-an-integer"),
            "invalid Content-Length",
        ),
        (
            lambda payload: _artifact(payload, sha256="0" * 64),
            lambda payload: _Response(payload),
            "SHA-256",
        ),
    ],
)
def test_download_fails_closed_on_size_or_checksum_mismatch(
    tmp_path: Path, artifact_factory: object, response_factory: object, match: str
) -> None:
    payload = b"runtime payload"
    artifact = artifact_factory(payload)  # type: ignore[operator]
    response = response_factory(payload)  # type: ignore[operator]

    with pytest.raises(RuntimeInstallError, match=match):
        _download(tmp_path, artifact, response)


def test_download_rejects_https_to_http_redirect(tmp_path: Path) -> None:
    payload = b"runtime payload"
    with pytest.raises(RuntimeInstallError, match="HTTPS|redirect"):
        _download(
            tmp_path,
            _artifact(payload),
            _Response(payload, final_url="http://downloads.example.test/runtime.zip"),
        )


def _zip_bytes(entries: list[tuple[str, bytes, int]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as package:
        for name, data, mode in entries:
            member = zipfile.ZipInfo(name)
            member.create_system = 3
            member.external_attr = mode << 16
            package.writestr(member, data)
    return output.getvalue()


def _extract_payload(
    tmp_path: Path,
    payload: bytes,
    *,
    unpacked_size: int,
    platform: str = "linux-x64",
) -> Path:
    archive = tmp_path / "archive.zip"
    destination = tmp_path / "extracted"
    archive.write_bytes(payload)
    destination.mkdir()
    runtime_install._extract_archive(
        archive,
        destination,
        _artifact(
            payload,
            unpacked_size=unpacked_size,
            platform=platform,
        ),
        progress=lambda _event: None,
        canceled=threading.Event(),
    )
    return destination


def test_safe_zip_extraction_preserves_file_contents_and_executable_mode(
    tmp_path: Path,
) -> None:
    data = b"worker"
    payload = _zip_bytes(
        [
            ("runtime/", b"", stat.S_IFDIR | 0o755),
            ("runtime/linux-x64/", b"", stat.S_IFDIR | 0o755),
            (
                "runtime/linux-x64/bin/audio2face_worker",
                data,
                stat.S_IFREG | 0o755,
            ),
        ]
    )

    destination = _extract_payload(tmp_path, payload, unpacked_size=len(data))
    worker = destination / "runtime/linux-x64/bin/audio2face_worker"
    assert worker.read_bytes() == data
    if os.name != "nt":
        assert stat.S_IMODE(worker.stat().st_mode) == 0o755


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute/path",
        "C:/windows/path",
        "runtime/linux-x64/../../escape",
        "runtime\\linux-x64\\escape",
        "runtime//linux-x64/bin/worker",
        "runtime/./linux-x64/bin/worker",
        "runtime/linux-x64/bin/worker:stream",
        "runtime/linux-x64:stream/bin/worker",
    ],
)
def test_zip_extraction_rejects_unsafe_member_paths(
    tmp_path: Path, name: str
) -> None:
    payload = _zip_bytes([(name, b"bad", stat.S_IFREG | 0o644)])
    with pytest.raises(RuntimeInstallError, match="unsafe path"):
        _extract_payload(tmp_path, payload, unpacked_size=3)
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("name", ["runtime//", "runtime/./"])
def test_zip_extraction_rejects_noncanonical_directory_paths(
    tmp_path: Path, name: str
) -> None:
    payload = _zip_bytes([(name, b"", stat.S_IFDIR | 0o755)])
    with pytest.raises(RuntimeInstallError, match="unsafe path"):
        _extract_payload(tmp_path, payload, unpacked_size=0)


def test_zip_extraction_rejects_symbolic_links(tmp_path: Path) -> None:
    payload = _zip_bytes(
        [
            (
                "runtime/linux-x64/bin/worker-link",
                b"../../outside",
                stat.S_IFLNK | 0o777,
            )
        ]
    )
    with pytest.raises(RuntimeInstallError, match="symbolic links"):
        _extract_payload(tmp_path, payload, unpacked_size=len(b"../../outside"))


def test_zip_extraction_rejects_duplicate_members(tmp_path: Path) -> None:
    entries = [
        ("runtime/linux-x64/bin/worker", b"one", stat.S_IFREG | 0o755),
        ("runtime/linux-x64/bin/worker", b"two", stat.S_IFREG | 0o755),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        payload = _zip_bytes(entries)
    with pytest.raises(RuntimeInstallError, match="duplicate path"):
        _extract_payload(tmp_path, payload, unpacked_size=6)


def test_windows_zip_rejects_case_colliding_members(tmp_path: Path) -> None:
    payload = _zip_bytes(
        [
            ("runtime/windows-x64/Bin/worker.exe", b"one", stat.S_IFREG | 0o755),
            ("runtime/windows-x64/bin/worker.exe", b"two", stat.S_IFREG | 0o755),
        ]
    )
    with pytest.raises(RuntimeInstallError, match="duplicate path|case-colliding"):
        _extract_payload(
            tmp_path,
            payload,
            unpacked_size=6,
            platform="windows-x64",
        )


def _launch_spec(tmp_path: Path, trt_info: object) -> BundleLaunchSpec:
    for model_name in ("audio2face", "audio2emotion"):
        model_directory = tmp_path / "models" / model_name
        model_directory.mkdir(parents=True)
        (model_directory / "model.json").write_text(
            '{"networkPath":"network.trt"}', encoding="utf-8"
        )
        (model_directory / "network.onnx").write_bytes(b"onnx")
        (model_directory / "trt_info.json").write_text(
            json.dumps(trt_info), encoding="utf-8"
        )
    return BundleLaunchSpec(
        platform="linux-x64",
        root=tmp_path,
        executable=tmp_path / "bin/audio2face_worker",
        trtexec=tmp_path / "bin/trtexec",
        env={"PATH": "/usr/bin"},
        audio2face_model=tmp_path / "models/audio2face/model.json",
        audio2emotion_model=tmp_path / "models/audio2emotion/model.json",
    )


def test_trt_build_plan_is_shell_free_pinned_and_forces_batch_one(tmp_path: Path) -> None:
    spec = _launch_spec(
        tmp_path,
        {
            "trt_build_param": {
                "precision": ["--fp16"],
                "batch": [
                    "--minShapes=input:{MIN_BATCH_SIZE}x{MIN_DURATION}x{FEATURE_SIZE}",
                    "--optShapes=input:{OPT_BATCH_SIZE}x{OPT_DURATION}x{FEATURE_SIZE}",
                    "--maxShapes=input:{MAX_BATCH_SIZE}x{MAX_DURATION}x{FEATURE_SIZE}",
                ]
            },
            "defaults": {
                "MIN_DURATION": 100,
                "OPT_DURATION": 500,
                "MAX_DURATION": 1000,
                "FEATURE_SIZE": 1024,
                "OPT_BATCH_SIZE": 8,
                "MAX_BATCH_SIZE": 16,
            },
        },
    )

    command, message = runtime_install._trt_build_plan(
        spec, spec.audio2face_model, "Audio2Face"
    )

    assert command == [
        str(spec.trtexec),
        f"--onnx={spec.audio2face_model.parent / 'network.onnx'}",
        f"--saveEngine={spec.audio2face_model.parent / 'network.trt'}",
        "--device=0",
        "--fp16",
        "--minShapes=input:1x100x1024",
        "--optShapes=input:1x500x1024",
        "--maxShapes=input:1x1000x1024",
    ]
    assert isinstance(command, list)
    assert message == "Optimizing the Audio2Face model for this GPU"


@pytest.mark.parametrize(
    ("trt_info", "match"),
    [
        ([], "must be an object"),
        ({"trt_build_param": [], "defaults": {}}, "invalid structure"),
        ({"trt_build_param": {"batch": "--fp16"}, "defaults": {}}, "group 'batch'"),
        ({"trt_build_param": {"batch": ["fp16"]}, "defaults": {}}, "group 'batch'"),
        (
            {"trt_build_param": {"batch": ["--x={MISSING}"]}, "defaults": {}},
            "cannot be formatted",
        ),
        (
            {"trt_build_param": {"batch": []}, "defaults": {"FLAG": True}},
            "defaults are invalid",
        ),
    ],
)
def test_trt_build_plan_rejects_invalid_model_metadata(
    tmp_path: Path, trt_info: object, match: str
) -> None:
    with pytest.raises(RuntimeInstallError, match=match):
        spec = _launch_spec(tmp_path, trt_info)
        runtime_install._trt_build_plan(
            spec, spec.audio2emotion_model, "Audio2Emotion"
        )


@pytest.mark.parametrize(
    "reserved",
    [
        "--onnx=/tmp/other.onnx",
        "--saveEngine=/tmp/other.trt",
        "--device=1",
        "--device 1",
    ],
)
def test_trt_build_plan_rejects_model_attempt_to_override_managed_paths(
    tmp_path: Path, reserved: str
) -> None:
    spec = _launch_spec(
        tmp_path,
        {"trt_build_param": {"batch": [reserved]}, "defaults": {}},
    )
    with pytest.raises(RuntimeInstallError, match="override|onnx|saveEngine|device"):
        runtime_install._trt_build_plan(
            spec, spec.audio2face_model, "Audio2Face"
        )


def test_trt_build_plan_rejects_a_formatted_device_override(tmp_path: Path) -> None:
    spec = _launch_spec(
        tmp_path,
        {
            "trt_build_param": {"batch": ["--{OPTION}=1"]},
            "defaults": {"OPTION": "device"},
        },
    )
    with pytest.raises(RuntimeInstallError, match="device"):
        runtime_install._trt_build_plan(
            spec, spec.audio2face_model, "Audio2Face"
        )


def test_failed_model_rebuild_preserves_the_existing_gpu_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _launch_spec(
        tmp_path,
        {"trt_build_param": {"batch": []}, "defaults": {}},
    )
    output = spec.audio2emotion_model.with_name("network.trt")
    output.write_bytes(b"existing engine")

    class FailedBuild:
        returncode = 1

        def poll(self) -> int:
            return 1

    monkeypatch.setattr(
        runtime_install.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FailedBuild(),
    )

    with pytest.raises(RuntimeInstallError, match="optimization failed"):
        runtime_install._build_trt_engine(
            spec,
            spec.audio2emotion_model,
            "audio2emotion",
            "Audio2Emotion",
            progress_value=0.93,
            progress=lambda _event: None,
            canceled=threading.Event(),
        )
    assert output.read_bytes() == b"existing engine"
    assert not list(output.parent.glob(".audio2face-*.network.trt"))


def test_model_builds_use_distinct_progress_and_log_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _launch_spec(
        tmp_path,
        {"trt_build_param": {"batch": []}, "defaults": {}},
    )

    class FinishedBuild:
        returncode = 0

        def poll(self) -> int:
            return 0

    def fake_popen(command: list[str], **kwargs: object) -> FinishedBuild:
        output_option = next(item for item in command if item.startswith("--saveEngine="))
        Path(output_option.partition("=")[2]).write_bytes(b"local engine")
        log = kwargs["stdout"]
        log.write("completed\n")  # type: ignore[union-attr]
        return FinishedBuild()

    monkeypatch.setattr(runtime_install.subprocess, "Popen", fake_popen)
    spec.audio2face_model.with_name("network.trt").write_bytes(b"old face engine")
    spec.audio2emotion_model.with_name("network.trt").write_bytes(b"old emotion engine")
    progress: list[object] = []
    audio2face_candidate = runtime_install._build_trt_engine(
        spec,
        spec.audio2face_model,
        "audio2face",
        "Audio2Face",
        progress_value=0.87,
        progress=progress.append,
        canceled=threading.Event(),
    )
    audio2emotion_candidate = runtime_install._build_trt_engine(
        spec,
        spec.audio2emotion_model,
        "audio2emotion",
        "Audio2Emotion",
        progress_value=0.93,
        progress=progress.append,
        canceled=threading.Event(),
    )

    assert [event.stage for event in progress] == [  # type: ignore[attr-defined]
        "building_audio2face_model",
        "building_audio2emotion_model",
    ]
    assert [event.progress for event in progress] == [0.87, 0.93]  # type: ignore[attr-defined]
    assert "Audio2Face" in progress[0].message  # type: ignore[attr-defined]
    assert "Audio2Emotion" in progress[1].message  # type: ignore[attr-defined]
    assert spec.audio2face_model.with_name("network.trt").read_bytes() == b"old face engine"
    assert spec.audio2emotion_model.with_name("network.trt").read_bytes() == b"old emotion engine"
    assert audio2face_candidate.temporary.read_bytes() == b"local engine"
    assert audio2emotion_candidate.temporary.read_bytes() == b"local engine"
    assert (spec.root / "trtexec-audio2face-install.log").read_text() == "completed\n"
    assert (spec.root / "trtexec-audio2emotion-install.log").read_text() == "completed\n"
    assert not (spec.root / "trtexec-install.log").exists()
    runtime_install._cleanup_engine_candidates(
        (audio2face_candidate, audio2emotion_candidate)
    )


def test_windows_unicode_model_path_builds_through_an_ascii_temp_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _launch_spec(
        tmp_path / "模型",
        {"trt_build_param": {"batch": []}, "defaults": {}},
    )
    spec = BundleLaunchSpec(
        platform="windows-x64",
        root=original.root,
        executable=original.executable,
        trtexec=original.trtexec,
        env=original.env,
        audio2face_model=original.audio2face_model,
        audio2emotion_model=original.audio2emotion_model,
    )
    ascii_directory = tmp_path / "ascii-trt"

    def make_ascii_directory() -> Path:
        ascii_directory.mkdir()
        return ascii_directory

    class FinishedBuild:
        returncode = 0

        def poll(self) -> int:
            return 0

    def fake_popen(command: list[str], **_kwargs: object) -> FinishedBuild:
        onnx = next(item.partition("=")[2] for item in command if item.startswith("--onnx="))
        output = next(
            item.partition("=")[2] for item in command if item.startswith("--saveEngine=")
        )
        assert onnx.isascii()
        assert output.isascii()
        assert Path(onnx).read_bytes() == b"onnx"
        Path(output).write_bytes(b"windows engine")
        return FinishedBuild()

    monkeypatch.setattr(
        runtime_install,
        "_windows_ascii_build_directory",
        make_ascii_directory,
    )
    monkeypatch.setattr(runtime_install.subprocess, "Popen", fake_popen)
    candidate = runtime_install._build_trt_engine(
        spec,
        spec.audio2face_model,
        "audio2face",
        "Audio2Face",
        progress_value=0.87,
        progress=lambda _event: None,
        canceled=threading.Event(),
    )

    assert candidate.temporary.read_bytes() == b"windows engine"
    assert candidate.destination == spec.audio2face_model.with_name("network.trt")
    assert not ascii_directory.exists()
    runtime_install._cleanup_engine_candidates((candidate,))


def _elf_x64() -> bytes:
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<H", header, 18, 62)
    return bytes(header)


def _valid_linux_runtime_zip() -> tuple[bytes, int]:
    manifest = {
        "schema": RUNTIME_SCHEMA,
        "platform": "linux-x64",
        "worker": "bin/audio2face_worker",
        "trtexec": "bin/trtexec",
        "library_directories": ["lib"],
        "licenses": ["licenses/THIRD_PARTY.txt"],
    }
    files = [
        ("runtime/", b"", stat.S_IFDIR | 0o755),
        ("runtime/linux-x64/", b"", stat.S_IFDIR | 0o755),
        ("runtime/linux-x64/bin/", b"", stat.S_IFDIR | 0o755),
        ("runtime/linux-x64/lib/", b"", stat.S_IFDIR | 0o755),
        ("runtime/linux-x64/licenses/", b"", stat.S_IFDIR | 0o755),
        (
            "runtime/linux-x64/bundle.json",
            json.dumps(manifest).encode("utf-8"),
            stat.S_IFREG | 0o644,
        ),
        (
            "runtime/linux-x64/bin/audio2face_worker",
            _elf_x64(),
            stat.S_IFREG | 0o755,
        ),
        (
            "runtime/linux-x64/bin/trtexec",
            _elf_x64(),
            stat.S_IFREG | 0o755,
        ),
        (
            "runtime/linux-x64/licenses/THIRD_PARTY.txt",
            b"notices",
            stat.S_IFREG | 0o644,
        ),
    ]
    return _zip_bytes(files), sum(len(data) for name, data, _mode in files if not name.endswith("/"))


def _external_model_directories(root: Path) -> tuple[Path, Path]:
    trt_info = {
        "trt_build_param": {"batch": ["--optShapes=input:{OPT_BATCH_SIZE}x4"]},
        "defaults": {"OPT_BATCH_SIZE": 4},
    }
    directories: list[Path] = []
    for model_name in ("audio2face", "audio2emotion"):
        directory = root / model_name
        directory.mkdir(parents=True)
        model = directory / "model.json"
        model.write_text('{"networkPath":"network.trt"}', encoding="utf-8")
        (directory / "network.onnx").write_bytes(b"onnx")
        (directory / "trt_info.json").write_text(
            json.dumps(trt_info), encoding="utf-8"
        )
        directories.append(directory)
    return directories[0], directories[1]


def test_atomic_activation_replaces_existing_runtime(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    data_root = tmp_path / "data"
    staged = staging / "runtime/linux-x64"
    active = data_root / "runtime/linux-x64"
    staged.mkdir(parents=True)
    active.mkdir(parents=True)
    (staged / "marker").write_text("new", encoding="utf-8")
    (active / "marker").write_text("old", encoding="utf-8")

    plan = runtime_install._prepare_activation(staging, data_root, "linux-x64")
    backup = runtime_install._atomic_activate(plan)

    assert (active / "marker").read_text(encoding="utf-8") == "new"
    assert backup == data_root / "runtime/.linux-x64.previous"
    assert (backup / "marker").read_text(encoding="utf-8") == "old"

    runtime_install._cleanup_activation_backup(backup)

    assert not (data_root / "runtime/.linux-x64.previous").exists()


def test_atomic_activation_rolls_back_if_new_runtime_cannot_be_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    data_root = tmp_path / "data"
    staged = staging / "runtime/linux-x64"
    active = data_root / "runtime/linux-x64"
    staged.mkdir(parents=True)
    active.mkdir(parents=True)
    (staged / "marker").write_text("new", encoding="utf-8")
    (active / "marker").write_text("old", encoding="utf-8")
    real_replace = os.replace
    calls = 0

    def fail_second_replace(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated activation failure")
        real_replace(source, destination)

    monkeypatch.setattr(runtime_install.os, "replace", fail_second_replace)
    plan = runtime_install._prepare_activation(staging, data_root, "linux-x64")
    with pytest.raises(OSError, match="simulated activation failure"):
        runtime_install._atomic_activate(plan)

    assert (active / "marker").read_text(encoding="utf-8") == "old"
    assert (staged / "marker").read_text(encoding="utf-8") == "new"
    assert not (data_root / "runtime/.linux-x64.previous").exists()


def test_combined_activation_rolls_back_both_external_engines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    data_root = tmp_path / "data"
    staged_worker = staging / "runtime/linux-x64"
    active_worker = data_root / "runtime/linux-x64"
    staged_worker.mkdir(parents=True)
    active_worker.mkdir(parents=True)
    (staged_worker / "marker").write_text("new worker", encoding="utf-8")
    (active_worker / "marker").write_text("old worker", encoding="utf-8")

    engine_directory = tmp_path / "models"
    engine_directory.mkdir()
    destinations = (
        engine_directory / "face.network.trt",
        engine_directory / "emotion.network.trt",
    )
    temporaries = (
        engine_directory / ".face.candidate.trt",
        engine_directory / ".emotion.candidate.trt",
    )
    for index, destination in enumerate(destinations):
        destination.write_bytes(f"old {index}".encode())
        temporaries[index].write_bytes(f"new {index}".encode())
    candidates = tuple(
        runtime_install._EngineCandidate(temporary, destination)
        for temporary, destination in zip(temporaries, destinations, strict=True)
    )

    real_replace = os.replace

    def fail_second_candidate(source: object, destination: object) -> None:
        if Path(source) == temporaries[1]:
            raise OSError("simulated second-engine activation failure")
        real_replace(source, destination)

    monkeypatch.setattr(runtime_install.os, "replace", fail_second_candidate)
    worker_plan = runtime_install._prepare_activation(
        staging, data_root, "linux-x64"
    )
    with pytest.raises(OSError, match="second-engine"):
        runtime_install._activate_worker_and_engines(
            worker_plan,
            candidates,  # type: ignore[arg-type]
        )

    assert (active_worker / "marker").read_text(encoding="utf-8") == "old worker"
    assert destinations[0].read_bytes() == b"old 0"
    assert destinations[1].read_bytes() == b"old 1"
    assert not list(engine_directory.glob(".audio2face-backup-*"))


def _start_lock_holder(lock_path: Path, ready_path: Path) -> subprocess.Popen[str]:
    script = """
import sys
import threading
import time
from pathlib import Path
from audio2face.runtime_install import _InterprocessInstallLock

lock_path = Path(sys.argv[1])
ready_path = Path(sys.argv[2])
with _InterprocessInstallLock(lock_path, canceled=threading.Event(), timeout=5.0):
    ready_path.write_text("locked", encoding="ascii")
    time.sleep(60.0)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(lock_path), str(ready_path)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5.0
    while not ready_path.is_file():
        if process.poll() is not None:
            _stdout, stderr = process.communicate()
            pytest.fail(f"interprocess lock holder exited early: {stderr}")
        if time.monotonic() >= deadline:
            process.kill()
            process.wait(timeout=5.0)
            pytest.fail("interprocess lock holder did not become ready")
        time.sleep(0.01)
    return process


def test_install_lock_blocks_a_second_process_and_crash_releases_it(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "managed"
    data_root.mkdir()
    lock_path = data_root / runtime_install.INSTALL_LOCK_FILENAME
    process = _start_lock_holder(lock_path, tmp_path / "holder-ready")
    download_called = False

    def unexpected_download(_request: object, timeout: float) -> _Response:
        nonlocal download_called
        download_called = True
        return _Response(b"unused")

    try:
        started = time.monotonic()
        with pytest.raises(RuntimeInstallError, match="another Blender instance|timed out"):
            runtime_install.install_managed_runtime(
                _artifact(b"unused"),
                data_root,
                audio2face_model_directory=tmp_path / "unused-face",
                audio2emotion_model_directory=tmp_path / "unused-emotion",
                progress=lambda _event: None,
                canceled=threading.Event(),
                activation_lock=threading.Lock(),
                open_url=unexpected_download,
                interprocess_lock_timeout=0.10,
            )
        assert time.monotonic() - started < 2.0
        assert not download_called
        assert not list(data_root.glob(".a2f-install-*"))
    finally:
        process.kill()
        process.wait(timeout=5.0)

    # The file remains, but the OS releases its lock when the holder dies.
    assert lock_path.is_file()
    with runtime_install._InterprocessInstallLock(
        lock_path,
        canceled=threading.Event(),
        timeout=1.0,
        poll_interval=0.01,
    ):
        pass


def test_install_lock_wait_is_cancellable(tmp_path: Path) -> None:
    lock_path = tmp_path / runtime_install.INSTALL_LOCK_FILENAME
    process = _start_lock_holder(lock_path, tmp_path / "holder-ready")
    canceled = threading.Event()
    timer = threading.Timer(0.05, canceled.set)
    try:
        timer.start()
        started = time.monotonic()
        with pytest.raises(runtime_install.RuntimeInstallCancelled, match="canceled"):
            with runtime_install._InterprocessInstallLock(
                lock_path,
                canceled=canceled,
                timeout=5.0,
                poll_interval=0.01,
            ):
                pass
        assert time.monotonic() - started < 1.0
    finally:
        timer.cancel()
        process.kill()
        process.wait(timeout=5.0)


def test_install_downloads_builds_and_atomically_activates_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, unpacked_size = _valid_linux_runtime_zip()
    artifact = _artifact(payload, unpacked_size=unpacked_size)
    data_root = tmp_path / "managed data"
    old_runtime = data_root / "runtime/linux-x64"
    old_runtime.mkdir(parents=True)
    (old_runtime / "old-marker").write_text("old", encoding="utf-8")
    audio2face_directory, audio2emotion_directory = _external_model_directories(
        tmp_path / "selected models"
    )

    built_models: list[str] = []

    def fake_build(
        _spec: BundleLaunchSpec,
        model: Path,
        model_id: str,
        _model_label: str,
        **_kwargs: object,
    ) -> object:
        built_models.append(model_id)
        candidate = model.parent / f".{model_id}.candidate.trt"
        candidate.write_bytes(f"{model_id} engine".encode())
        return runtime_install._EngineCandidate(
            temporary=candidate,
            destination=model.parent / "network.trt",
        )

    monkeypatch.setattr(runtime_install, "_build_trt_engine", fake_build)
    progress: list[object] = []
    result = runtime_install.install_managed_runtime(
        artifact,
        data_root,
        audio2face_model_directory=audio2face_directory,
        audio2emotion_model_directory=audio2emotion_directory,
        progress=progress.append,
        canceled=threading.Event(),
        activation_lock=threading.Lock(),
        open_url=lambda _request, timeout: _Response(
            payload, content_length=str(len(payload))
        ),
    )

    assert result.root == data_root.resolve() / "runtime/linux-x64"
    assert built_models == ["audio2face", "audio2emotion"]
    assert result.audio2face_model.name == "model.json"
    assert result.audio2face_model.with_name("network.trt").read_bytes() == (
        b"audio2face engine"
    )
    assert result.audio2emotion_model.with_name("network.trt").read_bytes() == (
        b"audio2emotion engine"
    )
    assert (result.root / RUNTIME_RECEIPT_FILENAME).read_text(encoding="ascii") == (
        f"{artifact.sha256}\n"
    )
    validate_install_receipt(result, artifact)
    assert not (result.root / "old-marker").exists()
    assert not list(data_root.glob(".a2f-install-*"))
    assert progress[-1].stage == "complete"  # type: ignore[attr-defined]


def test_activation_gate_only_covers_cancellation_check_and_renames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, unpacked_size = _valid_linux_runtime_zip()
    artifact = _artifact(payload, unpacked_size=unpacked_size)
    data_root = tmp_path / "managed"
    active = data_root / "runtime/linux-x64"
    stale_backup = data_root / "runtime/.linux-x64.previous"
    active.mkdir(parents=True)
    stale_backup.mkdir(parents=True)
    (active / "marker").write_text("old active", encoding="utf-8")
    (stale_backup / "marker").write_text("stale backup", encoding="utf-8")
    audio2face_directory, audio2emotion_directory = _external_model_directories(
        tmp_path / "selected models"
    )

    class TrackingGate:
        held = False

        def __enter__(self) -> None:
            assert not self.held
            self.held = True

        def __exit__(self, *_args: object) -> None:
            self.held = False

    gate = TrackingGate()

    def fake_build(
        _spec: BundleLaunchSpec,
        model: Path,
        _model_id: str,
        _model_label: str,
        **_kwargs: object,
    ) -> object:
        candidate = model.parent / f".{_model_id}.candidate.trt"
        candidate.write_bytes(b"local engine")
        return runtime_install._EngineCandidate(
            temporary=candidate,
            destination=model.parent / "network.trt",
        )

    real_remove = runtime_install._remove_activation_path
    backup_removals = 0

    def tracked_remove(path: Path) -> None:
        nonlocal backup_removals
        assert not gate.held
        if path.name == ".linux-x64.previous":
            backup_removals += 1
            if backup_removals == 2:
                # The new runtime is already active.  Cleanup failure must not
                # make the installer report that activation failed.
                raise OSError("simulated slow-backup cleanup failure")
        real_remove(path)

    real_replace = os.replace
    rename_gate_states: list[bool] = []

    def tracked_replace(source: object, destination: object) -> None:
        rename_gate_states.append(gate.held)
        real_replace(source, destination)

    def track_progress(event: object) -> None:
        if getattr(event, "stage", None) == "activating":
            assert not gate.held

    monkeypatch.setattr(runtime_install, "_build_trt_engine", fake_build)
    monkeypatch.setattr(runtime_install, "_remove_activation_path", tracked_remove)
    monkeypatch.setattr(runtime_install.os, "replace", tracked_replace)

    result = runtime_install.install_managed_runtime(
        artifact,
        data_root,
        audio2face_model_directory=audio2face_directory,
        audio2emotion_model_directory=audio2emotion_directory,
        activation_lock=gate,
        progress=track_progress,
        canceled=threading.Event(),
        open_url=lambda _request, timeout: _Response(payload),
    )

    assert not gate.held
    assert backup_removals == 2
    assert rename_gate_states == [True, True, True, True]
    assert result.audio2face_model.with_name("network.trt").read_bytes() == b"local engine"
    assert result.audio2emotion_model.with_name("network.trt").read_bytes() == b"local engine"
    assert (stale_backup / "marker").read_text(encoding="utf-8") == "old active"


def test_install_receipt_rejects_a_different_pinned_archive(tmp_path: Path) -> None:
    spec = _launch_spec(tmp_path, {"trt_build_param": {"batch": []}, "defaults": {}})
    artifact = _artifact(b"current")
    (spec.root / RUNTIME_RECEIPT_FILENAME).write_text("0" * 64 + "\n", encoding="ascii")

    with pytest.raises(RuntimeInstallError, match="does not match"):
        validate_install_receipt(spec, artifact)


def test_failed_model_build_keeps_previous_runtime_active_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, unpacked_size = _valid_linux_runtime_zip()
    artifact = _artifact(payload, unpacked_size=unpacked_size)
    data_root = tmp_path / "managed"
    old_runtime = data_root / "runtime/linux-x64"
    old_runtime.mkdir(parents=True)
    marker = old_runtime / "marker"
    marker.write_text("old", encoding="utf-8")
    audio2face_directory, audio2emotion_directory = _external_model_directories(
        tmp_path / "selected models"
    )

    built_models: list[str] = []

    def fail_build(
        _spec: BundleLaunchSpec,
        model: Path,
        model_id: str,
        _model_label: str,
        **_kwargs: object,
    ) -> object:
        built_models.append(model_id)
        if model_id == "audio2emotion":
            raise RuntimeInstallError("simulated TensorRT failure")
        candidate = model.parent / ".audio2face.candidate.trt"
        candidate.write_bytes(b"temporary engine")
        return runtime_install._EngineCandidate(
            temporary=candidate,
            destination=model.parent / "network.trt",
        )

    monkeypatch.setattr(runtime_install, "_build_trt_engine", fail_build)
    with pytest.raises(RuntimeInstallError, match="simulated TensorRT failure"):
        runtime_install.install_managed_runtime(
            artifact,
            data_root,
            audio2face_model_directory=audio2face_directory,
            audio2emotion_model_directory=audio2emotion_directory,
            progress=lambda _event: None,
            canceled=threading.Event(),
            activation_lock=threading.Lock(),
            open_url=lambda _request, timeout: _Response(payload),
        )

    assert marker.read_text(encoding="utf-8") == "old"
    assert built_models == ["audio2face", "audio2emotion"]
    assert not (audio2face_directory / "network.trt").exists()
    assert not (audio2emotion_directory / "network.trt").exists()
    assert not list(audio2face_directory.glob("*.candidate.trt"))
    assert not list(data_root.glob(".a2f-install-*"))
