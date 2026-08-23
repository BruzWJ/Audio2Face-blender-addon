from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from audio2face.frame_stream import sample_linear


class _ShapeKey:
    def __init__(self) -> None:
        self.value = 0.0


class _ShapeKeys:
    def __init__(self) -> None:
        self.key_blocks = {"jawOpen": _ShapeKey()}

    def as_pointer(self) -> int:
        return id(self)


class _MeshTarget:
    type = "MESH"

    def __init__(self) -> None:
        self.data = SimpleNamespace(shape_keys=_ShapeKeys())

    def as_pointer(self) -> int:
        return id(self)


class _Handle:
    __slots__ = (
        "_position",
        "loop_count",
        "position_writes",
        "status",
        "stop_count",
    )

    def __init__(self) -> None:
        self._position = 0.0
        self.loop_count = 0
        self.position_writes: list[float] = []
        self.status = "PLAYING"
        self.stop_count = 0

    @property
    def position(self) -> float:
        return self._position

    @position.setter
    def position(self, value: float) -> None:
        self._position = float(value)
        self.position_writes.append(self._position)

    def advance(self, position: float) -> None:
        self._position = position

    def pause(self) -> bool:
        self.status = "PAUSED"
        return True

    def resume(self) -> bool:
        self.status = "PLAYING"
        return True

    def stop(self) -> bool:
        self.status = "STOPPED"
        self.stop_count += 1
        return True


@pytest.fixture
def preview_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    bpy = ModuleType("bpy")
    bpy.types = SimpleNamespace(Object=object, Scene=object)  # type: ignore[attr-defined]
    bpy.data = SimpleNamespace(scenes={})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    module_name = "audio2face._preview_test"
    source = Path(__file__).resolve().parents[1] / "audio2face" / "preview.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _start_preview(
    preview_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[object, SimpleNamespace, _Handle, _MeshTarget]:
    handle = _Handle()
    aud = ModuleType("aud")
    aud.STATUS_STOPPED = "STOPPED"  # type: ignore[attr-defined]
    aud.STATUS_PLAYING = "PLAYING"  # type: ignore[attr-defined]
    aud.STATUS_PAUSED = "PAUSED"  # type: ignore[attr-defined]

    class Sound:
        length = 200
        specs = (100.0, 1)

        def __init__(self, path: str) -> None:
            self.path = path

    class Device:
        def play(self, _sound: Sound) -> _Handle:
            return handle

    aud.Sound = Sound  # type: ignore[attr-defined]
    aud.Device = Device  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aud", aud)

    target = _MeshTarget()
    settings = SimpleNamespace(
        target_meshes=[SimpleNamespace(object=target, enabled=True)],
        prediction_delay=0.0,
        preview_duration=0.0,
        preview_loop=False,
        preview_progress=0.0,
        preview_state="IDLE",
        preview_time=0.0,
        status_message="",
    )
    scene = SimpleNamespace(name="PreviewScene", audio2face=settings)
    preview_module.bpy.data.scenes[scene.name] = scene
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"RIFF")
    result = preview_module.AnimationResult(
        timestamps=[0, 200],
        sample_rate=100,
        weights=[[0.0], [1.0]],
        operation_id="operation-1",
        channels=["jawOpen"],
    )
    controller = preview_module.PreviewController()
    controller.start(scene, result, audio_path.resolve())
    return controller, settings, handle, target


def test_sampled_frame_is_the_exact_internal_preview_tuple(
    preview_module: ModuleType,
) -> None:
    shape_keys = _ShapeKeys()
    target = SimpleNamespace(data=SimpleNamespace(shape_keys=shape_keys))
    frame = sample_linear([0, 100], [[0.0], [1.0]], 25.0)

    assert type(frame) is tuple
    preview_module.apply_shape_key_frame((target,), ("jawOpen",), frame)

    assert shape_keys.key_blocks["jawOpen"].value == pytest.approx(0.25)


def test_shape_key_delivery_rejects_mutable_frame_alias(
    preview_module: ModuleType,
) -> None:
    target = SimpleNamespace(data=SimpleNamespace(shape_keys=_ShapeKeys()))

    with pytest.raises(preview_module.PreviewError, match="frozen frame tuple"):
        preview_module.apply_shape_key_frame((target,), ("jawOpen",), [0.5])


def test_preview_uses_audio_duration_and_seeks_only_changed_progress(
    preview_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller, settings, handle, target = _start_preview(
        preview_module,
        monkeypatch,
        tmp_path,
    )

    assert settings.preview_duration == pytest.approx(2.0)
    assert settings.preview_time == pytest.approx(0.0)
    assert settings.preview_progress == pytest.approx(0.0)
    assert handle.position_writes == []

    settings.preview_progress = 0.75
    assert controller.tick()

    assert handle.position == pytest.approx(1.5)
    assert handle.position_writes == [pytest.approx(1.5)]
    assert settings.preview_time == pytest.approx(1.5)
    assert target.data.shape_keys.key_blocks["jawOpen"].value == pytest.approx(0.75)

    assert controller.tick()
    assert handle.position_writes == [pytest.approx(1.5)]


@pytest.mark.parametrize(
    ("delay", "expected_weight"),
    ((0.25, 0.375), (-0.25, 0.125)),
)
def test_prediction_delay_offsets_model_sampling_not_audio_position(
    preview_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    delay: float,
    expected_weight: float,
) -> None:
    controller, settings, handle, target = _start_preview(
        preview_module,
        monkeypatch,
        tmp_path,
    )
    handle.advance(0.5)
    settings.prediction_delay = delay

    assert controller.tick()

    assert handle.position == pytest.approx(0.5)
    assert settings.preview_time == pytest.approx(0.5)
    assert target.data.shape_keys.key_blocks["jawOpen"].value == pytest.approx(
        expected_weight
    )


def test_rewind_preserves_pause_and_loop_state(
    preview_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller, settings, handle, target = _start_preview(
        preview_module,
        monkeypatch,
        tmp_path,
    )
    handle.advance(1.0)
    controller.tick()
    controller.pause()
    settings.preview_loop = True
    controller.tick()

    controller.rewind()

    assert handle.status == "PAUSED"
    assert settings.preview_state == "PAUSED"
    assert handle.loop_count == -1
    assert handle.position == pytest.approx(0.0)
    assert settings.preview_time == pytest.approx(0.0)
    assert settings.preview_progress == pytest.approx(0.0)
    assert target.data.shape_keys.key_blocks["jawOpen"].value == pytest.approx(0.0)

    settings.preview_loop = False
    controller.tick()
    assert handle.loop_count == 0


def test_natural_completion_needs_no_selected_reset_or_volume_properties(
    preview_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller, settings, handle, _target = _start_preview(
        preview_module,
        monkeypatch,
        tmp_path,
    )
    handle.status = "STOPPED"

    assert not controller.tick()
    assert not controller.active
    assert settings.preview_state == "IDLE"
    assert settings.preview_progress == pytest.approx(0.0)
