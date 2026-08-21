from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

MODEL_CHANNELS = ("sdkJawOpen", *(f"sdkOutput{index:02d}" for index in range(51)))
AppliedFrame = tuple[tuple[str, ...], list[float]]


@dataclass
class _Settings:
    stream_time: float = 7.0
    stream_reset_on_stop: bool = True


@dataclass
class _Scene:
    name: str
    audio2face: _Settings
    is_editable: bool = True


class _Scenes(list[_Scene]):
    def get(self, name: str | None) -> _Scene | None:
        return next((scene for scene in self if scene.name == name), None)


@pytest.fixture
def live_module(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, _Scene, list[AppliedFrame]]:
    settings = _Settings()
    scene = _Scene("Scene", settings)
    bpy = ModuleType("bpy")
    bpy.types = SimpleNamespace(Scene=object)  # type: ignore[attr-defined]
    bpy.data = SimpleNamespace(scenes=_Scenes([scene]))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    applied: list[AppliedFrame] = []
    preview = ModuleType("audio2face.preview")
    preview.PreviewError = RuntimeError  # type: ignore[attr-defined]
    preview.build_subscriptions = lambda _settings: (object(),)  # type: ignore[attr-defined]
    preview.apply_shape_key_frame = (  # type: ignore[attr-defined]
        lambda _subscriptions, channels, weights: applied.append(
            (tuple(channels), list(weights))
        )
    )
    monkeypatch.setitem(sys.modules, preview.__name__, preview)

    module_name = "audio2face._live_stream_test"
    source = Path(__file__).resolve().parents[1] / "audio2face" / "live_stream.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, scene, applied


def test_source_free_stream_applies_negative_timestamp_frame_immediately(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
) -> None:
    live, scene, applied = live_module
    controller = live.LiveStreamController()
    controller.prepare(scene, "stream-1", 16_000, MODEL_CHANNELS)
    weights = [0.0] * len(MODEL_CHANNELS)
    weights[MODEL_CHANNELS.index("sdkJawOpen")] = 0.625

    controller.receive("stream-1", -320, weights)

    assert applied == [(MODEL_CHANNELS, weights)]
    assert scene.audio2face.stream_time == 0.0
    assert controller.active is True
    assert controller.stream_id == "stream-1"


def test_prepare_reports_only_the_missing_enabled_mesh_requirement(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, scene, _applied = live_module
    monkeypatch.setattr(live, "build_subscriptions", lambda _settings: ())

    with pytest.raises(
        live.LiveStreamError,
        match="^no enabled target mesh is selected$",
    ):
        live.LiveStreamController().prepare(
            scene,
            "stream-1",
            16_000,
            MODEL_CHANNELS,
        )


def test_prepare_translates_invalid_channel_contract(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
) -> None:
    live, scene, _applied = live_module

    with pytest.raises(live.LiveStreamError, match="exactly 52 names"):
        live.LiveStreamController().prepare(
            scene,
            "stream-1",
            16_000,
            MODEL_CHANNELS[:-1],
        )


def test_source_free_stream_interpolates_bursted_frames_on_a_monotonic_clock(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, scene, applied = live_module
    now = [10.0]
    monkeypatch.setattr(live.time, "monotonic", lambda: now[0])
    controller = live.LiveStreamController()
    controller.prepare(scene, "stream-1", 16_000, MODEL_CHANNELS)
    closed = [0.0] * len(MODEL_CHANNELS)
    open_frame = closed.copy()
    jaw = MODEL_CHANNELS.index("sdkJawOpen")
    open_frame[jaw] = 1.0

    controller.receive("stream-1", 0, closed)
    controller.receive("stream-1", 1600, open_frame)
    assert applied == [(MODEL_CHANNELS, closed)]

    now[0] += 0.05
    assert controller.tick() is True

    assert applied[-1][0] == MODEL_CHANNELS
    assert applied[-1][1][jaw] == pytest.approx(0.5)
    assert scene.audio2face.stream_time == pytest.approx(0.05)


def test_live_stream_requires_strictly_increasing_signed_64_bit_timestamps(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
) -> None:
    live, scene, _applied = live_module
    weights = [0.0] * len(MODEL_CHANNELS)

    for invalid in (True, -(1 << 63) - 1, 1 << 63, 0.5):
        controller = live.LiveStreamController()
        controller.prepare(scene, "stream-1", 16_000, MODEL_CHANNELS)
        with pytest.raises(live.LiveStreamError, match="signed 64-bit"):
            controller.receive("stream-1", invalid, weights)

    controller = live.LiveStreamController()
    controller.prepare(scene, "stream-1", 16_000, MODEL_CHANNELS)
    controller.receive("stream-1", -1, weights)
    with pytest.raises(live.LiveStreamError, match="strictly increasing"):
        controller.receive("stream-1", -1, weights)


@pytest.mark.parametrize(
    "weights",
    (
        [0.0] * (len(MODEL_CHANNELS) - 1),
        [0.0] * (len(MODEL_CHANNELS) - 1) + [True],
        [0.0] * (len(MODEL_CHANNELS) - 1) + [float("nan")],
        [0.0] * (len(MODEL_CHANNELS) - 1) + [-0.01],
        [0.0] * (len(MODEL_CHANNELS) - 1) + [1.01],
    ),
)
def test_live_stream_rejects_invalid_model_width_or_weight(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
    weights: list[object],
) -> None:
    live, scene, applied = live_module
    controller = live.LiveStreamController()
    controller.prepare(scene, "stream-1", 16_000, MODEL_CHANNELS)

    with pytest.raises(live.LiveStreamError):
        controller.receive("stream-1", 0, weights)

    assert applied == []


def test_terminal_event_cleans_external_stream_and_resets_values(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
) -> None:
    live, scene, applied = live_module
    controller = live.LiveStreamController()
    stopped: list[str] = []
    controller.prepare(
        scene,
        "stream-1",
        16_000,
        MODEL_CHANNELS,
        playback_stopped=lambda: stopped.append("stream-1"),
    )
    weights = [0.5] * len(MODEL_CHANNELS)
    controller.receive("stream-1", 1600, weights)

    controller.mark_terminal("stream-1")

    assert applied == [
        (MODEL_CHANNELS, weights),
        (MODEL_CHANNELS, [0.0] * len(MODEL_CHANNELS)),
    ]
    assert controller.active is False
    assert controller.stream_id is None
    assert scene.audio2face.stream_time == 0.0
    assert stopped == ["stream-1"]
