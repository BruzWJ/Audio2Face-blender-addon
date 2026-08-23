from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

MODEL_CHANNELS = ["sdkJawOpen", *(f"sdkOutput{index:02d}" for index in range(51))]
AppliedFrame = tuple[tuple[str, ...], tuple[float, ...]]


@dataclass
class _Settings:
    stream_time: float = 7.0
    prediction_delay: float = 0.0
    playback_state: str = "IDLE"
    playback_duration: float = 0.0
    playback_progress: float = 0.0
    status: str = "MODEL_READY"
    status_message: str = ""


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
    shape_keys = ModuleType("audio2face.shape_keys")
    shape_keys.ShapeKeyStreamError = ValueError  # type: ignore[attr-defined]
    def validate_output_channels(channels: object) -> tuple[str, ...]:
        if type(channels) is not list:
            raise ValueError("channels must be a JSON array")
        if len(channels) != 52:
            raise ValueError("channels must contain exactly 52 names")
        return tuple(channels)

    shape_keys.validate_output_channels = validate_output_channels  # type: ignore[attr-defined]
    shape_keys.resolve_target_meshes = lambda _settings: (object(),)  # type: ignore[attr-defined]

    def apply_shape_key_frame(
        _subscriptions: object,
        channels: tuple[str, ...],
        weights: tuple[float, ...],
    ) -> None:
        assert type(channels) is tuple
        assert type(weights) is tuple
        applied.append((channels, weights))

    shape_keys.apply_shape_key_frame = apply_shape_key_frame  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, shape_keys.__name__, shape_keys)

    module_name = "audio2face._live_stream_test"
    source = Path(__file__).resolve().parents[1] / "audio2face" / "live_stream.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, scene, applied


def _prepare_external(controller: object, scene: _Scene, channels: list[str]) -> None:
    controller.prepare(  # type: ignore[attr-defined]
        scene,
        "stream-1",
        16_000,
        channels.copy(),
        audio_path=None,
        playback_started=None,
        playback_stopped=None,
    )


def test_source_free_stream_applies_negative_timestamp_frame_immediately(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
) -> None:
    live, scene, applied = live_module
    controller = live.LiveStreamController()
    _prepare_external(controller, scene, MODEL_CHANNELS)
    weights = [0.0] * len(MODEL_CHANNELS)
    weights[MODEL_CHANNELS.index("sdkJawOpen")] = 0.625

    controller.receive("stream-1", -320, weights)

    assert applied == [(tuple(MODEL_CHANNELS), tuple(weights))]
    assert scene.audio2face.stream_time == 0.0
    assert controller.active is True
    assert controller.operation_id == "stream-1"


def test_prepare_allows_targets_to_be_added_after_stream_start(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, scene, _applied = live_module
    monkeypatch.setattr(live, "resolve_target_meshes", lambda _settings: ())
    controller = live.LiveStreamController()

    controller.prepare(
        scene,
        "stream-1",
        16_000,
        MODEL_CHANNELS.copy(),
        audio_path=None,
        playback_started=None,
        playback_stopped=None,
    )

    assert controller.active is True


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
            audio_path=None,
            playback_started=None,
            playback_stopped=None,
        )


def test_source_free_stream_interpolates_bursted_frames_on_a_monotonic_clock(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, scene, applied = live_module
    now = [10.0]
    monkeypatch.setattr(live.time, "monotonic", lambda: now[0])
    controller = live.LiveStreamController()
    _prepare_external(controller, scene, MODEL_CHANNELS)
    closed = [0.0] * len(MODEL_CHANNELS)
    open_frame = closed.copy()
    jaw = MODEL_CHANNELS.index("sdkJawOpen")
    open_frame[jaw] = 1.0

    controller.receive("stream-1", 0, closed)
    controller.receive("stream-1", 1600, open_frame)
    assert applied == [(tuple(MODEL_CHANNELS), tuple(closed))]

    now[0] += 0.05
    assert controller.tick() is True

    assert applied[-1][0] == tuple(MODEL_CHANNELS)
    assert applied[-1][1][jaw] == pytest.approx(0.5)
    assert scene.audio2face.stream_time == pytest.approx(0.05)


def test_live_frames_resolve_the_current_mesh_targets(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, scene, _applied = live_module
    now = [10.0]
    first_target = object()
    second_target = object()
    current_targets = [(first_target,)]
    delivered_targets: list[tuple[object, ...]] = []
    monkeypatch.setattr(live.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        live,
        "resolve_target_meshes",
        lambda _settings: current_targets[0],
    )
    monkeypatch.setattr(
        live,
        "apply_shape_key_frame",
        lambda targets, _channels, _weights: delivered_targets.append(targets),
    )
    controller = live.LiveStreamController()
    _prepare_external(controller, scene, MODEL_CHANNELS)
    weights = [0.25] * len(MODEL_CHANNELS)

    controller.receive("stream-1", 0, weights)
    current_targets[0] = (second_target,)
    controller.receive("stream-1", 1600, weights)
    now[0] += 0.05
    controller.tick()
    current_targets[0] = ()
    now[0] += 0.01
    controller.tick()
    current_targets[0] = (first_target,)
    now[0] += 0.01
    controller.tick()

    assert delivered_targets == [
        (first_target,),
        (second_target,),
        (),
        (first_target,),
    ]


def test_paused_selected_audio_uses_the_current_mesh_targets(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    live, scene, _applied = live_module
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"RIFF")
    first_target = object()
    second_target = object()
    current_targets = [(first_target,)]
    delivered_targets: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        live,
        "resolve_target_meshes",
        lambda _settings: current_targets[0],
    )
    monkeypatch.setattr(
        live,
        "apply_shape_key_frame",
        lambda targets, _channels, _weights: delivered_targets.append(targets),
    )
    controller = live.LiveStreamController()
    controller.prepare(
        scene,
        "stream-1",
        16_000,
        MODEL_CHANNELS.copy(),
        audio_path=audio_path,
        playback_started=None,
        playback_stopped=None,
    )
    controller._handle = SimpleNamespace(status=True, position=0.0)
    controller._duration = 2.0
    scene.audio2face.playback_state = "PAUSED"
    controller.receive("stream-1", 0, [0.25] * len(MODEL_CHANNELS))

    controller.tick()
    current_targets[0] = (second_target,)
    controller.tick()

    assert delivered_targets == [(first_target,), (second_target,)]


def test_frame_reset_starts_a_new_timestamp_epoch_without_stopping_stream(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
) -> None:
    live, scene, _applied = live_module
    controller = live.LiveStreamController()
    _prepare_external(controller, scene, MODEL_CHANNELS)
    weights = [0.25] * len(MODEL_CHANNELS)
    controller.receive("stream-1", 100, weights)

    controller.reset_frames("stream-1")
    controller.receive("stream-1", -50, weights)

    assert controller.active is True
    assert controller.operation_id == "stream-1"


def test_live_stream_requires_strictly_increasing_signed_64_bit_timestamps(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
) -> None:
    live, scene, _applied = live_module
    weights = [0.0] * len(MODEL_CHANNELS)

    for invalid in (True, -(1 << 63) - 1, 1 << 63, 0.5):
        controller = live.LiveStreamController()
        _prepare_external(controller, scene, MODEL_CHANNELS)
        with pytest.raises(live.LiveStreamError, match="signed 64-bit"):
            controller.receive("stream-1", invalid, weights)

    controller = live.LiveStreamController()
    _prepare_external(controller, scene, MODEL_CHANNELS)
    controller.receive("stream-1", -1, weights)
    with pytest.raises(live.LiveStreamError, match="strictly increasing"):
        controller.receive("stream-1", -1, weights)


@pytest.mark.parametrize(
    "weights",
    (
        [0.0] * (len(MODEL_CHANNELS) - 1),
        [0.0] * (len(MODEL_CHANNELS) - 1) + [True],
        [0.0] * (len(MODEL_CHANNELS) - 1) + [float("nan")],
        [0.0] * (len(MODEL_CHANNELS) - 1) + [0],
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
    _prepare_external(controller, scene, MODEL_CHANNELS)

    with pytest.raises(live.LiveStreamError):
        controller.receive("stream-1", 0, weights)

    assert applied == []


def test_live_stream_rejects_tuple_aliases_at_json_array_boundaries(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
) -> None:
    live, scene, _applied = live_module
    controller = live.LiveStreamController()

    with pytest.raises(live.LiveStreamError, match="channels must be a JSON array"):
        controller.prepare(
            scene,
            "stream-1",
            16_000,
            tuple(MODEL_CHANNELS),
            audio_path=None,
            playback_started=None,
            playback_stopped=None,
        )

    _prepare_external(controller, scene, MODEL_CHANNELS)
    with pytest.raises(live.LiveStreamError, match="weights must be a JSON array"):
        controller.receive("stream-1", 0, tuple([0.0] * len(MODEL_CHANNELS)))


def test_live_stream_rejects_terminal_event_for_inactive_stream(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
) -> None:
    live, _scene, _applied = live_module

    with pytest.raises(live.LiveStreamError, match="inactive stream"):
        live.LiveStreamController().mark_terminal("stream-1")


def test_terminal_event_cleans_external_stream_and_holds_final_values(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
) -> None:
    live, scene, applied = live_module
    controller = live.LiveStreamController()
    stopped: list[str] = []
    controller.prepare(
        scene,
        "stream-1",
        16_000,
        MODEL_CHANNELS.copy(),
        audio_path=None,
        playback_started=None,
        playback_stopped=lambda natural: stopped.append(
            f"stream-1:{natural}"
        ),
    )
    weights = [0.5] * len(MODEL_CHANNELS)
    controller.receive("stream-1", 1600, weights)

    controller.mark_terminal("stream-1")

    assert applied == [(tuple(MODEL_CHANNELS), tuple(weights))]
    assert controller.active is False
    assert controller.operation_id is None
    assert scene.audio2face.stream_time == 0.0
    assert stopped == ["stream-1:True"]


def test_selected_audio_waits_for_worker_terminal_after_device_stops(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
    tmp_path: Path,
) -> None:
    live, scene, _applied = live_module
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"RIFF")
    stopped: list[bool] = []
    controller = live.LiveStreamController()
    controller.prepare(
        scene,
        "stream-1",
        16_000,
        MODEL_CHANNELS.copy(),
        audio_path=audio_path,
        playback_started=None,
        playback_stopped=stopped.append,
    )

    class Handle:
        status = False
        position = 2.0
        stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    handle = Handle()
    controller._handle = handle
    controller._duration = 2.0
    scene.audio2face.playback_duration = 2.0

    assert controller.tick() is True
    assert controller.active is True
    assert scene.audio2face.playback_progress == pytest.approx(1.0)
    assert stopped == []

    controller.mark_terminal("stream-1")

    assert controller.active is True
    assert controller.tick() is False
    assert controller.active is False
    assert handle.stop_calls == 1
    assert stopped == [True]


def test_boolean_handle_status_does_not_overwrite_paused_state(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
    tmp_path: Path,
) -> None:
    live, scene, _applied = live_module
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"RIFF")
    controller = live.LiveStreamController()
    controller.prepare(
        scene,
        "stream-1",
        16_000,
        MODEL_CHANNELS.copy(),
        audio_path=audio_path,
        playback_started=None,
        playback_stopped=None,
    )
    controller._handle = SimpleNamespace(status=True, position=0.5)
    controller._duration = 2.0
    scene.audio2face.playback_state = "PAUSED"

    assert controller.tick() is True

    assert scene.audio2face.playback_state == "PAUSED"


def test_progress_edit_requests_seek_without_snapping_back(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
    tmp_path: Path,
) -> None:
    live, scene, _applied = live_module
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"RIFF")
    seeks: list[tuple[float, bool]] = []
    controller = live.LiveStreamController()
    controller.prepare(
        scene,
        "stream-1",
        16_000,
        MODEL_CHANNELS.copy(),
        audio_path=audio_path,
        playback_started=None,
        playback_seeked=lambda position, paused: seeks.append((position, paused)),
        playback_stopped=None,
    )
    controller._handle = SimpleNamespace(status=True, position=0.25)
    controller._duration = 2.0
    controller._published_progress = 0.125
    scene.audio2face.playback_state = "PAUSED"
    scene.audio2face.playback_progress = 0.75

    assert controller.tick() is True

    assert seeks == [(1.5, True)]
    assert scene.audio2face.playback_progress == 0.75


def test_seek_stop_preserves_requested_playback_presentation(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
    tmp_path: Path,
) -> None:
    live, scene, _applied = live_module
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"RIFF")
    controller = live.LiveStreamController()
    controller.prepare(
        scene,
        "stream-1",
        16_000,
        MODEL_CHANNELS.copy(),
        audio_path=audio_path,
        playback_started=None,
        playback_stopped=None,
    )

    class Handle:
        status = True

        def stop(self) -> None:
            pass

    controller._handle = Handle()
    controller._duration = 2.0

    controller.stop_for_seek(1.25, paused=True)

    assert controller.active is False
    assert scene.audio2face.playback_state == "PAUSED"
    assert scene.audio2face.playback_duration == 2.0
    assert scene.audio2face.playback_progress == 0.625


def test_selected_audio_seek_endpoint_resolves_to_final_model_sample(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
    tmp_path: Path,
) -> None:
    live, scene, _applied = live_module
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"RIFF")
    seeks: list[tuple[float, bool]] = []
    controller = live.LiveStreamController()
    controller.prepare(
        scene,
        "stream-1",
        16_000,
        MODEL_CHANNELS.copy(),
        audio_path=audio_path,
        playback_started=None,
        playback_seeked=lambda position, paused: seeks.append((position, paused)),
        playback_stopped=None,
    )
    controller._handle = SimpleNamespace()
    controller._duration = 2.0
    scene.audio2face.playback_state = "PAUSED"

    controller.request_seek(2.0)

    assert seeks == [(pytest.approx(2.0 - (1.0 / 16_000)), True)]
