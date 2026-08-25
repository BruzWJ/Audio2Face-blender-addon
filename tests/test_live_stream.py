from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

MODEL_CHANNELS = ["sdkJawOpen", *(f"sdkOutput{index:02d}" for index in range(51))]
MODEL_EMOTION_CHANNELS = ["Neutral", "Joy"]
MODEL_EMOTIONS = [0.75, 0.25]
AppliedFrame = tuple[tuple[str, ...], tuple[float, ...]]


@dataclass
class _Settings:
    stream_time: float = 7.0
    prediction_delay: float = 0.0
    playback_state: str = "IDLE"
    status: str = "MODEL_READY"
    status_message: str = ""
    custom_properties: dict[str, object] = field(default_factory=dict)
    custom_property_ui: dict[str, dict[str, object]] = field(default_factory=dict)
    preferred_emotions: tuple[SimpleNamespace, ...] = field(
        default_factory=lambda: (
            SimpleNamespace(name="Neutral", value=0.1),
            SimpleNamespace(name="Joy", value=0.9),
        )
    )
    mixed_emotions: tuple[SimpleNamespace, ...] = field(
        default_factory=lambda: (
            SimpleNamespace(name="Neutral", value=0.0),
            SimpleNamespace(name="Joy", value=0.0),
        )
    )

    def __contains__(self, key: str) -> bool:
        return key in self.custom_properties

    def __getitem__(self, key: str) -> object:
        return self.custom_properties[key]

    def __setitem__(self, key: str, value: object) -> None:
        self.custom_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.custom_properties[key]

    def id_properties_ui(self, key: str) -> object:
        metadata = self.custom_property_ui.setdefault(key, {})
        return SimpleNamespace(
            update=lambda **values: metadata.update(values),
            as_dict=lambda: metadata.copy(),
        )


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
    shape_keys.resolve_target_objects = lambda _settings: (object(),)  # type: ignore[attr-defined]

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

    properties = ModuleType("audio2face.properties")

    def apply_mixed_emotions(
        settings: _Settings,
        channels: tuple[str, ...],
        values: tuple[float, ...],
    ) -> None:
        assert channels == tuple(item.name for item in settings.mixed_emotions)
        for item, value in zip(settings.mixed_emotions, values):
            item.value = min(max(value, 0.0), 1.0)

    properties.apply_mixed_emotions = apply_mixed_emotions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, properties.__name__, properties)

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
        tuple(channels),
        tuple(MODEL_EMOTION_CHANNELS),
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

    controller.receive("stream-1", -320, weights, MODEL_EMOTIONS.copy())

    assert applied == [(tuple(MODEL_CHANNELS), tuple(weights))]
    assert scene.audio2face.stream_time == 0.0
    assert controller.active is True
    assert controller.operation_id == "stream-1"


def test_prepare_allows_targets_to_be_added_after_stream_start(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, scene, _applied = live_module
    monkeypatch.setattr(live, "resolve_target_objects", lambda _settings: ())
    controller = live.LiveStreamController()

    controller.prepare(
        scene,
        "stream-1",
        16_000,
        tuple(MODEL_CHANNELS),
        tuple(MODEL_EMOTION_CHANNELS),
        audio_path=None,
        playback_started=None,
        playback_stopped=None,
    )

    assert controller.active is True

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

    controller.receive("stream-1", 0, closed, [1.0, 0.0])
    controller.receive("stream-1", 1600, open_frame, [0.0, 1.0])
    assert applied == [(tuple(MODEL_CHANNELS), tuple(closed))]

    now[0] += 0.05
    assert controller.tick() is True

    assert applied[-1][0] == tuple(MODEL_CHANNELS)
    assert applied[-1][1][jaw] == pytest.approx(0.5)
    assert [item.value for item in scene.audio2face.mixed_emotions] == pytest.approx(
        [0.5, 0.5]
    )
    assert scene.audio2face.stream_time == pytest.approx(0.05)


def test_live_frames_resolve_the_current_object_targets(
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
        "resolve_target_objects",
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

    controller.receive("stream-1", 0, weights, MODEL_EMOTIONS.copy())
    current_targets[0] = (second_target,)
    controller.receive("stream-1", 1600, weights, MODEL_EMOTIONS.copy())
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


def test_selected_audio_updates_mixed_emotions_without_mutating_preferred(
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
        "resolve_target_objects",
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
        tuple(MODEL_CHANNELS),
        tuple(MODEL_EMOTION_CHANNELS),
        audio_path=audio_path,
        playback_started=None,
        playback_stopped=None,
    )
    controller._handle = SimpleNamespace(status=True, position=0.0)
    controller._duration = 2.0
    live.configure_playback_position(scene.audio2face, 0.0, 2.0)
    controller._published_position = 0.0
    scene.audio2face.playback_state = "PAUSED"
    controller.receive(
        "stream-1",
        0,
        [0.25] * len(MODEL_CHANNELS),
        MODEL_EMOTIONS.copy(),
    )
    assert [item.value for item in scene.audio2face.mixed_emotions] == [0.0, 0.0]

    controller.tick()
    assert [item.value for item in scene.audio2face.mixed_emotions] == pytest.approx(
        MODEL_EMOTIONS
    )
    current_targets[0] = (second_target,)
    controller.tick()

    assert delivered_targets == [(first_target,), (second_target,)]
    assert [item.value for item in scene.audio2face.preferred_emotions] == [0.1, 0.9]

    scene.audio2face.playback_state = "PLAYING"
    controller.tick()

    assert delivered_targets == [(first_target,), (second_target,), (second_target,)]
    assert [item.value for item in scene.audio2face.mixed_emotions] == pytest.approx(
        MODEL_EMOTIONS
    )


def test_frame_reset_starts_a_new_timestamp_epoch_without_stopping_stream(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
) -> None:
    live, scene, _applied = live_module
    controller = live.LiveStreamController()
    _prepare_external(controller, scene, MODEL_CHANNELS)
    weights = [0.25] * len(MODEL_CHANNELS)
    controller.receive("stream-1", 100, weights, MODEL_EMOTIONS.copy())

    controller.reset_frames("stream-1")
    assert controller._emotions == []
    controller.receive("stream-1", -50, weights, MODEL_EMOTIONS.copy())

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
            controller.receive("stream-1", invalid, weights, MODEL_EMOTIONS.copy())

    controller = live.LiveStreamController()
    _prepare_external(controller, scene, MODEL_CHANNELS)
    controller.receive("stream-1", -1, weights, MODEL_EMOTIONS.copy())
    with pytest.raises(live.LiveStreamError, match="strictly increasing"):
        controller.receive("stream-1", -1, weights, MODEL_EMOTIONS.copy())


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
        controller.receive("stream-1", 0, weights, MODEL_EMOTIONS.copy())

    assert applied == []


@pytest.mark.parametrize(
    "emotions",
    (
        [0.0],
        [0.0, True],
        [0.0, float("nan")],
        [0.0, float("inf")],
        [0.0, float("-inf")],
        [0.0, 0],
    ),
)
def test_live_stream_rejects_invalid_emotion_width_or_value(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
    emotions: list[object],
) -> None:
    live, scene, applied = live_module
    controller = live.LiveStreamController()
    _prepare_external(controller, scene, MODEL_CHANNELS)

    with pytest.raises(live.LiveStreamError):
        controller.receive(
            "stream-1",
            0,
            [0.0] * len(MODEL_CHANNELS),
            emotions,
        )

    assert applied == []


def test_live_stream_accepts_finite_unbounded_emotions(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
) -> None:
    live, scene, applied = live_module
    controller = live.LiveStreamController()
    _prepare_external(controller, scene, MODEL_CHANNELS)
    weights = [0.25] * len(MODEL_CHANNELS)

    controller.receive("stream-1", 0, weights, [-0.5, 1.5])

    assert applied == [(tuple(MODEL_CHANNELS), tuple(weights))]


def test_live_stream_rejects_tuples_at_worker_frame_json_array_boundaries(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
) -> None:
    live, scene, _applied = live_module
    controller = live.LiveStreamController()

    _prepare_external(controller, scene, MODEL_CHANNELS)
    with pytest.raises(live.LiveStreamError, match="weights must be a JSON array"):
        controller.receive(
            "stream-1",
            0,
            tuple([0.0] * len(MODEL_CHANNELS)),
            MODEL_EMOTIONS.copy(),
        )

    with pytest.raises(live.LiveStreamError, match="emotions must be a JSON array"):
        controller.receive(
            "stream-1",
            0,
            [0.0] * len(MODEL_CHANNELS),
            tuple(MODEL_EMOTIONS),
        )


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
        tuple(MODEL_CHANNELS),
        tuple(MODEL_EMOTION_CHANNELS),
        audio_path=None,
        playback_started=None,
        playback_stopped=lambda natural: stopped.append(
            f"stream-1:{natural}"
        ),
    )
    weights = [0.5] * len(MODEL_CHANNELS)
    controller.receive("stream-1", 1600, weights, MODEL_EMOTIONS.copy())

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
        tuple(MODEL_CHANNELS),
        tuple(MODEL_EMOTION_CHANNELS),
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
    live.configure_playback_position(scene.audio2face, 0.0, 2.0)

    assert controller.tick() is True
    assert controller.active is True
    assert live.playback_position(scene.audio2face) == pytest.approx(2.0)
    assert stopped == []

    controller.mark_terminal("stream-1")

    assert controller.active is True
    assert controller.tick() is False
    assert controller.active is False
    assert handle.stop_calls == 1
    assert stopped == [True]
    assert live.playback_position(scene.audio2face) == pytest.approx(2.0)
    assert live.playback_position_maximum(scene.audio2face) == pytest.approx(2.0)


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
        tuple(MODEL_CHANNELS),
        tuple(MODEL_EMOTION_CHANNELS),
        audio_path=audio_path,
        playback_started=None,
        playback_stopped=None,
    )
    controller._handle = SimpleNamespace(status=True, position=0.5)
    controller._duration = 2.0
    live.configure_playback_position(scene.audio2face, 0.5, 2.0)
    controller._published_position = 0.5
    scene.audio2face.playback_state = "PAUSED"

    assert controller.tick() is True

    assert scene.audio2face.playback_state == "PAUSED"


def test_position_slider_coalesces_edits_without_snapping_back(
    live_module: tuple[ModuleType, _Scene, list[AppliedFrame]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live, scene, _applied = live_module
    now = [10.0]
    monkeypatch.setattr(live.time, "monotonic", lambda: now[0])
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"RIFF")
    seeks: list[tuple[float, bool]] = []
    controller = live.LiveStreamController()
    controller.prepare(
        scene,
        "stream-1",
        16_000,
        tuple(MODEL_CHANNELS),
        tuple(MODEL_EMOTION_CHANNELS),
        audio_path=audio_path,
        playback_started=None,
        playback_seeked=lambda position, paused: seeks.append((position, paused)),
        playback_stopped=None,
    )
    controller._handle = SimpleNamespace(status=True, position=0.25)
    controller._duration = 2.0
    live.configure_playback_position(scene.audio2face, 0.25, 2.0)
    controller._published_position = 0.25
    scene.audio2face.playback_state = "PAUSED"
    scene.audio2face[live.PLAYBACK_POSITION_KEY] = 1.0

    assert controller.tick() is True
    assert seeks == []
    assert live.playback_position(scene.audio2face) == 1.0

    scene.audio2face[live.PLAYBACK_POSITION_KEY] = 1.5
    now[0] += 0.10
    assert controller.tick() is True
    assert seeks == []
    assert live.playback_position(scene.audio2face) == 1.5

    now[0] += live.SEEK_SETTLE_SECONDS
    assert controller.tick() is True

    assert seeks == [(1.5, True)]
    assert live.playback_position(scene.audio2face) == 1.5


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
        tuple(MODEL_CHANNELS),
        tuple(MODEL_EMOTION_CHANNELS),
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
    live.configure_playback_position(scene.audio2face, 0.0, 2.0)

    controller.stop_for_seek(1.25, paused=True)

    assert controller.active is False
    assert scene.audio2face.playback_state == "PAUSED"
    assert live.playback_position(scene.audio2face) == 1.25
    assert scene.audio2face.custom_property_ui[live.PLAYBACK_POSITION_KEY] == {
        "min": 0.0,
        "max": 2.0,
        "soft_min": 0.0,
        "soft_max": 2.0,
        "subtype": "TIME",
        "description": "Seek within the selected audio playback",
    }


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
        tuple(MODEL_CHANNELS),
        tuple(MODEL_EMOTION_CHANNELS),
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
