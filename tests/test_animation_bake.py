from __future__ import annotations

from types import SimpleNamespace

import pytest

from audio2face.animation_bake import (
    ACTION_LAYER_NAME,
    ACTION_OWNER_KEY,
    ACTION_OWNER_VALUE,
    AnimationBakeError,
    bake_shape_key_actions,
    is_addon_bake_action,
    plan_bake_targets,
)


class _KeyframePoint:
    def __init__(self) -> None:
        self.co: tuple[float, float] | None = None
        self.interpolation = ""


class _KeyframePoints(list[_KeyframePoint]):
    def add(self, count: int) -> None:
        self.extend(_KeyframePoint() for _index in range(count))


class _FCurve:
    def __init__(self, data_path: str, index: int) -> None:
        self.data_path = data_path
        self.array_index = index
        self.keyframe_points = _KeyframePoints()

    def update(self) -> None:
        pass


class _FCurves(list[_FCurve]):
    def new(self, *, data_path: str, index: int) -> _FCurve:
        curve = _FCurve(data_path, index)
        self.append(curve)
        return curve


class _Channelbag:
    def __init__(self) -> None:
        self.fcurves = _FCurves()


class _Strip:
    def __init__(self, strip_type: str) -> None:
        self.type = strip_type
        self.channelbags: list[_Channelbag] = []

    def channelbag(self, _slot: object, *, ensure: bool) -> _Channelbag:
        assert ensure is True
        channelbag = _Channelbag()
        self.channelbags.append(channelbag)
        return channelbag


class _Strips(list[_Strip]):
    def new(self, *, type: str) -> _Strip:
        strip = _Strip(type)
        self.append(strip)
        return strip


class _Layer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.strips = _Strips()


class _Layers(list[_Layer]):
    def new(self, *, name: str) -> _Layer:
        layer = _Layer(name)
        self.append(layer)
        return layer


class _Slot:
    def __init__(self, id_type: str, name: str) -> None:
        self.target_id_type = id_type
        self.name_display = name


class _Slots(list[_Slot]):
    def new(self, *, id_type: str, name: str) -> _Slot:
        slot = _Slot(id_type, name)
        self.append(slot)
        return slot


class _Action(dict[str, object]):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.slots = _Slots()
        self.layers = _Layers()


class _Actions(list[_Action]):
    def new(self, *, name: str) -> _Action:
        base = name
        suffix = 1
        existing = {action.name for action in self}
        while name in existing:
            name = f"{base}.{suffix:03d}"
            suffix += 1
        action = _Action(name)
        self.append(action)
        return action


class _ShapeKey:
    def __init__(self, name: str) -> None:
        self.name = name

    def path_from_id(self, property_name: str) -> str:
        assert property_name == "value"
        return f'key_blocks["{self.name}"].value'


class _ShapeKeys:
    def __init__(self, name: str, *channels: str) -> None:
        self.name = name
        self.key_blocks = {channel: _ShapeKey(channel) for channel in channels}
        self.animation_data: SimpleNamespace | None = None

    def as_pointer(self) -> int:
        return id(self)

    def animation_data_create(self) -> SimpleNamespace:
        if self.animation_data is None:
            self.animation_data = SimpleNamespace(action=None, action_slot=None)
        return self.animation_data


def _target(shape_keys: _ShapeKeys | None) -> object:
    return SimpleNamespace(data=SimpleNamespace(shape_keys=shape_keys))


def _curves(action: _Action) -> _FCurves:
    return action.layers[0].strips[0].channelbags[0].fcurves


def test_plan_deduplicates_shared_keys_and_skips_unmatched_targets() -> None:
    shared = _ShapeKeys("Shared", "jawOpen")
    unmatched = _ShapeKeys("Unmatched", "other")

    plans = plan_bake_targets(
        ("jawOpen", "missing"),
        (_target(shared), _target(shared), _target(unmatched), _target(None)),
    )

    assert len(plans) == 1
    assert plans[0].shape_keys is shared
    assert plans[0].channels == ((0, 'key_blocks["jawOpen"].value'),)


def test_plan_rejects_an_artist_action_before_baking() -> None:
    shape_keys = _ShapeKeys("Face", "jawOpen")
    shape_keys.animation_data = SimpleNamespace(
        action=_Action("Artist Action"),
        action_slot=object(),
    )

    with pytest.raises(AnimationBakeError, match="non-Audio2Face active Action"):
        plan_bake_targets(("jawOpen",), (_target(shape_keys),))


def test_bake_builds_layered_linear_action_and_assigns_its_slot() -> None:
    actions = _Actions()
    shape_keys = _ShapeKeys("Face Keys", "jawOpen", "mouthSmile")
    plans = plan_bake_targets(
        ("jawOpen", "mouthSmile", "missing"),
        (_target(shape_keys),),
    )

    result = bake_shape_key_actions(
        (10, 12, 15),
        ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), (0.7, 0.8, 0.9)),
        plans,
        actions,
    )

    assert result == tuple(actions)
    action = actions[0]
    assert action[ACTION_OWNER_KEY] == ACTION_OWNER_VALUE
    assert is_addon_bake_action(action)
    assert action.layers[0].name == ACTION_LAYER_NAME
    assert action.layers[0].strips[0].type == "KEYFRAME"
    slot = action.slots[0]
    assert (slot.target_id_type, slot.name_display) == ("KEY", "Face Keys")
    curves = _curves(action)
    assert [curve.data_path for curve in curves] == [
        'key_blocks["jawOpen"].value',
        'key_blocks["mouthSmile"].value',
    ]
    assert [point.co for point in curves[0].keyframe_points] == [
        (10.0, 0.1),
        (12.0, 0.4),
        (15.0, 0.7),
    ]
    assert all(
        point.interpolation == "LINEAR"
        for curve in curves
        for point in curve.keyframe_points
    )
    assert shape_keys.animation_data.action is action  # type: ignore[union-attr]
    assert shape_keys.animation_data.action_slot is slot  # type: ignore[union-attr]


def test_repeated_bake_preserves_the_previous_owned_action() -> None:
    actions = _Actions()
    shape_keys = _ShapeKeys("Face", "jawOpen")
    plans = plan_bake_targets(("jawOpen",), (_target(shape_keys),))

    first = bake_shape_key_actions((1,), ((0.1,),), plans, actions)[0]
    second_plans = plan_bake_targets(("jawOpen",), (_target(shape_keys),))
    second = bake_shape_key_actions((2,), ((0.9,),), second_plans, actions)[0]

    assert actions == [first, second]
    assert shape_keys.animation_data.action is second  # type: ignore[union-attr]
    assert [point.co for point in _curves(first)[0].keyframe_points] == [(1.0, 0.1)]


@pytest.mark.parametrize(
    ("frames", "weights", "message"),
    [
        ((), (), "nonempty and aligned"),
        ((1, 2), ((0.5,),), "nonempty and aligned"),
        ((1,), ((),), "do not match"),
    ],
)
def test_bake_rejects_misaligned_internal_results(
    frames: tuple[int, ...],
    weights: tuple[tuple[float, ...], ...],
    message: str,
) -> None:
    plans = plan_bake_targets(
        ("jawOpen",),
        (_target(_ShapeKeys("Face", "jawOpen")),),
    )
    with pytest.raises(AnimationBakeError, match=message):
        bake_shape_key_actions(frames, weights, plans, _Actions())
