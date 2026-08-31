from __future__ import annotations

from types import SimpleNamespace

import pytest

from audio2face.animation_bake import (
    ACTION_GROUP_NAME,
    ACTION_OWNER_KEY,
    AnimationBakeError,
    bake_shape_key_actions,
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
    def __init__(self, data_path: str, index: int, group_name: str) -> None:
        self.data_path = data_path
        self.array_index = index
        self.group_name = group_name
        self.keyframe_points = _KeyframePoints()
        self.modifiers: list[str] = []

    def update(self) -> None:
        self.keyframe_points.sort(key=lambda point: point.co[0])  # type: ignore[index]


class _Layers(list[object]):
    def __init__(self, action: _Action) -> None:
        super().__init__()
        self.action = action

    def remove(self, layer: object) -> None:
        super().remove(layer)
        self.action.fcurves.clear()


class _Action:
    def __init__(self, name: str) -> None:
        self.name = name
        self.fcurves: list[_FCurve] = []
        self.layers = _Layers(self)
        self.properties: dict[str, object] = {}

    def __getitem__(self, key: str) -> object:
        return self.properties[key]

    def __setitem__(self, key: str, value: object) -> None:
        self.properties[key] = value

    def get(self, key: str) -> object | None:
        return self.properties.get(key)

    def fcurve_ensure_for_datablock(
        self,
        shape_keys: object,
        data_path: str,
        *,
        index: int,
        group_name: str,
    ) -> _FCurve:
        animation_data = getattr(shape_keys, "animation_data", None)
        if animation_data is None or animation_data.action is not self:
            raise RuntimeError("assign the Action before creating its F-Curves")
        if not self.layers:
            self.layers.append(object())
        for curve in self.fcurves:
            if curve.data_path == data_path and curve.array_index == index:
                return curve
        curve = _FCurve(data_path, index, group_name)
        self.fcurves.append(curve)
        return curve


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
            self.animation_data = SimpleNamespace(action=None)
        return self.animation_data


def _target(shape_keys: _ShapeKeys | None) -> object:
    return SimpleNamespace(data=SimpleNamespace(shape_keys=shape_keys))


def _curves(action: _Action) -> list[_FCurve]:
    return action.fcurves


def _add_point(curve: _FCurve, frame: float, value: float) -> _KeyframePoint:
    point = _KeyframePoint()
    point.co = (frame, value)
    point.interpolation = "BEZIER"
    curve.keyframe_points.append(point)
    return point


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


def test_bake_replaces_an_unowned_active_action_without_mutating_it() -> None:
    shape_keys = _ShapeKeys("Face", "jawOpen")
    artist_action = _Action("Audio2Face Shape Key Bake - Face")
    shape_keys.animation_data = SimpleNamespace(action=artist_action)
    jaw_curve = artist_action.fcurve_ensure_for_datablock(
        shape_keys,
        'key_blocks["jawOpen"].value',
        index=0,
        group_name="Artist Curves",
    )
    before = _add_point(jaw_curve, 1.0, 0.1)
    middle = _add_point(jaw_curve, 11.0, 0.2)
    after = _add_point(jaw_curve, 20.0, 0.3)
    unrelated = artist_action.fcurve_ensure_for_datablock(
        shape_keys,
        'key_blocks["unrelated"].value',
        index=0,
        group_name="Artist Curves",
    )
    unrelated_point = _add_point(unrelated, 11.0, 0.7)
    actions = _Actions((artist_action,))

    plans = plan_bake_targets(("jawOpen",), (_target(shape_keys),))
    baked = bake_shape_key_actions((10, 12), ((0.5,), (0.9,)), plans, actions)

    assert len(baked) == 1
    baked_action = baked[0]
    assert baked_action is not artist_action
    assert actions == [artist_action, baked_action]
    assert shape_keys.animation_data.action is baked_action
    assert baked_action[ACTION_OWNER_KEY] is True
    assert [point.co for point in jaw_curve.keyframe_points] == [
        (1.0, 0.1),
        (11.0, 0.2),
        (20.0, 0.3),
    ]
    assert before in jaw_curve.keyframe_points
    assert middle in jaw_curve.keyframe_points
    assert after in jaw_curve.keyframe_points
    assert unrelated.keyframe_points == [unrelated_point]
    assert [point.co for point in _curves(baked_action)[0].keyframe_points] == [
        (10.0, 0.5),
        (12.0, 0.9),
    ]


def test_bake_creates_linear_curves_in_one_native_action() -> None:
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
    assert action[ACTION_OWNER_KEY] is True
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
    assert {curve.group_name for curve in curves} == {ACTION_GROUP_NAME}
    assert shape_keys.animation_data.action is action  # type: ignore[union-attr]


def test_repeated_bake_reuses_owned_action_and_replaces_all_curves() -> None:
    actions = _Actions()
    shape_keys = _ShapeKeys("Face", "jawOpen", "mouthSmile")
    plans = plan_bake_targets(("jawOpen", "mouthSmile"), (_target(shape_keys),))

    action = bake_shape_key_actions(
        (1, 2),
        ((0.1, 0.3), (0.2, 0.4)),
        plans,
        actions,
    )[0]
    old_jaw_curve = _curves(action)[0]
    old_jaw_curve.modifiers.append("NOISE")
    second_plans = plan_bake_targets(("jawOpen",), (_target(shape_keys),))
    repeated = bake_shape_key_actions((2,), ((0.9,),), second_plans, actions)[0]

    assert actions == [action]
    assert repeated is action
    assert old_jaw_curve not in _curves(action)
    assert len(_curves(action)) == 1
    assert _curves(action)[0].modifiers == []
    assert [point.co for point in _curves(action)[0].keyframe_points] == [
        (2.0, 0.9),
    ]


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
