from __future__ import annotations

import re
from pathlib import Path

from audio2face.arkit import ARKIT_52_CHANNELS


EXPECTED_ARKIT_52 = (
    "EyeBlinkLeft",
    "EyeLookDownLeft",
    "EyeLookInLeft",
    "EyeLookOutLeft",
    "EyeLookUpLeft",
    "EyeSquintLeft",
    "EyeWideLeft",
    "EyeBlinkRight",
    "EyeLookDownRight",
    "EyeLookInRight",
    "EyeLookOutRight",
    "EyeLookUpRight",
    "EyeSquintRight",
    "EyeWideRight",
    "JawForward",
    "JawLeft",
    "JawRight",
    "JawOpen",
    "MouthClose",
    "MouthFunnel",
    "MouthPucker",
    "MouthLeft",
    "MouthRight",
    "MouthSmileLeft",
    "MouthSmileRight",
    "MouthFrownLeft",
    "MouthFrownRight",
    "MouthDimpleLeft",
    "MouthDimpleRight",
    "MouthStretchLeft",
    "MouthStretchRight",
    "MouthRollLower",
    "MouthRollUpper",
    "MouthShrugLower",
    "MouthShrugUpper",
    "MouthPressLeft",
    "MouthPressRight",
    "MouthLowerDownLeft",
    "MouthLowerDownRight",
    "MouthUpperUpLeft",
    "MouthUpperUpRight",
    "BrowDownLeft",
    "BrowDownRight",
    "BrowInnerUp",
    "BrowOuterUpLeft",
    "BrowOuterUpRight",
    "CheekPuff",
    "CheekSquintLeft",
    "CheekSquintRight",
    "NoseSneerLeft",
    "NoseSneerRight",
    "TongueOut",
)


def test_canonical_pascal_case_order_is_stable() -> None:
    assert ARKIT_52_CHANNELS == EXPECTED_ARKIT_52
    assert len(ARKIT_52_CHANNELS) == 52
    assert len(set(ARKIT_52_CHANNELS)) == 52


def test_python_and_native_worker_use_the_same_channel_contract() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "worker" / "src" / "a2f_backend.cpp"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"kArkit52SdkNames\s*=\s*\{\{(?P<items>.*?)\}\};",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    sdk_names = tuple(re.findall(r'"([A-Za-z0-9]+)"', match.group("items")))

    assert sdk_names == tuple(name[0].lower() + name[1:] for name in ARKIT_52_CHANNELS)
