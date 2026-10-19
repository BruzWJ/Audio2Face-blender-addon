from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from audio2face.ui_text import draw_wrapped_label, wrap_width_for_region


class _Layout:
    def __init__(self) -> None:
        self.labels: list[dict[str, str]] = []

    def label(self, **values: str) -> None:
        self.labels.append(values)


def test_wrapped_label_splits_prose_paths_and_explicit_lines() -> None:
    layout = _Layout()
    text = "Runtime is missing\nC:/Users/example/very-long-runtime/network.trt"

    draw_wrapped_label(layout, text, width=12, icon="ERROR")

    assert len(layout.labels) > 2
    assert all(len(label["text"]) <= 12 for label in layout.labels)
    assert layout.labels[0]["icon"] == "ERROR"
    assert all(label["icon"] == "BLANK1" for label in layout.labels[1:])
    rendered = "".join(label["text"].replace(" ", "") for label in layout.labels)
    assert rendered == text.replace(" ", "").replace("\n", "")


def test_wrapped_label_preserves_one_short_row() -> None:
    layout = _Layout()

    draw_wrapped_label(layout, "Models ready", width=42, icon="INFO")

    assert layout.labels == [{"text": "Models ready", "icon": "INFO"}]


def test_region_width_reflows_text_responsively() -> None:
    narrow_width = wrap_width_for_region(240, 1.0)
    wide_width = wrap_width_for_region(480, 1.0)
    narrow = _Layout()
    wide = _Layout()
    text = "Worker diagnostics reflow whenever the Blender region is resized"

    draw_wrapped_label(narrow, text, width=narrow_width)
    draw_wrapped_label(wide, text, width=wide_width)

    assert narrow_width < wide_width
    assert len(narrow.labels) > len(wide.labels)


@pytest.fixture
def ui_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    bpy = ModuleType("bpy")
    bpy.types = SimpleNamespace(Menu=object, UIList=object, Panel=object)
    properties = ModuleType("audio2face.properties")
    properties.AUDIO2FACE_SETTING_GROUPS = ()
    runtime = ModuleType("audio2face.runtime")
    runtime.RuntimeController = object
    runtime.get_controller = lambda: None
    for module in (bpy, properties, runtime):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    name = "audio2face._ui_test"
    path = Path(__file__).resolve().parents[1] / "audio2face" / "ui.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _channel_scene(*used: int, selected: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        audio2face=SimpleNamespace(audio_channel=selected),
        sequence_editor=SimpleNamespace(
            strips=[SimpleNamespace(channel=number) for number in used],
            channels=[
                SimpleNamespace(number=number, name=f"Channel {number}")
                for number in range(129)
            ],
        ),
    )


def test_channel_dropdown_keeps_empty_selection_without_listing_unused_defaults(ui_module: ModuleType) -> None:
    scene = _channel_scene(selected=5)
    assert ui_module._audio_channel_choices(scene) == ((5, "Channel 5"),)
    scene.sequence_editor = None
    assert ui_module._audio_channel_choices(scene) == ((5, "Channel 5"),)
    assert scene.sequence_editor is None


def test_channel_dropdown_tracks_used_channels_names_and_deleted_strips(ui_module: ModuleType) -> None:
    scene = _channel_scene(7, 3, 7, selected=5)
    editor = scene.sequence_editor
    editor.channels[0].name = "Internal channel"
    editor.channels[2].name = "Ambience"
    assert ui_module._audio_channel_choices(scene) == (
        (2, "2: Ambience"), (3, "Channel 3"), (5, "Channel 5"), (7, "Channel 7"),
    )
    editor.strips[0].channel = 4
    editor.strips.pop()
    editor.channels[2].name = "Dialogue"
    assert ui_module._audio_channel_choices(scene) == (
        (2, "2: Dialogue"), (3, "Channel 3"), (4, "Channel 4"), (5, "Channel 5"),
    )
    scene.audio2face.audio_channel = 4
    editor.strips.clear()
    assert ui_module._audio_channel_choices(scene) == ((2, "2: Dialogue"), (4, "Channel 4"))


def test_channel_menu_writes_existing_integer_property_through_blender_operator(ui_module: ModuleType) -> None:
    scene = _channel_scene(3, 7, selected=3)
    choices = []

    def operator(identifier: str, **kwargs: str) -> SimpleNamespace:
        choice = SimpleNamespace(identifier=identifier, **kwargs)
        choices.append(choice)
        return choice

    menu = ui_module.A2F_MT_audio_channels()
    menu.layout = SimpleNamespace(operator=operator)
    menu.draw(SimpleNamespace(scene=scene))
    assert [vars(choice) for choice in choices] == [
        {
            "identifier": "wm.context_set_int", "text": "Channel 3",
            "icon": "RADIOBUT_ON", "data_path": "scene.audio2face.audio_channel", "value": 3,
        },
        {
            "identifier": "wm.context_set_int", "text": "Channel 7",
            "icon": "RADIOBUT_OFF", "data_path": "scene.audio2face.audio_channel", "value": 7,
        },
    ]
    assert scene.audio2face.audio_channel == 3
