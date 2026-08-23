from __future__ import annotations

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
