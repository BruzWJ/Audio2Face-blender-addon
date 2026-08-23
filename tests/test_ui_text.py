from __future__ import annotations

from audio2face.ui_text import draw_wrapped_label


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
    assert all(label["icon"] == "NONE" for label in layout.labels[1:])
    rendered = "".join(label["text"].replace(" ", "") for label in layout.labels)
    assert rendered == text.replace(" ", "").replace("\n", "")


def test_wrapped_label_preserves_one_short_row() -> None:
    layout = _Layout()

    draw_wrapped_label(layout, "Models ready", width=42, icon="INFO")

    assert layout.labels == [{"text": "Models ready", "icon": "INFO"}]
