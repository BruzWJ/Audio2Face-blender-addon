"""Shared text wrapping for Blender layouts."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bpy


def draw_wrapped_label(
    layout: bpy.types.UILayout,
    text: str,
    *,
    width: int,
    icon: str = "NONE",
) -> None:
    """Draw one label row per wrapped segment, including long path tokens."""

    first_segment = True
    for logical_line in text.splitlines() or ("",):
        segments = textwrap.wrap(
            logical_line,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or ("",)
        for segment in segments:
            layout.label(
                text=segment,
                icon=icon if first_segment else "NONE",
            )
            first_segment = False
