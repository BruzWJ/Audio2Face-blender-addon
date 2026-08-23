"""Shared text wrapping for Blender layouts."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bpy


MINIMUM_TEXT_WIDTH = 12
REGION_GUTTER_PIXELS = 48.0
AVERAGE_GLYPH_PIXELS = 7.0


def wrap_width_for_region(
    region_width: int,
    ui_scale: float,
) -> int:
    """Convert the current Blender region width to a readable character budget."""

    usable_pixels = max(
        0.0,
        (float(region_width) / float(ui_scale)) - REGION_GUTTER_PIXELS,
    )
    responsive_width = int(usable_pixels / AVERAGE_GLYPH_PIXELS)
    return max(MINIMUM_TEXT_WIDTH, responsive_width)


def context_wrap_width(context: bpy.types.Context) -> int:
    """Return a character budget that follows the active Blender region."""

    return wrap_width_for_region(
        context.region.width,
        context.preferences.system.ui_scale,
    )


def draw_wrapped_label(
    layout: bpy.types.UILayout,
    text: str,
    *,
    width: int,
    icon: str = "NONE",
) -> None:
    """Draw one label row per wrapped segment, including long path tokens."""

    first_segment = True
    continuation_icon = "BLANK1" if icon != "NONE" else "NONE"
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
                icon=icon if first_segment else continuation_icon,
            )
            first_segment = False
