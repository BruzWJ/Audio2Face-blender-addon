"""Canonical lexical paths and filesystem-alias rejection."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any


def require_unaliased_path(
    value: Any,
    *,
    description: str,
    error_type: type[Exception],
) -> Path:
    """Return an existing lexical absolute path with no aliased component."""

    try:
        spelling = os.fspath(value)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise error_type(f"{description} path is invalid") from exc
    if type(spelling) is not str or not spelling or "\0" in spelling:
        raise error_type(f"{description} path is invalid")
    canonical_spelling = os.path.abspath(spelling)
    if os.name != "nt" and canonical_spelling.startswith("//"):
        raise error_type(f"{description} must be one canonical absolute path")
    if spelling != canonical_spelling:
        raise error_type(f"{description} must be one canonical absolute path")
    absolute = Path(spelling)
    for component in reversed((absolute, *absolute.parents)):
        try:
            details = component.lstat()
        except OSError as exc:
            raise error_type(
                f"{description} is missing or inaccessible: {absolute}"
            ) from exc
        is_reparse_point = bool(
            os.name == "nt"
            and details.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
        )
        if stat.S_ISLNK(details.st_mode) or is_reparse_point:
            raise error_type(
                f"{description} must not use a filesystem alias: {component}"
            )
    return absolute
