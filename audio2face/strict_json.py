"""One duplicate-key rule for every JSON document owned by the add-on."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn


def duplicate_key_hook(
    error_type: type[Exception],
    subject: str,
) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]:
    """Create a JSON object hook that rejects duplicate field names."""

    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise error_type(f"{subject} contains duplicate field {key!r}")
            result[key] = value
        return result

    return reject


def invalid_constant_hook(
    error_type: type[Exception],
    subject: str,
) -> Callable[[str], NoReturn]:
    """Create a JSON number hook that rejects NaN and both infinities."""

    def reject(token: str) -> NoReturn:
        raise error_type(f"{subject} contains invalid number {token}")

    return reject
