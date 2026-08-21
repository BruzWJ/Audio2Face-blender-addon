from __future__ import annotations

import os
from pathlib import Path

import pytest

from audio2face.path_contract import require_unaliased_path


@pytest.mark.parametrize("spelling", ["dot", "repeated", "trailing"])
def test_existing_paths_require_one_exact_lexical_spelling(
    tmp_path: Path,
    spelling: str,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    if spelling == "dot":
        value = f"{tmp_path}{os.sep}.{os.sep}target"
    elif spelling == "repeated":
        value = f"{tmp_path}{os.sep}{os.sep}target"
    else:
        value = f"{target}{os.sep}"

    with pytest.raises(ValueError, match="canonical absolute path"):
        require_unaliased_path(
            value,
            description="test target",
            error_type=ValueError,
        )


def test_path_contract_rejects_bytes_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path is invalid"):
        require_unaliased_path(
            os.fsencode(tmp_path),
            description="test target",
            error_type=ValueError,
        )
