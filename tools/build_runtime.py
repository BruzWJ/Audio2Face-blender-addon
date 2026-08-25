#!/usr/bin/env python3
"""Route one native Audio2Face runtime build to its platform implementation."""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import runtime_build_common as common


def build_runtime(platform_id: str, work_root: Path) -> Path:
    if platform_id == "windows-x64":
        from build_windows_runtime import build_windows_runtime

        return build_windows_runtime(work_root)
    if platform_id == "linux-x64":
        from build_linux_runtime import build_linux_runtime

        return build_linux_runtime(work_root)
    raise common.BuildError(f"unsupported runtime platform {platform_id!r}")


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one pinned native Audio2Face runtime at "
            "build/runtime/<platform> for extension embedding."
        )
    )
    parser.add_argument(
        "--platform",
        required=True,
        choices=common.SUPPORTED_PLATFORMS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = create_argument_parser().parse_args(argv)
    try:
        work_parent = common.REPOSITORY_ROOT / "build"
        work_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="audio2face-runtime-", dir=work_parent
        ) as temporary:
            work_root = Path(temporary).resolve()
            runtime = build_runtime(arguments.platform, work_root)
            output = common.publish_runtime(runtime, arguments.platform)
    except common.BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Built {arguments.platform} runtime for extension embedding: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
