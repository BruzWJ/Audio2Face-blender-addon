"""Pure timing helpers for replaying buffered animation frames as a stream."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence


def sample_linear(
    timestamps_samples: Sequence[int],
    weights: Sequence[Sequence[float]],
    sample_position: float,
) -> tuple[float, ...]:
    """Sample a result-loader-validated frame matrix by its timestamp clock."""

    if sample_position <= timestamps_samples[0]:
        return tuple(float(value) for value in weights[0])
    if sample_position >= timestamps_samples[-1]:
        return tuple(float(value) for value in weights[-1])

    upper = bisect_right(timestamps_samples, sample_position)
    lower = upper - 1
    start = timestamps_samples[lower]
    end = timestamps_samples[upper]
    mix = (sample_position - start) / (end - start)
    return tuple(
        float(left) + (float(right) - float(left)) * mix
        for left, right in zip(weights[lower], weights[upper])
    )
