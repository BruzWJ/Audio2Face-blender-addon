from __future__ import annotations

import pytest

from a2f_blender.frame_stream import sample_linear


def test_sample_linear_holds_edges_and_interpolates() -> None:
    timestamps = [-10, 0, 10]
    weights = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]

    assert sample_linear(timestamps, weights, -20) == (0.0, 1.0)
    assert sample_linear(timestamps, weights, -5) == pytest.approx((0.25, 0.75))
    assert sample_linear(timestamps, weights, 5) == pytest.approx((0.75, 0.25))
    assert sample_linear(timestamps, weights, 20) == (1.0, 0.0)
