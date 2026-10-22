"""Test suite for SAT geometry engine."""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np

from src.geometry.sat import (
    compute_half_extents,
    compute_bounding_box,
    compute_enclosing_side,
    get_square_corners,
    pairwise_sat_vectorized,
    validate_packing_sat
)


def test_single_square():
    # Square at (0, 0) with angle 0
    coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    res = validate_packing_sat(coords)
    assert res["valid"] is True
    assert abs(res["side"] - 1.0) < 1e-12
    assert abs(res["score"] - 0.0) < 1e-12


def test_two_squares_touching_edge():
    # Two axis-aligned squares touching along vertical edge: x=0.5 and x=1.5
    coords = np.array([
        [0.5, 0.5, 0.0],
        [1.5, 0.5, 0.0]
    ], dtype=np.float64)
    res = validate_packing_sat(coords, eps=1e-10)
    assert res["valid"] is True
    assert abs(res["min_margin"]) < 1e-12
    assert abs(res["side"] - 2.0) < 1e-12


def test_two_squares_overlapping():
    # Two axis-aligned squares overlapping slightly: x=0.5 and x=1.49
    coords = np.array([
        [0.5, 0.5, 0.0],
        [1.49, 0.5, 0.0]
    ], dtype=np.float64)
    res = validate_packing_sat(coords, eps=1e-10)
    assert res["valid"] is False
    assert res["overlaps_found"] > 0
    assert res["min_margin"] < -0.009


def test_corner_touching_rotated():
    # Square at (0, 0) angle 0: corner at (0.5, 0.5)
    # Square rotated 45 deg centered at (0.5 + 0.5/sqrt(2), 0.5 + 0.5/sqrt(2))
    # Its corner touches (0.5, 0.5)
    diag = 0.5 * math.sqrt(2.0)
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.5 + diag, 0.5 + diag, math.pi / 4]
    ], dtype=np.float64)
    res = validate_packing_sat(coords, eps=1e-10)
    assert res["valid"] is True
    assert res["min_margin"] >= -1e-10


def test_n5_diamond_construction():
    # n=5 diamond packing: 4 squares at corners + 1 diamond at center
    # Container side = 2 + 1/sqrt(2) approx 2.70710678
    R = 1.0 / math.sqrt(2.0)
    k = 2
    coords = [
        [0.5, 0.5, 0.0],
        [1.5 + R, 0.5, 0.0],
        [0.5, 1.5 + R, 0.0],
        [1.5 + R, 1.5 + R, 0.0],
        [1.0 + R / 2.0, 1.0 + R / 2.0, math.pi / 4]
    ]
    arr = np.array(coords, dtype=np.float64)
    res = validate_packing_sat(arr, eps=1e-10, exhaustive=True)
    assert res["valid"] is True
    expected_side = 2.0 + R
    assert abs(res["side"] - expected_side) < 1e-10


if __name__ == "__main__":
    test_single_square()
    test_two_squares_touching_edge()
    test_two_squares_overlapping()
    test_corner_touching_rotated()
    test_n5_diamond_construction()
    print("All SAT tests passed successfully!")
