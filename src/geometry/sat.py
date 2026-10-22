"""Separating Axis Theorem (SAT) geometry engine for unit squares in 2D.

Mathematically exact collision detection and enclosing square calculations
for N congruent unit squares with arbitrary positions and orientations.
"""

from __future__ import annotations

import math
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.spatial import cKDTree

SQRT2 = math.sqrt(2.0)
EPSILON_DEFAULT = 1e-10


def compute_half_extents(theta: np.ndarray) -> np.ndarray:
    """Compute axis-aligned half-extent for unit squares rotated by theta.
    
    For a unit square rotated by theta, its projections onto the X and Y axes
    are both equal to 0.5 * (|cos(theta)| + |sin(theta)|).
    """
    return 0.5 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))


def compute_bounding_box(coords: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute (xmin, xmax, ymin, ymax) of the tight enclosing axis-aligned bounding box."""
    if len(coords) == 0:
        return 0.0, 0.0, 0.0, 0.0
    h = compute_half_extents(coords[:, 2])
    xmin = float(np.min(coords[:, 0] - h))
    xmax = float(np.max(coords[:, 0] + h))
    ymin = float(np.min(coords[:, 1] - h))
    ymax = float(np.max(coords[:, 1] + h))
    return xmin, xmax, ymin, ymax


def compute_enclosing_side(coords: np.ndarray) -> float:
    """Compute the minimum side length of an axis-aligned container square."""
    xmin, xmax, ymin, ymax = compute_bounding_box(coords)
    return max(xmax - xmin, ymax - ymin)


def get_square_corners(x: float, y: float, theta: float) -> np.ndarray:
    """Compute the 4 corner coordinates of a unit square at (x, y) with angle theta.
    
    Returns 4x2 array of corners in counter-clockwise order.
    """
    c = math.cos(theta) * 0.5
    s = math.sin(theta) * 0.5
    # Corners: (x + c - s, y + s + c), (x - c - s, y - s + c),
    #          (x - c + s, y - s - c), (x + c + s, y + s - c)
    return np.array([
        [x + c - s, y + s + c],
        [x - c - s, y - s + c],
        [x - c + s, y - s - c],
        [x + c + s, y + s - c]
    ], dtype=np.float64)


def pairwise_sat_vectorized(
    coords: np.ndarray,
    pairs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute separating margin for given pairs of unit squares using SAT.
    
    Returns:
        gaps: 1D array of separation margins (positive = separated, negative = penetrating)
        critical_axes: 1D array indicating which of the 4 candidate axes achieved the max separation
    """
    if len(pairs) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int32)
        
    i = pairs[:, 0]
    j = pairs[:, 1]
    
    dx = coords[j, 0] - coords[i, 0]
    dy = coords[j, 1] - coords[i, 1]
    
    ti = coords[i, 2]
    tj = coords[j, 2]
    
    ci = np.cos(ti)
    si = np.sin(ti)
    cj = np.cos(tj)
    sj = np.sin(tj)
    
    delta = ti - tj
    # Combined support along edge normals: 0.5 * (1 + |cos(delta)| + |sin(delta)|)
    support = 0.5 * (1.0 + np.abs(np.cos(delta)) + np.abs(np.sin(delta)))
    
    # 4 candidate separating axes (normals of square i and square j)
    # Axis 0: normal 1 of square i (ci, si)
    # Axis 1: normal 2 of square i (-si, ci)
    # Axis 2: normal 1 of square j (cj, sj)
    # Axis 3: normal 2 of square j (-sj, cj)
    sep0 = np.abs(dx * ci + dy * si) - support
    sep1 = np.abs(-dx * si + dy * ci) - support
    sep2 = np.abs(dx * cj + dy * sj) - support
    sep3 = np.abs(-dx * sj + dy * cj) - support
    
    all_seps = np.stack([sep0, sep1, sep2, sep3], axis=1)
    crit_axis = np.argmax(all_seps, axis=1)
    gaps = all_seps[np.arange(len(pairs)), crit_axis]
    
    return gaps, crit_axis


def validate_packing_sat(
    coords: np.ndarray,
    eps: float = EPSILON_DEFAULT,
    exhaustive: bool = False
) -> Dict[str, Any]:
    """Validate a candidate packing of N squares.
    
    Args:
        coords: (N, 3) array [x, y, theta]
        eps: tolerance for overlap (strict: 1e-10)
        exhaustive: if True, check all N*(N-1)/2 pairs instead of KD-tree pruning
        
    Returns:
        dict with validation results:
            - valid: bool
            - n: int
            - side: float
            - score: float (side / sqrt(n) - 1)
            - min_margin: float
            - num_pairs_checked: int
            - overlaps_found: int
            - error_message: Optional[str]
    """
    n = len(coords)
    if n == 0 or coords.shape != (n, 3) or not np.isfinite(coords).all():
        return {
            "valid": False,
            "n": n,
            "side": float("inf"),
            "score": float("inf"),
            "min_margin": -float("inf"),
            "num_pairs_checked": 0,
            "overlaps_found": 0,
            "error_message": "Invalid or non-finite coordinate array"
        }
        
    side = compute_enclosing_side(coords)
    area_lower_bound = math.sqrt(n)
    if side + 1e-8 < area_lower_bound:
        return {
            "valid": False,
            "n": n,
            "side": side,
            "score": side / area_lower_bound - 1.0,
            "min_margin": -1.0,
            "num_pairs_checked": 0,
            "overlaps_found": 1,
            "error_message": f"Container side {side:.12f} violates area lower bound {area_lower_bound:.12f}"
        }
        
    if exhaustive or n <= 10:
        # Generate all pairs
        i_idx, j_idx = np.triu_indices(n, k=1)
        pairs = np.stack([i_idx, j_idx], axis=1)
    else:
        # Spatial indexing: centers further than sqrt(2) cannot collide
        # Use sqrt(2) + 1e-6 as safety radius
        tree = cKDTree(coords[:, :2])
        pairs = tree.query_pairs(r=SQRT2 + 1e-6, output_type='ndarray')
        
    if len(pairs) == 0:
        min_margin = float("inf")
        overlaps = 0
    else:
        gaps, _ = pairwise_sat_vectorized(coords, pairs)
        min_margin = float(np.min(gaps))
        bad_mask = gaps < -eps
        overlaps = int(np.sum(bad_mask))
        if overlaps > 0:
            first_bad = np.where(bad_mask)[0][0]
            sq_i = int(pairs[first_bad, 0])
            sq_j = int(pairs[first_bad, 1])
            penetration = -float(gaps[first_bad])
            return {
                "valid": False,
                "n": n,
                "side": side,
                "score": side / area_lower_bound - 1.0,
                "min_margin": min_margin,
                "num_pairs_checked": len(pairs),
                "overlaps_found": overlaps,
                "error_message": f"Overlap between squares {sq_i} and {sq_j} by {penetration:.3e} > {eps:.1e}"
            }
            
    return {
        "valid": True,
        "n": n,
        "side": side,
        "score": side / area_lower_bound - 1.0,
        "min_margin": min_margin,
        "num_pairs_checked": len(pairs),
        "overlaps_found": 0,
        "error_message": None
    }
