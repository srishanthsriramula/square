"""Fault-line Gadget Dynamic Programming (DP) Composer.

Implements Chris Deotte's multi-gadget fault-line composition:
Partitions an enclosing square of side K + f into two continents:
1. Lower-left continent: upright unit squares on integer grid.
2. Upper-right continent: upright unit squares shifted by (f, f).
3. Fault-line interface: gadgets of size w x h with surplus dm and overhang f.

Given a catalog of gadgets (crops and dense packings), uses DP to find
the composition that minimizes f for each N = K^2 + D.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.geometry.sat import (
    compute_half_extents,
    compute_enclosing_side,
    validate_packing_sat,
    EPSILON_DEFAULT
)


class Gadget:
    def __init__(
        self,
        name: str,
        w: int,
        h: int,
        dm: int,
        f: float,
        coords: np.ndarray,
        source_n: int
    ):
        self.name = name
        self.w = w
        self.h = h
        self.dm = dm
        self.f = f
        self.coords = coords.copy()
        self.source_n = source_n

    def __repr__(self):
        return f"Gadget({self.name}: {self.w}x{self.h}, dm=+{self.dm}, f={self.f:.6f})"


def extract_full_packings_as_gadgets(packings: Dict[int, np.ndarray]) -> List[Gadget]:
    """Convert whole packings that have surplus into gadgets."""
    gadgets = []
    for n, coords in packings.items():
        if n < 4:
            continue
        h_ext = compute_half_extents(coords[:, 2])
        xmin, ymin = np.min(coords[:, 0] - h_ext), np.min(coords[:, 1] - h_ext)
        xmax, ymax = np.max(coords[:, 0] + h_ext), np.max(coords[:, 1] + h_ext)
        W, H = xmax - xmin, ymax - ymin
        w, h_int = int(math.floor(W)), int(math.floor(H))
        
        dm = n - w * h_int
        if dm > 0:
            fx = W - w
            fy = H - h_int
            f = max(fx, fy)
            if f < 0.999999:
                norm_coords = coords.copy()
                norm_coords[:, 0] -= xmin
                norm_coords[:, 1] -= ymin
                gadgets.append(Gadget(
                    name=f"full_n{n}",
                    w=w,
                    h=h_int,
                    dm=dm,
                    f=f,
                    coords=norm_coords,
                    source_n=n
                ))
    return gadgets


def extract_crops_from_donors(
    packings: Dict[int, np.ndarray],
    donor_ns: Optional[List[int]] = None,
    max_w: int = 15,
    max_h: int = 15
) -> List[Gadget]:
    """Extract rectangular window crops that contain surplus squares."""
    if donor_ns is None:
        # High density and diverse rotated packings
        donor_ns = [
            5, 11, 14, 18, 26, 28, 40, 50, 65, 82, 85, 105, 126,
            147, 170, 207, 265, 301, 372, 451, 602, 641, 653, 699
        ]
        
    crops: List[Gadget] = []
    
    for n in donor_ns:
        if n not in packings:
            continue
        coords = packings[n]
        h_ext = compute_half_extents(coords[:, 2])
        xmin, ymin = float(np.min(coords[:, 0] - h_ext)), float(np.min(coords[:, 1] - h_ext))
        xmax, ymax = float(np.max(coords[:, 0] + h_ext)), float(np.max(coords[:, 1] + h_ext))
        total_W = xmax - xmin
        total_H = ymax - ymin
        
        cx = coords[:, 0] - xmin
        cy = coords[:, 1] - ymin
        
        for w in range(2, min(max_w + 1, int(math.floor(total_W)))):
            for h_dim in range(2, min(max_h + 1, int(math.floor(total_H)))):
                for f_target in [0.5, 0.70710678, 0.75, 0.8, 0.85, 0.9, 0.95]:
                    W = w + f_target
                    H = h_dim + f_target
                    if W > total_W or H > total_H:
                        continue
                    
                    found_crop = False
                    for x0 in np.arange(0.0, total_W - W + 1e-4, 1.0):
                        for y0 in np.arange(0.0, total_H - H + 1e-4, 1.0):
                            inside = (
                                (cx - h_ext >= x0) &
                                (cx + h_ext <= x0 + W) &
                                (cy - h_ext >= y0) &
                                (cy + h_ext <= y0 + H)
                            )
                            M = int(np.sum(inside))
                            dm = M - w * h_dim
                            if dm >= 1:
                                sub = coords[inside].copy()
                                sub_h = h_ext[inside]
                                sub_xmin = np.min(sub[:, 0] - sub_h)
                                sub_ymin = np.min(sub[:, 1] - sub_h)
                                sub[:, 0] -= sub_xmin
                                sub[:, 1] -= sub_ymin
                                act_W = np.max(sub[:, 0] + sub_h)
                                act_H = np.max(sub[:, 1] + sub_h)
                                act_f = max(act_W - w, act_H - h_dim)
                                
                                crops.append(Gadget(
                                    name=f"crop_n{n}_{w}x{h_dim}_dm{dm}",
                                    w=w,
                                    h=h_dim,
                                    dm=dm,
                                    f=act_f,
                                    coords=sub,
                                    source_n=n
                                ))
                                found_crop = True
                                break
                        if found_crop:
                            break
                            
    return crops


def extract_all_gadgets(packings: Dict[int, np.ndarray]) -> List[Gadget]:
    """Extract both full packings and crop gadgets, pruned to Pareto-optimal."""
    full_g = extract_full_packings_as_gadgets(packings)
    crop_g = extract_crops_from_donors(packings)
    all_g = full_g + crop_g
    
    # Pareto prune: for each (w, h, dm), keep the one with minimal f
    best_by_whd: Dict[Tuple[int, int, int], Gadget] = {}
    for g in all_g:
        key = (g.w, g.h, g.dm)
        if key not in best_by_whd or g.f < best_by_whd[key].f:
            best_by_whd[key] = g
            
    # Also add transposed (h, w) under 90-degree rotation if beneficial
    result = list(best_by_whd.values())
    return result


def run_fault_line_dp(
    gadgets: List[Gadget],
    max_k: int = 32,
    max_f: float = 0.99
) -> Dict[Tuple[int, int], Tuple[float, List[Gadget]]]:
    """Run DP to find optimal gadget sequences for each (K, D).
    
    Returns:
        {(K, D): (best_f, list_of_gadgets)}
    """
    best_for_kd: Dict[Tuple[int, int], Tuple[float, List[Gadget]]] = {}
    usable_gadgets = [g for g in gadgets if g.f <= max_f and g.w <= max_k and g.h <= max_k]

    for K in range(2, max_k + 1):
        dp: Dict[Tuple[int, int, int], Tuple[float, List[Gadget]]] = {
            (0, 0, 0): (0.0, [])
        }

        cand_g = [g for g in usable_gadgets if g.w <= K and g.h <= K]

        # Single-gadget compositions
        for g in cand_g:
            state = (g.w, g.h, g.dm)
            if state not in dp or g.f < dp[state][0]:
                dp[state] = (g.f, [g])
                
        # Two-gadget compositions
        current_states = list(dp.items())
        for (c, r, d), (f_val, glist) in current_states:
            if len(glist) >= 3:
                continue
            for g in cand_g:
                new_c = c + g.w + 1
                new_r = r + g.h + 1
                new_d = d + g.dm
                new_f = max(f_val, g.f)
                
                if new_c <= K and new_r <= K and new_f <= max_f:
                    st = (new_c, new_r, new_d)
                    if st not in dp or new_f < dp[st][0]:
                        dp[st] = (new_f, glist + [g])

        for (c, r, d), (f_val, glist) in dp.items():
            if d > 0:
                key = (K, d)
                if key not in best_for_kd or f_val < best_for_kd[key][0]:
                    best_for_kd[key] = (f_val, glist)

    return best_for_kd


def compose_packing(
    K: int,
    D: int,
    f: float,
    gadget_chain: List[Gadget]
) -> np.ndarray:
    """Build concrete coordinates for N = K^2 + D from fault-line composition."""
    N = K * K + D
    squares = []
    
    gadget_boxes = []
    curr_c = 0
    curr_r = K
    for g in gadget_chain:
        curr_r -= g.h
        gadget_boxes.append((curr_c, curr_r, g.w, g.h, g))
        curr_c += g.w + 1
        curr_r -= 1

    for (C, R, w, h, g) in gadget_boxes:
        for sq in g.coords:
            squares.append([sq[0] + C, sq[1] + R, sq[2]])

    for i in range(K):
        for j in range(K):
            inside_gadget = False
            for (C, R, w, h, g) in gadget_boxes:
                if C <= i < C + w and R <= j < R + h:
                    inside_gadget = True
                    break
            if inside_gadget:
                continue

            is_lower = True
            for (C, R, w, h, g) in gadget_boxes:
                if i >= C and j >= R:
                    is_lower = False
                    break
            
            if is_lower:
                squares.append([i + 0.5, j + 0.5, 0.0])
            else:
                squares.append([i + f + 0.5, j + f + 0.5, 0.0])

    arr = np.array(squares, dtype=np.float64)
    return arr
