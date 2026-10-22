"""Extract crop gadgets from dense packing configurations."""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.scoring.registry import PackingRegistry
from src.geometry.sat import compute_half_extents

def extract_crops_from_packing(
    coords: np.ndarray,
    source_n: int,
    max_w: int = 12,
    max_h: int = 12,
    grid_step: float = 0.5
) -> list:
    crops = []
    n = len(coords)
    h_extents = compute_half_extents(coords[:, 2])
    
    xmin = float(np.min(coords[:, 0] - h_extents))
    ymin = float(np.min(coords[:, 1] - h_extents))
    xmax = float(np.max(coords[:, 0] + h_extents))
    ymax = float(np.max(coords[:, 1] + h_extents))
    
    # Normalized coords
    cx = coords[:, 0] - xmin
    cy = coords[:, 1] - ymin
    
    total_W = xmax - xmin
    total_H = ymax - ymin
    
    # Scan small integer spans w, h
    for w in range(2, min(max_w + 1, int(math.floor(total_W)))):
        for h_dim in range(2, min(max_h + 1, int(math.floor(total_H)))):
            # Target window size (w + f, h + f) for f in [0.5, 0.7071, 0.85, 0.95]
            for f in [0.5, 0.70710678, 0.75, 0.8, 0.85, 0.9]:
                W = w + f
                H = h_dim + f
                if W > total_W or H > total_H:
                    continue
                    
                # Scan a few anchor positions
                for x0 in np.arange(0.0, total_W - W + 1e-4, 1.0):
                    for y0 in np.arange(0.0, total_H - H + 1e-4, 1.0):
                        # Mask of squares fully inside [x0, x0 + W] x [y0, y0 + H]
                        inside = (
                            (cx - h_extents >= x0) &
                            (cx + h_extents <= x0 + W) &
                            (cy - h_extents >= y0) &
                            (cy + h_extents <= y0 + H)
                        )
                        M = int(np.sum(inside))
                        dm = M - w * h_dim
                        if dm >= 1:
                            sub_coords = coords[inside].copy()
                            # Shift to (0, 0)
                            sub_h = h_extents[inside]
                            sub_xmin = np.min(sub_coords[:, 0] - sub_h)
                            sub_ymin = np.min(sub_coords[:, 1] - sub_h)
                            sub_coords[:, 0] -= sub_xmin
                            sub_coords[:, 1] -= sub_ymin
                            actual_W = np.max(sub_coords[:, 0] + sub_h)
                            actual_H = np.max(sub_coords[:, 1] + sub_h)
                            actual_f = max(actual_W - w, actual_H - h_dim)
                            
                            crops.append({
                                "source_n": source_n,
                                "w": w,
                                "h": h_dim,
                                "dm": dm,
                                "f": actual_f,
                                "M": M,
                                "coords": sub_coords
                            })
                            # Stop scanning this (w, h, f) if we found a good crop
                            break
                    if crops and crops[-1]["source_n"] == source_n and crops[-1]["w"] == w and crops[-1]["h"] == h_dim:
                        break
                        
    return crops

if __name__ == "__main__":
    reg = PackingRegistry()
    print("Testing crop extraction on key donor instances...")
    donors = [5, 26, 28, 40, 50, 65, 82, 105, 126, 170, 207, 265, 301, 372, 451, 699]
    all_crops = []
    for d in donors:
        cr = extract_crops_from_packing(reg.get_packing(d), source_n=d)
        print(f"Donor N={d:3d}: found {len(cr)} crops with dm >= 1")
        all_crops.extend(cr)
        
    print(f"\nTotal crops found: {len(all_crops)}")
    all_crops.sort(key=lambda x: x["f"] / x["dm"])
    print("Top 15 most efficient crops (lowest f/dm):")
    for c in all_crops[:15]:
        print(f"From N={c['source_n']:3d}: {c['w']}x{c['h']}, M={c['M']}, dm=+{c['dm']}, f={c['f']:.6f}, f/dm={c['f']/c['dm']:.6f}")
