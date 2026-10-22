"""Fast, high-yield mass crop extractor for Pareto-optimal fault-line gadgets.

Extracts optimal (w, h, dm, f) gadgets across multi-scale crystal donors
using single-pass bounding-box analysis and 90-degree rotational symmetry.
"""

from __future__ import annotations

import glob
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.sat import compute_half_extents
from src.scoring.registry import PackingRegistry
from src.constructions.fault_dp import Gadget


def load_external_packing(filepath: str) -> Tuple[int, float, np.ndarray]:
    """Load packing from external n*.txt file."""
    with open(filepath) as f:
        lines = [line.strip() for line in f if line.strip()]
        
    n = None
    side = None
    coords = []
    
    for line in lines:
        if line.startswith("# n =") or line.startswith("# n="):
            n = int(line.split("=")[1].strip())
        elif line.startswith("# s =") or line.startswith("# s="):
            side = float(line.split("=")[1].strip())
        elif not line.startswith("#"):
            parts = line.split()
            x = float(parts[0])
            y = float(parts[1])
            theta = float(parts[2])
            coords.append([x, y, theta])
            
    arr = np.array(coords, dtype=np.float64)
    if n is None:
        n = len(arr)
    if side is None:
        h_ext = compute_half_extents(arr[:, 2])
        side = float(max(np.max(arr[:, 0] + h_ext), np.max(arr[:, 1] + h_ext)))
        
    return n, side, arr


def fast_extract_from_coords(
    coords: np.ndarray,
    source_name: str,
    source_n: int,
    max_w: int = 15,
    max_h: int = 15,
    anchor_step: float = 0.5
) -> Dict[Tuple[int, int, int], Gadget]:
    """Extract Pareto-optimal crops for each (w, h, dm) from a single packing."""
    # Fast mathematically sound skip: upright grid packings cannot yield dm >= 1
    if not np.any(np.abs(np.sin(2.0 * coords[:, 2])) > 1e-4):
        return {}
        
    n = len(coords)
    h_ext = compute_half_extents(coords[:, 2])
    xmin, ymin = float(np.min(coords[:, 0] - h_ext)), float(np.min(coords[:, 1] - h_ext))
    xmax, ymax = float(np.max(coords[:, 0] + h_ext)), float(np.max(coords[:, 1] + h_ext))
    total_W = xmax - xmin
    total_H = ymax - ymin
    
    cx = coords[:, 0] - xmin
    cy = coords[:, 1] - ymin
    
    best_crops: Dict[Tuple[int, int, int], Gadget] = {}
    
    max_w_clamped = min(max_w, int(math.floor(total_W)))
    max_h_clamped = min(max_h, int(math.floor(total_H)))
    
    for w in range(2, max_w_clamped + 1):
        for h_dim in range(2, max_h_clamped + 1):
            W = w + 0.95
            H = h_dim + 0.95
            if W > total_W or H > total_H:
                continue
                
            x_anchors = np.arange(0.0, total_W - W + 1e-4, anchor_step)
            y_anchors = np.arange(0.0, total_H - H + 1e-4, anchor_step)
            
            for x0 in x_anchors:
                for y0 in y_anchors:
                    mask = (
                        (cx - h_ext >= x0 - 1e-6) &
                        (cx + h_ext <= x0 + W + 1e-6) &
                        (cy - h_ext >= y0 - 1e-6) &
                        (cy + h_ext <= y0 + H + 1e-6)
                    )
                    M = int(np.sum(mask))
                    dm = M - w * h_dim
                    if dm >= 1:
                        sub = coords[mask].copy()
                        sub_h = h_ext[mask]
                        sub_xmin = float(np.min(sub[:, 0] - sub_h))
                        sub_ymin = float(np.min(sub[:, 1] - sub_h))
                        sub[:, 0] -= sub_xmin
                        sub[:, 1] -= sub_ymin
                        act_W = float(np.max(sub[:, 0] + sub_h))
                        act_H = float(np.max(sub[:, 1] + sub_h))
                        act_f = max(act_W - w, act_H - h_dim)
                        
                        if act_f < 0.999:
                            key = (w, h_dim, dm)
                            if key not in best_crops or act_f < best_crops[key].f:
                                best_crops[key] = Gadget(
                                    name=f"crop_{source_name}_{w}x{h_dim}_dm{dm}",
                                    w=w,
                                    h=h_dim,
                                    dm=dm,
                                    f=act_f,
                                    coords=sub,
                                    source_n=source_n
                                )
                                
                                # Add 90-degree rotated counterpart
                                rot = sub.copy()
                                rot[:, 0] = sub[:, 1]
                                rot[:, 1] = -sub[:, 0]
                                rot[:, 2] += math.pi / 2
                                rot_h = compute_half_extents(rot[:, 2])
                                rot[:, 0] -= np.min(rot[:, 0] - rot_h)
                                rot[:, 1] -= np.min(rot[:, 1] - rot_h)
                                rot_W = float(np.max(rot[:, 0] + rot_h))
                                rot_H = float(np.max(rot[:, 1] + rot_h))
                                rot_f = max(rot_W - h_dim, rot_H - w)
                                
                                rot_key = (h_dim, w, dm)
                                if rot_key not in best_crops or rot_f < best_crops[rot_key].f:
                                    best_crops[rot_key] = Gadget(
                                        name=f"crop_{source_name}_{h_dim}x{w}_dm{dm}_r90",
                                        w=h_dim,
                                        h=w,
                                        dm=dm,
                                        f=rot_f,
                                        coords=rot,
                                        source_n=source_n
                                    )
                                    
    return best_crops


def run_fast_mass_extraction():
    reg = PackingRegistry()
    print("Starting fast mass crop extraction...")
    t0 = time.time()
    
    pareto_catalog: Dict[Tuple[int, int, int], Gadget] = {}
    
    # 1. Whole packings
    print("Cataloging whole packings as gadgets...")
    for n in range(4, 1001):
        coords = reg.get_packing(n)
        h_ext = compute_half_extents(coords[:, 2])
        xmin, ymin = np.min(coords[:, 0] - h_ext), np.min(coords[:, 1] - h_ext)
        xmax, ymax = np.max(coords[:, 0] + h_ext), np.max(coords[:, 1] + h_ext)
        W, H = xmax - xmin, ymax - ymin
        w, h_int = int(math.floor(W)), int(math.floor(H))
        dm = n - w * h_int
        if dm > 0:
            f = max(W - w, H - h_int)
            if f < 0.999999:
                norm = coords.copy()
                norm[:, 0] -= xmin
                norm[:, 1] -= ymin
                key = (w, h_int, dm)
                if key not in pareto_catalog or f < pareto_catalog[key].f:
                    pareto_catalog[key] = Gadget(
                        name=f"full_n{n}",
                        w=w,
                        h=h_int,
                        dm=dm,
                        f=f,
                        coords=norm,
                        source_n=n
                    )
    print(f"Cataloged {len(pareto_catalog)} whole packing gadgets.")
    
    # 2. Key diverse donors across all size scales
    donors = [
        5, 10, 11, 14, 18, 19, 26, 28, 37, 39, 40, 41, 50, 65, 82, 85,
        105, 126, 147, 170, 207, 230, 265, 301, 328, 364, 368, 372, 400,
        451, 500, 550, 602, 641, 653, 679, 699, 750, 803, 849, 850, 903, 950, 975
    ]
    print(f"\nExtracting crops from {len(donors)} curated crystal donors...")
    for idx, d in enumerate(donors):
        coords = reg.get_packing(d)
        crops = fast_extract_from_coords(coords, source_name=f"n{d}", source_n=d, max_w=15, max_h=15, anchor_step=0.5)
        new_bests = 0
        for key, g in crops.items():
            if key not in pareto_catalog or g.f < pareto_catalog[key].f:
                pareto_catalog[key] = g
                new_bests += 1
        print(f"  Donor N={d:3d} ({idx+1}/{len(donors)}): {len(crops)} crops, {new_bests} new Pareto records.")
        
    # 3. Dense large external records from Francisco Uzo
    print("\nScanning large external crystal packings from Francisco Uzo...")
    external_targets = [1061, 1263, 1335, 1413, 1489, 1564, 1644, 1727, 1811, 1897, 1986, 2076, 2168, 2262, 2358, 2457]
    for n_ext in external_targets:
        fpath = f"external/franciscouzo/n{n_ext}.txt"
        if os.path.exists(fpath):
            try:
                _, _, c_ext = load_external_packing(fpath)
                crops = fast_extract_from_coords(c_ext, source_name=f"ext{n_ext}", source_n=n_ext, max_w=15, max_h=15, anchor_step=0.5)
                new_bests = 0
                for key, g in crops.items():
                    if key not in pareto_catalog or g.f < pareto_catalog[key].f:
                        pareto_catalog[key] = g
                        new_bests += 1
                print(f"  External N={n_ext}: {len(crops)} crops, {new_bests} new Pareto records.")
            except Exception as e:
                print(f"  Error with {fpath}: {e}")
                
    elapsed = time.time() - t0
    print(f"\nExtraction complete in {elapsed:.1f}s!")
    print(f"Total Pareto-optimal gadgets in catalog: {len(pareto_catalog)}")
    
    # Sort by (w, h, -dm, f)
    sorted_gadgets = sorted(pareto_catalog.values(), key=lambda g: (g.w, g.h, -g.dm, g.f))
    
    # Save to data/generated
    out_dir = (Path(__file__).resolve().parent.parent / "data" / "generated").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "gadgets.txt"
    npz_path = out_dir / "gadgets_coords.npz"
    
    coords_dict = {}
    with open(txt_path, "w") as f:
        f.write("# id w h dm f name\n")
        for idx, g in enumerate(sorted_gadgets):
            f.write(f"{idx} {g.w} {g.h} {g.dm} {g.f:.8f} {g.name}\n")
            coords_dict[f"g_{idx}"] = g.coords
            coords_dict[f"meta_{idx}"] = np.array([g.w, g.h, g.dm, g.f, g.source_n])
            
    np.savez_compressed(npz_path, **coords_dict)
    print(f"Saved {len(sorted_gadgets)} gadgets to {txt_path} and {npz_path}")


if __name__ == "__main__":
    run_fast_mass_extraction()
