"""Large-Neighborhood Destroy-and-Repair / Packing Surgery Optimizer.

Selectively destroys a congested cluster of squares to break rigid jammed
contact graphs, relaxes the remaining cavity at a descending target,
and repacks the removed squares with new topological contacts.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.geometry.sat import validate_packing_sat, compute_half_extents, compute_enclosing_side
from src.optimizers.fq_engine import FQ_BIN, PACKER_BIN, read_engine_config, write_engine_config
from src.scoring.registry import PackingRegistry


def find_congested_cluster(
    coords: np.ndarray,
    side: float,
    k_remove: int = 6,
    seed: int = 42
) -> List[int]:
    """Find indices of k mutually close squares near a high-contact region."""
    rng = np.random.default_rng(seed)
    n = len(coords)
    
    # Measure local contact density (number of neighbors within distance 1.45)
    dists = np.hypot(
        coords[:, 0, None] - coords[:, 0],
        coords[:, 1, None] - coords[:, 1]
    )
    # Exclude diagonal
    np.fill_diagonal(dists, np.inf)
    contacts = np.sum(dists < 1.42, axis=1)
    
    # Pick candidate center from top 20% most congested squares
    top_indices = np.argsort(-contacts)[:max(10, n // 5)]
    center_idx = rng.choice(top_indices)
    
    # Pick k closest squares to this center
    center_dists = np.hypot(
        coords[:, 0] - coords[center_idx, 0],
        coords[:, 1] - coords[center_idx, 1]
    )
    cluster = np.argsort(center_dists)[:k_remove].tolist()
    return cluster


def destroy_and_repair(
    coords: np.ndarray,
    cur_side: float,
    target_side: float,
    k_remove: int = 6,
    seed: int = 42,
    timeout_sec: float = 60.0
) -> Tuple[bool, float, Optional[np.ndarray]]:
    """Execute destroy-and-repair move at target_side."""
    n = len(coords)
    cluster = find_congested_cluster(coords, cur_side, k_remove=k_remove, seed=seed)
    
    # 1. Destroy: remove k squares
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[cluster] = False
    
    remaining_coords = coords[keep_mask].copy()
    removed_coords = coords[cluster].copy()
    
    # 2. Relax remaining squares in reduced container
    in_rem = f"/tmp/dr_rem_in_{seed}.txt"
    out_rem = f"/tmp/dr_rem_out_{seed}.txt"
    write_engine_config(in_rem, remaining_coords, target_side)
    
    cmd_rem = [
        str(PACKER_BIN), "relax",
        "--in", in_rem,
        "--s", str(target_side),
        "--out", out_rem
    ]
    res_rem = subprocess.run(cmd_rem, capture_output=True, text=True)
    if res_rem.returncode != 0 or not os.path.exists(out_rem):
        return False, cur_side, None
        
    _, relaxed_rem = read_engine_config(out_rem)
    
    # 3. Repair: re-insert removed squares with randomized rotations and positions
    rng = np.random.default_rng(seed + 101)
    new_placed = []
    for sq in removed_coords:
        # Small perturbation around original location or random void
        px = np.clip(sq[0] + rng.uniform(-0.4, 0.4), 0.5, target_side - 0.5)
        py = np.clip(sq[1] + rng.uniform(-0.4, 0.4), 0.5, target_side - 0.5)
        ptheta = rng.choice([0.0, 11.3, 45.0, 78.7, sq[2]])
        new_placed.append([px, py, ptheta])
        
    combined_coords = np.vstack([relaxed_rem, new_placed])
    
    # 4. Joint penalty relaxation
    in_comb = f"/tmp/dr_comb_in_{seed}.txt"
    out_comb = f"/tmp/dr_comb_out_{seed}.txt"
    write_engine_config(in_comb, combined_coords, target_side)
    
    cmd_comb = [
        str(PACKER_BIN), "relax",
        "--in", in_comb,
        "--s", str(target_side),
        "--out", out_comb
    ]
    res_comb = subprocess.run(cmd_comb, capture_output=True, text=True)
    energy = float("inf")
    if res_comb.returncode == 0:
        for part in res_comb.stdout.split():
            if part.startswith("E="):
                try: energy = float(part.split("=")[1])
                except: pass
                
    if energy > 1e-4 or not os.path.exists(out_comb):
        return False, cur_side, None
        
    # 5. Direct polish with fq quench
    q_out = f"/tmp/dr_q_out_{seed}.txt"
    cmd_q = [
        str(FQ_BIN), "quench",
        "--in", out_comb,
        "--out", q_out,
        "--loosen", "1.000",
        "--no-alm",
        "--seed", str(seed)
    ]
    try:
        subprocess.run(cmd_q, capture_output=True, text=True, timeout=timeout_sec)
        if os.path.exists(q_out):
            final_s, final_c = read_engine_config(q_out)
            if final_s < cur_side - 1e-10:
                val = validate_packing_sat(final_c, eps=1e-10)
                if val["valid"] and val["side"] < cur_side - 1e-10:
                    return True, val["side"], final_c
    except Exception as e:
        pass
        
    return False, cur_side, None
