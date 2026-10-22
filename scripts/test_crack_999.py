#!/usr/bin/env python3
"""Attempt to crack N=999 by void insertion into certified N=998 (L=31.99687592)."""

import json
import math
import os
import subprocess
import sys
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.sat import validate_packing_sat, compute_enclosing_side
from src.optimizers.fq_engine import FQ_BIN, PACKER_BIN, read_engine_config, write_engine_config
from src.scoring.registry import PackingRegistry

reg = PackingRegistry()
c998 = reg.get_packing(998)
cur_side_998 = reg.get_side(998)

print(f"Loaded certified N=998 packing with side {cur_side_998:.8f}")

# Target container sides strictly below 32.00000000
target_sides = [31.9975, 31.9985, 31.9995]

# Find largest voids in 998
gx = np.linspace(0.6, cur_side_998 - 0.6, 250)
gy = np.linspace(0.6, cur_side_998 - 0.6, 250)
GX, GY = np.meshgrid(gx, gy)
pts = np.column_stack([GX.ravel(), GY.ravel()])

tree = cKDTree(c998[:, :2])
dists, _ = tree.query(pts)

# Top 10 largest voids
top_idxs = np.argsort(dists)[-10:]
void_candidates = []
for idx in reversed(top_idxs):
    void_candidates.append((pts[idx, 0], pts[idx, 1], 0.0))
    void_candidates.append((pts[idx, 0], pts[idx, 1], math.pi / 4))

print(f"Found {len(void_candidates)} candidate insertion poses. Max clearance: {np.max(dists):.4f}")

in_file = "/tmp/insert_999_in.txt"
out_file = "/tmp/insert_999_out.txt"

best_found_side = 32.000000000000
best_found_coords = None

for side_target in target_sides:
    print(f"\n--- Testing Target Side L = {side_target:.6f} ---")
    scale = side_target / cur_side_998
    scaled_coords = c998.copy()
    scaled_coords[:, :2] *= scale

    for v_idx, (vx, vy, vt) in enumerate(void_candidates[:10]):
        new_sq = np.array([[vx * scale, vy * scale, vt]])
        cand_999 = np.vstack([scaled_coords, new_sq])

        write_engine_config(in_file, cand_999, side_target)

        for trial in range(2):
            seed = 2000 + v_idx * 17 + trial * 31
            cmd = [
                str(FQ_BIN), "quench",
                "--in", in_file,
                "--out", out_file,
                "--seed", str(seed),
                "--loosen", "1.002",
                "--round-r0", "0.03",
                "--round-compress",
                "--round-steps", "25"
            ]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if res.returncode == 0 and os.path.exists(out_file):
                    cand_side, cand_out = read_engine_config(out_file)
                    val = validate_packing_sat(cand_out, eps=1e-10, exhaustive=False)
                    if val["valid"] and val["side"] < 32.0 - 1e-10:
                        print(f"  [SUCCESS] N=999 CRACKED! L = {val['side']:.10f} (Target: {side_target})")
                        if val["side"] < best_found_side - 1e-10:
                            best_found_side = val["side"]
                            best_found_coords = cand_out.copy()
                            print(f"  *** BREAKTHROUGH: New Best L = {best_found_side:.10f} ***")
                            break
            except Exception as e:
                pass
        if best_found_coords is not None:
            break
    if best_found_coords is not None:
        break

if best_found_coords is not None and best_found_side < 32.0:
    gain = (32.0 - best_found_side) / math.sqrt(999)
    print(f"\n=======================================================")
    print(f"N=999 CRACKED BELOW 32.0!")
    print(f"Old: 32.000000000000")
    print(f"New: {best_found_side:.12f}")
    print(f"Gain: +{gain:.10f}")
    print(f"=======================================================")
    reg.register_candidate(999, best_found_coords, source="void_insertion_crack_999")
    reg.save_submission()
else:
    print(f"\nN=999 not cracked in this quick pass.")
