#!/usr/bin/env python3
"""Attempt to crack N=56 integer grid barrier (L < 8.0) by void insertion into N=55."""

import json
import math
import os
import subprocess
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.sat import validate_packing_sat, compute_enclosing_side
from src.optimizers.fq_engine import FQ_BIN, PACKER_BIN, read_engine_config, write_engine_config
from src.scoring.registry import PackingRegistry

reg = PackingRegistry()
c55 = reg.get_packing(55)
cur_side_55 = reg.get_side(55)

print(f"Loaded N=55 packing with side {cur_side_55:.8f}")

# Target container sides strictly below 8.0
target_sides = [7.98, 7.985, 7.99, 7.995, 7.998]

# Candidate void insertion positions
void_candidates = [
    (7.3458, 6.5322, 0.0),
    (7.3458, 6.5322, math.pi / 4),
    (2.3966, 5.5492, 0.0),
    (2.3966, 5.5492, -0.389),
    (0.65, 0.65, 0.0),
    (7.35, 0.65, 0.0),
    (0.65, 7.35, 0.0),
    (7.35, 7.35, 0.0)
]

best_found_side = 8.000000000000
best_found_coords = None

in_file = "/tmp/insert_56_in.txt"
out_file = "/tmp/insert_56_out.txt"

for side_target in target_sides:
    print(f"\n--- Testing Target Side L = {side_target:.4f} ---")
    scale = side_target / cur_side_55
    scaled_coords = c55.copy()
    scaled_coords[:, :2] *= scale

    for v_idx, (vx, vy, vt) in enumerate(void_candidates):
        # Scale insertion position
        new_sq = np.array([[vx * scale, vy * scale, vt]])
        cand_56 = np.vstack([scaled_coords, new_sq])

        write_engine_config(in_file, cand_56, side_target)

        # Run fq quench with rounded square homotopy and active set
        for trial in range(3):
            seed = 1000 + v_idx * 13 + trial * 7
            cmd = [
                str(FQ_BIN), "quench",
                "--in", in_file,
                "--out", out_file,
                "--seed", str(seed),
                "--loosen", "1.01",
                "--round-r0", "0.08",
                "--round-compress",
                "--round-steps", "35"
            ]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                if res.returncode == 0 and os.path.exists(out_file):
                    cand_side, cand_out = read_engine_config(out_file)
                    val = validate_packing_sat(cand_out, eps=1e-10, exhaustive=True)
                    if val["valid"]:
                        print(f"  [SUCCESS] Valid packing found! L = {val['side']:.10f} (Target was {side_target})")
                        if val["side"] < best_found_side - 1e-10:
                            best_found_side = val["side"]
                            best_found_coords = cand_out.copy()
                            print(f"  *** BREAKTHROUGH: N=56 CRACKED! New Best L = {best_found_side:.10f} ***")
                    else:
                        pass
            except Exception as e:
                pass

if best_found_coords is not None and best_found_side < 8.0:
    print(f"\n=======================================================")
    print(f"FINAL RESULT: N=56 CRACKED BELOW 8.0!")
    print(f"Side: {best_found_side:.12f}")
    print(f"Gain: {(8.0 - best_found_side) / math.sqrt(56):.8f}")
    print(f"=======================================================")
    reg.register_candidate(56, best_found_coords, source="void_insertion_crack")
    reg.save_submission()
else:
    print(f"\nNo valid packing found strictly below 8.0 with these void insertions.")
