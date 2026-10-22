"""Engineered crystal extension to pack N=56 below 8.00000000.

Key Mathematical Insight:
Any row or column containing 8 upright squares rigidly bounds the container to side >= 8.0.
Therefore, square 56 MUST be added into the oblique crystal band (theta ~ 67.7 deg),
preserving max upright count <= 7 in every row and column.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.sat import validate_packing_sat, compute_half_extents
from src.optimizers.fq_engine import FQ_BIN, PACKER_BIN, read_engine_config, write_engine_config
from src.scoring.registry import PackingRegistry


def test_crack_56_crystal():
    reg = PackingRegistry()
    cur_56 = reg.get_side(56)
    cur_55 = reg.get_side(55)
    c55 = reg.get_packing(55)
    
    print(f"Current N=56 side: {cur_56:.10f}")
    print(f"Seed N=55 side:    {cur_55:.10f}")
    
    target_side = 7.985000
    
    # Candidate positions along the boundary of the oblique crystal band
    # Crystal lattice vector ~ (0.88, 0.72) or (-0.72, 0.88)
    theta_rad = np.deg2rad(67.71)
    
    candidates = [
        [3.3, 5.2, theta_rad],
        [4.2, 4.8, theta_rad],
        [5.1, 3.8, theta_rad],
        [5.1, 2.8, theta_rad],
        [2.4, 5.6, theta_rad],
        [1.5, 4.9, theta_rad],
        [0.6, 4.2, theta_rad],
        [5.5, 5.8, np.deg2rad(45.0)],
        [6.2, 5.2, np.deg2rad(45.0)],
        [3.8, 1.5, theta_rad],
        [4.8, 2.0, theta_rad],
        [2.0, 0.8, theta_rad]
    ]
    
    best_cand_e = float("inf")
    best_coords = None
    best_idx = -1
    
    print(f"\nEvaluating {len(candidates)} crystal extension seeds at target s={target_side:.6f}...")
    for idx, cand in enumerate(candidates):
        c_test = np.vstack([c55, cand])
        in_p = f"/tmp/crystal56_in_{idx}.txt"
        out_p = f"/tmp/crystal56_out_{idx}.txt"
        write_engine_config(in_p, c_test, target_side)
        
        cmd = [
            str(PACKER_BIN), "relax",
            "--in", in_p,
            "--s", str(target_side),
            "--out", out_p
        ]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        energy = float("inf")
        if res.returncode == 0:
            for part in res.stdout.split():
                if part.startswith("E="):
                    try: energy = float(part.split("=")[1])
                    except: pass
        print(f"  Seed {idx:2d} ({cand[0]:.2f}, {cand[1]:.2f}, {np.rad2deg(cand[2]):.1f} deg): E={energy:.4e} ({time.time()-t0:.2f}s)")
        if energy < best_cand_e and os.path.exists(out_p):
            best_cand_e = energy
            _, best_coords = read_engine_config(out_p)
            best_idx = idx
            
    print(f"\nBest crystal seed: idx {best_idx} with E={best_cand_e:.4e}")
    if best_coords is not None:
        best_in = "/tmp/crystal56_best.txt"
        write_engine_config(best_in, best_coords, target_side)
        
        # Basin hopping with descending target
        hop_out = "/tmp/crystal56_hop.txt"
        cmd_hop = [
            str(PACKER_BIN), "hop",
            "--in", best_in,
            "--s", str(target_side),
            "--iters", "3000",
            "--descend", "1e-5",
            "--seed", "9876",
            "--amp", "0.3",
            "--out", hop_out
        ]
        print("Running packer hop (3000 iters)...")
        t0 = time.time()
        res_hop = subprocess.run(cmd_hop, capture_output=True, text=True)
        print(f"Hop finished in {time.time()-t0:.2f}s")
        print(res_hop.stdout)
        
        # Check if .inf was written or best
        target_file = hop_out if os.path.exists(hop_out) else f"{hop_out}.inf"
        if not os.path.exists(target_file):
            target_file = best_in
            
        q_out = "/tmp/crystal56_final.txt"
        cmd_q = [
            str(FQ_BIN), "quench",
            "--in", target_file,
            "--out", q_out,
            "--round-r0", "0.06",
            "--round-compress",
            "--loosen", "1.000",
            "--seed", "42"
        ]
        print("Running fq quench with rounded square Minkowski relaxation...")
        t0 = time.time()
        res_q = subprocess.run(cmd_q, capture_output=True, text=True)
        print(f"Quench finished in {time.time()-t0:.2f}s")
        print(res_q.stdout)
        
        if os.path.exists(q_out):
            final_s, final_c = read_engine_config(q_out)
            val = validate_packing_sat(final_c, eps=1e-10)
            print(f"\nFinal side reported: {final_s:.12f}")
            print(f"SAT Validation: valid={val['valid']}, side={val['side']:.12f}, margin={val.get('min_margin')}")
            if val["valid"] and val["side"] < 8.0 - 1e-10:
                gain = (cur_56 - val["side"]) / math.sqrt(56)
                reg.register_candidate(56, final_c, source="crystal56_from_n55")
                reg.save_submission()
                print(f"\n========================================================")
                print(f"WORLD FIRST BREAKTHROUGH: N=56 PACKED BELOW 8.0!")
                print(f"New side: {val['side']:.12f}, Score gain: {gain:.10f}")
                print(f"========================================================")


if __name__ == "__main__":
    test_crack_56_crystal()
