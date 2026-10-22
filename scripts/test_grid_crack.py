"""Engineered trial to crack high-deficit grid instances below integer plateaus.

Target: Pack N=999 into L < 32.00000000 (currently locked at 32.00000000).
Strategy:
1. Seed from N=998 (L=31.99687592, which has 25.8 square units of free space).
2. Sample diverse insertion candidates for square 999.
3. Rapid penalty relaxation with packer relax at target s = 31.9985.
4. Soft-to-rigid Minkowski quench with fq quench.
5. Strict SAT validation (eps=1e-10).
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

from src.geometry.sat import validate_packing_sat
from src.optimizers.fq_engine import FQ_BIN, PACKER_BIN, read_engine_config, write_engine_config
from src.scoring.registry import PackingRegistry


def test_crack_999():
    reg = PackingRegistry()
    cur_999_side = reg.get_side(999)
    cur_998_side = reg.get_side(998)
    coords_998 = reg.get_packing(998)
    
    print(f"Current N=999 side: {cur_999_side:.10f}")
    print(f"Seed N=998 side:    {cur_998_side:.10f}")
    
    target_side = 31.9985
    
    # Candidate insertion coordinates (x, y, theta_deg)
    candidates = [
        [0.5, 0.5, 0.0],
        [31.4, 31.4, 0.0],
        [0.5, 31.4, 0.0],
        [31.4, 0.5, 0.0],
        [16.0, 16.0, 0.0],
        [15.5, 16.5, 11.3],
        [8.0, 24.0, 0.0],
        [24.0, 8.0, 0.0],
        [10.0, 10.0, 0.0],
        [22.0, 22.0, 0.0]
    ]
    
    best_cand_e = float("inf")
    best_cand_coords = None
    best_cand_idx = -1
    
    print(f"\nEvaluating {len(candidates)} candidate insertion seeds at target s={target_side:.6f}...")
    for idx, cand in enumerate(candidates):
        c_test = np.vstack([coords_998, cand])
        in_path = f"/tmp/crack_in_{idx}.txt"
        out_path = f"/tmp/crack_out_{idx}.txt"
        write_engine_config(in_path, c_test, target_side)
        
        cmd = [
            str(PACKER_BIN), "relax",
            "--in", in_path,
            "--s", str(target_side),
            "--out", out_path
        ]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        # Parse energy from output line: E=...
        energy = float("inf")
        if res.returncode == 0:
            for part in res.stdout.split():
                if part.startswith("E="):
                    try:
                        energy = float(part.split("=")[1])
                    except:
                        pass
        print(f"  Seed {idx:2d} ({cand[0]:4.1f}, {cand[1]:4.1f}): E={energy:.4e} ({time.time()-t0:.2f}s)")
        if energy < best_cand_e and os.path.exists(out_path):
            best_cand_e = energy
            _, best_cand_coords = read_engine_config(out_path)
            best_cand_idx = idx
            
    print(f"\nBest seed: idx {best_cand_idx} with E={best_cand_e:.4e}")
    if best_cand_coords is not None:
        # Polish with fq quench
        q_in = "/tmp/crack_best_in.txt"
        q_out = "/tmp/crack_best_out.txt"
        write_engine_config(q_in, best_cand_coords, target_side)
        
        cmd_q = [
            str(FQ_BIN), "quench",
            "--in", q_in,
            "--out", q_out,
            "--round-r0", "0.05",
            "--round-compress",
            "--seed", "42"
        ]
        print("Running fq quench with rounded square Minkowski relaxation...")
        t0 = time.time()
        res_q = subprocess.run(cmd_q, capture_output=True, text=True)
        print(f"Quench finished in {time.time()-t0:.2f}s")
        print(res_q.stdout)
        
        if os.path.exists(q_out):
            final_s, final_c = read_engine_config(q_out)
            print(f"Final side reported: {final_s:.12f}")
            val = validate_packing_sat(final_c, eps=1e-10)
            print(f"SAT Validation: valid={val['valid']}, side={val['side']:.12f}, margin={val.get('min_margin')}")
            if val["valid"] and val["side"] < 32.0 - 1e-10:
                gain = (cur_999_side - val["side"]) / math.sqrt(999)
                reg.register_candidate(999, final_c, source="grid_crack_from_n998")
                reg.save_submission()
                print(f"\nHUGE SUCCESS! N=999 BROKEN BELOW 32.0! Score gain: {gain:.10f}")


if __name__ == "__main__":
    test_crack_999()
