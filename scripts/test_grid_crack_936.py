"""Engineered trial to crack N=936 below the integer 31.0 grid bound.

Target: Pack N=936 into L < 31.00000000.
Seed: N=935 (L=30.99417406, free space = 25.6 units).
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


def test_crack_936():
    reg = PackingRegistry()
    cur_936_side = reg.get_side(936)
    cur_935_side = reg.get_side(935)
    coords_935 = reg.get_packing(935)
    
    print(f"Current N=936 side: {cur_936_side:.10f}")
    print(f"Seed N=935 side:    {cur_935_side:.10f}")
    
    target_side = 30.9980
    
    candidates = [
        [0.5, 0.5, 0.0],
        [30.4, 30.4, 0.0],
        [0.5, 30.4, 0.0],
        [30.4, 0.5, 0.0],
        [15.5, 15.5, 0.0],
        [15.0, 16.0, 11.3],
        [7.5, 23.5, 0.0],
        [23.5, 7.5, 0.0]
    ]
    
    best_cand_e = float("inf")
    best_cand_coords = None
    best_cand_idx = -1
    
    print(f"\nEvaluating {len(candidates)} candidate insertion seeds at target s={target_side:.6f}...")
    for idx, cand in enumerate(candidates):
        c_test = np.vstack([coords_935, cand])
        in_path = f"/tmp/crack936_in_{idx}.txt"
        out_path = f"/tmp/crack936_out_{idx}.txt"
        write_engine_config(in_path, c_test, target_side)
        
        cmd = [
            str(PACKER_BIN), "relax",
            "--in", in_path,
            "--s", str(target_side),
            "--out", out_path
        ]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
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
            
    print(f"\nBest seed for N=936: idx {best_cand_idx} with E={best_cand_e:.4e}")
    if best_cand_coords is not None:
        q_in = "/tmp/crack936_best_in.txt"
        q_out = "/tmp/crack936_best_out.txt"
        write_engine_config(q_in, best_cand_coords, target_side)
        
        cmd_q = [
            str(FQ_BIN), "quench",
            "--in", q_in,
            "--out", q_out,
            "--round-r0", "0.06",
            "--round-compress",
            "--seed", "42"
        ]
        print("Running fq quench on N=936...")
        t0 = time.time()
        res_q = subprocess.run(cmd_q, capture_output=True, text=True)
        print(f"Quench finished in {time.time()-t0:.2f}s")
        print(res_q.stdout)
        
        if os.path.exists(q_out):
            final_s, final_c = read_engine_config(q_out)
            print(f"Final side reported: {final_s:.12f}")
            val = validate_packing_sat(final_c, eps=1e-10)
            print(f"SAT Validation: valid={val['valid']}, side={val['side']:.12f}, margin={val.get('min_margin')}")
            if val["valid"] and val["side"] < 31.0 - 1e-10:
                gain = (cur_936_side - val["side"]) / math.sqrt(936)
                reg.register_candidate(936, final_c, source="grid_crack_from_n935")
                reg.save_submission()
                print(f"\nHUGE SUCCESS! N=936 BROKEN BELOW 31.0! Score gain: {gain:.10f}")


if __name__ == "__main__":
    test_crack_936()
