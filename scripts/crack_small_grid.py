"""Engineered trial to pack N=56 into L < 8.00000000.

Seed from N=55 (side 7.945771).
Target: side <= 7.99500000.
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


def test_crack_56():
    reg = PackingRegistry()
    cur_56 = reg.get_side(56)
    cur_55 = reg.get_side(55)
    c55 = reg.get_packing(55)
    
    print(f"Current N=56 side: {cur_56:.10f}")
    print(f"Seed N=55 side:    {cur_55:.10f}")
    
    # Target side: 7.990000
    target_side = 7.990000
    
    # Try 16 diverse insertion positions for square 56
    xs = np.linspace(0.5, target_side - 0.5, 4)
    ys = np.linspace(0.5, target_side - 0.5, 4)
    
    candidates = []
    for x in xs:
        for y in ys:
            candidates.append([x, y, 0.0])
            
    best_cand_e = float("inf")
    best_coords = None
    best_idx = -1
    
    print(f"\nTesting {len(candidates)} insertion seeds at target s={target_side:.6f}...")
    for idx, cand in enumerate(candidates):
        c_test = np.vstack([c55, cand])
        in_p = f"/tmp/crack56_in_{idx}.txt"
        out_p = f"/tmp/crack56_out_{idx}.txt"
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
        print(f"  Seed {idx:2d} ({cand[0]:.2f}, {cand[1]:.2f}): E={energy:.4e} ({time.time()-t0:.2f}s)")
        if energy < best_cand_e and os.path.exists(out_p):
            best_cand_e = energy
            _, best_coords = read_engine_config(out_p)
            best_idx = idx
            
    print(f"\nBest seed: idx {best_idx} with E={best_cand_e:.4e}")
    if best_coords is not None:
        # Run hopper + quench
        hop_out = "/tmp/crack56_hop.txt"
        write_engine_config("/tmp/crack56_best.txt", best_coords, target_side)
        cmd_hop = [
            str(PACKER_BIN), "hop",
            "--in", "/tmp/crack56_best.txt",
            "--s", str(target_side),
            "--iters", "2000",
            "--descend", "1e-5",
            "--seed", "12345",
            "--out", hop_out
        ]
        print("Running packer hop (2000 iters) with descending target...")
        t0 = time.time()
        res_hop = subprocess.run(cmd_hop, capture_output=True, text=True)
        print(f"Hop finished in {time.time()-t0:.2f}s")
        print(res_hop.stdout)
        
        q_in = hop_out if os.path.exists(hop_out) else "/tmp/crack56_best.txt"
        q_out = "/tmp/crack56_final.txt"
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
            val = validate_packing_sat(final_c, eps=1e-10)
            print(f"\nFinal side reported: {final_s:.12f}")
            print(f"SAT Validation: valid={val['valid']}, side={val['side']:.12f}, margin={val.get('min_margin')}")
            if val["valid"] and val["side"] < 8.0 - 1e-10:
                gain = (cur_56 - val["side"]) / math.sqrt(56)
                reg.register_candidate(56, final_c, source="crack56_from_n55")
                reg.save_submission()
                print(f"\n========================================================")
                print(f"WORLD FIRST BREAKTHROUGH: N=56 PACKED BELOW 8.0!")
                print(f"New side: {val['side']:.12f}, Score gain: {gain:.10f}")
                print(f"========================================================")


if __name__ == "__main__":
    test_crack_56()
