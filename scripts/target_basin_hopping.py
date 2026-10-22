"""Targeted basin hopping and active-set polish for continuous non-grid instances.

Executes:
1. packer hop (event-chain & L-BFGS basin hopping with descending squeeze)
2. fq quench (Augmented Lagrangian Method + Clarabel IPM polish)
3. Strict SAT validation (eps=1e-10)
4. Monotonicity cascade & Fault-line DP propagation
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.sat import compute_enclosing_side, validate_packing_sat
from src.optimizers.fq_engine import FQ_BIN, PACKER_BIN, read_engine_config, write_engine_config
from src.scoring.registry import PackingRegistry


def optimize_puzzle(
    n: int,
    reg: PackingRegistry,
    iters: int = 400,
    seed: int = 42,
    timeout_sec: float = 120.0
) -> Tuple[bool, float, float]:
    """Run hop + quench on instance n."""
    cur_side = reg.get_side(n)
    coords = reg.get_packing(n)
    
    in_path = f"/tmp/hop_in_{n}.txt"
    hop_out_path = f"/tmp/hop_out_{n}.txt"
    q_out_path = f"/tmp/q_out_{n}.txt"
    
    write_engine_config(in_path, coords, cur_side)
    
    # 1. Basin hopping with descending target
    cmd_hop = [
        str(PACKER_BIN), "hop",
        "--in", in_path,
        "--s", str(cur_side),
        "--seed", str(seed),
        "--iters", str(iters),
        "--out", hop_out_path,
        "--descend", "2e-6",
        "--amp", "0.5"
    ]
    
    t0 = time.time()
    try:
        res_hop = subprocess.run(cmd_hop, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return False, cur_side, 0.0
        
    cand_side = cur_side
    cand_coords = coords
    
    if os.path.exists(hop_out_path):
        h_side, h_coords = read_engine_config(hop_out_path)
        if h_side < cur_side - 1e-10:
            cand_side = h_side
            cand_coords = h_coords
            
    # 2. ALM + IPM Polish (fq quench) with strict 1.0000 looseness (no dilation)
    write_engine_config(hop_out_path, cand_coords, cand_side)
    cmd_q = [
        str(FQ_BIN), "quench",
        "--in", hop_out_path,
        "--out", q_out_path,
        "--loosen", "1.0000",
        "--outer", "25",
        "--pit", "80",
        "--seed", str(seed)
    ]
    try:
        res_q = subprocess.run(cmd_q, capture_output=True, text=True, timeout=45.0)
    except subprocess.TimeoutExpired:
        pass
        
    final_side = cand_side
    final_coords = cand_coords
    
    if os.path.exists(q_out_path):
        q_side, q_coords = read_engine_config(q_out_path)
        if q_side < final_side - 1e-10:
            final_side = q_side
            final_coords = q_coords
            
    # 3. SAT Validation
    gain = (cur_side - final_side) / math.sqrt(n)
    if gain > 1e-10:
        val = validate_packing_sat(final_coords, eps=1e-10)
        if val["valid"] and val["side"] < cur_side - 1e-10:
            success, g, msg = reg.register_candidate(
                n,
                final_coords,
                source=f"hop{iters}_quench_seed{seed}"
            )
            if success:
                print(f"SUCCESS N={n}: cur={cur_side:.10f} -> new={val['side']:.10f}, gain={g:.10f} ({time.time()-t0:.1f}s)")
                reg.save_submission()
                return True, val["side"], g
                
    return False, cur_side, 0.0


def run_target_puzzles(target_ns: List[int], iters: int = 350):
    reg = PackingRegistry()
    print(f"Starting targeted basin hopping on {len(target_ns)} candidate instances...")
    print(f"Current Verified Score: {reg.get_score():.14f}")
    
    total_gain = 0.0
    improved = []
    
    for idx, n in enumerate(target_ns):
        cur_s = reg.get_side(n)
        print(f"[{idx+1}/{len(target_ns)}] Optimizing N={n:3d} (cur_side={cur_s:.8f})...", flush=True)
        ok, new_s, g = optimize_puzzle(n, reg, iters=iters, seed=42 + idx * 7)
        if ok:
            improved.append(n)
            total_gain += g
        else:
            print(f"  N={n:3d}: no gain")
            
    print("\n=== BATCH SUMMARY ===")
    print(f"Total improved: {len(improved)} ({improved})")
    print(f"Total score gain: {total_gain:.10f}")
    print(f"New score: {reg.get_score():.14f}")


if __name__ == "__main__":
    # Prioritize top continuous instances with high rotational freedom
    puzzles = [103, 105, 106, 110, 123, 127, 128, 129, 130, 131, 132, 146, 151, 152, 154, 155, 156, 171, 172, 175, 176]
    run_target_puzzles(puzzles[:8], iters=300)
