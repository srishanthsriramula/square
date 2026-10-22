"""Systematic donor refinement engine using soft-to-rigid Minkowski and aspect relaxation.

Applies:
1. fq quench with rounded square Minkowski relaxation (--round-r0, --round-compress)
2. fq quench with soft square log-aspect relaxation (--soft-a0, --soft-compress)
3. Controlled kicks to escape local jamming basins
4. Strict SAT geometry validation (eps=1e-10)
5. Monotonicity cascades across adjacent instances
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.sat import validate_packing_sat, compute_enclosing_side
from src.optimizers.fq_engine import FQ_BIN, PACKER_BIN, read_engine_config, write_engine_config
from src.scoring.registry import PackingRegistry


def refine_instance(
    n: int,
    reg: PackingRegistry,
    num_trials: int = 5,
    timeout_sec: float = 60.0
) -> Tuple[bool, float, float]:
    """Attempt multi-strategy active-set and relaxation refinement on instance n."""
    cur_side = reg.get_side(n)
    coords = reg.get_packing(n)
    
    in_file = f"/tmp/refine_{n}_in.txt"
    out_file = f"/tmp/refine_{n}_out.txt"
    write_engine_config(in_file, coords, cur_side)
    
    strategies = [
        # Strategy 1: Soft-to-rigid rounded square annealing
        ["--round-r0", "0.05", "--round-compress", "--round-steps", "25"],
        ["--round-r0", "0.08", "--round-compress", "--round-steps", "30"],
        ["--round-r0", "0.03", "--round-compress", "--round-steps", "20", "--round-kick", "0.001"],
        # Strategy 2: Soft aspect ratio relaxation
        ["--soft-a0", "0.05", "--soft-compress", "--soft-steps", "25"],
        ["--soft-a0", "0.08", "--soft-compress", "--soft-steps", "30"],
        # Strategy 3: Gentle kick followed by direct quench
        ["--kick", "0.002"],
        ["--kick", "0.005"]
    ]
    
    best_side = cur_side
    best_coords = coords
    improved = False
    total_gain = 0.0
    
    for trial_idx in range(min(num_trials, len(strategies))):
        strat = strategies[trial_idx]
        seed = 1000 + trial_idx * 17
        cmd = [
            str(FQ_BIN), "quench",
            "--in", in_file,
            "--out", out_file,
            "--seed", str(seed)
        ] + strat
        
        try:
            t0 = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            if res.returncode == 0 and os.path.exists(out_file):
                cand_side, cand_coords = read_engine_config(out_file)
                if cand_side < best_side - 1e-10:
                    val = validate_packing_sat(cand_coords, eps=1e-10)
                    if val["valid"] and val["side"] < cur_side - 1e-10:
                        success, g, msg = reg.register_candidate(
                            n,
                            cand_coords,
                            source=f"fq_strat{trial_idx}_seed{seed}"
                        )
                        if success:
                            best_side = val["side"]
                            best_coords = cand_coords
                            improved = True
                            total_gain += g
                            print(f"  [+] N={n}: IMPROVED! cur={cur_side:.10f} -> new={best_side:.10f} (gain={g:.10f}, {time.time()-t0:.2f}s)")
                            # Update in_file for subsequent passes
                            write_engine_config(in_file, best_coords, best_side)
                            cur_side = best_side
        except subprocess.TimeoutExpired:
            continue
        except Exception as e:
            print(f"  Error on N={n} trial {trial_idx}: {e}")
            
    if improved:
        reg.save_submission()
        
    return improved, best_side, total_gain


def run_refinement_batch(instances: List[int], trials_per_instance: int = 5):
    reg = PackingRegistry()
    print(f"Starting refinement batch on {len(instances)} instances...")
    print(f"Current Verified Score: {reg.get_score():.14f}\n")
    
    improved_list = []
    total_gain = 0.0
    
    for idx, n in enumerate(instances):
        cur_s = reg.get_side(n)
        print(f"[{idx+1}/{len(instances)}] Refining N={n:3d} (current side={cur_s:.8f})...", flush=True)
        ok, new_s, g = refine_instance(n, reg, num_trials=trials_per_instance)
        if ok:
            improved_list.append(n)
            total_gain += g
        else:
            print(f"  N={n:3d}: no improvement with {trials_per_instance} trials.")
            
    print("\n" + "=" * 50)
    print(f"BATCH COMPLETE:")
    print(f"Total instances improved: {len(improved_list)} ({improved_list})")
    print(f"Total score gain: {total_gain:.12f}")
    print(f"New Verified Score: {reg.get_score():.14f}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=str, default="66,67,68,69,70,71,83,84,85,86,87,88")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    
    ns = [int(x.strip()) for x in args.instances.split(",") if x.strip()]
    run_refinement_batch(ns, trials_per_instance=args.trials)
