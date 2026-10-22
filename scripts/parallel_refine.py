"""High-throughput multi-worker active-set and Minkowski relaxation engine.

Parallelizes active-set donor refinement across multiple CPU cores on Apple M3:
- Each worker tackles independent instances in /tmp/refine_{n}_*.txt
- Multiple relaxation strategies (Minkowski rounded square, log-aspect soft square, gentle kicks)
- Strict double-precision SAT geometry validation (eps=1e-10)
- Process-safe centralized registration with downward monotonicity cascades
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.sat import validate_packing_sat, compute_enclosing_side
from src.optimizers.fq_engine import FQ_BIN, PACKER_BIN, read_engine_config, write_engine_config
from src.scoring.registry import PackingRegistry


def worker_refine(n: int, coords: np.ndarray, cur_side: float, num_trials: int = 4, timeout_sec: float = 60.0) -> Optional[Tuple[int, np.ndarray, float, str]]:
    """Worker task: optimize instance n using multiple fq relaxation strategies."""
    in_file = f"/tmp/par_ref_{n}_in.txt"
    out_file = f"/tmp/par_ref_{n}_out.txt"
    write_engine_config(in_file, coords, cur_side)
    
    strategies = [
        # Strategy 1: Soft-to-rigid Minkowski rounded square
        ["--round-r0", "0.05", "--round-compress", "--round-steps", "25"],
        ["--round-r0", "0.08", "--round-compress", "--round-steps", "30"],
        ["--round-r0", "0.03", "--round-compress", "--round-steps", "20", "--round-kick", "0.001"],
        # Strategy 2: Soft log-aspect ratio relaxation
        ["--soft-a0", "0.05", "--soft-compress", "--soft-steps", "25"],
        ["--soft-a0", "0.08", "--soft-compress", "--soft-steps", "30"],
        # Strategy 3: Gentle kick followed by direct quench
        ["--kick", "0.002"],
        ["--kick", "0.005"]
    ]
    
    best_side = cur_side
    best_coords = None
    best_source = None
    
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
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            if res.returncode == 0 and os.path.exists(out_file):
                cand_side, cand_coords = read_engine_config(out_file)
                if cand_side < best_side - 1e-10:
                    val = validate_packing_sat(cand_coords, eps=1e-10, exhaustive=(n <= 50))
                    if val["valid"] and val["side"] < best_side - 1e-10:
                        best_side = val["side"]
                        best_coords = cand_coords.copy()
                        best_source = f"fq_strat{trial_idx}_seed{seed}"
                        # Update in_file for subsequent trials to build on the gain
                        write_engine_config(in_file, best_coords, best_side)
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue
            
    # Cleanup temp worker files
    for f in [in_file, out_file]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass
            
    if best_coords is not None and best_side < cur_side - 1e-10:
        return (n, best_coords, best_side, best_source)
    return None


def run_parallel_refine(instances: List[int], max_workers: int = 3, trials_per_instance: int = 4):
    reg = PackingRegistry()
    initial_score = reg.get_score()
    print(f"============================================================")
    print(f"PARALLEL ACTIVE-SET REFINEMENT ENGINE")
    print(f"Target instances: {len(instances)} (Workers: {max_workers}, Trials: {trials_per_instance})")
    print(f"Initial Certified Score: {initial_score:.14f}")
    print(f"============================================================\n", flush=True)
    
    t0 = time.time()
    improved_count = 0
    total_score_gain = 0.0
    
    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for n in instances:
            cur_s = reg.get_side(n)
            coords = reg.get_packing(n)
            # Adjust timeout dynamically: 30s for small n, 90s for large n
            timeout = 30.0 if n <= 300 else (60.0 if n <= 700 else 90.0)
            fut = executor.submit(worker_refine, n, coords, cur_s, trials_per_instance, timeout)
            futures[fut] = (n, cur_s)
            
        print(f"Submitted {len(futures)} tasks to executor pool...\n", flush=True)
        
        for fut in as_completed(futures):
            n, old_side = futures[fut]
            try:
                result = fut.result()
                if result is not None:
                    res_n, new_coords, new_side, source = result
                    # Double-check against current registry side (in case monotonicity cascaded)
                    cur_reg_side = reg.get_side(res_n)
                    if new_side < cur_reg_side - 1e-10:
                        success, gain, msg = reg.register_candidate(res_n, new_coords, source=source)
                        if success:
                            improved_count += 1
                            total_score_gain += gain
                            cur_score = reg.get_score()
                            print(f"[+] N={res_n:3d} IMPROVED! side: {cur_reg_side:.10f} -> {new_side:.10f} (gain={gain:.10f}) | New Score: {cur_score:.14f}", flush=True)
                            reg.save_submission()
                    else:
                        print(f"[-] N={n:3d} candidate side {new_side:.8f} not better than current registry {cur_reg_side:.8f}")
                else:
                    print(f"[-] N={n:3d} finished with no improvement.", flush=True)
            except Exception as e:
                print(f"[!] Error on N={n:3d}: {e}", flush=True)
                
    elapsed = time.time() - t0
    final_score = reg.get_score()
    print(f"\n============================================================")
    print(f"PARALLEL SWEEP COMPLETE ({elapsed:.1f}s)")
    print(f"Total instances improved: {improved_count}")
    print(f"Total score reduction:    {total_score_gain:.14f}")
    print(f"Final Certified Score:    {final_score:.14f}")
    print(f"============================================================", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=str, default="")
    parser.add_argument("--range", type=str, default="", help="e.g. 50-200")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--trials", type=int, default=4)
    args = parser.parse_args()
    
    reg = PackingRegistry()
    target_ns = []
    
    if args.instances:
        target_ns = [int(x.strip()) for x in args.instances.split(",") if x.strip()]
    elif args.range:
        lo, hi = map(int, args.range.split("-"))
        # Filter for untouched non-grid in range
        from src.validation.validator import load_submission_csv
        from src.geometry.sat import compute_enclosing_side
        ref_packings = load_submission_csv('data/reference/improved_submission_cpu.csv')
        for n in range(lo, hi + 1):
            reg_s = reg.get_side(n)
            sqn = math.sqrt(n)
            ceil_k = math.ceil(sqn)
            ref_s = compute_enclosing_side(ref_packings[n])
            is_improved = reg_s < ref_s - 1e-9
            is_non_grid = abs(reg_s - ceil_k) > 1e-5
            if is_non_grid and not is_improved:
                target_ns.append(n)
    else:
        print("Specify either --instances or --range")
        sys.exit(1)
        
    run_parallel_refine(target_ns, max_workers=args.workers, trials_per_instance=args.trials)
