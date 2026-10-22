#!/usr/bin/env python3
"""High-Throughput Batch Cracking Orchestrator for AMD Instinct MI300X (ROCm / HIP).

Tackles high-deficit integer grid instances and difficult packing frontiers:
1. Dispatches massively parallel GPU simulated annealing chains (4,096 - 16,384 chains)
2. Collects near-feasible candidate packings from GPU
3. Polishes candidates on CPU via Active-Set ALM (fq quench) and Trust-Region SLP
4. Validates geometry with strict double-precision Separating Axis Theorem (eps=1e-10)
5. Registers improvements, triggers downward monotonicity cascades, and pushes git commits
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.sat import validate_packing_sat, compute_enclosing_side
from src.optimizers.fq_engine import FQ_BIN, PACKER_BIN, read_engine_config, write_engine_config
from src.scoring.registry import PackingRegistry

MI300X_DIR = Path(__file__).parent
BUILD_BIN = MI300X_DIR / "build" / "sa_hip"
CANDIDATES_DIR = MI300X_DIR / "candidates"


def get_high_deficit_grid_targets(min_deficit: int = 5) -> List[int]:
    """Return all grid instances with (K+1)^2 - N >= min_deficit, sorted by deficit descending."""
    targets = []
    for K in range(2, 32):
        for n in range(K * (K + 1), (K + 1) ** 2):
            deficit = (K + 1) ** 2 - n
            if deficit >= min_deficit:
                targets.append((n, K + 1, deficit))
    
    # Sort primarily by potential score gain (deficit / sqrt(n)) descending
    targets.sort(key=lambda x: x[2] / math.sqrt(x[0]), reverse=True)
    return [t[0] for t in targets]


def crack_instance_on_gpu(
    n: int,
    cur_side: float,
    seed_file: Optional[str] = None,
    blocks: int = 256,
    sweeps: int = 500,
    target_side: Optional[float] = None
) -> List[Tuple[float, np.ndarray]]:
    """Run GPU simulated annealing on MI300X for instance n, returning candidate packings."""
    if not BUILD_BIN.exists():
        print(f"Error: GPU binary {BUILD_BIN} not found! Compile with hipcc first.")
        return []

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    out_prefix = str(CANDIDATES_DIR / f"cand_{n}")
    
    if target_side is None:
        # Target a compression strictly below the integer grid bound
        if cur_side == round(cur_side):
            target_side = cur_side - 0.05
        else:
            target_side = cur_side - 0.01

    cmd = [
        str(BUILD_BIN),
        "--n", str(n),
        "--side", f"{target_side:.9f}",
        "--blocks", str(blocks),
        "--sweeps", str(sweeps),
        "--keep", "8",
        "--out", out_prefix,
        "--seed", str(int(time.time() * 1000) % 2000000000)
    ]
    if seed_file and os.path.exists(seed_file):
        cmd.extend(["--in", seed_file])

    print(f"[GPU MI300X] Launching {blocks * 32} annealing chains for N={n} (target L={target_side:.6f})...")
    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - t0
        print(f"[GPU MI300X] Finished in {elapsed:.2f}s. Output summary: {res.stdout.splitlines()[-3:] if res.stdout else 'None'}")
    except subprocess.TimeoutExpired:
        print(f"[GPU MI300X] Timed out on N={n}")
        return []
    except Exception as e:
        print(f"[GPU MI300X] Error running kernel: {e}")
        return []

    # Read output candidate files
    candidates = []
    for k in range(8):
        cfile = f"{out_prefix}.{k}.txt"
        if os.path.exists(cfile):
            try:
                coords = []
                with open(cfile) as f:
                    for line in f:
                        if line.startswith("#"):
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if len(coords) == n:
                    arr = np.array(coords)
                    side = compute_enclosing_side(arr)
                    candidates.append((side, arr))
                os.remove(cfile)
            except Exception:
                pass
    return candidates


def polish_and_certify(n: int, coords: np.ndarray, best_cur_side: float) -> Optional[Tuple[float, np.ndarray]]:
    """Polish a candidate packing using fq quench and certify with double-precision SAT."""
    in_file = f"/tmp/polish_{n}_in.txt"
    out_file = f"/tmp/polish_{n}_out.txt"
    cur_side = compute_enclosing_side(coords)
    write_engine_config(in_file, coords, cur_side)

    # Active-set ALM polish
    cmd = [
        str(FQ_BIN), "quench",
        "--in", in_file,
        "--out", out_file,
        "--loosen", "1.01",
        "--round-r0", "0.05",
        "--round-compress",
        "--round-steps", "30"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode == 0 and os.path.exists(out_file):
            cand_side, cand_coords = read_engine_config(out_file)
            val = validate_packing_sat(cand_coords, eps=1e-10, exhaustive=(n <= 50))
            if val["valid"] and val["side"] < best_cur_side - 1e-10:
                return (val["side"], cand_coords)
    except Exception:
        pass
    finally:
        for f in [in_file, out_file]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
    return None


def run_mi300x_campaign(targets: List[int], blocks: int = 256, sweeps: int = 500):
    reg = PackingRegistry()
    initial_score = reg.get_score()
    print("=" * 70)
    print("AMD INSTINCT MI300X MASSIVE BATCH PACKING ENGINE")
    print(f"Target count: {len(targets)} | Blocks: {blocks} ({blocks*32} parallel chains/target)")
    print(f"Initial Certified Score: {initial_score:.14f}")
    print("=" * 70 + "\n", flush=True)

    improved_count = 0
    total_gain = 0.0

    for idx, n in enumerate(targets):
        cur_side = reg.get_side(n)
        cur_coords = reg.get_packing(n)
        seed_file = f"/tmp/seed_{n}.txt"
        write_engine_config(seed_file, cur_coords, cur_side)

        print(f"\n[{idx+1}/{len(targets)}] Processing N={n} (Current L={cur_side:.6f})...")
        candidates = crack_instance_on_gpu(n, cur_side, seed_file=seed_file, blocks=blocks, sweeps=sweeps)

        best_found_side = cur_side
        best_found_coords = None

        for cand_side, cand_coords in candidates:
            # Polish on CPU
            polished = polish_and_certify(n, cand_coords, best_found_side)
            if polished is not None:
                p_side, p_coords = polished
                if p_side < best_found_side - 1e-10:
                    best_found_side = p_side
                    best_found_coords = p_coords

        if best_found_coords is not None and best_found_side < cur_side - 1e-10:
            gain = (cur_side - best_found_side) / math.sqrt(n)
            print(f"  [CRACKED] N={n}: {cur_side:.10f} -> {best_found_side:.10f} (Gain: +{gain:.8f})")
            reg.register_candidate(n, best_found_coords, source=f"mi300x_hip_sa")
            improved_count += 1
            total_gain += gain
            
            # Auto-save submission and cascade
            reg.save_submission()
            print(f"  --> Updated Certified Score: {reg.get_score():.14f}")
        else:
            print(f"  [-] N={n}: No breakthrough this iteration.")

        if os.path.exists(seed_file):
            try: os.remove(seed_file)
            except: pass

    final_score = reg.get_score()
    print("\n" + "=" * 70)
    print("MI300X CAMPAIGN COMPLETE")
    print(f"Instances cracked: {improved_count}/{len(targets)}")
    print(f"Total Score Drop:  -{total_gain:.14f}")
    print(f"Final Score:       {final_score:.14f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MI300X Batch Cracker")
    parser.add_argument("--targets", type=str, default="", help="Comma-separated target Ns")
    parser.add_argument("--all-grid", action="store_true", help="Target all 184 high-deficit grid instances")
    parser.add_argument("--blocks", type=int, default=256, help="GPU blocks (PPB=32, so total chains = blocks*32)")
    parser.add_argument("--sweeps", type=int, default=500, help="Simulated annealing sweeps")
    args = parser.parse_args()

    if args.targets:
        targets = [int(x.strip()) for x in args.targets.split(",") if x.strip()]
    elif args.all_grid:
        targets = get_high_deficit_grid_targets(min_deficit=10)
    else:
        # Default to top 30 highest deficit instances
        targets = get_high_deficit_grid_targets(min_deficit=15)[:30]

    run_mi300x_campaign(targets, blocks=args.blocks, sweeps=args.sweeps)
