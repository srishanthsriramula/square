#!/usr/bin/env python3
"""High-Performance Parallel Basin Hopping Engine (Xeon Parallel Event-Chain + Penalty Softening).

Uses compiled `packer hhop` and `packer hop` to escape jammed local minima
via event-chain displacements, penalty softening, and Clarabel/QDLDL active-set squeeze.
Runs concurrently across Xeon Platinum cores.
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
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.sat import validate_packing_sat
from src.optimizers.fq_engine import PACKER_BIN, FQ_BIN, read_engine_config, write_engine_config
from src.scoring.registry import PackingRegistry

LOG_FILE = Path(__file__).parent.parent / "live_engine.log"


def log(msg: str):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def run_single_hop(
    n: int,
    seed: int,
    cur_side: float,
    coords: np.ndarray,
    iters: int = 300,
    soft: float = 0.35,
    timeout: float = 120.0
) -> tuple[int, bool, float, Optional[np.ndarray], str]:
    """Execute a single packer hhop basin hopping run on instance n."""
    in_file = f"/tmp/p_hop_{n}_{seed}_in.txt"
    out_file = f"/tmp/p_hop_{n}_{seed}_out.txt"
    
    try:
        write_engine_config(in_file, coords, cur_side)
        cmd = [
            str(PACKER_BIN), "hhop",
            "--in", in_file,
            "--iters", str(iters),
            "--soft", str(soft),
            "--chains", "8",
            "--temp", "2e-4",
            "--gmax", "0.015",
            "--out", out_file,
            "--seed", str(seed)
        ]
        
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if res.returncode == 0 and os.path.exists(out_file):
            new_side, new_coords = read_engine_config(out_file)
            if new_side < cur_side - 1e-10:
                val = validate_packing_sat(new_coords, eps=1e-10)
                if val["valid"] and val["side"] < cur_side - 1e-10:
                    return n, True, val["side"], new_coords, f"hhop_s{seed}"
                    
        # Fallback: if not improved, check if packer relax --squeeze tightens directly
        sq_out = f"/tmp/p_sq_{n}_{seed}_out.txt"
        sq_cmd = [
            str(PACKER_BIN), "relax",
            "--in", in_file,
            "--squeeze",
            "--out", sq_out
        ]
        sq_res = subprocess.run(sq_cmd, capture_output=True, text=True, timeout=30)
        if sq_res.returncode == 0 and os.path.exists(sq_out):
            sq_side, sq_coords = read_engine_config(sq_out)
            if sq_side < cur_side - 1e-10:
                val = validate_packing_sat(sq_coords, eps=1e-10)
                if val["valid"] and val["side"] < cur_side - 1e-10:
                    return n, True, val["side"], sq_coords, f"relax_squeeze_s{seed}"
                    
        return n, False, cur_side, None, "no_improvement"
    except Exception as e:
        return n, False, cur_side, None, f"error: {e}"
    finally:
        for f in [in_file, out_file, f"/tmp/p_sq_{n}_{seed}_out.txt"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description="Parallel Basin Hopping Engine")
    parser.add_argument("--workers", type=int, default=16, help="Worker processes")
    parser.add_argument("--iters", type=int, default=300, help="Hhop iterations per trial")
    parser.add_argument("--trials", type=int, default=3, help="Seeds per instance")
    args = parser.parse_args()

    reg = PackingRegistry()
    init_score = reg.get_score()
    
    log("============================================================")
    log("PARALLEL BASIN HOPPING ENGINE ONLINE (XEON MULTI-CORE)")
    log(f"Workers: {args.workers} | Hhop Iters: {args.iters} | Trials/Target: {args.trials}")
    log(f"Current Certified Score: {init_score:.12f}")
    log("============================================================")

    # Prioritize targets by score leverage and non-grid potential
    priority_targets = [
        # Top score leverage instances
        51, 52, 53, 54, 55, 65, 66, 67, 68, 69, 70, 71,
        82, 83, 84, 85, 86, 87, 88, 89,
        101, 102, 103, 105, 106, 107, 108, 109, 110,
        122, 123, 124, 126, 127, 128, 129, 130, 131, 132,
        146, 147, 148, 150, 153, 154, 155, 156,
        172, 173, 175, 176, 180, 181, 182, 183,
        199, 200, 203, 204, 208, 209, 210, 211,
        227, 228, 229, 230, 232, 233, 237, 238, 239, 240, 241,
        258, 260, 261, 263, 264, 265, 268, 270, 271, 272, 273,
        293, 294, 295, 297, 298, 306, 307, 308,
        # Transition boundaries
        111, 133, 157, 184, 212, 242, 274, 309, 345, 383, 423
    ]

    round_num = 0
    while True:
        round_num += 1
        log(f"\n--- Starting Parallel Basin Hop Round {round_num} ({len(priority_targets)} targets) ---")
        
        tasks = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for n in priority_targets:
                cur_s = reg.get_side(n)
                coords = reg.get_packing(n)
                for t in range(args.trials):
                    seed = int(time.time() * 1000 + n * 100 + t) % 1000000007
                    tasks.append(
                        executor.submit(run_single_hop, n, seed, cur_s, coords, args.iters)
                    )

            for future in as_completed(tasks):
                try:
                    n, improved, new_side, new_coords, src = future.result()
                    if improved and new_coords is not None:
                        old_s = reg.get_side(n)
                        if new_side < old_s - 1e-10:
                            success, gain, msg = reg.register_candidate(n, new_coords, source=f"hhop_{src}")
                            if success:
                                new_score = reg.get_score()
                                log(f"[+] [XEON BREAKTHROUGH] N={n} IMPROVED! side: {old_s:.8f} -> {new_side:.8f} (gain={gain:.8f}) | Score: {new_score:.12f}")
                                reg.save_submission()
                except Exception as e:
                    log(f"[!] Worker exception: {e}")

        time.sleep(1)


if __name__ == "__main__":
    main()
