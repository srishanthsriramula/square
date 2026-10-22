#!/usr/bin/env python3
"""Local High-Depth Boundary Cluster Attacker for macOS M3.

Attacks high-deficit transition clusters (K^2 - D for D in [18, 26]) where
density is high but vacancies allow large structural drops (e.g. N=460).
Uses deep packer hop (iters=600) + Clarabel IPM active-set polish.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.sat import validate_packing_sat
from src.optimizers.fq_engine import FQ_BIN, PACKER_BIN, read_engine_config, write_engine_config
from src.scoring.registry import PackingRegistry

LOG_FILE = Path(__file__).parent.parent / "local_attacker.log"


def log(msg: str):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def run_single_attack(
    n: int,
    seed: int,
    cur_side: float,
    coords: np.ndarray,
    iters: int = 600
) -> Tuple[int, bool, float, Optional[np.ndarray], str]:
    """Run deep basin hop and IPM quench on instance n."""
    in_file = f"/tmp/loc_in_{n}_{seed}.txt"
    hop_out = f"/tmp/loc_hop_{n}_{seed}.txt"
    q_out = f"/tmp/loc_q_{n}_{seed}.txt"

    try:
        write_engine_config(in_file, coords, cur_side)
        target_s = cur_side - 0.0012

        cmd_hop = [
            str(PACKER_BIN), "hop",
            "--in", in_file,
            "--s", f"{target_s:.9f}",
            "--etol", "1e-6",
            "--iters", str(iters),
            "--out", hop_out,
            "--seed", str(seed)
        ]

        subprocess.run(cmd_hop, capture_output=True, text=True, timeout=45)
        cand_file = hop_out if os.path.exists(hop_out) else f"{hop_out}.inf"

        if os.path.exists(cand_file):
            use_no_alm = os.path.exists(hop_out)
            cmd_q = [
                str(FQ_BIN), "quench",
                "--in", cand_file,
                "--out", q_out,
                "--loosen", "1.0000",
                "--outer", "25",
                "--pit", "80"
            ]
            if use_no_alm:
                cmd_q.append("--no-alm")

            subprocess.run(cmd_q, capture_output=True, text=True, timeout=25)
            if os.path.exists(q_out):
                new_side, new_coords = read_engine_config(q_out)
                if new_side < cur_side - 1e-10:
                    val = validate_packing_sat(new_coords, eps=1e-10)
                    if val["valid"] and val["side"] < cur_side - 1e-10:
                        return n, True, val["side"], new_coords, f"local_hop_s{seed}"

        return n, False, cur_side, None, "no_improvement"
    except Exception as e:
        return n, False, cur_side, None, f"error: {e}"
    finally:
        for f in [in_file, hop_out, f"{hop_out}.inf", q_out]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass


def main():
    reg = PackingRegistry()
    init_score = reg.get_score()

    log("============================================================")
    log("LOCAL DEEP BOUNDARY CLUSTER ATTACK ENGINE (APPLE SILICON)")
    log(f"Current Verified Score: {init_score:.12f}")
    log("============================================================")

    # High-leverage boundary clusters where large drops occur
    boundary_targets = [
        # K=16..20 (N=240..395)
        240, 241, 271, 272, 273, 305, 306, 307, 308, 341, 342, 343, 344,
        378, 379, 380, 381, 382,
        # K=21..25 (N=415..615)
        418, 419, 420, 421, 422, 458, 459, 461, 462, 463, 464,
        504, 505, 506, 507, 508, 509, 551, 552, 553, 554, 555,
        599, 600, 601, 602, 603,
        # K=26..32 (N=650..998)
        651, 652, 653, 654, 703, 704, 705, 706,
        757, 758, 759, 760, 814, 815, 816, 817,
        872, 873, 874, 875, 931, 932, 933, 934, 935,
        994, 995, 996, 997, 998
    ]

    workers = 6
    log(f"Targeting {len(boundary_targets)} high-leverage instances across {workers} parallel workers...")

    tasks = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for n in boundary_targets:
            cur_s = reg.get_side(n)
            coords = reg.get_packing(n)
            for trial in range(3):
                seed = 10000 + n * 37 + trial * 101
                tasks.append(executor.submit(run_single_attack, n, seed, cur_s, coords, 600))

        log(f"Submitted {len(tasks)} deep attack tasks to pool.")

        improved_count = 0
        total_gain = 0.0

        for future in as_completed(tasks):
            try:
                n, success, new_side, new_coords, source = future.result()
                if success:
                    cur_s = reg.get_side(n)
                    if new_side < cur_s - 1e-10:
                        registered, gain, msg = reg.register_candidate(n, new_coords, source=source)
                        if registered:
                            improved_count += 1
                            total_gain += gain
                            log(f"[+] [LOCAL BREAKTHROUGH] N={n} IMPROVED! side: {cur_s:.8f} -> {new_side:.8f} (gain={gain:.8f}) | New Score: {reg.get_score():.12f}")
                            reg.save_submission()
            except Exception as e:
                log(f"[!] Task error: {e}")

    log(f"\nCompleted! Improvements: {improved_count}, Total Gain: {total_gain:.10f}, Final Score: {reg.get_score():.12f}")


if __name__ == "__main__":
    main()
