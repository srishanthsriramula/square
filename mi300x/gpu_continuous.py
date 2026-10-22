#!/usr/bin/env python3
"""Continuous MI300X GPU Simulated Annealing Engine (ROCm / HIP).
Runs massively parallel CDNA3 chains (1024 blocks = 2048 chains) on high-value targets,
utilizing donor crystal seeding and continuous compression.
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

from src.geometry.sat import validate_packing_sat, compute_enclosing_side
from src.optimizers.fq_engine import FQ_BIN, PACKER_BIN, read_engine_config, write_engine_config
from src.scoring.registry import PackingRegistry

BUILD_BIN = Path(__file__).parent / "build" / "sa_hip"
LOG_FILE = Path(__file__).parent.parent / "live_engine.log"


def log(msg: str):
    print(msg, flush=True)


def get_donor_seed(n: int, reg: PackingRegistry, target_s: float) -> tuple[np.ndarray, float, bool]:
    """Provide donor crystal seed for integer-trapped instances from adjacent tilted packing."""
    cur_s = reg.get_side(n)
    ceil_s = math.ceil(math.sqrt(n))
    
    # If instance is currently locked at or above the integer ceiling, check donor n-1
    if cur_s >= ceil_s - 1e-4 and n > 1:
        prev_s = reg.get_side(n - 1)
        if prev_s < ceil_s - 1e-4:
            donor_coords = reg.get_packing(n - 1)
            centers = donor_coords[:, :2]
            
            # Find largest void in donor container
            grid_n = 50
            xs = np.linspace(0.6, target_s - 0.6, grid_n)
            ys = np.linspace(0.6, target_s - 0.6, grid_n)
            best_d, best_pt = -1.0, None
            for x in xs:
                for y in ys:
                    d = np.min(np.hypot(centers[:, 0] - x, centers[:, 1] - y))
                    if d > best_d:
                        best_d = d
                        best_pt = (x, y)
                        
            # Determine typical tilt of donor crystal
            non_zero = donor_coords[abs(donor_coords[:, 2]) > 0.05, 2]
            typical_tilt = float(np.median(non_zero)) if len(non_zero) > 0 else 0.45
            
            # Append inserted square into the void
            cand = np.vstack([donor_coords, [[best_pt[0], best_pt[1], typical_tilt]]])
            actual_target = ceil_s - 0.001
            return cand, actual_target, True
            
    # For general instances, target a reachable incremental step (e.g. 0.0006)
    return reg.get_packing(n), cur_s - 0.0006, False


def run_gpu_continuous():
    log("============================================================")
    log("[GPU MI300X] CONTINUOUS CDNA3 ACCELERATOR ONLINE (192GB VRAM)")
    log("[GPU MI300X] Dispersing 2,048 parallel chains @ 98 G-pairtests/s")
    log("============================================================")

    reg = PackingRegistry()
    
    # High-leverage targets and transition boundaries
    targets = [
        # Sweet-spot top score contributors
        51, 52, 53, 54, 55, 65, 66, 67, 68, 69, 70, 71,
        82, 83, 84, 85, 86, 87, 88, 89,
        101, 102, 103, 105, 106, 107, 108, 109, 110,
        122, 123, 124, 126, 127, 128, 129, 130, 131, 132,
        # Key transition boundaries
        111, 133, 157, 184, 212, 242, 274, 309, 345, 383, 423,
        603, 604, 654, 655, 706, 707, 760, 761, 816, 818, 875, 876, 935, 936, 998, 999
    ]

    while True:
        for n in targets:
            cur_s = reg.get_side(n)
            nominal_target = cur_s - 0.003
            coords, target_s, is_donor = get_donor_seed(n, reg, nominal_target)
            
            seed_file = f"/tmp/gpu_seed_{n}.txt"
            write_engine_config(seed_file, coords, cur_s if not is_donor else math.ceil(math.sqrt(n)))
            out_prefix = f"/tmp/gpu_cand_{n}"

            perturb = "0.010" if is_donor else "0.005"
            if n <= 150:
                sweeps = "4000"
            elif n <= 350:
                sweeps = "2000"
            elif n <= 650:
                sweeps = "1200"
            else:
                sweeps = "800"

            cmd = [
                str(BUILD_BIN),
                "--n", str(n),
                "--side", f"{target_s:.8f}",
                "--blocks", "2048",
                "--sweeps", sweeps,
                "--perturb", perturb,
                "--tele", "0.005",
                "--temp", "5e-3",
                "--quench", "0",
                "--keep", "4",
                "--out", out_prefix,
                "--in", seed_file,
                "--seed", str(int(time.time() * 1000) % 2000000000)
            ]

            log(f"[GPU MI300X] Launching 2048 annealing chains on N={n} (target L={target_s:.6f}, donor={is_donor}, sweeps={sweeps})...")
            t0 = time.time()
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                elapsed = time.time() - t0
                log(f"[GPU MI300X] N={n} evaluated across 2048 chains in {elapsed:.1f}s.")
            except Exception as e:
                log(f"[GPU MI300X] Error running N={n}: {e}")
                continue

            # Read candidates
            for k in range(4):
                cfile = f"{out_prefix}.{k}.txt"
                if os.path.exists(cfile):
                    try:
                        c_side, c_coords = read_engine_config(cfile)
                        
                        # 1. Immediate SAT check
                        val = validate_packing_sat(c_coords, eps=1e-10)
                        if val["valid"] and val["side"] < cur_s - 1e-10:
                            success, gain, msg = reg.register_candidate(n, c_coords, source=f"mi300x_gpu_raw_chain{k}")
                            if success:
                              log(
                                  f"[+] [GPU MI300X BREAKTHROUGH] N={n} IMPROVED"
                                  f" RAW! side: {cur_s:.8f} -> {val['side']:.8f}"
                                  f" (gain={gain:.8f}) | Score:"
                                  f" {reg.get_score():.12f}"
                              )
                              reg.save_submission()
                              cur_s = val["side"]
                              continue

                        # 2. Fast Polish via fq quench (Clarabel IPM active-set)
                        q_out = f"/tmp/gpu_q_{n}_{k}_out.txt"
                        
                        # First try instant polish via --no-alm (takes ~0.1s)
                        qcmd_fast = [
                            str(FQ_BIN), "quench",
                            "--in", cfile,
                            "--out", q_out,
                            "--loosen", "1.0000",
                            "--no-alm",
                            "--pit", "80"
                        ]
                        subprocess.run(qcmd_fast, capture_output=True, text=True, timeout=10)
                        if os.path.exists(q_out):
                            q_side, q_coords = read_engine_config(q_out)
                            if q_side < cur_s - 1e-10:
                                val = validate_packing_sat(q_coords, eps=1e-10)
                                if val["valid"] and val["side"] < cur_s - 1e-10:
                                    success, gain, msg = reg.register_candidate(n, q_coords, source=f"mi300x_gpu_noalm_chain{k}")
                                    if success:
                                        log(f"[+] [GPU MI300X BREAKTHROUGH] N={n} IMPROVED FAST! side: {cur_s:.8f} -> {val['side']:.8f} (gain={gain:.8f}) | Score: {reg.get_score():.12f}")
                                        reg.save_submission()
                                        cur_s = val["side"]
                                        if os.path.exists(q_out): os.remove(q_out)
                                        if os.path.exists(cfile): os.remove(cfile)
                                        continue

                        # Full ALM polish fallback: only for near-feasible candidates (margin >= -0.05) and top 2 chains
                        if val.get("min_margin", -1.0) < -0.05 or k > 1:
                            if os.path.exists(q_out): os.remove(q_out)
                            if os.path.exists(cfile): os.remove(cfile)
                            continue

                        qcmd = [
                            str(FQ_BIN), "quench",
                            "--in", cfile,
                            "--out", q_out,
                            "--loosen", "1.0000",
                            "--outer", "25",
                            "--pit", "80"
                        ]
                        qres = subprocess.run(qcmd, capture_output=True, text=True, timeout=45)
                        if qres.returncode == 0 and os.path.exists(q_out):
                            q_side, q_coords = read_engine_config(q_out)
                            if q_side < cur_s - 1e-10:
                                val = validate_packing_sat(q_coords, eps=1e-10)
                                if val["valid"] and val["side"] < cur_s - 1e-10:
                                    success, gain, msg = reg.register_candidate(n, q_coords, source=f"mi300x_gpu_chain{k}")
                                    if success:
                                      log(
                                          "[+] [GPU MI300X BREAKTHROUGH] N="
                                          f"{n} IMPROVED! side: {cur_s:.8f} ->"
                                          f" {val['side']:.8f}"
                                          f" (gain={gain:.8f}) | Score:"
                                          f" {reg.get_score():.12f}"
                                      )
                                      reg.save_submission()
                                      cur_s = val["side"]
                        if os.path.exists(q_out):
                            os.remove(q_out)
                        if os.path.exists(cfile):
                            os.remove(cfile)
                    except Exception as err:
                        log(f"[!] Error processing candidate N={n} chain {k}: {err}")


if __name__ == "__main__":
    run_gpu_continuous()
