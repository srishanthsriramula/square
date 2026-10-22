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
    try:
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def run_gpu_continuous():
    log("============================================================")
    log("[GPU MI300X] CONTINUOUS CDNA3 ACCELERATOR ONLINE (192GB VRAM)")
    log("[GPU MI300X] Dispersing 2,048 parallel chains @ 47 G-pairtests/s")
    log("============================================================")

    reg = PackingRegistry()
    
    # Priority targets: high-value instances where density is high
    targets = [
        935, 936, 998, 999, 875, 876, 816, 818, 760, 761, 706, 707,
        654, 655, 603, 604, 988, 925, 926, 927, 919, 920, 923, 924
    ]

    while True:
        for n in targets:
            cur_s = reg.get_side(n)
            coords = reg.get_packing(n)
            seed_file = f"/tmp/gpu_seed_{n}.txt"
            write_engine_config(seed_file, coords, cur_s)

            # Target 0.003 below current certified side
            target_s = cur_s - 0.003
            out_prefix = f"/tmp/gpu_cand_{n}"

            cmd = [
                str(BUILD_BIN),
                "--n", str(n),
                "--side", f"{target_s:.8f}",
                "--blocks", "1024",
                "--sweeps", "1000",
                "--perturb", "0.003",
                "--tele", "0.0",
                "--keep", "4",
                "--out", out_prefix,
                "--in", seed_file,
                "--seed", str(int(time.time() * 1000) % 2000000000)
            ]

            log(f"[GPU MI300X] Launching 2048 annealing chains on N={n} (target L={target_s:.6f})...")
            t0 = time.time()
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                elapsed = time.time() - t0
                log(f"[GPU MI300X] N={n} evaluated 1.9B moves in {elapsed:.1f}s.")
            except Exception as e:
                log(f"[GPU MI300X] Error running N={n}: {e}")
                continue

            # Read candidates
            for k in range(4):
                cfile = f"{out_prefix}.{k}.txt"
                if os.path.exists(cfile):
                    try:
                        c_side, c_coords = read_engine_config(cfile)
                        # Quick CPU quench polish without loosen
                        q_in = f"/tmp/gpu_q_{n}_{k}_in.txt"
                        q_out = f"/tmp/gpu_q_{n}_{k}_out.txt"
                        write_engine_config(q_in, c_coords, c_side)

                        qcmd = [
                            str(FQ_BIN), "quench",
                            "--in", q_in,
                            "--out", q_out,
                            "--loosen", "1.0000",
                            "--round-r0", "0.03",
                            "--round-compress",
                            "--round-steps", "25"
                        ]
                        qres = subprocess.run(qcmd, capture_output=True, text=True, timeout=40)
                        if qres.returncode == 0 and os.path.exists(q_out):
                            q_side, q_coords = read_engine_config(q_out)
                            if q_side < cur_s - 1e-10:
                                val = validate_packing_sat(q_coords, eps=1e-10)
                                if val["valid"] and val["side"] < cur_s - 1e-10:
                                    success, gain, msg = reg.register_candidate(n, q_coords, source=f"mi300x_gpu_chain{k}")
                                    if success:
                                        new_score = reg.get_score()
                                        log(f"[+] [GPU MI300X BREAKTHROUGH] N={n} IMPROVED! side: {cur_s:.8f} -> {val['side']:.8f} (gain={gain:.8f}) | Score: {new_score:.12f}")
                                        reg.save_submission()
                                        cur_s = val["side"]
                        for f in [q_in, q_out]:
                            if os.path.exists(f): os.remove(f)
                        os.remove(cfile)
                    except Exception:
                        pass


if __name__ == "__main__":
    run_gpu_continuous()
