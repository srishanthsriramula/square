#!/usr/bin/env python3
"""Continuous Multi-Range Active-Set Sweep Driver for 20-Core Xeon Cluster.
Continuously cycles through high-value instance ranges with multi-strategy active-set unjamming.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PYTHON_BIN = sys.executable
SCRIPT_PATH = Path(__file__).parent / "parallel_refine.py"

RANGES = [
    (800, 1000),
    (500, 800),
    (200, 500),
    (50, 200),
    (1, 1000)
]

def main():
    print(f"============================================================", flush=True)
    print(f"CONTINUOUS MULTI-RANGE AUTONOMOUS SWEEP ONLINE", flush=True)
    print(f"============================================================\n", flush=True)

    while True:
        for r_start, r_end in RANGES:
            print(f"\n>>> LAUNCHING SWEEP FOR RANGE {r_start}-{r_end} <<<", flush=True)
            cmd = [
                PYTHON_BIN,
                str(SCRIPT_PATH),
                "--range", f"{r_start}-{r_end}",
                "--workers", "16",
                "--trials", "4"
            ]
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(f"Error in sweep {r_start}-{r_end}: {e}", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
