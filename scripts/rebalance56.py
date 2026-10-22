"""Rebalance N=56 crystal packing to break the Integer Chain Obstruction.

Moves Square 1 out of Column 0 into the void at (2.5, 5.5),
and moves Square 18 out of Row 0 into the void at (7.4, 6.4),
ensuring that EVERY row and column has at most 7 squares.
Then relaxes at target side S < 8.000000000000.
"""

import math
import os
import subprocess
import numpy as np

from src.geometry.sat import validate_packing_sat
from src.optimizers.fq_engine import FQ_BIN, PACKER_BIN, read_engine_config, write_engine_config
from src.scoring.registry import PackingRegistry

def main():
    reg = PackingRegistry()
    cur_56 = reg.get_side(56)
    print(f"Current N=56 side in registry: {cur_56:.10f}")

    in_file = "/tmp/crystal56_final.txt"
    if not os.path.exists(in_file):
        print(f"File {in_file} does not exist.")
        return

    # Read configuration (degrees)
    s_stated, coords = read_engine_config(in_file)
    print(f"Loaded {len(coords)} squares from {in_file} with side {s_stated:.12f}")

    # Inspect square 1 and square 18
    # Square 1 is at x ~ 0.5, y ~ 7.5
    # Square 18 is at x ~ 7.5, y ~ 0.5
    print(f"Square 1 original:  x={coords[1, 0]:.4f}, y={coords[1, 1]:.4f}, th={np.rad2deg(coords[1, 2]):.2f} deg")
    print(f"Square 18 original: x={coords[18, 0]:.4f}, y={coords[18, 1]:.4f}, th={np.rad2deg(coords[18, 2]):.2f} deg")

    # Shift Square 1 to (2.50, 5.55, 45.0 deg)
    coords[1] = [2.50, 5.55, np.deg2rad(45.0)]

    # Shift Square 18 to (7.35, 6.35, 45.0 deg)
    coords[18] = [7.35, 6.35, np.deg2rad(45.0)]

    # Verify column and row counts
    print("\nVerifying row/column counts after rebalancing:")
    for k in range(8):
        col_cnt = np.sum((coords[:, 0] >= k) & (coords[:, 0] < k + 1))
        row_cnt = np.sum((coords[:, 1] >= k) & (coords[:, 1] < k + 1))
        print(f"  Index {k}: Col count = {col_cnt}, Row count = {row_cnt}")
        assert col_cnt <= 7, f"Column {k} still has > 7 squares ({col_cnt})"
        assert row_cnt <= 7, f"Row {k} still has > 7 squares ({row_cnt})"

    print("\nSUCCESS: All columns and rows have at most 7 squares! The integer chain is broken.")

    # Target sides to test: 7.995, 7.990, 7.985
    targets = [7.995, 7.990, 7.985, 7.980]
    best_c = None
    best_s = cur_56

    for target_s in targets:
        in_p = f"/tmp/reb56_in_{target_s:.3f}.txt"
        out_p = f"/tmp/reb56_out_{target_s:.3f}.txt"
        write_engine_config(in_p, coords, target_s)

        cmd_rel = [
            str(PACKER_BIN), "relax",
            "--in", in_p,
            "--s", str(target_s),
            "--out", out_p
        ]
        res = subprocess.run(cmd_rel, capture_output=True, text=True)
        energy = float("inf")
        if res.returncode == 0:
            for part in res.stdout.split():
                if part.startswith("E="):
                    try: energy = float(part.split("=")[1])
                    except: pass
        print(f"Target s={target_s:.3f}: relax E={energy:.4e}")

        if os.path.exists(out_p) and energy < 1e-3:
            # Basin hopping around this configuration
            hop_out = f"/tmp/reb56_hop_{target_s:.3f}.txt"
            cmd_hop = [
                str(PACKER_BIN), "hop",
                "--in", out_p,
                "--s", str(target_s),
                "--iters", "2000",
                "--seed", "1234",
                "--amp", "0.2",
                "--out", hop_out
            ]
            res_hop = subprocess.run(cmd_hop, capture_output=True, text=True)
            t_file = hop_out if os.path.exists(hop_out) else f"{hop_out}.inf"
            if not os.path.exists(t_file):
                t_file = out_p

            # Quench with fq
            q_out = f"/tmp/reb56_q_{target_s:.3f}.txt"
            cmd_q = [
                str(FQ_BIN), "quench",
                "--in", t_file,
                "--out", q_out,
                "--round-r0", "0.06",
                "--round-compress",
                "--loosen", "1.000",
                "--seed", "42"
            ]
            res_q = subprocess.run(cmd_q, capture_output=True, text=True)
            if os.path.exists(q_out):
                fin_s, fin_c = read_engine_config(q_out)
                val = validate_packing_sat(fin_c, eps=1e-10)
                print(f"  Quench result: side={fin_s:.10f}, SAT valid={val['valid']}, actual_side={val['side']:.10f}, margin={val.get('min_margin')}")
                if val["valid"] and val["side"] < 8.0 - 1e-10:
                    gain = (cur_56 - val["side"]) / math.sqrt(56)
                    print(f"\n*** BREAKTHROUGH: N=56 PACKED IN {val['side']:.10f} < 8.0! ***")
                    reg.register_candidate(56, fin_c, source="rebalance_crystal_56")
                    reg.save_submission()
                    break

if __name__ == "__main__":
    main()
