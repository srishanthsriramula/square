"""Mondrian Dual-Piece Composer (itsnaka SQUISH method).
Places two Pareto-optimal corner pieces at opposite corners (LL and UR)
and uses exact MILP (HiGHS mis_fill2) to pack all intervening upright squares.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "evand" / "search" / "packer"))

from gen import mis_fill2, corners, pen, wall_viol
from src.geometry.sat import validate_packing_sat, compute_enclosing_side
from src.optimizers.fq_engine import FQ_BIN, write_engine_config, read_engine_config
from src.scoring.registry import PackingRegistry
from src.constructions.fault_dp import Gadget


def load_curated_gadgets():
    npz_data = np.load("data/generated/gadgets_coords.npz")
    gadgets = []
    with open("data/generated/gadgets.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            idx = int(parts[0])
            w, h, dm, f_val = int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4])
            name = parts[5]
            coords = npz_data[f"g_{idx}"]
            source_n = int(npz_data[f"meta_{idx}"][4])
            if dm >= 1 and f_val < 0.95 and w <= 16 and h <= 16:
                gadgets.append(Gadget(name=name, w=w, h=h, dm=dm, f=f_val, coords=coords, source_n=source_n))
    return gadgets


def run_dual_piece_search(target_n_list: list[int], max_candidates: int = 50):
    reg = PackingRegistry()
    gadgets = load_curated_gadgets()
    print(f"Loaded {len(gadgets)} curated corner gadgets for dual-piece composition.")
    
    # Sort gadgets by density / deficit
    gadgets.sort(key=lambda g: (-g.dm / (g.w * g.h + 1e-5), g.f))
    top_g = gadgets[:40]
    
    for n in target_n_list:
        cur_s = reg.get_side(n)
        target_s = cur_s - 0.005
        print(f"\nTargeting N={n} (current certified L={cur_s:.8f}, target L={target_s:.8f})...")
        
        found = False
        for i, gA in enumerate(top_g):
            if found: break
            for j, gB in enumerate(top_g):
                if gA.w + gB.w >= target_s or gA.h + gB.h >= target_s:
                    continue
                
                # Piece A at (0, 0) LL
                # Piece B at (target_s - gB_w, target_s - gB_h) UR
                sqA = [(float(x), float(y), math.degrees(th)) for x, y, th in gA.coords]
                
                # Shift B to UR corner
                dx_B = target_s - (gB.w + gB.f)
                dy_B = target_s - (gB.h + gB.f)
                sqB = [(float(x) + dx_B, float(y) + dy_B, math.degrees(th)) for x, y, th in gB.coords]
                
                # Check for overlap between A and B
                tilted = sqA + sqB
                has_overlap = False
                for qa in sqA:
                    for qb in sqB:
                        if pen(qa, qb) > 1e-6:
                            has_overlap = True
                            break
                    if has_overlap: break
                if has_overlap:
                    continue
                
                # Run exact MILP fill
                try:
                    ax = mis_fill2(target_s, tilted, tol=1e-5)
                    tot = len(tilted) + len(ax)
                    if tot >= n:
                        print(f"  [+] COMPOSITION FEASIBLE: gA={gA.name}, gB={gB.name} -> {tot} squares at s={target_s:.6f}!")
                        allsq = tilted + [tuple(q[:3]) if len(q) >= 3 else (q[0], q[1], 0.0) for q in ax][:n - len(tilted)]
                        
                        # Convert to radians
                        cand_coords = np.array([[x, y, math.radians(deg)] for x, y, deg in allsq])
                        
                        # Polish with fq quench
                        in_tmp = f"/tmp/dp_{n}_in.txt"
                        out_tmp = f"/tmp/dp_{n}_out.txt"
                        write_engine_config(in_tmp, cand_coords, target_s + 0.005)
                        
                        import subprocess
                        qcmd = [str(FQ_BIN), "quench", "--in", in_tmp, "--out", out_tmp, "--loosen", "1.0000", "--round-r0", "0.02", "--round-compress", "--round-steps", "25"]
                        r = subprocess.run(qcmd, capture_output=True, text=True, timeout=60)
                        if os.path.exists(out_tmp):
                            s_out, c_out = read_engine_config(out_tmp)
                            val = validate_packing_sat(c_out, eps=1e-10)
                            if val["valid"] and val["side"] < cur_s - 1e-10:
                                success, gain, msg = reg.register_candidate(n, c_out, source=f"mondrian_dual_{gA.name}_{gB.name}")
                                if success:
                                    print(f"  [*** BREAKTHROUGH ***] N={n} CERTIFIED! {cur_s:.8f} -> {val['side']:.8f} (gain={gain:.8f})")
                                    reg.save_submission()
                                    cur_s = val["side"]
                                    found = True
                                    break
                except Exception as e:
                    pass


if __name__ == "__main__":
    targets = [105, 106, 107, 108, 109, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 147, 150, 151]
    run_dual_piece_search(targets)
