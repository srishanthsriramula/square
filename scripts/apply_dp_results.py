"""Apply and verify C++ DP solver results against current best packings."""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.scoring.registry import PackingRegistry
from src.geometry.sat import validate_packing_sat, compute_enclosing_side
from src.constructions.fault_dp import Gadget, compose_packing

def apply_dp():
    reg = PackingRegistry()
    print(f"Current score before C++ DP merge: {reg.get_score():.14f}")
    
    df_dp = pd.read_csv("data/generated/dp_results.csv")
    npz_data = np.load("data/generated/gadgets_coords.npz")
    
    # Map name -> (Gadget)
    gadgets_by_name = {}
    # Read gadgets.txt
    with open("data/generated/gadgets.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            idx = int(parts[0])
            w = int(parts[1])
            h = int(parts[2])
            dm = int(parts[3])
            f_val = float(parts[4])
            name = parts[5]
            coords = npz_data[f"g_{idx}"]
            gadgets_by_name[name] = Gadget(
                name=name,
                w=w,
                h=h,
                dm=dm,
                f=f_val,
                coords=coords,
                source_n=int(npz_data[f"meta_{idx}"][4])
            )
            
    registered_count = 0
    total_gain = 0.0
    
    for _, row in df_dp.iterrows():
        K = int(row["K"])
        D = int(row["D"])
        f_val = float(row["f"])
        n = K * K + D
        if not (1 <= n <= 1000):
            continue
            
        cur_side = reg.get_side(n)
        cand_side = K + f_val
        gain = (cur_side - cand_side) / math.sqrt(n)
        
        if gain > 1e-10:
            names = [s.strip() for s in str(row["gadget_names"]).split(";") if s.strip()]
            chain = [gadgets_by_name[nm] for nm in names if nm in gadgets_by_name]
            if len(chain) != len(names):
                continue
                
            try:
                f_actual = max([f_val] + [g.f for g in chain]) + 1e-12
                cand_arr = compose_packing(K, D, f_actual, chain)
                # SAT validation
                val = validate_packing_sat(cand_arr, eps=1e-10)
                if val["valid"] and val["side"] < cur_side - 1e-10:
                    success, g, msg = reg.register_candidate(
                        n,
                        cand_arr,
                        source=f"cpp_fault_dp_K{K}_D{D}"
                    )
                    if success:
                        registered_count += 1
                        total_gain += g
                        print(f"IMPROVED n={n}: cur={cur_side:.8f} -> new={val['side']:.8f}, gain={g:.10f}")
                elif not val["valid"]:
                    # print debug for potential improvements
                    if val.get("overlap_depth", 0) > 1e-8:
                        pass
            except Exception as e:
                print(f"Error composing n={n}: {e}")
                
    print(f"\nTotal new registered improvements: {registered_count}")
    print(f"Total score gain: {total_gain:.10f}")
    print(f"New competition score: {reg.get_score():.14f}")
    reg.save_submission()
    print("Saved submission to submissions/best_submission.csv")

if __name__ == "__main__":
    apply_dp()
