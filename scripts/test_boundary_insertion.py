import os
import math
import subprocess
import time
import numpy as np
import pandas as pd
from pathlib import Path

from src.geometry.sat import validate_packing_sat, compute_enclosing_side
from src.optimizers.fq_engine import FQ_BIN, PACKER_BIN, read_engine_config, write_engine_config

def test_insert_n(donor_n: int, target_n: int, target_side: float = None):
    df = pd.read_csv('submissions/best_submission.csv')
    donor_df = df[df['n'] == donor_n]
    coords = donor_df[['x', 'y', 'angle']].to_numpy()
    
    val_donor = validate_packing_sat(coords, eps=1e-10)
    assert val_donor['valid'], f"Donor N={donor_n} is invalid!"
    donor_side = val_donor['side']
    
    if target_side is None:
        target_side = donor_side + 0.005 # slightly larger than donor, but still strictly below integer!
        
    print(f"Donor N={donor_n} side: {donor_side:.6f}. Target N={target_n} side: {target_side:.6f}")
    
    # Grid scan for candidate (x, y, theta) with minimum overlap
    grid_pts = 60
    xs = np.linspace(0.6, target_side - 0.6, grid_pts)
    ys = np.linspace(0.6, target_side - 0.6, grid_pts)
    thetas = [0.0, 0.45, -0.45, 0.785]
    
    best_cand = None
    min_dist_score = -1e9
    
    # We want a candidate point (cx, cy) that maximizes distance to nearest square center
    centers = coords[:, :2]
    for x in xs:
        for y in ys:
            dists = np.hypot(centers[:, 0] - x, centers[:, 1] - y)
            min_d = np.min(dists)
            if min_d > min_dist_score:
                min_dist_score = min_d
                best_cand = (x, y)
                
    print(f"Best void found at {best_cand} with clearance to nearest center = {min_dist_score:.4f}")
    
    # Try inserting with different initial angles
    success = False
    best_res_side = 1e9
    best_res_coords = None
    
    for theta in thetas:
        new_coords = np.vstack([coords, [[best_cand[0], best_cand[1], theta]]])
        
        in_file = f"/tmp/test_insert_in_{target_n}_{theta:.2f}.txt"
        out_file = f"/tmp/test_insert_out_{target_n}_{theta:.2f}.txt"
        write_engine_config(in_file, new_coords, target_side)
        
        # 1. Run packer relax with penalties
        cmd_rel = [
            str(PACKER_BIN), "relax",
            "--in", in_file,
            "--s", str(target_side),
            "--out", out_file
        ]
        res_rel = subprocess.run(cmd_rel, capture_output=True, text=True)
        if res_rel.returncode != 0 or not os.path.exists(out_file):
            continue
            
        # 2. Run fq quench
        q_out = f"/tmp/test_insert_q_{target_n}_{theta:.2f}.txt"
        cmd_q = [
            str(FQ_BIN), "quench",
            "--in", out_file,
            "--out", q_out,
            "--loosen", "1.002",
            "--round-r0", "0.03",
            "--round-compress",
            "--round-steps", "30"
        ]
        res_q = subprocess.run(cmd_q, capture_output=True, text=True, timeout=30)
        if res_q.returncode == 0 and os.path.exists(q_out):
            s_q, c_q = read_engine_config(q_out)
            val = validate_packing_sat(c_q, eps=1e-10)
            if val['valid']:
                print(f"[+] VALID PACKING FOUND for N={target_n}! Side = {val['side']:.8f}")
                if val['side'] < best_res_side:
                    best_res_side = val['side']
                    best_res_coords = c_q
                    success = True
            else:
                print(f"[-] Quench produced invalid packing (overlap {val['max_overlap']:.2e}) for theta={theta:.2f}")
                
        # Clean up
        for f in [in_file, out_file, q_out]:
            if os.path.exists(f): os.remove(f)
            
    if success:
        print(f"SUCCESS: N={target_n} packed in side {best_res_side:.8f} (vs integer {math.ceil(math.sqrt(target_n))})!")
    else:
        print(f"FAILED: Could not pack N={target_n} in side < {math.ceil(math.sqrt(target_n))}")

if __name__ == '__main__':
    test_insert_n(132, 133, target_side=11.999)
