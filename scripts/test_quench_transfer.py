"""Cross-instance transfer with ALM + IPM quench relaxation.

Hypothesis: Removing a square from packing N+1 creates a void.
Running fq quench allows the remaining N squares to relax into the void,
potentially compressing the container side length significantly below L_N.
"""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.scoring.registry import PackingRegistry
from src.geometry.sat import compute_enclosing_side, compute_half_extents, validate_packing_sat
from src.optimizers.fq_engine import quench_packing

def test_transfer_quench(sample_ns: list, timeout_per_instance: float = 20.0):
    reg = PackingRegistry()
    print(f"Testing Transfer-and-Quench on {len(sample_ns)} instances...")
    print(f"Current Total Score: {reg.get_score():.14f}")
    
    total_gain = 0.0
    improved_list = []
    
    for n in sample_ns:
        if n >= 1000:
            continue
        cur_side = reg.get_side(n)
        higher_packing = reg.get_packing(n + 1)
        higher_side = reg.get_side(n + 1)
        
        # Don't try if higher side is vastly larger than current side
        if higher_side > cur_side + 0.5:
            continue
            
        # Find which squares are on the boundary
        h = compute_half_extents(higher_packing[:, 2])
        xmin = np.min(higher_packing[:, 0] - h)
        ymin = np.min(higher_packing[:, 1] - h)
        xmax = np.max(higher_packing[:, 0] + h)
        ymax = np.max(higher_packing[:, 1] + h)
        
        # Candidate squares to drop: those closest to the 4 walls
        dist_to_wall = np.minimum.reduce([
            higher_packing[:, 0] - h - xmin,
            xmax - (higher_packing[:, 0] + h),
            higher_packing[:, 1] - h - ymin,
            ymax - (higher_packing[:, 1] + h)
        ])
        
        # Sort squares by proximity to boundary
        boundary_candidates = np.argsort(dist_to_wall)[:5]
        
        best_cand_side = cur_side
        best_cand_arr = None
        
        for drop_idx in boundary_candidates:
            sub = np.delete(higher_packing, drop_idx, axis=0)
            raw_side = compute_enclosing_side(sub)
            
            # Quench the remaining N squares
            ok, q_side, q_arr, _ = quench_packing(sub, kick=0.0005, timeout_sec=timeout_per_instance)
            test_side = q_side if ok else raw_side
            test_arr = q_arr if ok else sub
            
            if test_side < best_cand_side - 1e-9:
                # SAT verify
                val = validate_packing_sat(test_arr, eps=1e-10)
                if val["valid"]:
                    best_cand_side = test_side
                    best_cand_arr = test_arr
                    
        gain = (cur_side - best_cand_side) / math.sqrt(n)
        if gain > 1e-10 and best_cand_arr is not None:
            success, g, msg = reg.register_candidate(
                n,
                best_cand_arr,
                source=f"transfer_quench_from_n{n+1}"
            )
            if success:
                improved_list.append(n)
                total_gain += g
                print(f"SUCCESS n={n}: cur={cur_side:.8f} -> new={best_cand_side:.8f}, gain={g:.10f}")
                reg.save_submission()
        else:
            print(f"n={n}: no gain (best={best_cand_side:.8f}, cur={cur_side:.8f})")
            
    print(f"\nCompleted Transfer-and-Quench: {len(improved_list)} improved, total gain: {total_gain:.10f}")
    print(f"New score: {reg.get_score():.14f}")

if __name__ == "__main__":
    # Test on non-grid cases between 100 and 250
    reg = PackingRegistry()
    test_ns = []
    for n in range(50, 300):
        L = reg.get_side(n)
        if L < math.ceil(math.sqrt(n)) - 1e-4:
            test_ns.append(n)
    print(f"Found {len(test_ns)} non-grid candidates in [50, 300]")
    # Run first 20 candidates
    test_transfer_quench(test_ns[:20], timeout_per_instance=10.0)
