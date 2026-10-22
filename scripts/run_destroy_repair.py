"""Driver for Large-Neighborhood Destroy-and-Repair optimization."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimizers.destroy_repair import destroy_and_repair
from src.scoring.registry import PackingRegistry
from src.geometry.sat import validate_packing_sat


def run_dr_search(instances: list[int], num_passes: int = 4):
    reg = PackingRegistry()
    print(f"Starting Destroy-and-Repair search on {len(instances)} instances...")
    print(f"Current Verified Score: {reg.get_score():.14f}\n")
    
    total_gain = 0.0
    improved = []
    
    for idx, n in enumerate(instances):
        cur_s = reg.get_side(n)
        print(f"[{idx+1}/{len(instances)}] Optimizing N={n:3d} (cur_side={cur_s:.8f})...", flush=True)
        
        coords = reg.get_packing(n)
        best_s = cur_s
        best_c = coords
        
        # Test targets: descend by 1e-5, 2e-5, 5e-5
        for p in range(num_passes):
            seed = 5000 + p * 13
            k_rem = 4 + (p % 4) * 2  # 4, 6, 8, 10
            target_s = best_s - 1e-5 * (p + 1)
            t0 = time.time()
            ok, new_s, new_c = destroy_and_repair(
                best_c, best_s, target_s,
                k_remove=k_rem, seed=seed
            )
            if ok:
                g = (best_s - new_s) / math.sqrt(n)
                success, _, _ = reg.register_candidate(n, new_c, source=f"destroy_repair_k{k_rem}_seed{seed}")
                if success:
                    best_s = new_s
                    best_c = new_c
                    improved.append(n)
                    total_gain += g
                    print(f"  [+] Pass {p+1}: SUCCESS! N={n}: side {cur_s:.8f} -> {best_s:.8f} (gain={g:.10f}, {time.time()-t0:.2f}s)")
            else:
                pass
                
    if improved:
        reg.save_submission()
        
    print("\n" + "=" * 50)
    print(f"DR Search Complete:")
    print(f"Total instances improved: {len(improved)} ({improved})")
    print(f"Total score gain: {total_gain:.10f}")
    print(f"New Verified Score: {reg.get_score():.14f}")
    print("=" * 50)


if __name__ == "__main__":
    test_ns = [203, 295, 376, 233, 124, 497, 330, 372, 416, 489, 368]
    run_dr_search(test_ns[:4], num_passes=3)
