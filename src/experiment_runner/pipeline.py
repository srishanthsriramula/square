"""Autonomous research pipeline for square packing optimization.

Orchestrates:
1. Candidate target prioritization (rigid 0/45 constructions, key donors)
2. High-performance Basin Hopping (packer hop)
3. Augmented Lagrangian + IPM Polish (fq quench)
4. Strict SAT geometry validation (eps=1e-10)
5. Monotonicity propagation
6. Multi-gadget Fault-line DP composition
7. Atomic registry updates and checkpoint logging
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.scoring.registry import PackingRegistry
from src.geometry.sat import (
    compute_enclosing_side,
    validate_packing_sat,
    EPSILON_DEFAULT
)
from src.optimizers.packer_engine import run_packer_hop
from src.optimizers.fq_engine import quench_packing
from src.constructions.fault_dp import (
    extract_all_gadgets,
    extract_full_packings_as_gadgets,
    run_fault_line_dp,
    compose_packing
)


class ResearchEngine:
    def __init__(self, registry: Optional[PackingRegistry] = None):
        self.reg = registry or PackingRegistry()

    def find_rigid_0_45_targets(self, min_n: int = 20, max_n: int = 1000) -> List[Tuple[int, float]]:
        """Identify instances that are rigid 0/45 degree constructions with high potential."""
        targets = []
        for n in range(min_n, max_n + 1):
            L = self.reg.get_side(n)
            ceil_s = math.ceil(math.sqrt(n))
            if L < ceil_s - 1e-4:
                a = self.reg.get_packing(n)
                angles = a[:, 2] % (math.pi / 2)
                diff0 = np.abs(angles)
                diff45 = np.abs(angles - math.pi / 4)
                is_0_45 = np.all((diff0 < 1e-4) | (diff45 < 1e-4) | (np.abs(diff0 - math.pi/2) < 1e-4))
                if is_0_45:
                    potential = (L - math.sqrt(n)) / math.sqrt(n)
                    targets.append((n, potential))
                    
        # Sort by improvement potential
        targets.sort(key=lambda x: x[1], reverse=True)
        return targets

    def optimize_instance(
        self,
        n: int,
        iters: int = 1500,
        hop_timeout_sec: float = 60.0,
        seed: int = 42
    ) -> Tuple[bool, float, str]:
        """Run single-instance optimization: Hop -> Quench -> SAT -> Registry."""
        orig_side = self.reg.get_side(n)
        a = self.reg.get_packing(n)

        # 1. Basin Hopping with descending target
        # Target slightly below current side
        target_side = orig_side - 5e-5
        hop_ok, hop_side, hop_a, hop_stats = run_packer_hop(
            a,
            target_side=target_side,
            iters=iters,
            seed=seed,
            timeout_sec=hop_timeout_sec
        )

        cand_a = hop_a if hop_ok else a
        cand_side = hop_side if hop_ok else orig_side

        # 2. Augmented Lagrangian + IPM polish (fq quench)
        quench_ok, q_side, q_a, q_stats = quench_packing(
            cand_a,
            kick=0.001 if not hop_ok else 0.0,
            seed=seed,
            timeout_sec=30.0
        )

        final_a = q_a if quench_ok else cand_a
        final_side = compute_enclosing_side(final_a)

        gain = (orig_side - final_side) / math.sqrt(n)
        if gain > 1e-10:
            success, g, msg = self.reg.register_candidate(
                n,
                final_a,
                source=f"hop_{iters}_quench_seed{seed}"
            )
            if success:
                self.reg.save_submission()
                return True, g, msg

        return False, 0.0, f"No improvement for n={n}: cur={orig_side:.10f}, best={final_side:.10f}"

    def run_fault_line_dp_pass(self, max_k: int = 31) -> Tuple[int, float]:
        """Run a full Fault-Line DP composition pass over all (K, D) pairs using all full and crop gadgets."""
        gadgets = extract_all_gadgets(self.reg.packings)
        best_kd = run_fault_line_dp(gadgets, max_k=max_k)

        registered = 0
        total_gain = 0.0

        for (K, D), (f_val, chain) in best_kd.items():
            n = K * K + D
            if not (1 <= n <= 1000):
                continue
            cur_side = self.reg.get_side(n)
            cand_side = K + f_val
            gain = (cur_side - cand_side) / math.sqrt(n)
            if gain > 1e-10:
                try:
                    cand_arr = compose_packing(K, D, f_val, chain)
                    success, g, msg = self.reg.register_candidate(
                        n,
                        cand_arr,
                        source=f"fault_dp_pass_K{K}_D{D}"
                    )
                    if success:
                        registered += 1
                        total_gain += g
                except Exception:
                    pass

        if registered > 0:
            self.reg.save_submission()

        return registered, total_gain

    def optimize_batch(
        self,
        target_ns: List[int],
        iters_per_n: int = 1500,
        time_limit_sec: float = 600.0,
        seed_base: int = 2026
    ) -> Dict[str, Any]:
        """Run batch optimization over selected targets with early stopping."""
        t0 = time.time()
        start_score = self.reg.get_score()
        improved_instances = []
        total_gain = 0.0

        for idx, n in enumerate(target_ns):
            elapsed = time.time() - t0
            if elapsed >= time_limit_sec:
                print(f"Time limit reached ({elapsed:.1f}s >= {time_limit_sec}s). Stopping batch.")
                break

            seed = seed_base + idx * 17
            print(f"[{idx+1}/{len(target_ns)}] Optimizing n={n} (current side={self.reg.get_side(n):.8f})...", end=" ", flush=True)
            
            t_n = time.time()
            ok, gain, msg = self.optimize_instance(n, iters=iters_per_n, seed=seed)
            dt = time.time() - t_n

            if ok:
                improved_instances.append(n)
                total_gain += gain
                print(f"SUCCESS! {msg} ({dt:.1f}s)")
                # Run quick DP pass to propagate
                dp_count, dp_gain = self.run_fault_line_dp_pass()
                if dp_count > 0:
                    total_gain += dp_gain
                    print(f"   -> DP pass propagated: {dp_count} more instances improved (+{dp_gain:.10f})")
            else:
                print(f"no gain ({dt:.1f}s)")

        # Final DP pass to ensure all gains are shared
        final_dp_count, final_dp_gain = self.run_fault_line_dp_pass()
        total_gain += final_dp_gain

        end_score = self.reg.get_score()
        return {
            "start_score": start_score,
            "end_score": end_score,
            "total_gain": total_gain,
            "improved_count": len(improved_instances),
            "improved_instances": improved_instances,
            "elapsed_sec": time.time() - t0
        }
