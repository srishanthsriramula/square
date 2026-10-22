"""Registry and transaction manager for best known valid square packings.

Ensures scientific integrity:
1. Every candidate must pass SAT validation at strict eps (1e-10).
2. Every candidate must strictly improve enclosing side (by at least 1e-10).
3. Maintains atomic checkpoints and history log.
4. Auto-propagates downward monotonicity (N -> N-1).
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from src.geometry.sat import (
    validate_packing_sat,
    compute_enclosing_side,
    EPSILON_DEFAULT
)
from src.validation.validator import (
    load_submission_csv,
    EXPECTED_COLUMNS,
    EXPECTED_TOTAL_ROWS
)


class PackingRegistry:
    def __init__(
        self,
        best_csv_path: str | Path = "data/best_known/current_best_submission.csv",
        results_dir: str | Path = "results",
        submissions_dir: str | Path = "submissions"
    ):
        self.best_csv_path = Path(best_csv_path)
        self.results_dir = Path(results_dir)
        self.submissions_dir = Path(submissions_dir)
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.submissions_dir.mkdir(parents=True, exist_ok=True)
        
        self.history_csv = self.results_dir / "score_history.csv"
        self.current_best_json = self.results_dir / "current_best.json"
        self.export_csv = self.submissions_dir / "best_submission.csv"

        self.packings: Dict[int, np.ndarray] = {}
        self.sides: Dict[int, float] = {}
        self.scores: Dict[int, float] = {}
        self.total_score: float = 0.0

        self._load()

    def _load(self):
        if not self.best_csv_path.exists():
            raise FileNotFoundError(f"Base submission {self.best_csv_path} not found")
        self.packings = load_submission_csv(self.best_csv_path)
        self._recompute_stats()

    def _recompute_stats(self):
        total = 0.0
        for n in range(1, 1001):
            a = self.packings[n]
            s = compute_enclosing_side(a)
            score_n = s / math.sqrt(n) - 1.0
            self.sides[n] = s
            self.scores[n] = score_n
            total += score_n
        self.total_score = total

    def get_packing(self, n: int) -> np.ndarray:
        return self.packings[n].copy()

    def get_side(self, n: int) -> float:
        return self.sides[n]

    def get_score(self) -> float:
        return self.total_score

    def register_candidate(
        self,
        n: int,
        coords: np.ndarray,
        source: str = "optimizer",
        eps: float = EPSILON_DEFAULT,
        auto_propagate_down: bool = True
    ) -> Tuple[bool, float, str]:
        """Attempt to register an improved packing for instance n.
        
        Returns:
            (success: bool, gain: float, message: str)
        """
        if len(coords) != n:
            return False, 0.0, f"Size mismatch: expected {n} squares, got {len(coords)}"

        cur_side = self.sides[n]
        cand_side = compute_enclosing_side(coords)
        gain = (cur_side - cand_side) / math.sqrt(n)

        if gain <= 1e-10:
            return False, 0.0, f"No significant improvement: cur={cur_side:.12f}, cand={cand_side:.12f}, gain={gain:.2e}"

        # Strict SAT validation
        val = validate_packing_sat(coords, eps=eps, exhaustive=(n <= 50))
        if not val["valid"]:
            return False, 0.0, f"SAT validation failed: {val['error_message']}"

        # Verified! Update state
        old_total = self.total_score
        self.packings[n] = coords.copy()
        self.sides[n] = cand_side
        self.scores[n] = cand_side / math.sqrt(n) - 1.0
        self.total_score -= gain

        msg = f"IMPROVED n={n}: side {cur_side:.10f} -> {cand_side:.10f}, score gain = {gain:.10f}"
        self._record_improvement(n, cur_side, cand_side, gain, source)

        # Monotonicity propagation: check if n can improve n-1
        if auto_propagate_down and n > 1:
            self._propagate_down(n, eps=eps)

        return True, gain, msg

    def _propagate_down(self, start_n: int, eps: float = EPSILON_DEFAULT):
        """If packing for n is smaller than current packing for n-1, try dropping squares."""
        for target_n in range(start_n - 1, max(0, start_n - 10), -1):
            if self.sides[target_n] <= self.sides[start_n]:
                break
            # Find square to drop
            higher = self.packings[target_n + 1]
            best_side = float("inf")
            best_idx = -1
            for i in range(len(higher)):
                sub = np.delete(higher, i, axis=0)
                s = compute_enclosing_side(sub)
                if s < best_side:
                    best_side = s
                    best_idx = i
            
            if best_side < self.sides[target_n] - 1e-10:
                best_sub = np.delete(higher, best_idx, axis=0)
                # Register
                self.register_candidate(
                    target_n,
                    best_sub,
                    source=f"subset_drop_from_{target_n+1}",
                    eps=eps,
                    auto_propagate_down=False
                )

    def _record_improvement(self, n: int, old_side: float, new_side: float, gain: float, source: str):
        # Append to score_history.csv
        file_exists = self.history_csv.exists()
        with open(self.history_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "n", "old_side", "new_side", "score_gain", "new_total_score", "source"])
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                n,
                f"{old_side:.14f}",
                f"{new_side:.14f}",
                f"{gain:.14f}",
                f"{self.total_score:.14f}",
                source
            ])

        # Write current_best.json
        info = {
            "total_score": self.total_score,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_improved_n": n,
            "last_gain": gain,
            "last_source": source
        }
        self.current_best_json.write_text(json.dumps(info, indent=2) + "\n")

    def save_submission(self, dest_path: Optional[str | Path] = None):
        """Export current best packings to submission CSV atomically."""
        dest = Path(dest_path) if dest_path else self.best_csv_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Use PID + nanosecond timestamp to prevent multi-worker collisions
        temp_dest = dest.with_name(f"{dest.stem}_{os.getpid()}_{time.time_ns()}.tmp")
        try:
            with open(temp_dest, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(EXPECTED_COLUMNS)
                row_count = 0
                for n in range(1, 1001):
                    arr = self.packings[n]
                    for sq_id, (x, y, theta) in enumerate(arr):
                        writer.writerow([
                            f"{n}_{sq_id}",
                            n,
                            sq_id,
                            f"{x:.16f}",
                            f"{y:.16f}",
                            f"{theta:.16f}"
                        ])
                        row_count += 1
                        
            assert row_count == EXPECTED_TOTAL_ROWS, f"Exported {row_count} rows, expected {EXPECTED_TOTAL_ROWS}"
            temp_dest.replace(dest)
            
            # Also copy to submissions/best_submission.csv
            if dest != self.export_csv:
                shutil.copy2(dest, self.export_csv)
        finally:
            if temp_dest.exists():
                try:
                    temp_dest.unlink()
                except Exception:
                    pass
