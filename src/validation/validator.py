"""Independent, numerically robust validation module for square packings.

Verifies complete submissions and individual instances against all rules:
1. Valid CSV schema: id, n, square_id, x, y, angle
2. Complete 1..1000 problem set (500,500 rows)
3. Exactly n squares per instance with contiguous IDs 0..n-1
4. Strict non-overlap via Separating Axis Theorem (SAT)
5. Enclosing container bounds and side calculation
6. Exact competition metric evaluation
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.geometry.sat import (
    validate_packing_sat,
    compute_enclosing_side,
    compute_bounding_box,
    EPSILON_DEFAULT
)


EXPECTED_TOTAL_ROWS = 500500
EXPECTED_COLUMNS = ["id", "n", "square_id", "x", "y", "angle"]


def load_submission_csv(csv_path: str | Path) -> Dict[int, np.ndarray]:
    """Load and parse submission CSV into a dictionary {n: (n, 3) ndarray}."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Submission file not found: {csv_path}")

    packings: Dict[int, List[Tuple[int, float, float, float]]] = {n: [] for n in range(1, 1001)}
    seen_keys = set()
    total_rows = 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"Invalid columns: {reader.fieldnames}. Expected: {EXPECTED_COLUMNS}")

        for row in reader:
            total_rows += 1
            n = int(row["n"])
            sq_id = int(row["square_id"])
            
            key = (n, sq_id)
            if key in seen_keys:
                raise ValueError(f"Duplicate entry for n={n}, square_id={sq_id}")
            seen_keys.add(key)

            x = float(row["x"])
            y = float(row["y"])
            theta = float(row["angle"])
            packings[n].append((sq_id, x, y, theta))

    if total_rows != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_ROWS} rows, got {total_rows}")

    result: Dict[int, np.ndarray] = {}
    for n in range(1, 1001):
        rows = packings[n]
        if len(rows) != n:
            raise ValueError(f"Instance n={n} has {len(rows)} squares, expected {n}")
        rows.sort(key=lambda item: item[0])
        sq_ids = [item[0] for item in rows]
        if sq_ids != list(range(n)):
            raise ValueError(f"Instance n={n} does not have contiguous square_ids 0..{n-1}")
        
        arr = np.array([[item[1], item[2], item[3]] for item in rows], dtype=np.float64)
        result[n] = arr

    return result


def validate_submission(
    csv_path: str | Path,
    eps: float = EPSILON_DEFAULT,
    exhaustive_up_to: int = 50,
    verbose: bool = False
) -> Dict[str, Any]:
    """Full independent validation of a 1000-instance submission CSV."""
    t0 = time.time()
    packings = load_submission_csv(csv_path)

    total_score = 0.0
    per_n_results = {}
    failed_instances = []
    total_pairs_checked = 0

    for n in range(1, 1001):
        coords = packings[n]
        exhaustive = (n <= exhaustive_up_to)
        res = validate_packing_sat(coords, eps=eps, exhaustive=exhaustive)
        
        score_n = res["score"]
        total_score += score_n
        total_pairs_checked += res["num_pairs_checked"]
        per_n_results[n] = {
            "side": res["side"],
            "score": score_n,
            "min_margin": res["min_margin"],
            "valid": res["valid"]
        }

        if not res["valid"]:
            failed_instances.append((n, res["error_message"]))

        if verbose and n % 100 == 0:
            print(f"Validated n={n}/1000, cumulative score={total_score:.8f}")

    elapsed = time.time() - t0
    is_valid = len(failed_instances) == 0

    return {
        "valid": is_valid,
        "total_score": total_score,
        "total_instances": 1000,
        "failed_count": len(failed_instances),
        "failed_instances": failed_instances[:10],
        "total_pairs_checked": total_pairs_checked,
        "validation_time_sec": elapsed,
        "per_n": per_n_results
    }


def compute_score_from_dict(packings: Dict[int, np.ndarray]) -> float:
    """Fast score computation without full pairwise SAT check."""
    total = 0.0
    for n, coords in packings.items():
        side = compute_enclosing_side(coords)
        total += (side / math.sqrt(n) - 1.0)
    return total
