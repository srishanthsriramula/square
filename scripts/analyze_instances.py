"""Analyze score distribution, instance categories, and optimization potential."""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.validation.validator import load_submission_csv
from src.geometry.sat import compute_enclosing_side

def analyze(csv_path: str):
    packings = load_submission_csv(csv_path)
    records = []
    
    total_score = 0.0
    for n in range(1, 1001):
        a = packings[n]
        L = compute_enclosing_side(a)
        sqrt_n = math.sqrt(n)
        ceil_sqrt_n = math.ceil(sqrt_n)
        score_n = L / sqrt_n - 1.0
        total_score += score_n
        
        # Rotations
        angles = a[:, 2]
        # Modulo pi/2
        mod_angles = np.abs(np.sin(2 * angles))
        tilted = np.sum(mod_angles > 1e-4)
        is_grid = (abs(L - ceil_sqrt_n) < 1e-5) and (tilted == 0)
        
        # Categorization
        k = math.isqrt(n)
        is_square = (k * k == n)
        diff_from_square = (k + 1)**2 - n if not is_square else 0
        
        records.append({
            "n": n,
            "L": L,
            "sqrt_n": sqrt_n,
            "ceil_sqrt_n": ceil_sqrt_n,
            "score_n": score_n,
            "density": n / (L * L),
            "tilted_count": int(tilted),
            "is_grid": is_grid,
            "is_square": is_square,
            "diff_from_next_square": diff_from_square
        })
        
    df = pd.DataFrame(records)
    print(f"Total verified score: {total_score:.8f}")
    
    # Grid cases
    grid_df = df[df.is_grid]
    non_grid_df = df[~df.is_grid]
    
    print("\n=== INSTANCE BREAKDOWN ===")
    print(f"Total instances: {len(df)}")
    print(f"Plain grid instances (L == ceil(sqrt(N))): {len(grid_df)}, score contribution: {grid_df.score_n.sum():.6f}")
    print(f"Non-grid / optimized instances: {len(non_grid_df)}, score contribution: {non_grid_df.score_n.sum():.6f}")
    print(f"Perfect square instances (score == 0): {len(df[df.is_square])}")
    
    print("\n=== TOP 25 HIGHEST SCORE CONTRIBUTORS ===")
    top25 = df.sort_values("score_n", ascending=False).head(25)
    print(top25[["n", "L", "ceil_sqrt_n", "score_n", "is_grid", "tilted_count"]].to_string(index=False))
    
    print("\n=== SCORE CONTRIBUTION BY N RANGES ===")
    ranges = [(1, 100), (101, 250), (251, 500), (501, 750), (751, 1000)]
    for r_lo, r_hi in ranges:
        sub = df[(df.n >= r_lo) & (df.n <= r_hi)]
        print(f"N in [{r_lo:3d}, {r_hi:4d}]: sum score = {sub.score_n.sum():.6f}, avg score = {sub.score_n.mean():.6f}, max = {sub.score_n.max():.6f} (N={sub.loc[sub.score_n.idxmax(), 'n']})")

    # Save detailed per-instance metrics
    out_path = Path("results/score_by_n.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nDetailed metrics written to {out_path}")

if __name__ == "__main__":
    target_csv = sys.argv[1] if len(sys.argv) > 1 else "submissions/best_submission.csv"
    analyze(target_csv)
