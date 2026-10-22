"""Analyze dual-piece composed layouts from Mondrian compose."""

import json
import math
import os
from src.scoring.registry import PackingRegistry

def main():
    json_path = "/tmp/mondrian_compose.json"
    if not os.path.exists(json_path):
        print(f"File {json_path} does not exist yet.")
        return

    with open(json_path) as fp:
        try:
            records = json.load(fp)
        except Exception as e:
            print("Error loading JSON:", e)
            return

    reg = PackingRegistry()
    improvements = []

    for r in records:
        n = r["n"]
        s_cand = r["s"]
        s_cur = reg.get_side(n)
        diff = s_cur - s_cand
        if diff > 1e-6:
            gain = diff / math.sqrt(n)
            improvements.append((n, s_cur, s_cand, diff, gain, r))

    print(f"Total instances where composed layout is strictly smaller: {len(improvements)}")
    total_gain = sum(x[4] for x in improvements)
    print(f"Total potential score reduction: {total_gain:.8f}")

    improvements.sort(key=lambda x: -x[4])
    print(f"\nTop 25 highest-gain composed layouts:")
    print(f"{'N':>5} | {'Current Side':>14} | {'Composed Side':>14} | {'Delta L':>12} | {'Score Gain':>12} | {'Left Piece':>20} | {'Right Piece':>20}")
    print("-" * 105)
    for n, cur_s, cand_s, diff, gain, r in improvements[:25]:
        L = r.get("left")
        R = r.get("right")
        l_str = f"{L['m']}sq from {L['src']}" if L else "None"
        r_str = f"{R['m']}sq from {R['src']}" if R else "None"
        print(f"{n:5d} | {cur_s:14.8f} | {cand_s:14.8f} | {diff:12.6f} | {gain:12.8f} | {l_str:>20} | {r_str:>20}")

if __name__ == "__main__":
    main()
