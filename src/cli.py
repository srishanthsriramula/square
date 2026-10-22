"""Command-line interface for Squares in Squares Sub-15 Research Engine.

Usage examples:
  python -m src.cli verify --input submissions/best_submission.csv
  python -m src.cli score --input submissions/best_submission.csv
  python -m src.cli optimize --range 325:400 --budget-minutes 10
  python -m src.cli dp
  python -m src.cli report
  python -m src.cli export --output submissions/best_submission.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from src.scoring.registry import PackingRegistry
from src.validation.validator import validate_submission, load_submission_csv
from src.experiment_runner.pipeline import ResearchEngine


def cmd_verify(args):
    path = args.input or "submissions/best_submission.csv"
    print(f"Running strict SAT validation on: {path}")
    t0 = time.time()
    res = validate_submission(path, eps=args.eps, verbose=args.verbose)
    elapsed = time.time() - t0
    
    print("\n=== VALIDATION RESULT ===")
    print(f"Valid: {res['valid']}")
    print(f"Total Score: {res['total_score']:.14f}")
    print(f"Total instances checked: {res['total_instances']}")
    print(f"Failed instances: {res['failed_count']}")
    print(f"Total pair checks: {res['total_pairs_checked']}")
    print(f"Time: {elapsed:.2f}s")
    if not res['valid']:
        print("Failures:")
        for n, msg in res['failed_instances']:
            print(f"  n={n}: {msg}")
        sys.exit(1)


def cmd_score(args):
    path = args.input or "submissions/best_submission.csv"
    reg = PackingRegistry(best_csv_path=path)
    score = reg.get_score()
    print(f"Competition Score for {path}: {score:.14f}")


def cmd_optimize(args):
    reg = PackingRegistry()
    engine = ResearchEngine(registry=reg)
    
    target_ns = []
    if args.range:
        parts = args.range.split(":")
        lo = int(parts[0])
        hi = int(parts[1]) if len(parts) > 1 else lo
        target_ns = list(range(lo, hi + 1))
    elif args.ns:
        target_ns = [int(x.strip()) for x in args.ns.split(",")]
    else:
        # Prioritize 0/45 degree rigid constructions
        targets = engine.find_rigid_0_45_targets(min_n=args.min_n, max_n=args.max_n)
        target_ns = [t[0] for t in targets]

    print(f"Starting batch optimization on {len(target_ns)} instances...")
    print(f"Current score: {reg.get_score():.14f}")
    time_limit_sec = args.budget_minutes * 60.0
    
    res = engine.optimize_batch(
        target_ns=target_ns,
        iters_per_n=args.iters,
        time_limit_sec=time_limit_sec
    )
    
    print("\n=== OPTIMIZATION SUMMARY ===")
    print(f"Start Score: {res['start_score']:.14f}")
    print(f"End Score:   {res['end_score']:.14f}")
    print(f"Total Gain:  {res['total_gain']:.14f}")
    print(f"Improved instances: {res['improved_count']} ({res['improved_instances']})")
    print(f"Elapsed Time: {res['elapsed_sec']:.1f}s")


def cmd_dp(args):
    reg = PackingRegistry()
    engine = ResearchEngine(registry=reg)
    print(f"Running Fault-Line DP pass on current best packings...")
    print(f"Current score: {reg.get_score():.14f}")
    count, gain = engine.run_fault_line_dp_pass(max_k=args.max_k)
    print(f"DP pass complete. Registered {count} improvements, total gain: {gain:.10f}")
    print(f"New score: {reg.get_score():.14f}")


def cmd_report(args):
    reg = PackingRegistry()
    score = reg.get_score()
    baseline = 22.4772458938304
    improvement = baseline - score
    dist_sub15 = score - 15.0
    
    print("=================================================================")
    print("      SQUARES IN SQUARES — SUB-15 RESEARCH PROGRESS REPORT       ")
    print("=================================================================")
    print(f"Current Verified Total Score : {score:.14f}")
    print(f"Starting Benchmark Score     : {baseline:.14f}")
    print(f"Verified Score Improvement   : {improvement:.14f}")
    print(f"Distance to Sub-15 Target    : {dist_sub15:.14f}")
    print(f"Lower Bound (Theoretical)    : 5.59669230000000")
    print("-----------------------------------------------------------------")
    
    # Check history
    if reg.history_csv.exists():
        lines = reg.history_csv.read_text().splitlines()
        print(f"Total registered improvements logged: {len(lines) - 1}")
        if len(lines) > 1:
            print("Recent improvements:")
            for l in lines[-5:]:
                print("  ", l)
    print("=================================================================")


def cmd_export(args):
    reg = PackingRegistry()
    out_path = Path(args.output)
    reg.save_submission(out_path)
    print(f"Exported current best submission to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Squares in Squares Sub-15 Research Engine")
    subparsers = parser.add_subparsers(dest="command")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify submission CSV with strict SAT")
    p_verify.add_argument("--input", "-i", type=str, default="submissions/best_submission.csv")
    p_verify.add_argument("--eps", type=float, default=1e-10)
    p_verify.add_argument("--verbose", "-v", action="store_true")

    # score
    p_score = subparsers.add_parser("score", help="Compute competition score")
    p_score.add_argument("--input", "-i", type=str, default="submissions/best_submission.csv")

    # optimize
    p_opt = subparsers.add_parser("optimize", help="Run optimization pipeline")
    p_opt.add_argument("--range", type=str, help="Range of n e.g. 325:400")
    p_opt.add_argument("--ns", type=str, help="Comma-separated n e.g. 372,451,494")
    p_opt.add_argument("--min-n", type=int, default=20)
    p_opt.add_argument("--max-n", type=int, default=1000)
    p_opt.add_argument("--iters", type=int, default=1200)
    p_opt.add_argument("--budget-minutes", type=float, default=10.0)

    # dp
    p_dp = subparsers.add_parser("dp", help="Run Fault-Line DP composition pass")
    p_dp.add_argument("--max-k", type=int, default=31)

    # report
    subparsers.add_parser("report", help="Display progress and status report")

    # export
    p_exp = subparsers.add_parser("export", help="Export submission CSV")
    p_exp.add_argument("--output", "-o", type=str, default="submissions/best_submission.csv")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "verify":
        cmd_verify(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "dp":
        cmd_dp(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "export":
        cmd_export(args)


if __name__ == "__main__":
    main()
