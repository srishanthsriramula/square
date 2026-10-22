"""Python interface to packer basin hopping and descending target search."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np

from src.geometry.sat import (
    compute_enclosing_side,
    validate_packing_sat,
    EPSILON_DEFAULT
)
from src.optimizers.fq_engine import (
    PACKER_BIN,
    write_engine_config,
    read_engine_config
)


def run_packer_hop(
    coords: np.ndarray,
    target_side: Optional[float] = None,
    iters: int = 1000,
    seed: int = 42,
    timeout_sec: float = 120.0
) -> Tuple[bool, float, np.ndarray, Dict[str, Any]]:
    """Run packer basin hopping with descending target.
    
    Returns:
        (improved: bool, best_side: float, best_coords: np.ndarray, stats: dict)
    """
    if not PACKER_BIN.exists():
        raise FileNotFoundError(f"packer binary not found at {PACKER_BIN}")
        
    n = len(coords)
    orig_side = compute_enclosing_side(coords)
    if target_side is None:
        target_side = orig_side - 1e-6
        
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, f"in_{n}.txt")
        out_path = os.path.join(tmpdir, f"out_{n}.txt")
        
        write_engine_config(in_path, coords, orig_side)
        
        cmd = [
            str(PACKER_BIN),
            "hop",
            "--in", in_path,
            "--s", f"{target_side:.12f}",
            "--seed", str(seed),
            "--iters", str(iters),
            "--out", out_path
        ]
        
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec
            )
            
            if not os.path.exists(out_path):
                return False, orig_side, coords, {
                    "error": "No output produced (no feasible squeeze found)",
                    "stdout": res.stdout,
                    "stderr": res.stderr
                }
                
            new_side, new_coords = read_engine_config(out_path)
            
            # Post-check SAT validation
            val = validate_packing_sat(new_coords, eps=EPSILON_DEFAULT)
            if not val["valid"]:
                return False, orig_side, coords, {
                    "error": f"Result failed SAT: {val['error_message']}",
                    "stdout": res.stdout
                }
                
            measured_side = compute_enclosing_side(new_coords)
            improved = (measured_side < orig_side - 1e-10)
            return improved, measured_side, new_coords, {
                "stdout": res.stdout,
                "measured_side": measured_side,
                "reported_side": new_side
            }
            
        except subprocess.TimeoutExpired:
            return False, orig_side, coords, {"error": "timeout"}
        except Exception as e:
            return False, orig_side, coords, {"error": str(e)}
