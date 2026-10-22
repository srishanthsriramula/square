"""Python interface to the high-performance compiled fq and packer engines.

Provides:
- Conversion between Kaggle coordinate formats (radians) and engine formats (degrees)
- Invocation of fq quench (Augmented Lagrangian Method + Clarabel IPM polish)
- Invocation of packer relax / hop (L-BFGS penalty relaxation)
- Strict SAT post-validation
"""

from __future__ import annotations

import json
import math
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

PROJECT_ROOT = Path(__file__).parent.parent.parent
FQ_BIN = PROJECT_ROOT / "external/evand/search/packer/target/release/fq"
PACKER_BIN = PROJECT_ROOT / "external/evand/search/packer/target/release/packer"


def write_engine_config(path: str | Path, coords: np.ndarray, side: Optional[float] = None) -> float:
    """Write coordinates to engine text format (N s \n x y angle_deg)."""
    n = len(coords)
    if side is None:
        side = compute_enclosing_side(coords)
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        f.write(f"{n} {side:.17f}\n")
        for x, y, theta in coords:
            deg = math.degrees(theta) % 360.0
            f.write(f"{x:.17f} {y:.17f} {deg:.17f}\n")
            
    return side


def read_engine_config(path: str | Path) -> Tuple[float, np.ndarray]:
    """Read coordinates from engine text format, converting degrees back to radians."""
    path = Path(path)
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
    head = lines[0].split()
    n = int(head[0])
    side = float(head[1])
    
    coords = []
    for line in lines[1:n+1]:
        parts = line.split()
        x = float(parts[0])
        y = float(parts[1])
        deg = float(parts[2])
        coords.append([x, y, math.radians(deg)])
        
    arr = np.array(coords, dtype=np.float64)
    return side, arr


def quench_packing(
    coords: np.ndarray,
    kick: float = 0.0,
    round_r0: float = 0.0,
    seed: int = 42,
    extra_args: Optional[list] = None,
    timeout_sec: float = 120.0
) -> Tuple[bool, float, np.ndarray, Dict[str, Any]]:
    """Run fq quench on a packing.
    
    Returns:
        (improved: bool, new_side: float, new_coords: np.ndarray, stats: dict)
    """
    if not FQ_BIN.exists():
        raise FileNotFoundError(f"fq binary not found at {FQ_BIN}")
        
    n = len(coords)
    orig_side = compute_enclosing_side(coords)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, f"in_{n}.txt")
        out_path = os.path.join(tmpdir, f"out_{n}.txt")
        
        write_engine_config(in_path, coords, orig_side)
        
        cmd = [
            str(FQ_BIN),
            "quench",
            "--in", in_path,
            "--out", out_path,
            "--seed", str(seed)
        ]
        if kick > 0.0:
            cmd.extend(["--kick", str(kick)])
        if round_r0 > 0.0:
            cmd.extend(["--round-r0", str(round_r0)])
        if extra_args:
            cmd.extend(extra_args)
            
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec
            )
            if res.returncode != 0:
                return False, orig_side, coords, {"error": res.stderr}
                
            stats = {}
            if res.stdout.strip():
                try:
                    stats = json.loads(res.stdout.splitlines()[0])
                except Exception:
                    stats = {"stdout": res.stdout}
                    
            if not os.path.exists(out_path):
                return False, orig_side, coords, {"error": "output file not created", **stats}
                
            new_side, new_coords = read_engine_config(out_path)
            
            # Post-check SAT validation
            val = validate_packing_sat(new_coords, eps=EPSILON_DEFAULT)
            if not val["valid"]:
                return False, orig_side, coords, {
                    "error": f"Quenched result failed SAT validation: {val['error_message']}",
                    **stats
                }
                
            measured_side = compute_enclosing_side(new_coords)
            improved = (measured_side < orig_side - 1e-10)
            return improved, measured_side, new_coords, stats
            
        except subprocess.TimeoutExpired:
            return False, orig_side, coords, {"error": "timeout"}
        except Exception as e:
            return False, orig_side, coords, {"error": str(e)}
