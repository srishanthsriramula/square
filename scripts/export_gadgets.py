"""Export extracted gadgets for high-performance C++ DP solver."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.scoring.registry import PackingRegistry
from src.constructions.fault_dp import extract_all_gadgets

def export_gadgets():
    reg = PackingRegistry()
    print("Extracting all gadgets from packings and crops...")
    gadgets = extract_all_gadgets(reg.packings)
    print(f"Extracted {len(gadgets)} Pareto-optimal gadgets.")
    
    out_dir = Path("data/generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    txt_path = out_dir / "gadgets.txt"
    coords_dict = {}
    
    with open(txt_path, "w") as f:
        f.write("# id w h dm f name\n")
        for idx, g in enumerate(gadgets):
            f.write(f"{idx} {g.w} {g.h} {g.dm} {g.f:.8f} {g.name}\n")
            coords_dict[f"g_{idx}"] = g.coords
            coords_dict[f"meta_{idx}"] = np.array([g.w, g.h, g.dm, g.f, g.source_n])
            
    np.savez_compressed(out_dir / "gadgets_coords.npz", **coords_dict)
    print(f"Exported gadgets to {txt_path} and {out_dir / 'gadgets_coords.npz'}")

if __name__ == "__main__":
    export_gadgets()
