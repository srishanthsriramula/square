#!/usr/bin/env bash
# ==============================================================================
# ONE-CLICK TURNKEY DEPLOYMENT SCRIPT FOR AMD INSTINCT MI300X (ROCm / HIP)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "========================================================================"
echo "      AMD INSTINCT MI300X DEPLOYMENT & RESEARCH ENGINE"
echo "========================================================================"
echo "Host: $(hostname)"
echo "Date: $(date -u)"
echo "Root: ${ROOT_DIR}"
echo "========================================================================"

# 1. Hardware & ROCm Environment Inspection
if command -v rocm-smi >/dev/null 2>&1; then
    echo "[1/5] ROCm detected. GPU Status:"
    rocm-smi || true
elif command -v nvidia-smi >/dev/null 2>&1; then
    echo "[1/5] NVIDIA GPU detected (CUDA fallback)."
    nvidia-smi || true
else
    echo "[1/5] Warning: Neither rocm-smi nor nvidia-smi detected."
fi

# 2. Python Environment Verification
echo "[2/5] Checking Python packages..."
python3 -c "import numpy, scipy, pandas; print('NumPy:', numpy.__version__, '| SciPy:', scipy.__version__)" || {
    echo "Installing missing Python dependencies..."
    pip install numpy scipy pandas
}

# 3. Rust Binaries Check & Build
echo "[3/5] Verifying Rust ALM Solver (fq & packer)..."
if [ ! -f "external/evand/search/packer/target/release/fq" ]; then
    echo "Building fq and packer with cargo..."
    cd external/evand/search/packer
    cargo build --release
    cd "${ROOT_DIR}"
fi
echo "fq binary: $(ls -lh external/evand/search/packer/target/release/fq)"

# 4. Compile HIP / CUDA Simulated Annealing Engine
echo "[4/5] Compiling High-Performance GPU Kernel (sa_hip)..."
mkdir -p "${SCRIPT_DIR}/build" "${SCRIPT_DIR}/candidates"

if command -v hipcc >/dev/null 2>&1; then
    echo "Compiling with hipcc for AMD CDNA3 (MI300X)..."
    hipcc -O3 -x hip "${SCRIPT_DIR}/src/sa_hip.cpp" -o "${SCRIPT_DIR}/build/sa_hip" -D__HIP_PLATFORM_AMD__
elif command -v nvcc >/dev/null 2>&1; then
    echo "Compiling with nvcc for NVIDIA CUDA..."
    nvcc -O3 "${SCRIPT_DIR}/src/sa_hip.cpp" -o "${SCRIPT_DIR}/build/sa_hip"
else
    echo "Error: Neither hipcc nor nvcc compiler found on PATH!"
    exit 1
fi
echo "GPU Binary successfully compiled: $(ls -lh ${SCRIPT_DIR}/build/sa_hip)"

# 5. Launch Full 1-Hour Research Campaign
echo "[5/5] Launching 1-Hour Massive Packing Campaign..."
echo "Starting Certified Score: $(python3 -c 'from src.scoring.registry import PackingRegistry; print(PackingRegistry().get_score())')"

# Phase 1: High-Deficit Grid Cracking (Targeting 184 grid instances)
python3 "${SCRIPT_DIR}/batch_cracker.py" --all-grid --blocks 512 --sweeps 800

# Push any breakthrough commits to GitHub
if git diff --quiet submissions/best_submission.csv; then
    echo "No submission changes to push."
else
    echo "BREAKTHROUGH ACHIEVED! Pushing to GitHub..."
    git add submissions/best_submission.csv results/
    git commit -m "MI300X Breakthrough: Cracking high-deficit grid instances"
    git push origin main || true
fi

echo "========================================================================"
echo "CAMPAIGN COMPLETED SUCCESSFULLY!"
echo "Final Certified Score: $(python3 -c 'from src.scoring.registry import PackingRegistry; print(PackingRegistry().get_score())')"
echo "========================================================================"
