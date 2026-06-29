#!/bin/bash
# Run this once inside the container (./docker/run.sh) to compile the three
# CUDA extensions that require GPU access at build time. Compiled .so files
# land in the mounted /workspace and persist across container restarts.
set -e

cd /workspace

echo "==> Building vid2sim-rasterizer (diff_gauss)..."
TORCH_CUDA_ARCH_LIST="8.9" pip install --no-build-isolation -e submodules/vid2sim-rasterizer

echo "==> Building groundingdino CUDA extension..."
TORCH_CUDA_ARCH_LIST="8.9" pip install --no-build-isolation -e submodules/Grounded-SAM-2/grounding_dino

echo "==> Building simple-knn..."
TORCH_CUDA_ARCH_LIST="8.9" pip install --no-build-isolation -e submodules/simple-knn

echo ""
echo "All CUDA extensions built successfully."
