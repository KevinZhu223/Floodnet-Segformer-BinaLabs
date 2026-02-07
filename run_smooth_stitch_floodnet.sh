#!/usr/bin/env bash
set -e
DOCKER_IMAGE="letatanu/semseg_2d:latest"
SCRIPT_NAME="viz_smooth_stitch_floodnet.py"
# Default Arguments
SPLIT=${1:-"val"}
MODEL_PATH=${2:-"/working/runs/floodnet_final_b4_ohem_cosine_V2/BEST_MODELS_ARCHIVE/checkpoint-mIoU-0.7667-Ep225.0"}
OUT_DIR="/working/runs/viz_floodnet_SMOOTH_${SPLIT}"

# Wipe the folder for a clean run
echo "🧹 Cleaning output directory: ${OUT_DIR}"
# Use a path relative to the host if possible, but here we use the container's view
# Note: OUT_DIR is inside /working (which is /media/volume/Data_Kevin_Zhu/FloodNet-Segformer/)
HOST_OUT_DIR="/media/volume/Data_Kevin_Zhu/FloodNet-Segformer/runs/viz_floodnet_SMOOTH_${SPLIT}"
sudo rm -rf "${HOST_OUT_DIR}"
mkdir -p "${HOST_OUT_DIR}"

echo "🚀 Starting FloodNet Smooth Stitch Visualization for split: ${SPLIT}"

docker run --rm \
  -v /dev/shm:/dev/shm \
  --gpus "all" \
  -w /working \
  -v /media/volume/Data_Kevin_Zhu/FloodNet-Segformer/:/working \
  -v /media/volume/Data_Kevin_Zhu/:/data \
  "${DOCKER_IMAGE}" \
  bash -lc "
    # Install dependencies inside container
    /opt/conda/envs/semseg/bin/pip install opencv-python-headless albumentations > /dev/null 2>&1
    
    # Run visualization
    /opt/conda/envs/semseg/bin/python /working/${SCRIPT_NAME} --split ${SPLIT} --model ${MODEL_PATH} --outdir ${OUT_DIR}
  "

echo "✅ Done! Visuals saved to: ${HOST_OUT_DIR}"
