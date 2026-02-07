#!/usr/bin/env bash
set -e

## --------------------------------------------------------- ##
DOCKER_IMAGE="letatanu/semseg_2d:latest"

docker run --rm -it \
  -v /dev/shm:/dev/shm \
  --gpus "all" \
  -w /working \
  -v /media/volume/Data_Kevin_Zhu/FloodNet-Segformer/:/working \
  -v /media/volume/Data_Kevin_Zhu/:/data \
  "${DOCKER_IMAGE}" \
  bash -lc "
        set -euo pipefail
        # Activate conda
        if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
            . /opt/conda/etc/profile.d/conda.sh
        fi
        conda activate semseg
        
        # Install dependencies
        /opt/conda/envs/semseg/bin/pip install opencv-python-headless albumentations > /dev/null 2>&1
        
        # Run Standard Visualization
        python viz_segformer.py \
            --model runs/floodnet_final_b4_ohem_cosine_V2/BEST_MODELS_ARCHIVE/checkpoint-mIoU-0.7667-Ep225.0 \
            --folder /data/FloodNet-Supervised_v1.0/test/test-org-img/ \
            --gt_folder /data/FloodNet-Supervised_v1.0/test/test-label-img/ \
            --outdir runs/viz_floodnet_test_standard
        "