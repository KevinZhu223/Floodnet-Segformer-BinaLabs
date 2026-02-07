#!/usr/bin/env bash
set -e
DEVICES="0,1,2,3" # Adjust based on available GPUs
NPROC=$(( $(tr -cd ',' <<<"$DEVICES" | wc -c) + 1 ))
DOCKER_IMAGE="letatanu/semseg_2d:latest"

docker run --rm \
  -v /dev/shm:/dev/shm \
  --gpus "\"device=${DEVICES}\"" \
  -w /working \
  -v /media/volume/Data_Kevin_Zhu/FloodNet-Segformer/:/working \
  -v /media/volume/Data_Kevin_Zhu/:/data \
  "${DOCKER_IMAGE}" \
  bash -lc "
        set -euo pipefail
        export OMP_NUM_THREADS=16

        # 1. Activate Conda Environment
        if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
            . /opt/conda/etc/profile.d/conda.sh
        elif [ -f $HOME/miniconda3/etc/profile.d/conda.sh ]; then
            . $HOME/miniconda3/etc/profile.d/conda.sh
        else
            echo 'conda source not found'
        fi
        conda activate semseg
        
        # 2. Install Dependencies (Missing from base image)
        echo 'Installing dependencies...'
        pip install albumentations
        
        # 3. Start Training
        echo 'Starting Optimized FloodNet Training: 1024px, B4, Class-Aware...'
        
        torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
        /working/train_segformer.py \
        --config_file /working/nh_datasets/configs/segformer_floodnet_v2.py
        "
