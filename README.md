# FloodNet-Segformer: High-Resolution UAV Damage Assessment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/transformers-HuggingFace-yellow.svg)](https://huggingface.co/docs/transformers/index)

A specialized semantic segmentation pipeline for the **FloodNet Dataset**, utilizing SegFormer-B4 for high-resolution UAV imagery analysis. This repository implements class-aware training, advanced data augmentation, and optimized "smooth-stitching" visualization to eliminate tiling artifacts.

## 🚀 Key Features

- **SegFormer-B4 Backbone**: Optimized for dense semantic prediction in top-down UAV views.
- **High-Res Training**: 1024x1024 random crops to preserve minute details (vehicles, pools, building damage types).
- **Class-Aware Cropping**: Dynamically weights rare classes (e.g., flooded buildings, vehicles) during training to solve dataset imbalance.
- **Smooth Overlapping Inference**: Overlapping sliding window (25% overlap) prevents the "grid-line" artifacts common in large-scale UAV stitching.
- **Real-time Monitoring**: Automated recording of the best checkpoints based on mIoU.

## 📁 Repository Structure

```text
├── nh_datasets/           # Dataset loaders and registry
│   ├── configs/           # Training configuration files
│   └── floodnet.py        # Core FloodNet dataset logic
├── runs/                  # Saved checkpoints and visualizations (ignored)
├── train_segformer.py     # Main training entry point
├── viz_smooth_stitch_floodnet.py # Optimized visualization logic
├── run_train_floodnet_v2.sh # One-click training script (Docker)
└── run_smooth_stitch_floodnet.sh # One-click visualization script
```

## 🛠️ Installation & Usage

### Prerequisites
- Docker (with NVIDIA Container Toolkit)
- Dataset placed at `/media/volume/Data_Kevin_Zhu/FloodNet-Supervised_v1.0`

### Training
Start the optimized training pipeline:
```bash
./run_train_floodnet_v2.sh
```

### Visualization
Generate high-quality overlapping stitching visualizations for the validation set:
```bash
./run_smooth_stitch_floodnet.sh val
```

## 📊 Results Summary

The model achieved a peak performance of **0.7667 mIoU** on the validation set at Epoch 225, with significant visual improvements in distinguishing flooded vs. non-flooded infrastructure.

---
*Developed for BinaLabs UAV damage assessment.*
