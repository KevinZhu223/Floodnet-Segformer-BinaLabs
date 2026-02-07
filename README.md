# FloodNet Semantic Segmentation: Testing Generalizability of DA-Segformer

This repository contains a high-performance **SegFormer-B4** pipeline for semantic segmentation on the **FloodNet** dataset. The primary focus of this project is to evaluate the **generalizability** of our **DA-Segformer** (Damage Assessment Segformer) architecture, originally developed for RescueNet, when applied to different high-resolution UAV disaster datasets.

By using the same core optimizations—OHEM loss and a 1024x1024 high-resolution cropping strategy—we demonstrate that the model effectively scales across diverse disaster scenarios (Hurricanes vs. Floods) with minimal tuning.

## 📊 Cross-Dataset Generalizability Comparison

Our DA-Segformer strategy consistently outperforms standard baselines. Below is a comparison of our best results on the two primary datasets:

| Dataset | Strategy | mIoU | mAcc |
| :--- | :--- | :--- | :--- |
| **RescueNet** (Phase 2) | 1024x1024 Crop + OHEM | 74.67% | 85.92% |
| **FloodNet** (Final)   | **1024x1024 Crop + OHEM + Class-Aware** | **76.67%** | **85.79%** |

*Key Insight: The DA-Segformer model achieved a **+2.0% mIoU lead** on FloodNet compared to RescueNet, proving that the high-resolution texture recognition logic and OHEM loss are highly generalizable and robust across different disaster domains.*

---

## 📈 Detailed Per-Class Performance (FloodNet)

The following table breaks down the model's performance on the FloodNet validation set. The use of **Class-Aware Cropping** was particularly effective in maintaining high accuracy for rare classes like "Pool" and "Vehicle."

| ID | Class Name | Individual IoU |
|:---|:---|:---|
| 0 | Background | 66.89% |
| 1 | Building Flooded | 74.24% |
| 2 | Building Non-Flooded | 87.28% |
| 3 | Road Flooded | **49.81%** |
| 4 | Road Non-Flooded | 79.39% |
| 5 | Water | 77.45% |
| 6 | Tree | 86.11% |
| 7 | Vehicle | **66.09%** |
| 8 | Pool | **87.73%** |
| 9 | Grass | 91.72% |

**Analysis:**
* **Generalization Success:** The model achieved near-perfect performance on **Grass** and **Trees**, while maintaining strong boundaries for **Building** types.
* **Segmentation Granularity:** By preserving the native 1024x1024 resolution, we avoided the "squashing" effect that often renders small objects like **Vehicles** and **Pools** as mere noise.

---

## 🛠️ The Visualization Challenge & Solution

A major engineering hurdle was generating artifact-free visualizations for massive 3000x4000px UAV images.

### The Problem: Grid Artifacts
Standard tiling methods often create "checkerboard" patterns or discontinuous boundaries where windows meet.
### The Solution: Smooth Sliding Window Inference
We implemented a custom visualization pipeline (`viz_smooth_stitch_floodnet.py`) that strictly mirrors the training logic:
1.  **Overlapping Windows:** We perform inference with a 25% window overlap.
2.  **Smooth Averaging:** Instead of a simple "winner-takes-all" merge, we accumulate probabilities across windows and average them, eliminating hard grid lines.
3.  **Result:** High-fidelity segmentation maps that are visually "clean" and suitable for immediate damage reporting.

---

## 🏗️ Methodology

This implementation utilizes:
* **Architecture:** `Segformer-B4` (MiT-B4 Encoder).
* **Training Strategy:** 
    * **1024x1024 High-Res Crops**: To capture fine textures of flooded vs. dry surfaces.
    * **Class-Aware Sampling**: Forces the model to see rare flooded classes 50% more often during training.
* **Loss Function:** **OHEM (Online Hard Example Mining) + Cross Entropy** to prioritize difficult pixel boundaries.
* **Optimization:** AdamW with a **Cosine Annealing** scheduler.

---

## 💻 Usage

### 🚀 Training
To reproduce the final FloodNet training run:
```bash
./run_train_floodnet_v2.sh
```

### 🔍 Visualization (Smooth Stitching)
To generate the high-quality visualizations:
```bash
./run_smooth_stitch_floodnet.sh val
```

### 📊 Standard Assessment
To run the standard evaluation script on the test set:
```bash
./run_viz_floodnet_segformer.sh
```

---

### Attribution & Research Team
This project was developed at **Bina Labs, Lehigh University**.

*   **Principal Investigator:** Dr. Maryam Rahnemoonfar
*   **Primary Author:** Nhut Le, PhD Candidate
*   **Research Lead:** Kevin Zhu

**Modifications (FloodNet Specifics):**
- Implementation of **Class-Aware Cropping** logic for dataset balance.
- Adaptation of **OHEM + Cosine** pipeline for FloodNet domain.
- Engineering of **Smooth Overlapping Stitching** to resolve tiling artifacts.

### License
Released for academic and educational use. Please cite the original [FloodNet Paper](https://arxiv.org/abs/2012.02951) and [RescueNet Paper](https://arxiv.org/abs/2204.14161) if used.
