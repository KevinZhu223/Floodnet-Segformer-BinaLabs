import torch
import numpy as np
import os
import cv2
import math
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm
from PIL import Image, ImageOps

# Import the dataset class to ensure identical loading logic
from nh_datasets.floodnet import FloodNetSegDataset

import argparse

# --- CONFIGURATION (Defaults) ---
DEFAULT_MODEL = "/working/runs/floodnet_final_b4_ohem_cosine_V2/BEST_MODELS_ARCHIVE/checkpoint-mIoU-0.7667-Ep225.0"
DATA_ROOT  = "/data/FloodNet-Supervised_v1.0"
DEFAULT_OUT = "/working/runs/viz_floodnet_SMOOTH"

# --- FLOODNET PALETTE (BGR for OpenCV) ---
PALETTE_BGR = {
    0: (0, 0, 0),       # background
    1: (0, 0, 255),     # building flooded (Red in RGB)
    2: (120, 120, 180), # building non-flooded
    3: (20, 150, 160),  # road flooded
    4: (140, 140, 140), # road non-flooded
    5: (250, 230, 61),  # water
    6: (255, 82, 0),    # tree
    7: (245, 0, 255),   # vehicle
    8: (0, 235, 255),   # pool
    9: (7, 250, 4),     # grass
}

def colorize_mask(mask):
    """Paints the index mask into BGR for OpenCV"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for index, color in PALETTE_BGR.items():
        color_mask[mask == index] = color
    return color_mask

@torch.inference_mode()
def predict_smooth_window(model, processor, image_np, device, crop_size=1024, stride=768):
    """
    Predicts using overlapping windows to remove grid lines.
    """
    h, w, _ = image_np.shape
    num_classes = 10 
    
    full_probs = torch.zeros((num_classes, h, w), device=device)
    count_map = torch.zeros((1, h, w), device=device)
    
    # Pre-calculate steps
    rows = range(0, h - crop_size + stride, stride)
    cols = range(0, w - crop_size + stride, stride)
    
    # Handle images smaller than crop
    if h <= crop_size or w <= crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        img_padded = cv2.copyMakeBorder(image_np, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
        inputs = processor(images=img_padded, return_tensors="pt")
        outputs = model(inputs.pixel_values.to(device))
        logits = F.interpolate(outputs.logits, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)[0, :, :h, :w]
        return torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)

    # Sliding Window
    for y in rows:
        for x in cols:
            y1, x1 = y, x
            y2, x2 = y1 + crop_size, x1 + crop_size
            
            # Boundary check
            if y2 > h:
                y2 = h
                y1 = h - crop_size
            if x2 > w:
                x2 = w
                x1 = w - crop_size

            tile = image_np[y1:y2, x1:x2]
            inputs = processor(images=tile, return_tensors="pt")
            outputs = model(inputs.pixel_values.to(device))
            
            # Upsample logits to 1024x1024
            logits = F.interpolate(outputs.logits, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1)[0]
            
            full_probs[:, y1:y2, x1:x2] += probs
            count_map[:, y1:y2, x1:x2] += 1.0

    # Avoid division by zero
    count_map[count_map == 0] = 1.0
    full_probs /= count_map
    return torch.argmax(full_probs, dim=0).cpu().numpy().astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Smooth Stitching Visualization for FloodNet")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to Segformer model")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUT, help="Output directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Model & Processor: {args.model}")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(device)
    model.eval()
    
    # Crucial: Load processor from model dir to get exact normalization used in training
    try:
        processor = SegformerImageProcessor.from_pretrained(args.model)
    except:
        print("Model dir missing config, using default Segformer processor")
        processor = SegformerImageProcessor(do_resize=False, do_normalize=True)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Use the Dataset class for robust loading (handles EXIF transpose if we just added it)
    ds = FloodNetSegDataset(root=DATA_ROOT, split=args.split, image_processor=processor)
    
    print(f"Processing {len(ds)} images from split '{args.split}' (Refactored Slide & Stitch)...")

    for i in tqdm(range(len(ds))):
        sample = ds.samples[i]
        img_path, lbl_path = sample
        
        # Load using the dataset's logic to guarantee alignment
        image_pil, label_pil = ds._load_pair(img_path, lbl_path)
        
        image_np = np.array(image_pil)
        label_np = np.array(label_pil)
        
        # PREDICT
        preds = predict_smooth_window(model, processor, image_np, device, crop_size=1024, stride=768)

        # VISUALIZE
        viz_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) 
        gt_color = colorize_mask(label_np)
        pred_color = colorize_mask(preds)
        
        # Overlay
        overlay = cv2.addWeighted(viz_img, 0.6, pred_color, 0.4, 0)

        # Combine: [Original | GT | Pred | Overlay]
        combined = np.hstack([viz_img, gt_color, pred_color, overlay])
        
        # Save
        out_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(args.outdir, out_name), combined)

    print(f"Refactored FloodNet Visuals Saved to: {args.outdir}")

if __name__ == "__main__":
    main()
