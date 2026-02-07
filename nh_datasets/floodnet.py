
import os, random
import albumentations as A
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoImageProcessor,
    SegformerImageProcessor
)

from .registry import register_dataset


# -----------------------
# Dataset (semantic → class/mask + semantic GT for metrics)
# -----------------------

@register_dataset("floodnet_mask2former")
class FloodNetMask2FormerDataset(Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}
    CLASSES = [
        "background",
        "building flooded",
        "building non-flooded",
        "road flooded",
        "road non-flooded",
        "water",
        "tree",
        "vehicle",
        "pool",
        "grass",
    ]
    label2id = {name: i for i, name in enumerate(CLASSES)}
    id2label = {i: name for i, name in enumerate(CLASSES)}
    def __init__(
        self,
        root: str,
        split: str,
        image_processor: AutoImageProcessor,
        num_classes: int = 10,
        image_size: int = 512,
        augment: bool = False,
        ignore_index: int = 0,
    ):
        self.root = Path(root)
        self.split = split
        self.ip = image_processor
        self.num_classes = num_classes
        self.image_size = image_size
        self.augment = augment
        self.ignore_index = ignore_index

        self.img_dir = self.root / split / f"{split}-org-img"
        self.lbl_dir = self.root / split / f"{split}-label-img"
        if not self.img_dir.is_dir() or not self.lbl_dir.is_dir():
            raise FileNotFoundError(f"Missing directories: {self.img_dir} or {self.lbl_dir}")

        self.samples: List[Tuple[Path, Path]] = []
        for fname in os.listdir(self.img_dir):
            stem, ext = os.path.splitext(fname)
            if ext not in self.IMG_EXTS:
                continue
            img_p = self.img_dir / fname
            for ext2 in self.IMG_EXTS:
                lbl_p = self.lbl_dir / f"{stem}_lab{ext2}"
                if lbl_p.exists():
                    self.samples.append((img_p, lbl_p))
                    break
        if len(self.samples) == 0:
            raise RuntimeError(f"No pairs found under {self.img_dir} & {self.lbl_dir}")

        self._rng = random.Random(1337)

    def __len__(self):
        return len(self.samples)

    def _load_pair(self, img_path: Path, lbl_path: Path):
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        
        lab = Image.open(lbl_path)
        if lab.mode != "L":
            lab = lab.convert("L")
        # Transpose label too just in case it has metadata, though unlikely for PNG
        lab = ImageOps.exif_transpose(lab)
        
        if img.size != lab.size:
            # Fallback if there's a slight mismatch
            lab = lab.resize(img.size, Image.NEAREST)
            
        return img, lab

    def _maybe_flip(self, img: Image.Image, lab: Image.Image):
        if self.augment and self._rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
        return img, lab

    @staticmethod
    def _classwise_masks_from_semantic(lab_np: np.ndarray, num_classes: int, ignore_index: int):
        mask_list = []
        class_ids = []
        for c in range(num_classes):
            if c == ignore_index:
                continue
            m = (lab_np == c)
            if m.any():
                mask_list.append(torch.from_numpy(m.astype(np.float32)))  # float32 (0/1)
                class_ids.append(c)
        if not class_ids:
            m = (lab_np == 0).astype(np.float32)
            mask_list.append(torch.from_numpy(m))
            class_ids.append(0)
        mask_tensor = torch.stack(mask_list, dim=0)              # [K,H,W] float32
        class_tensor = torch.tensor(class_ids, dtype=torch.long)  # [K]
        return mask_tensor, class_tensor

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]
        img, lab = self._load_pair(img_path, lbl_path)
        img, lab = self._maybe_flip(img, lab)
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        lab = lab.resize((self.image_size, self.image_size), Image.NEAREST)

        lab_np = np.array(lab, dtype=np.int64)  # [H,W]
        encoded = self.ip(images=img, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)  # [3,H,W]
        mask_tensor, class_tensor = self._classwise_masks_from_semantic(
            lab_np, num_classes=self.num_classes, ignore_index=self.ignore_index
        )
        return {
            "pixel_values": pixel_values,                 # [3,H,W] float
            "class_labels": class_tensor,                 # [K] long
            "mask_labels": mask_tensor,                   # [K,H,W] float32
            "labels_semantic": torch.from_numpy(lab_np),  # [H,W] long (metrics only)
            "id": img_path.stem,
        }

@register_dataset("floodnet_segformer")
class FloodNetSegDataset(Dataset):
    """
    Read JPEG images and PNG label masks (indexed class IDs).
    Expects pairs like:
      val/val-org-img/6336.jpg
      val/val-label-img/6336_lab.png
    """
    IMG_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}
    CLASSES = [
        "background",
        "building flooded",
        "building non-flooded",
        "road flooded",
        "road non-flooded",
        "water",
        "tree",
        "vehicle",
        "pool",
        "grass",
    ]
    label2id = {name: i for i, name in enumerate(CLASSES)}
    id2label = {i: name for i, name in enumerate(CLASSES)}

    def __init__(
        self,
        root: str,
        split: str,
        image_processor: SegformerImageProcessor,
        image_size: int = 512,
        augment: bool = False,
        ignore_index: int = 255,
        num_classes: int = 10
    ):
        self.root = Path(root)
        self.split = split
        self.image_processor = image_processor
        self.crop_size = image_size
        self.augment = augment
        self.ignore_index = ignore_index
        self.num_classes = num_classes

        self.img_dir = self.root / split / f"{split}-org-img"
        self.lbl_dir = self.root / split / f"{split}-label-img"

        if not self.img_dir.is_dir() or not self.lbl_dir.is_dir():
            raise FileNotFoundError(f"Missing directories: {self.img_dir} or {self.lbl_dir}")

        self.samples: List[Tuple[Path, Path]] = []
        for fname in os.listdir(self.img_dir):
            stem, ext = os.path.splitext(fname)
            if ext not in self.IMG_EXTS:
                continue
            img_p = self.img_dir / fname
            
            for ext2 in self.IMG_EXTS:
                lbl_p = self.lbl_dir / f"{stem}_lab{ext2}"
                if lbl_p.exists():
                    self.samples.append((img_p, lbl_p))
                    break

        if len(self.samples) == 0:
            raise RuntimeError(f"No pairs found under {self.img_dir} & {self.lbl_dir}")

        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Same as RescueNet
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.9),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.9),
                ], p=0.8),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, fill_value=0, mask_fill_value=None, p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            ])
        else:
            # Match RescueNet validation padding
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=self.crop_size, min_width=self.crop_size),
            ])

    def __len__(self):
        return len(self.samples)

    def _load_pair(self, img_path: Path, lbl_path: Path):
        # Image as RGB
        img = Image.open(img_path).convert("RGB")
        # Label as index map (Mode 'L' expected). If 'P' it’s also fine — convert('L') keeps indices.
        lab = Image.open(lbl_path)
        if lab.mode != "L":
            lab = lab.convert("L")
        return img, lab

    def get_class_aware_crop(self, img, mask):
        # Target classes for FloodNet: flooded stuff and vehicles/pools
        rare_classes = [1, 3, 7, 8] # building flooded, road flooded, vehicle, pool
        h, w = mask.shape
        crop_h, crop_w = self.crop_size, self.crop_size
        
        if h <= crop_h or w <= crop_w:
            return 0, 0, h, w

        # 50% chance to force a crop on a rare class
        if random.random() < 0.5:
            rare_pixels = np.isin(mask, rare_classes)
            y_indices, x_indices = np.where(rare_pixels)
            
            if len(y_indices) > 0:
                idx = random.randint(0, len(y_indices) - 1)
                center_y, center_x = y_indices[idx], x_indices[idx]
                
                top = max(0, min(center_y - crop_h // 2, h - crop_h))
                left = max(0, min(center_x - crop_w // 2, w - crop_w))
                return top, left, top + crop_h, left + crop_w

        # Fallback: Random crop
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        return top, left, top + crop_h, left + crop_w

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]
        img, lab = self._load_pair(img_path, lbl_path)
        
        # Convert to arrays for Albumentations and Cropping
        image = np.array(img)
        mask = np.array(lab, dtype=np.uint8)

        if self.augment:
            top, left, bottom, right = self.get_class_aware_crop(image, mask)
            image = image[top:bottom, left:right]
            mask = mask[top:bottom, left:right]
            
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            # Validation logic
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            
            h, w, _ = image.shape
            if h > self.crop_size and w > self.crop_size:
                top = (h - self.crop_size) // 2
                left = (w - self.crop_size) // 2
                image = image[top:top+self.crop_size, left:left+self.crop_size]
                mask = mask[top:top+self.crop_size, left:left+self.crop_size]

        # Convert back to Long for PyTorch
        lab_np = mask.astype(np.int64) 

        # Let image_processor normalize & convert to tensor
        encoded = self.image_processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)  # [3, H, W]

        return {
            "pixel_values": pixel_values,         # FloatTensor
            "labels": torch.from_numpy(lab_np),   # LongTensor [H, W]
            "id": img_path.stem,
        }
