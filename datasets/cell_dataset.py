import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import os
import json
import pandas as pd

# =========================
# CutMix augmentation
# =========================
def cutmix(image, label, image2, label2, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    H, W = image.size[1], image.size[0]  
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    w = int(W * np.sqrt(1 - lam))
    h = int(H * np.sqrt(1 - lam))
    x0 = np.clip(cx - w // 2, 0, W)
    y0 = np.clip(cy - h // 2, 0, H)
    x1 = np.clip(cx + w // 2, 0, W)
    y1 = np.clip(cy + h // 2, 0, H)

    image = np.array(image)
    image2 = np.array(image2)
    image[y0:y1, x0:x1] = image2[y0:y1, x0:x1]

    lam_adjusted = 1 - ((x1 - x0) * (y1 - y0) / (W * H))
    new_image = Image.fromarray(image.astype(np.uint8))
    new_label = lam_adjusted * label + (1 - lam_adjusted) * label2

    return new_image, new_label

# =========================
# Dataset
# =========================
class CellDataset(Dataset):
    def __init__(self, img_dir=None, labels_df=None, json_file=None, transform=None, split="train", apply_cutmix=False, rare_indices=None):
        # Load labels từ JSON nếu có
        if json_file is not None:
            with open(json_file, "r") as f:
                labels_dict = json.load(f)
            labels_df = pd.DataFrame.from_dict(labels_dict, orient="index")

        self.img_dir = img_dir
        self.labels_df = labels_df
        self.transform = transform
        self.split = split
        self.apply_cutmix = apply_cutmix
        self.rare_indices = rare_indices if rare_indices is not None else []
        self.img_paths = [os.path.join(img_dir, fname + ".jpg") for fname in labels_df.index]
        self.labels = labels_df.values.astype("float32")
        self.num_classes = self.labels.shape[1]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.apply_cutmix and idx in self.rare_indices and random.random() < 0.5:
            idx2 = random.randint(0, len(self.img_paths) - 1)
            img2 = Image.open(self.img_paths[idx2]).convert("RGB")
            label2 = self.labels[idx2]
            img, label = cutmix(img, label, img2, label2)

        if self.transform:
            img = self.transform(img)

        return img, label

# =========================
# Weighted sampler helper
# =========================
def compute_sample_weights(labels):
    label_counts = labels.sum(axis=0)
    class_weights = 1.0 / (label_counts + 1e-6)
    weights = (labels * class_weights).sum(axis=1)
    return weights
