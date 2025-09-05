import os
import json
import shutil
import random

# ---- CONFIG ----
IMG_DIR = "data/images"
LABEL_FILE = "data/labels.json"

OUTPUT_IMG_DIR = "data/split_images"
OUTPUT_JSON_DIR = "data/split_json"

SPLIT_RATIO = [0.7, 0.15, 0.15]  # train, val, test

# ---- CREATE FOLDERS ----
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_IMG_DIR, split), exist_ok=True)

os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# ---- LOAD LABELS ----
with open(LABEL_FILE, "r") as f:
    labels = json.load(f)

keys = list(labels.keys())
random.shuffle(keys)

n = len(keys)
n_train = int(SPLIT_RATIO[0] * n)
n_val = int(SPLIT_RATIO[1] * n)

train_keys = keys[:n_train]
val_keys = keys[n_train:n_train+n_val]
test_keys = keys[n_train+n_val:]

splits = {
    "train": train_keys,
    "val": val_keys,
    "test": test_keys,
}

# ---- SAVE SPLIT JSON + MOVE IMAGES ----
for split, split_keys in splits.items():
    split_labels = {k: labels[k] for k in split_keys}
    
    # Save JSON
    with open(os.path.join(OUTPUT_JSON_DIR, f"{split}.json"), "w") as f:
        json.dump(split_labels, f, indent=4)
    
    # Copy images
    for k in split_keys:
        img_name = k + ".jpg"  # nếu ảnh có đuôi khác thì chỉnh lại
        src_path = os.path.join(IMG_DIR, img_name)
        dst_path = os.path.join(OUTPUT_IMG_DIR, split, img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

print("✅ Done splitting dataset!")
print(f"Train: {len(train_keys)} | Val: {len(val_keys)} | Test: {len(test_keys)}")
