import torch
import os

# Paths (dùng folder/json đã chia sẵn)
DATA_DIR = "data/images"
JSON_DIR = "data/labels"

# Hyperparams
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42
NUM_CLASSES = 9
ACCUMULATION_STEPS = 2
LR = 1e-4
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 4

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
