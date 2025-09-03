import torch

# Paths
IMG_DIR = "/content/images/images"
LABEL_FILE = "/content/labels.json"

# Hyperparams
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42
NUM_CLASSES = 9
ACCUMULATION_STEPS = 2
LR = 1e-4

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
