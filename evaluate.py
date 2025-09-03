import torch
from torch.utils.data import DataLoader
from torchvision.models import convnext_base

from datasets.cell_dataset import CellDataset
from utils.transforms import get_basic_transform
from utils.trainer import evaluate, predict_with_thresholds
from utils.metrics import evaluate as eval_metrics, mean_average_precision

# =========================
# CONFIG
# =========================
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_JSON = "data/labels.json"
DATA_DIR = "data/images"

# =========================
# LOAD DATA
# =========================
test_dataset = CellDataset(
    json_file=DATA_JSON,
    img_dir=DATA_DIR,
    transform=get_basic_transform(IMG_SIZE),
    split="test"
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# =========================
# LOAD MODEL
# =========================
checkpoint = torch.load("best_model.pth", map_location=DEVICE)
thresholds = checkpoint["thresholds"]

model = convnext_base(weights=None)
model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, test_dataset.num_classes)
model.load_state_dict(checkpoint["model"])
model = model.to(DEVICE)

# =========================
# EVALUATE
# =========================
preds, y_true, avg_auc, y_probs, avg_loss = evaluate(model, test_loader, DEVICE)

results = eval_metrics(y_true, y_probs, thresholds)
maps = mean_average_precision(y_true, y_probs)

print("\n=== Test Results ===")
print(f"Loss: {avg_loss:.4f} | AUC: {avg_auc:.4f}")
print(f"Micro-F1: {results['micro_f1']:.4f} | Macro-F1: {results['macro_f1']:.4f} | EMA: {results['exact_match_acc']:.4f}")
print(f"maPx: {maps['maPx']:.4f} | maPy: {maps['maPy']:.4f}")
