import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import convnext_base

from datasets.cell_dataset import CellDataset
from losses.spa_lpr_loss import CombinedSPALoss
from utils.transforms import get_basic_transform, get_augmented_transform
from utils.trainer import train_one_epoch, evaluate
from utils.metrics import tune_thresholds, evaluate as eval_metrics

# =========================
# CONFIG
# =========================
BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 1e-4
EPOCHS = 10
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_JSON = "data/labels.json"
DATA_DIR = "data/images"

gc.collect()
torch.cuda.empty_cache()

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()  # for Windows
    print(torch.cuda.is_available())

    # =========================
    # DATA
    # =========================
    train_dataset = CellDataset(
        json_file=DATA_JSON,
        img_dir=DATA_DIR,
        transform=get_augmented_transform(IMG_SIZE),
        split="train"
    )
    val_dataset = CellDataset(
        json_file=DATA_JSON,
        img_dir=DATA_DIR,
        transform=get_basic_transform(IMG_SIZE),
        split="val"
    )

    # Weighted sampler (handle imbalance)
    if hasattr(train_dataset, "sample_weights"):
        sampler = WeightedRandomSampler(
            train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # =========================
    # MODEL
    # =========================
    model = convnext_base(weights="IMAGENET1K_V1")
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, train_dataset.num_classes)
    model = model.to(DEVICE)

    # =========================
    # LOSS & OPTIMIZER
    # =========================
    criterion = CombinedSPALoss(lambda_lpr=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # =========================
    # TRAIN LOOP
    # =========================
    best_macro_f1 = 0.0
    best_thresholds = None

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, y_true_train, y_probs_train = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )

        preds, y_true_val, avg_auc, y_probs_val, val_loss = evaluate(
            model, val_loader, DEVICE, criterion
        )

        thresholds = tune_thresholds(y_true_val, y_probs_val)
        results = eval_metrics(y_true_val, y_probs_val, thresholds)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | AUC: {avg_auc:.4f} | Micro-F1: {results['micro_f1']:.4f} | Macro-F1: {results['macro_f1']:.4f}")

        if results["macro_f1"] > best_macro_f1:
            best_macro_f1 = results["macro_f1"]
            best_thresholds = thresholds
            torch.save(
                {"model": model.state_dict(), "thresholds": thresholds},
                "best_model.pth"
            )
            print("✅ Saved best model!")

        scheduler.step()

        gc.collect()
        torch.cuda.empty_cache()

