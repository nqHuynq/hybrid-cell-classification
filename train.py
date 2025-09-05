import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from configs import config
from datasets.cell_dataset import CellDataset
from losses.spa_lpr_loss import CombinedSPALoss
from utils.transforms import get_basic_transform, get_augmented_transform
from utils.trainer import train_one_epoch, evaluate
from utils.metrics import tune_thresholds, evaluate as eval_metrics
from models.convnext_model import get_convnext_model 

def main():
    gc.collect()
    torch.cuda.empty_cache()
    print("CUDA available:", torch.cuda.is_available())

    # =========================
    # DATA
    # =========================
    train_dataset = CellDataset(
        json_file=os.path.join(config.JSON_DIR, "train.json"),
        img_dir=os.path.join(config.DATA_DIR, "train"),
        transform=get_augmented_transform(config.IMG_SIZE),
    )
    val_dataset = CellDataset(
        json_file=os.path.join(config.JSON_DIR, "val.json"),
        img_dir=os.path.join(config.DATA_DIR, "val"),
        transform=get_basic_transform(config.IMG_SIZE),
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
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # =========================
    # MODEL
    # =========================
    model = get_convnext_model().to(config.DEVICE)  

    # =========================
    # LOSS & OPTIMIZER
    # =========================
    criterion = CombinedSPALoss(lambda_lpr=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # =========================
    # TRAIN LOOP
    # =========================
    best_macro_f1 = 0.0
    best_thresholds = None

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")

        train_loss, _, _ = train_one_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE
        )

        preds, y_true_val, avg_auc, y_probs_val, val_loss = evaluate(
            model, val_loader, config.DEVICE, criterion
        )

        thresholds = tune_thresholds(y_true_val, y_probs_val)
        results = eval_metrics(y_true_val, y_probs_val, thresholds)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | AUC: {avg_auc:.4f} | "
              f"Micro-F1: {results['micro_f1']:.4f} | Macro-F1: {results['macro_f1']:.4f}")

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


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()  
    main()
