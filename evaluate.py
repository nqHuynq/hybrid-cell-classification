import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from configs import config
from datasets.cell_dataset import CellDataset
from utils.transforms import get_basic_transform
from utils.trainer import evaluate
from utils.metrics import evaluate as eval_metrics, mean_average_precision
from models.convnext_model import get_convnext_model


def main():
    # =========================
    # LOAD DATA
    # =========================
    test_dataset = CellDataset(
        json_file=os.path.join(config.JSON_DIR, "test.json"),
        img_dir=os.path.join(config.DATA_DIR, "test"),
        transform=get_basic_transform(config.IMG_SIZE),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # =========================
    # LOAD MODEL
    # =========================
    checkpoint = torch.load("best_model.pth", map_location=config.DEVICE, weights_only=False)
    thresholds = checkpoint["thresholds"]

    model = get_convnext_model()  # 🔑 dùng cùng kiến trúc với train.py
    model.load_state_dict(checkpoint["model"])
    model = model.to(config.DEVICE)

    # =========================
    # EVALUATE
    # =========================
    preds, y_true, avg_auc, y_probs, avg_loss = evaluate(model, test_loader, config.DEVICE)

    results = eval_metrics(y_true, y_probs, thresholds)
    maps = mean_average_precision(y_true, y_probs)

    print("\n=== Test Results ===")
    print(f"Loss: {avg_loss:.4f} | AUC: {avg_auc:.4f}")
    print(f"Micro-F1: {results['micro_f1']:.4f} | Macro-F1: {results['macro_f1']:.4f} | "
          f"EMA: {results['exact_match_acc']:.4f}")
    print(f"maPx: {maps['maPx']:.4f} | maPy: {maps['maPy']:.4f}")

    # =========================
    # CLASSIFICATION REPORT
    # =========================
    print("\n--- Final Classification Report (Test Set) ---")
    print(classification_report(
        y_true,
        preds,
        target_names=[f"Label_{i}" for i in range(y_true.shape[1])],
        digits=2
    ))


if __name__ == "__main__":
    main()
