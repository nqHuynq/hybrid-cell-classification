import gc
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def train_one_epoch(model, loader, optimizer, criterion, device, accumulation_steps=1):
    """
    Train model for one epoch
    Returns:
        avg_loss, y_true, y_probs
    """
    model.train()
    total_loss = 0.0
    all_targets, all_probs = [], []

    optimizer.zero_grad()

    for step, (imgs, labels) in enumerate(tqdm(loader, desc="Training")):
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient Accumulation
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            gc.collect()
            torch.cuda.empty_cache()

        total_loss += loss.item()
        all_targets.append(labels.detach().cpu().numpy())
        all_probs.append(torch.sigmoid(outputs).detach().cpu().numpy())

    avg_loss = total_loss / len(loader)
    y_true = np.vstack(all_targets)
    y_probs = np.vstack(all_probs)

    return avg_loss, y_true, y_probs



def evaluate(model, loader, device, criterion=None):
    """
    Evaluate model
    Returns:
        preds, y_true, avg_auc, y_probs, avg_loss
    """
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            if criterion is not None:
                total_loss += criterion(outputs, labels).item() * imgs.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_targets.append(labels.cpu().numpy())
            all_probs.append(probs)
            all_preds.append(preds)

    y_true = np.vstack(all_targets)
    y_probs = np.vstack(all_probs)
    y_preds = np.vstack(all_preds)

    # Compute average AUC
    auc_scores = []
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_probs[:, i])
            auc_scores.append(auc)
        except ValueError:
            continue
    avg_auc = np.mean(auc_scores) if len(auc_scores) > 0 else 0.0

    avg_loss = total_loss / len(loader.dataset) if criterion is not None else 0.0

    return y_preds, y_true, avg_auc, y_probs, avg_loss


def predict_with_thresholds(model, loader, thresholds, device):
    """
    Predict with custom thresholds
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Predicting"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > thresholds[None, :]).astype(int)
            all_preds.append(preds)
    return np.vstack(all_preds)
