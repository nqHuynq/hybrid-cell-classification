import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

# =========================
# Threshold tuning
# =========================
def tune_thresholds(y_true, y_pred_probs, step=0.01):
    """
    Tìm ngưỡng tối ưu cho từng class dựa trên F1-score
    """
    num_classes = y_true.shape[1]
    best_thresholds = []
    for c in range(num_classes):
        best_f1 = 0
        best_t = 0.5
        for t in np.arange(0.1, 0.9, step):
            preds = (y_pred_probs[:, c] >= t).astype(int)
            f1 = f1_score(y_true[:, c], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds.append(best_t)
    return np.array(best_thresholds)

# =========================
# Evaluation
# =========================
def evaluate(y_true, y_pred_probs, thresholds=None):
    """
    Trả về dict gồm nhiều metrics:
      - micro/macro F1
      - exact match accuracy
      - classification report
    """
    if thresholds is None:
        thresholds = np.array([0.5] * y_true.shape[1])

    y_pred = (y_pred_probs >= thresholds).astype(int)

    results = {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "exact_match_acc": accuracy_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }
    return results

# =========================
# maP-x & maP-y (MAP for multi-label)
# =========================
def mean_average_precision(y_true, y_pred_probs):
    """
    maPx: tính trung bình AP theo class (class-wise mAP)
    maPy: tính trung bình AP theo sample (sample-wise mAP)
    """
    from sklearn.metrics import average_precision_score

    # maPx (theo class)
    ap_class = []
    for i in range(y_true.shape[1]):
        ap = average_precision_score(y_true[:, i], y_pred_probs[:, i])
        ap_class.append(ap)
    maPx = np.nanmean(ap_class)

    # maPy (theo sample)
    ap_sample = []
    for i in range(y_true.shape[0]):
        ap = average_precision_score(y_true[i, :], y_pred_probs[i, :])
        ap_sample.append(ap)
    maPy = np.nanmean(ap_sample)

    return {"maPx": maPx, "maPy": maPy}
