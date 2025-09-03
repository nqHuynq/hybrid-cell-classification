import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# SPA Loss
# =========================
class SPALoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        loss = - targets * torch.log(probs + self.eps) - (1 - targets) * torch.log(1 - probs + self.eps)
        return loss.mean()

# =========================
# Pairwise Regularizer
# =========================
def pairwise_regularizer(logits, labels):
    probs = torch.sigmoid(logits)
    loss = 0.0
    count = 0

    for i in range(probs.size(0)):
        pos_indices = (labels[i] == 1).nonzero(as_tuple=True)[0]
        neg_indices = (labels[i] == 0).nonzero(as_tuple=True)[0]
        for pos_idx in pos_indices:
            for neg_idx in neg_indices:
                diff = probs[i][neg_idx] - probs[i][pos_idx]
                loss += torch.log(1 + torch.exp(diff))
                count += 1
    return loss / (count + 1e-8)

# =========================
# Combined SPA + PR
# =========================
class CombinedSPALoss(nn.Module):
    def __init__(self, lambda_lpr=0.1):
        super().__init__()
        self.spa = SPALoss()
        self.lambda_lpr = lambda_lpr

    def forward(self, logits, targets):
        spa_loss = self.spa(logits, targets)
        lpr_loss = pairwise_regularizer(logits, targets)
        return spa_loss + self.lambda_lpr * lpr_loss
