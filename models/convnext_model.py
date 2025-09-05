import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from configs.config import NUM_CLASSES

def get_convnext_model():
    model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.LayerNorm((num_ftrs,), eps=1e-6),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    return model
