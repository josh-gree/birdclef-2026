import timm
import torch.nn as nn


def build_frozen_efficientnet_b3_backbone() -> nn.Module:
    """EfficientNet-B3 backbone with 1-channel input, all parameters frozen."""
    backbone = timm.create_model(
        "efficientnet_b3", pretrained=True, in_chans=1, num_classes=0
    )
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


def build_head(in_features: int, n_classes: int, dropout: float = 0.0) -> nn.Module:
    """Classification head: Dropout → Linear."""
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, n_classes),
    )
