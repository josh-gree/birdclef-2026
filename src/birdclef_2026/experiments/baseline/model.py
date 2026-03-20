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


def build_mlp_head(
    in_features: int, n_classes: int, hidden_size: int = 512
) -> nn.Module:
    """2-layer MLP classification head."""
    return nn.Sequential(
        nn.Linear(in_features, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, n_classes),
    )
