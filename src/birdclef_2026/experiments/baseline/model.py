import timm
import torch.nn as nn


def build_efficientnet_b3_backbone(unfreeze_blocks: int = 0) -> nn.Module:
    """EfficientNet-B3 backbone with 1-channel input.

    All parameters are frozen by default. With unfreeze_blocks > 0, the last N
    block groups plus conv_head/bn2 are unfrozen for fine-tuning.
    """
    backbone = timm.create_model(
        "efficientnet_b3", pretrained=True, in_chans=1, num_classes=0
    )
    for param in backbone.parameters():
        param.requires_grad = False
    if unfreeze_blocks > 0:
        for block_group in backbone.blocks[-unfreeze_blocks:]:
            for param in block_group.parameters():
                param.requires_grad = True
        for param in backbone.conv_head.parameters():
            param.requires_grad = True
        for param in backbone.bn2.parameters():
            param.requires_grad = True
    return backbone


def build_frozen_efficientnet_b3_backbone() -> nn.Module:
    """EfficientNet-B3 backbone with all parameters frozen."""
    return build_efficientnet_b3_backbone(unfreeze_blocks=0)


def build_head(in_features: int, n_classes: int, dropout: float = 0.0, hidden: int = 0) -> nn.Module:
    """Classification head: Linear or MLP.

    When hidden > 0, uses Linear → ReLU → Dropout → Linear.
    When hidden == 0, uses Dropout → Linear (linear probe).
    """
    if hidden > 0:
        return nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, n_classes),
    )
