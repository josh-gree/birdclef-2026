import timm
import torch.nn as nn


def get_image_size(model_name: str) -> tuple[int, int]:
    """Look up the default input image size for a timm model.

    Returns (height, width) from the model's ``default_cfg['input_size']``.
    """
    cfg = timm.get_pretrained_cfg(model_name)
    if cfg is None:
        raise ValueError(f"No default config found for timm model '{model_name}'")
    _, h, w = cfg.input_size  # (channels, height, width)
    return h, w


def _unfreeze_efficientnet(backbone: nn.Module, unfreeze_blocks: int) -> None:
    """Unfreeze last N block groups + conv_head + bn2 on EfficientNet."""
    for block_group in backbone.blocks[-unfreeze_blocks:]:
        for param in block_group.parameters():
            param.requires_grad = True
    for param in backbone.conv_head.parameters():
        param.requires_grad = True
    for param in backbone.bn2.parameters():
        param.requires_grad = True


def _unfreeze_maxvit(backbone: nn.Module, unfreeze_blocks: int) -> None:
    """Unfreeze last N stages + norm on MaxViT."""
    for stage in backbone.stages[-unfreeze_blocks:]:
        for param in stage.parameters():
            param.requires_grad = True
    for param in backbone.norm.parameters():
        param.requires_grad = True
    for param in backbone.head.parameters():
        param.requires_grad = True


_UNFREEZE_FNS = {
    "efficientnet": _unfreeze_efficientnet,
    "tf_efficientnet": _unfreeze_efficientnet,
    "maxvit": _unfreeze_maxvit,
}


def _get_unfreeze_fn(model_name: str):
    for prefix, fn in _UNFREEZE_FNS.items():
        if model_name.startswith(prefix):
            return fn
    raise ValueError(
        f"No unfreezing strategy for '{model_name}'. "
        f"Supported families: {list(_UNFREEZE_FNS.keys())}. "
        f"Use unfreeze_blocks=0 (linear probe) or add a strategy."
    )


def build_backbone(
    model_name: str = "efficientnet_b3",
    pretrained: bool = True,
    unfreeze_blocks: int = 0,
) -> nn.Module:
    """Build a timm backbone with 1-channel input.

    All parameters are frozen by default. With unfreeze_blocks > 0, uses an
    architecture-specific strategy to unfreeze the last N stages.
    """
    backbone = timm.create_model(
        model_name, pretrained=pretrained, in_chans=1, num_classes=0
    )
    for param in backbone.parameters():
        param.requires_grad = False
    if unfreeze_blocks > 0:
        unfreeze_fn = _get_unfreeze_fn(model_name)
        unfreeze_fn(backbone, unfreeze_blocks)
    return backbone


# Alias for backwards compatibility with submission notebook and tests.
def build_efficientnet_b3_backbone(unfreeze_blocks: int = 0) -> nn.Module:
    return build_backbone("efficientnet_b3", pretrained=True, unfreeze_blocks=unfreeze_blocks)


def build_head(in_features: int, n_classes: int, dropout: float = 0.0, hidden: int = 0) -> nn.Module:
    """Classification head: Linear or MLP.

    When hidden > 0, uses Linear → ReLU → Dropout → Linear.
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
