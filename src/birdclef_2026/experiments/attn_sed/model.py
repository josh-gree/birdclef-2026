import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPooling(nn.Module):
    """Generalised Mean Pooling over the frequency axis only.

    Collapses dim=2 (frequency) of a (B, C, H', W') feature map, leaving
    the time axis (W') intact for the attention head downstream.

    p is a learnable scalar; p=1 → average pool, p→∞ → max pool.
    """

    def __init__(self, init_p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(init_p))
        self.eps = eps

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, C, H', W')
        p = self.p.clamp(min=1.0)
        return h.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)
        # returns (B, C, W') = (B, C, T)


class AttnSEDHead(nn.Module):
    """Attention Sound Event Detection head.

    Operates over T time steps. Two parallel branches:
    - att_fc: learns per-class attention weights over time
    - cls_fc: produces per-time-step logits

    Final logits are the attention-weighted sum over time.
    Also returns timewise_logits for the compound loss.
    """

    def __init__(self, num_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.pre_fc = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.att_fc = nn.Linear(num_features, num_classes)
        self.cls_fc = nn.Linear(num_features, num_classes)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: (B, C, T)
        h = h.permute(0, 2, 1)                                        # (B, T, C)
        h = self.pre_fc(h)                                             # (B, T, C)
        att_w = F.softmax(torch.tanh(self.att_fc(h)), dim=1)          # (B, T, n_classes)
        timewise = self.cls_fc(h)                                      # (B, T, n_classes)
        logits = (att_w * timewise).sum(dim=1)                         # (B, n_classes)
        return logits, timewise


class AttnSEDModel(nn.Module):
    """Configurable backbone + GeM freq pooling + AttnSEDHead.

    The full backbone is trained end-to-end (no freezing). global_pool=""
    and num_classes=0 are set so the backbone returns a spatial feature map
    (B, C, H', W') rather than a pooled vector.
    """

    def __init__(
        self,
        n_classes: int,
        dropout: float = 0.2,
        backbone_name: str = "hgnetv2_b0.ssld_stage2_ft_in1k",
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            in_chans=1,
            num_classes=0,
            global_pool="",
        )
        # Use dummy forward to determine actual output channels
        dummy = torch.zeros(1, 1, 256, 256)
        with torch.no_grad():
            num_features = self.backbone(dummy).shape[1]

        self.gem = GeMPooling()
        self.head = AttnSEDHead(num_features, n_classes, dropout=dropout)

    def forward_for_training(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)       # (B, C, H', W')
        h = self.gem(h)            # (B, C, T)
        return self.head(h)        # (B, n_classes), (B, T, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_for_training(x)
        return logits


def build_attn_sed_model(
    n_classes: int,
    dropout: float = 0.2,
    backbone_name: str = "hgnetv2_b0.ssld_stage2_ft_in1k",
) -> AttnSEDModel:
    return AttnSEDModel(n_classes=n_classes, dropout=dropout, backbone_name=backbone_name)
