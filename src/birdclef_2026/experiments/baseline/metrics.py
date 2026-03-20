import torch


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    """Number of samples in the batch where the true label appears in the top-k predictions."""
    topk_indices = logits.topk(k, dim=1).indices
    return (topk_indices == targets.unsqueeze(1)).any(dim=1).sum().item()
