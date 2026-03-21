import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    """Number of samples in the batch where the true label appears in the top-k predictions."""
    topk_indices = logits.topk(k, dim=1).indices
    return (topk_indices == targets.unsqueeze(1)).any(dim=1).sum().item()


def macro_roc_auc(
    all_logits: torch.Tensor,
    all_targets: torch.Tensor,
    n_classes: int,
) -> float:
    """Macro-averaged ROC-AUC, ignoring classes with no positive examples in the set.

    Mirrors the official BirdCLEF metric: classes absent from the solution are excluded
    so they don't artificially inflate or deflate the score.
    """
    probs = torch.softmax(all_logits, dim=1).cpu().numpy()  # (N, C)
    one_hot = np.zeros((len(all_targets), n_classes), dtype=np.float32)
    one_hot[np.arange(len(all_targets)), all_targets.cpu().numpy()] = 1.0

    # Only score classes that have at least one positive example
    class_sums = one_hot.sum(axis=0)
    scored = class_sums > 0
    assert scored.sum() > 0

    return roc_auc_score(one_hot[:, scored], probs[:, scored], average="macro")
