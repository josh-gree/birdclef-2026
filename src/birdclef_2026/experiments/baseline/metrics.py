import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def macro_roc_auc(
    all_logits: torch.Tensor,
    all_targets: torch.Tensor,
) -> float:
    """Macro-averaged ROC-AUC, ignoring classes with no positive examples in the set.

    Mirrors the official BirdCLEF metric: classes absent from the solution are excluded
    so they don't artificially inflate or deflate the score.
    """
    probs = torch.sigmoid(all_logits).cpu().numpy()  # (N, C)
    targets = all_targets.cpu().numpy()

    # Only score classes that have at least one positive example
    scored = targets.sum(axis=0) > 0
    assert scored.sum() > 0

    return roc_auc_score(targets[:, scored], probs[:, scored], average="macro")
