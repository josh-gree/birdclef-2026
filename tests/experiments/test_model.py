import pytest

from birdclef_2026.experiments.baseline.model import build_efficientnet_b3_backbone


@pytest.fixture(scope="module")
def backbone_frozen():
    return build_efficientnet_b3_backbone(unfreeze_blocks=0)


@pytest.fixture(scope="module")
def backbone_unfreeze_1():
    return build_efficientnet_b3_backbone(unfreeze_blocks=1)


@pytest.fixture(scope="module")
def backbone_unfreeze_2():
    return build_efficientnet_b3_backbone(unfreeze_blocks=2)


def test_fully_frozen_has_no_trainable_params(backbone_frozen):
    """unfreeze_blocks=0 should leave every backbone parameter frozen.

    Why: the baseline linear probe must not accidentally train backbone weights,
    which would change the features and invalidate comparisons.
    """
    trainable = [p for p in backbone_frozen.parameters() if p.requires_grad]
    assert len(trainable) == 0


def test_unfreeze_2_last_blocks_are_trainable(backbone_unfreeze_2):
    """blocks[-2:] parameters must be trainable with unfreeze_blocks=2.

    Why: these are the blocks we explicitly requested to unfreeze — if they
    remain frozen the backbone won't actually be fine-tuned.
    """
    for block_group in backbone_unfreeze_2.blocks[-2:]:
        for param in block_group.parameters():
            assert param.requires_grad


def test_unfreeze_2_earlier_blocks_are_frozen(backbone_unfreeze_2):
    """blocks[:-2] parameters must remain frozen with unfreeze_blocks=2.

    Why: unfreezing early layers with a high learning rate destabilises
    low-level features; only the requested tail should be trainable.
    """
    for block_group in backbone_unfreeze_2.blocks[:-2]:
        for param in block_group.parameters():
            assert not param.requires_grad


def test_unfreeze_2_conv_head_and_bn2_are_trainable(backbone_unfreeze_2):
    """conv_head and bn2 must be trainable with unfreeze_blocks=2.

    Why: these sit between the last block group and the pooling layer — leaving
    them frozen would bottleneck gradients from the unfrozen blocks.
    """
    for param in backbone_unfreeze_2.conv_head.parameters():
        assert param.requires_grad
    for param in backbone_unfreeze_2.bn2.parameters():
        assert param.requires_grad


def test_unfreeze_2_stem_is_frozen(backbone_unfreeze_2):
    """conv_stem and bn1 must remain frozen with unfreeze_blocks=2.

    Why: the stem is far from the output and should not be updated during
    the initial fine-tuning phase.
    """
    for param in backbone_unfreeze_2.conv_stem.parameters():
        assert not param.requires_grad
    for param in backbone_unfreeze_2.bn1.parameters():
        assert not param.requires_grad


def test_unfreeze_1_only_last_block_trainable(backbone_unfreeze_1):
    """With unfreeze_blocks=1, only blocks[-1] (plus conv_head/bn2) should be trainable.

    Why: unfreeze_blocks controls exactly how many tail block groups are opened;
    blocks[-2] must still be frozen when only 1 is requested.
    """
    for param in backbone_unfreeze_1.blocks[-1].parameters():
        assert param.requires_grad
    for block_group in backbone_unfreeze_1.blocks[:-1]:
        for param in block_group.parameters():
            assert not param.requires_grad


def test_unfreeze_2_has_more_trainable_params_than_unfreeze_1(backbone_unfreeze_1, backbone_unfreeze_2):
    """More unfreeze_blocks should yield strictly more trainable parameters.

    Why: a sanity check that the block count is actually respected — if both
    counts gave the same trainable set, the slicing logic would be broken.
    """
    n1 = sum(p.numel() for p in backbone_unfreeze_1.parameters() if p.requires_grad)
    n2 = sum(p.numel() for p in backbone_unfreeze_2.parameters() if p.requires_grad)
    assert n2 > n1
