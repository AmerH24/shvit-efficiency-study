"""
Experimental SHViT-S1 variants used in the scaled reproduction study.

The original SHViT-S1 uses an approximately 1/4.67 partial-channel
ratio for Single-Head Self-Attention (SHSA). This module defines
controlled variants for reproducing the partial-ratio ablation from
the SHViT paper, plus one stage-wise allocation extension.
"""

from timm.models.registry import register_model

from .shvit import SHViT


# ---------------------------------------------------------------------
# Shared SHViT-S1 architecture
# ---------------------------------------------------------------------

S1_EMBED_DIM = [128, 224, 320]
S1_DEPTH = [2, 4, 5]
S1_TYPES = ["i", "s", "s"]


def _build_s1(partial_dim, num_classes=1000, distillation=False):
    """
    Construct an SHViT-S1 model while changing only the number of
    channels processed by SHSA.

    This keeps depth, width, and stage types fixed so that our
    experiments isolate the effect of partial-channel allocation.
    """
    return SHViT(
        num_classes=num_classes,
        distillation=distillation,
        embed_dim=S1_EMBED_DIM,
        depth=S1_DEPTH,
        partial_dim=partial_dim,
        types=S1_TYPES,
    )


# ---------------------------------------------------------------------
# Paper reproduction variants
# ---------------------------------------------------------------------

@register_model
def shvit_s1_ratio_1_8(
    num_classes=1000,
    pretrained=False,
    distillation=False,
    **kwargs,
):
    """
    SHViT-S1 using approximately 1/8 of Stage 2 and Stage 3 channels
    for Single-Head Self-Attention.
    """
    return _build_s1(
        partial_dim=[16, 28, 40],
        num_classes=num_classes,
        distillation=distillation,
    )


@register_model
def shvit_s1_ratio_default(
    num_classes=1000,
    pretrained=False,
    distillation=False,
    **kwargs,
):
    """
    Official SHViT-S1 partial-channel configuration.

    Stage 2: 48 / 224 ~= 21.4%
    Stage 3: 68 / 320 ~= 21.3%

    This closely corresponds to the paper's default ratio of 1/4.67.
    """
    return _build_s1(
        partial_dim=[32, 48, 68],
        num_classes=num_classes,
        distillation=distillation,
    )


@register_model
def shvit_s1_ratio_1_2(
    num_classes=1000,
    pretrained=False,
    distillation=False,
    **kwargs,
):
    """
    SHViT-S1 using half of Stage 2 and Stage 3 channels for SHSA.
    """
    return _build_s1(
        partial_dim=[64, 112, 160],
        num_classes=num_classes,
        distillation=distillation,
    )


# ---------------------------------------------------------------------
# Project extension
# ---------------------------------------------------------------------

@register_model
def shvit_s1_progressive(
    num_classes=1000,
    pretrained=False,
    distillation=False,
    **kwargs,
):
    """
    Stage-wise partial-attention allocation.

    Instead of assigning approximately the same attention ratio to
    every attention stage, this variant allocates fewer SHSA channels
    to Stage 2 and more to Stage 3.

    Stage 2: 32 / 224 ~= 14.3%
    Stage 3: 80 / 320 = 25.0%

    The total channel-block attention budget is kept close to the
    original configuration:

        original:    4*48 + 5*68 = 532
        progressive: 4*32 + 5*80 = 528

    This allows us to test whether moving global-attention capacity
    toward later, semantically richer features improves the
    accuracy-efficiency trade-off.
    """
    return _build_s1(
        partial_dim=[32, 32, 80],
        num_classes=num_classes,
        distillation=distillation,
    )