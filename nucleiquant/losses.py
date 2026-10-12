"""Segmentation losses: BCE + soft Dice, with per-channel weighting."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Per-channel soft Dice loss averaged over batch and channels. Shapes (B, C, H, W)."""
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (probs * targets).sum(dims)
    denom = probs.sum(dims) + targets.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice  # (C,)


class BCEDiceLoss(nn.Module):
    """``bce_weight * BCE + dice_weight * Dice``, each channel scaled by ``channel_weights``."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        channel_weights: list[float] | None = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer(
            "channel_weights",
            torch.tensor(channel_weights, dtype=torch.float32) if channel_weights else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        c = logits.shape[1]
        w = self.channel_weights if self.channel_weights is not None else torch.ones(c, device=logits.device)
        w = w / w.sum() * c  # keep overall scale ≈ unweighted

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean((0, 2, 3))  # (C,)
        dice = soft_dice_loss(logits, targets)  # (C,)
        return (w * (self.bce_weight * bce + self.dice_weight * dice)).mean()


def build_loss(cfg: dict, out_channels: int) -> nn.Module:
    lc = cfg.get("loss", {})
    weights = [1.0] * out_channels
    if out_channels >= 2:
        weights[1] = float(lc.get("boundary_channel_weight", 1.0))
    return BCEDiceLoss(
        bce_weight=float(lc.get("bce_weight", 1.0)),
        dice_weight=float(lc.get("dice_weight", 1.0)),
        channel_weights=weights,
    )
