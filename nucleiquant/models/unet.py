"""U-Net (Ronneberger et al., 2015) with a few modern defaults.

* BatchNorm + ReLU in every conv block
* Padded convolutions, so output size == input size
* Bilinear upsampling + 1x1 conv instead of transposed convs (fewer checkerboard
  artefacts, fewer parameters)
* Optional dropout in the bottleneck

``out_channels=2`` gives the [foreground, boundary] heads used by NucleiQuant.
Set ``out_channels=1`` for a plain binary U-Net if you want the ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 1),
        )
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Guard against odd spatial sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        chs = [base_channels * 2**i for i in range(depth + 1)]  # e.g. 32,64,128,256,512

        self.inc = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList(
            [nn.Sequential(nn.MaxPool2d(2), DoubleConv(chs[i], chs[i + 1])) for i in range(depth)]
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.ups = nn.ModuleList(
            [Up(chs[i + 1], chs[i], chs[i]) for i in reversed(range(depth))]
        )
        self.outc = nn.Conv2d(chs[0], out_channels, 1)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = self.dropout(skips.pop())
        for up in self.ups:
            x = up(x, skips.pop())
        return self.outc(x)  # logits; apply sigmoid downstream

    @property
    def divisor(self) -> int:
        """Inputs should be padded to a multiple of this for clean skip shapes."""
        return 2**self.depth


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    if m["name"].lower() != "unet":
        raise ValueError(f"Unknown model {m['name']!r}; only 'unet' is implemented in Phase 1")
    return UNet(
        in_channels=m.get("in_channels", 3),
        out_channels=m.get("out_channels", 2),
        base_channels=m.get("base_channels", 32),
        depth=m.get("depth", 4),
        dropout=m.get("dropout", 0.0),
    )
