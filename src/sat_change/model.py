from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__(nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU(), nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU())


class SatelliteEncoder(nn.Module):
    """Small fully convolutional encoder retaining dense feature stages."""
    def __init__(self, in_channels: int = 4, width: int = 32, projection_dim: int = 128):
        super().__init__()
        self.stem = ConvBlock(in_channels, width)
        self.stage1 = ConvBlock(width, width * 2, 2)
        self.stage2 = ConvBlock(width * 2, width * 4, 2)
        self.stage3 = ConvBlock(width * 4, width * 8, 2)
        self.projector = nn.Sequential(nn.Linear(width * 8, width * 8), nn.GELU(), nn.Linear(width * 8, projection_dim))

    def features(self, x: torch.Tensor) -> list[torch.Tensor]:
        a = self.stem(x)
        b = self.stage1(a)
        c = self.stage2(b)
        d = self.stage3(c)
        return [b, c, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)[-1].mean((-2, -1))
        return F.normalize(self.projector(z), dim=1)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Numerically stable SimCLR loss for paired batches."""
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-torch.inf)
    targets = torch.cat([torch.arange(n, 2 * n, device=z.device), torch.arange(n, device=z.device)])
    return F.cross_entropy(logits, targets)


@torch.no_grad()
def feature_change(
    model: SatelliteEncoder,
    before: torch.Tensor,
    after: torch.Tensor,
    scale_weights: Sequence[float] | None = None,
    fusion: str = "mean",
) -> torch.Tensor:
    """Cosine distance between before and after features, fused over the three scales.

    ``scale_weights`` weights the fine, middle and coarse stage (default: equal).
    ``fusion`` is "mean" (weighted arithmetic mean) or "geomean" (weighted geometric mean).
    """
    target_size = before.shape[-2:]
    scores = []
    for f1, f2 in zip(model.features(before), model.features(after)):
        f1, f2 = F.normalize(f1, dim=1), F.normalize(f2, dim=1)
        distance = 1 - (f1 * f2).sum(1, keepdim=True)
        scores.append(F.interpolate(distance, target_size, mode="bilinear", align_corners=False))
    stack = torch.stack(scores)
    weights = torch.ones(len(scores)) if scale_weights is None else torch.tensor(list(scale_weights), dtype=torch.float32)
    if weights.numel() != len(scores) or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError(f"scale_weights needs {len(scores)} non-negative values that are not all zero")
    weights = (weights / weights.sum()).to(stack.device).view(-1, 1, 1, 1, 1)
    if fusion == "mean":
        fused = (stack * weights).sum(0)
    elif fusion == "geomean":
        fused = torch.exp((torch.log(stack.clamp_min(1e-6)) * weights).sum(0))
    else:
        raise ValueError(f"fusion must be 'mean' or 'geomean', not {fusion!r}")
    return fused.squeeze(1)
