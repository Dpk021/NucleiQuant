"""Run the model on arbitrary-sized images and load checkpoints."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data.dataset import normalize
from .models import build_model
from .postprocess import probs_to_instances
from .utils import get_device


def load_checkpoint(path: str | Path, device: torch.device | None = None):
    """Return (model, cfg) from a checkpoint saved by scripts/train.py."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    device = device or get_device(cfg.get("device", "auto"))
    return model.to(device).eval(), cfg


@torch.no_grad()
def predict_probs(model: torch.nn.Module, image: np.ndarray, device: torch.device,
                  tile: int | None = None, overlap: int = 64, tta: bool = False) -> np.ndarray:
    """uint8 RGB image (H, W, 3) -> probability maps (C, H, W) in [0, 1].

    Images are reflect-padded to a multiple of the network's downsampling factor.
    Large images are optionally processed in overlapping tiles so memory stays bounded.
    ``tta`` averages the 4 flip variants (slower, usually +0.5–1 Dice point).
    """
    h, w = image.shape[:2]
    div = getattr(model, "divisor", 16)

    if tile is None or (h <= tile and w <= tile):
        return _predict_padded(model, image, device, div, tta)[:, :h, :w]

    # Sliding-window with overlap; average predictions in the overlap region.
    stride = tile - overlap
    out = None
    count = np.zeros((1, h, w), dtype=np.float32)
    for y in range(0, max(h - overlap, 1), stride):
        for x in range(0, max(w - overlap, 1), stride):
            y0, x0 = min(y, max(h - tile, 0)), min(x, max(w - tile, 0))
            crop = image[y0:y0 + tile, x0:x0 + tile]
            p = _predict_padded(model, crop, device, div, tta)[:, :crop.shape[0], :crop.shape[1]]
            if out is None:
                out = np.zeros((p.shape[0], h, w), dtype=np.float32)
            out[:, y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] += p
            count[:, y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] += 1
    return out / count


def _predict_padded(model, image, device, div, tta):
    h, w = image.shape[:2]
    ph, pw = (-h) % div, (-w) % div
    x = torch.from_numpy(normalize(image)).unsqueeze(0).to(device)
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")

    variants = [(False, False)] if not tta else [(False, False), (True, False), (False, True), (True, True)]
    acc = None
    for fh, fw in variants:
        xi = x
        if fh:
            xi = xi.flip(-2)
        if fw:
            xi = xi.flip(-1)
        p = torch.sigmoid(model(xi))
        if fh:
            p = p.flip(-2)
        if fw:
            p = p.flip(-1)
        acc = p if acc is None else acc + p
    return (acc / len(variants))[0].float().cpu().numpy()


def segment_image(model, image: np.ndarray, device: torch.device, cfg: dict,
                  tta: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """Full pipeline: image -> (instance labels, prob maps, seconds elapsed)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    probs = predict_probs(model, image, device, tile=1024, tta=tta)
    labels = probs_to_instances(probs, **cfg.get("postprocess", {}))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return labels, probs, time.perf_counter() - t0
