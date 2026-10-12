"""Shared helpers: config loading, device selection, seeding, visualisation."""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply ``key.sub=value`` overrides."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        _set_nested(cfg, key.split("."), yaml.safe_load(raw))
    return cfg


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False)


# --------------------------------------------------------------------------- #
# Device / reproducibility
# --------------------------------------------------------------------------- #
def get_device(name: str = "auto") -> torch.device:
    """Pick CUDA > MPS > CPU unless a device is forced."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def label_to_color(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Map an instance label map to an RGB image with a random colour per nucleus."""
    rng = np.random.default_rng(seed)
    n = int(labels.max()) + 1
    palette = rng.integers(60, 255, size=(n, 3), dtype=np.uint8)
    palette[0] = 0
    return palette[labels]


def overlay_instances(
    image: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.35,
    draw_contours: bool = True,
    contour_color: tuple[int, int, int] = (255, 255, 0),
) -> np.ndarray:
    """Blend a coloured instance map onto the RGB image and outline each nucleus."""
    from skimage.segmentation import find_boundaries

    image = _to_uint8_rgb(image)
    color = label_to_color(labels)
    mask = labels > 0
    out = image.copy()
    out[mask] = (alpha * color[mask] + (1 - alpha) * image[mask]).astype(np.uint8)
    if draw_contours:
        edges = find_boundaries(labels, mode="outer")
        out[edges] = contour_color
    return out


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image[..., :3]
