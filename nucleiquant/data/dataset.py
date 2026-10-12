"""PyTorch dataset for processed nuclei data.

Each sample is a random ``patch_size`` crop of a training image with its two
target channels:

    targets[0] -> nucleus foreground (1 inside any nucleus)
    targets[1] -> nucleus boundary   (1 on the thin ring around each nucleus)

Predicting the boundary explicitly is the cheapest trick that lets a plain
U-Net separate touching nuclei: at inference we subtract the predicted boundary
from the predicted foreground to get one seed per nucleus, then grow them back
with watershed (see ``nucleiquant/postprocess.py``).

Augmentations are implemented with NumPy/OpenCV to keep the dependency list
short. They are applied identically to the image and the instance map so the
targets stay aligned.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.segmentation import find_boundaries
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_targets(instances: np.ndarray, boundary_width: int = 2) -> np.ndarray:
    """Instance label map -> (2, H, W) float32 targets [foreground, boundary]."""
    fg = (instances > 0).astype(np.float32)
    boundary = find_boundaries(instances, mode="inner").astype(np.uint8)
    if boundary_width > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boundary_width, boundary_width))
        boundary = cv2.dilate(boundary, k)
    return np.stack([fg, boundary.astype(np.float32)], axis=0)


def normalize(image: np.ndarray) -> np.ndarray:
    """uint8 RGB (H, W, 3) -> float32 (3, H, W) normalised with ImageNet stats."""
    x = image.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


class NucleiDataset(Dataset):
    """Random-crop training dataset or full-image evaluation dataset."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        patch_size: int | None = 256,
        patches_per_image: int = 8,
        boundary_width: int = 2,
        augment: bool = True,
    ):
        self.root = Path(processed_dir) / split
        self.stems = sorted(p.stem for p in (self.root / "images").glob("*.png"))
        if not self.stems:
            raise FileNotFoundError(
                f"No images in {self.root / 'images'}. Run nucleiquant.data.prepare_monuseg first."
            )
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image if patch_size else 1
        self.boundary_width = boundary_width
        self.augment = augment
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.stems) * self.patches_per_image

    def _load(self, stem: str) -> tuple[np.ndarray, np.ndarray]:
        if stem not in self._cache:
            img = cv2.cvtColor(cv2.imread(str(self.root / "images" / f"{stem}.png")), cv2.COLOR_BGR2RGB)
            inst = cv2.imread(str(self.root / "instances" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            self._cache[stem] = (img, inst.astype(np.int32))
        return self._cache[stem]

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx // self.patches_per_image]
        img, inst = self._load(stem)

        if self.patch_size:
            img, inst = self._random_crop(img, inst, self.patch_size)
        if self.augment:
            img, inst = self._augment(img, inst)

        targets = make_targets(inst, self.boundary_width)
        return {
            "image": torch.from_numpy(normalize(img)),
            "target": torch.from_numpy(targets),
            "instances": torch.from_numpy(inst.astype(np.int32)),
            "stem": stem,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _random_crop(img, inst, size):
        h, w = inst.shape
        if h < size or w < size:  # pad small images
            ph, pw = max(0, size - h), max(0, size - w)
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
            inst = cv2.copyMakeBorder(inst, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
            h, w = inst.shape
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
        return img[y:y + size, x:x + size], inst[y:y + size, x:x + size]

    @staticmethod
    def _augment(img, inst):
        # Geometric: flips + 90° rotations (dihedral group) — nuclei have no canonical orientation.
        if random.random() < 0.5:
            img, inst = img[:, ::-1], inst[:, ::-1]
        if random.random() < 0.5:
            img, inst = img[::-1], inst[::-1]
        k = random.randint(0, 3)
        if k:
            img, inst = np.rot90(img, k), np.rot90(inst, k)
        img, inst = np.ascontiguousarray(img), np.ascontiguousarray(inst)

        # Photometric: mimic stain variation across labs/scanners.
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-8, 8)) % 180        # hue shift
        hsv[..., 1] *= random.uniform(0.8, 1.2)                           # saturation
        hsv[..., 2] *= random.uniform(0.85, 1.15)                         # brightness
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

        if random.random() < 0.3:  # mild blur ≈ focus variation
            img = cv2.GaussianBlur(img, (3, 3), sigmaX=random.uniform(0.3, 0.8))
        return img, inst
