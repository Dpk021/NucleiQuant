#!/usr/bin/env python
"""Generate a small synthetic "tissue" dataset in the processed layout.

Lets you run train.py / evaluate.py / predict.py / the Streamlit app end-to-end
before downloading MoNuSeg, to verify the environment (MPS, CUDA, deps).

    python scripts/make_synthetic_data.py --out data/processed/synthetic --n-train 12 --n-test 4
    python scripts/train.py --config configs/unet_monuseg.yaml \
        --set data.processed_dir=data/processed/synthetic --set train.epochs=3 --set experiment=synthetic_smoke
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def synthetic_tissue(h: int, w: int, n: int, rng: np.random.Generator):
    labels = np.zeros((h, w), np.int32)
    img = np.full((h, w, 3), (236, 203, 218), np.uint8)
    # faint cytoplasm texture
    img = np.clip(img.astype(np.float32) + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
    for i in range(1, n + 1):
        cy, cx = rng.integers(8, h - 8), rng.integers(8, w - 8)
        ax, ay = rng.integers(5, 13), rng.integers(5, 13)
        cv2.ellipse(labels, (int(cx), int(cy)), (int(ax), int(ay)), float(rng.uniform(0, 180)), 0, 360, int(i), -1)
    for i in range(1, n + 1):
        m = labels == i
        img[m] = (rng.integers(55, 110), rng.integers(25, 70), rng.integers(105, 165))
    img = np.clip(img.astype(np.float32) + rng.normal(0, 7, img.shape), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    return img, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/processed/synthetic"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=4)
    ap.add_argument("--nuclei", type=int, default=160, help="approx nuclei per image")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for split, n in [("train", args.n_train), ("test", args.n_test)]:
        (args.out / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out / split / "instances").mkdir(parents=True, exist_ok=True)
        for k in range(n):
            n_nuc = int(rng.integers(int(args.nuclei * 0.7), int(args.nuclei * 1.3)))
            img, inst = synthetic_tissue(args.size, args.size, n_nuc, rng)
            stem = f"synthetic_{split}_{k:02d}"
            cv2.imwrite(str(args.out / split / "images" / f"{stem}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(args.out / split / "instances" / f"{stem}.png"), inst.astype(np.uint16))
            rows.append(dict(stem=stem, split=split, height=args.size, width=args.size,
                             n_nuclei=int(len(np.unique(inst)) - 1)))

    with open(args.out / "meta.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} synthetic images to {args.out}")


if __name__ == "__main__":
    main()
