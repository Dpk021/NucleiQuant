#!/usr/bin/env python
"""Segment and quantify one image or a folder of images from the command line.

    python scripts/predict.py --checkpoint outputs/checkpoints/unet_monuseg_baseline/best.pth \
        --input path/to/image.png --out outputs/predictions

For every image writes:
    <stem>_overlay.png     annotated image
    <stem>_labels.png      uint16 instance mask (open with skimage/ImageJ)
    <stem>_nuclei.csv      per-nucleus measurements
    <stem>_summary.json    counts, density, mean area, inference time
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nucleiquant.inference import load_checkpoint, segment_image  # noqa: E402
from nucleiquant.quantify import measure_nuclei, summarize  # noqa: E402
from nucleiquant.utils import get_device, overlay_instances  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is not None:
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:  # exotic TIFF variants
        import tifffile
        img = tifffile.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    return np.ascontiguousarray(img[..., :3]).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--out", default="outputs/predictions")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--mpp", type=float, default=None, help="microns per pixel (overrides config)")
    args = ap.parse_args()

    device = get_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    mpp = args.mpp if args.mpp is not None else cfg.get("quantify", {}).get("microns_per_pixel")

    src = Path(args.input)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for f in files:
        img = read_rgb(f)
        labels, _, secs = segment_image(model, img, device, cfg, tta=args.tta)
        df = measure_nuclei(labels, mpp, intensity_image=img)
        summary = {**summarize(df, img.shape, mpp), "inference_sec": secs, "source": f.name}

        cv2.imwrite(str(out / f"{f.stem}_overlay.png"), cv2.cvtColor(overlay_instances(img, labels), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{f.stem}_labels.png"), labels.astype(np.uint16))
        df.to_csv(out / f"{f.stem}_nuclei.csv", index=False)
        with open(out / f"{f.stem}_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"{f.name}: {summary['nuclei_count']} nuclei in {secs*1000:.0f} ms")

    print(f"wrote results to {out}")


if __name__ == "__main__":
    main()
