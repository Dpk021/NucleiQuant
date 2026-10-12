#!/usr/bin/env python
"""Evaluate a checkpoint on a processed split and write a report.

    python scripts/evaluate.py --checkpoint outputs/checkpoints/unet_monuseg_baseline/best.pth \
        --split test --out outputs/eval/baseline

Produces:
    <out>/per_image.csv     metrics per image (join with meta.csv for per-organ breakdown)
    <out>/summary.json      aggregated metrics + inference timing + model size
    <out>/overlays/*.png    prediction overlays for qualitative review (optional, --save-overlays)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nucleiquant.inference import load_checkpoint, segment_image  # noqa: E402
from nucleiquant.metrics import aggregate, evaluate_pair  # noqa: E402
from nucleiquant.utils import count_parameters, get_device, overlay_instances  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--processed-dir", default=None, help="defaults to the one in the checkpoint config")
    ap.add_argument("--out", default="outputs/eval/latest")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tta", action="store_true", help="flip test-time augmentation")
    ap.add_argument("--save-overlays", action="store_true")
    args = ap.parse_args()

    device = get_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    root = Path(args.processed_dir or cfg["data"]["processed_dir"]) / args.split
    out = Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in (root / "images").glob("*.png"))
    rows, times = [], []
    for stem in tqdm(stems, desc=f"eval {args.split}"):
        img = cv2.cvtColor(cv2.imread(str(root / "images" / f"{stem}.png")), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(root / "instances" / f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        pred, _, secs = segment_image(model, img, device, cfg, tta=args.tta)
        times.append(secs)
        rows.append({"stem": stem, **evaluate_pair(pred, gt), "inference_sec": secs})
        if args.save_overlays:
            cv2.imwrite(str(out / "overlays" / f"{stem}.png"),
                        cv2.cvtColor(overlay_instances(img, pred), cv2.COLOR_RGB2BGR))

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image.csv", index=False)

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "device": str(device),
        "tta": args.tta,
        "params_millions": count_parameters(model) / 1e6,
        "mean_inference_sec": float(np.mean(times)),
        "p95_inference_sec": float(np.percentile(times, 95)),
        **aggregate(rows),
    }
    if device.type == "cuda":
        summary["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out/'per_image.csv'} and {out/'summary.json'}")


if __name__ == "__main__":
    main()
