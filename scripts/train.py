#!/usr/bin/env python
"""Train the NucleiQuant U-Net.

    python scripts/train.py --config configs/unet_monuseg.yaml
    python scripts/train.py --config configs/unet_monuseg.yaml --set train.epochs=5 --set model.base_channels=16

Validation runs on full-size images every epoch and reports Dice/IoU/AJI/F1/
count MAE after the real post-processing, so the number you optimise is the
number the app will show.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nucleiquant.data import NucleiDataset  # noqa: E402
from nucleiquant.inference import predict_probs  # noqa: E402
from nucleiquant.losses import build_loss  # noqa: E402
from nucleiquant.metrics import aggregate, evaluate_pair  # noqa: E402
from nucleiquant.models import build_model  # noqa: E402
from nucleiquant.postprocess import probs_to_instances  # noqa: E402
from nucleiquant.utils import count_parameters, get_device, load_config, save_config, seed_everything  # noqa: E402


def _denorm_to_uint8(x: torch.Tensor):
    """Undo ImageNet normalisation -> uint8 RGB numpy for predict_probs."""
    import numpy as np
    from nucleiquant.data.dataset import IMAGENET_MEAN, IMAGENET_STD
    img = x.permute(1, 2, 0).cpu().numpy() * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


@torch.no_grad()
def validate(model, loader, device, cfg) -> dict:
    model.eval()
    per_image = []
    for batch in loader:
        for i in range(batch["image"].shape[0]):
            img = _denorm_to_uint8(batch["image"][i])
            probs = predict_probs(model, img, device, tile=1024)
            pred = probs_to_instances(probs, **cfg["postprocess"])
            per_image.append(evaluate_pair(pred, batch["instances"][i].numpy()))
    return aggregate(per_image)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. train.epochs=5")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    print(f"device: {device}")

    d, t = cfg["data"], cfg["train"]
    train_ds = NucleiDataset(d["processed_dir"], d["train_split"], d["patch_size"],
                             d["patches_per_image"], d["boundary_width"], augment=True)
    val_ds = NucleiDataset(d["processed_dir"], d["val_split"], patch_size=None,
                           boundary_width=d["boundary_width"], augment=False)
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                          num_workers=d["num_workers"], drop_last=True,
                          pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"train: {len(train_ds.stems)} images -> {len(train_ds)} patches/epoch | val: {len(val_ds)} images")

    model = build_model(cfg).to(device)
    print(f"model: {cfg['model']['name']} ({count_parameters(model)/1e6:.2f}M params)")
    criterion = build_loss(cfg, cfg["model"]["out_channels"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t["lr"], epochs=t["epochs"],
                                                steps_per_epoch=len(train_dl), pct_start=0.1)
    use_amp = bool(t.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(t["checkpoint_dir"]) / cfg["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    log_path = out_dir / "history.jsonl"

    start_epoch, best_score, bad_epochs = 0, -1.0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["opt_state"])
        start_epoch, best_score = ck["epoch"] + 1, ck.get("best_score", -1.0)
        print(f"resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        t0, running = time.time(), 0.0
        pbar = tqdm(train_dl, desc=f"epoch {epoch+1}/{t['epochs']}", leave=False)
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = running / len(train_dl)

        val = validate(model, val_dl, device, cfg)
        # Model selection on AJI (the MoNuSeg challenge metric); fall back to Dice if degenerate.
        score = val["aji"] if val["aji"] > 0 else val["dice"]
        rec = {"epoch": epoch + 1, "train_loss": train_loss, "lr": sched.get_last_lr()[0],
               "epoch_sec": time.time() - t0, **{f"val_{k}": v for k, v in val.items()}}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{epoch+1:3d}] loss {train_loss:.4f} | dice {val['dice']:.4f} iou {val['iou']:.4f} "
              f"aji {val['aji']:.4f} f1 {val['f1']:.4f} countMAE {val['count_mae']:.2f} | {rec['epoch_sec']:.0f}s")

        state = {"epoch": epoch, "model_state": model.state_dict(), "opt_state": opt.state_dict(),
                 "config": cfg, "val": val, "best_score": max(best_score, score)}
        torch.save(state, out_dir / "last.pth")
        if score > best_score:
            best_score, bad_epochs = score, 0
            torch.save(state, out_dir / "best.pth")
            print(f"      ↑ new best ({score:.4f}) saved to {out_dir/'best.pth'}")
        else:
            bad_epochs += 1
            if bad_epochs >= t.get("early_stopping_patience", 10**9):
                print(f"early stopping after {bad_epochs} epochs without improvement")
                break

    print(f"done. best score {best_score:.4f}. checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
