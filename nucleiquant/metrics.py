"""Evaluation metrics for nuclei segmentation.

Pixel-level (how good are the boundaries?):
    dice, iou

Instance-level (how many nuclei did we find correctly?):
    precision / recall / F1 at IoU >= 0.5  (a predicted nucleus is a true
    positive if it overlaps a distinct GT nucleus with IoU >= 0.5)
    AJI — Aggregated Jaccard Index (Kumar et al. 2017), the MoNuSeg challenge
    metric. Penalises both missed nuclei and over-segmentation.

Counting:
    count_mae — |#pred - #gt|, the number a pathologist cares about first.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def dice_iou(pred_fg: np.ndarray, gt_fg: np.ndarray) -> tuple[float, float]:
    pred_fg, gt_fg = pred_fg.astype(bool), gt_fg.astype(bool)
    inter = np.logical_and(pred_fg, gt_fg).sum()
    ps, gs = pred_fg.sum(), gt_fg.sum()
    union = ps + gs - inter
    dice = 2 * inter / (ps + gs) if (ps + gs) else 1.0
    iou = inter / union if union else 1.0
    return float(dice), float(iou)


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Make labels contiguous 1..N."""
    uniq = np.unique(labels)
    uniq = uniq[uniq != 0]
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    remap[uniq] = np.arange(1, len(uniq) + 1)
    return remap[labels]


def _overlap_matrix(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (intersection[G, P], gt_areas[G], pred_areas[P]); indices exclude background."""
    pred, gt = _relabel(pred), _relabel(gt)
    n_g, n_p = int(gt.max()), int(pred.max())
    inter = np.zeros((n_g + 1, n_p + 1), dtype=np.int64)
    np.add.at(inter, (gt.ravel(), pred.ravel()), 1)
    gt_areas = inter.sum(1)[1:]
    pred_areas = inter.sum(0)[1:]
    return inter[1:, 1:], gt_areas, pred_areas


def instance_prf(pred: np.ndarray, gt: np.ndarray, iou_thresh: float = 0.5) -> dict:
    inter, ga, pa = _overlap_matrix(pred, gt)
    n_g, n_p = len(ga), len(pa)
    if n_g == 0 and n_p == 0:
        return dict(tp=0, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0)
    if n_g == 0 or n_p == 0:
        return dict(tp=0, fp=n_p, fn=n_g, precision=0.0, recall=0.0, f1=0.0)

    union = ga[:, None] + pa[None, :] - inter
    iou = inter / np.maximum(union, 1)
    # One-to-one matching maximising total IoU, then threshold.
    rows, cols = linear_sum_assignment(-iou)
    tp = int((iou[rows, cols] >= iou_thresh).sum())
    fp, fn = n_p - tp, n_g - tp
    p = tp / n_p
    r = tp / n_g
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=float(p), recall=float(r), f1=float(f1))


def aggregated_jaccard_index(pred: np.ndarray, gt: np.ndarray) -> float:
    """AJI as defined in Kumar et al., IEEE TMI 2017."""
    inter, ga, pa = _overlap_matrix(pred, gt)
    n_g, n_p = len(ga), len(pa)
    if n_g == 0:
        return 1.0 if n_p == 0 else 0.0
    if n_p == 0:
        return 0.0

    union = ga[:, None] + pa[None, :] - inter
    iou = inter / np.maximum(union, 1)
    used = np.zeros(n_p, dtype=bool)
    c = u = 0
    for g in range(n_g):
        j = int(iou[g].argmax())
        if iou[g, j] > 0:
            c += inter[g, j]
            u += union[g, j]
            used[j] = True
        else:
            u += ga[g]
    u += pa[~used].sum()
    return float(c / u) if u else 0.0


def evaluate_pair(pred: np.ndarray, gt: np.ndarray) -> dict:
    """All metrics for one image."""
    dice, iou = dice_iou(pred > 0, gt > 0)
    prf = instance_prf(pred, gt)
    n_pred, n_gt = int(len(np.unique(pred)) - (1 if 0 in pred else 0)), int(len(np.unique(gt)) - (1 if 0 in gt else 0))
    return {
        "dice": dice,
        "iou": iou,
        "aji": aggregated_jaccard_index(pred, gt),
        **prf,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "count_error": n_pred - n_gt,
        "count_abs_error": abs(n_pred - n_gt),
    }


def aggregate(per_image: list[dict]) -> dict:
    """Mean over images for ratio metrics; MAE for counts."""
    if not per_image:
        return {}
    keys = ["dice", "iou", "aji", "precision", "recall", "f1"]
    out = {k: float(np.mean([m[k] for m in per_image])) for k in keys}
    out["count_mae"] = float(np.mean([m["count_abs_error"] for m in per_image]))
    out["count_bias"] = float(np.mean([m["count_error"] for m in per_image]))
    out["n_images"] = len(per_image)
    return out
