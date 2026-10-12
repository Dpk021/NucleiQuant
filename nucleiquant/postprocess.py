"""Turn probability maps into an instance label map.

Pipeline (marker-controlled watershed):

    1. foreground = P(fg) > fg_threshold
    2. seeds      = foreground AND NOT (P(boundary) > boundary_threshold)
                    -> touching nuclei are now disconnected because their shared
                       border has been carved out
    3. markers    = connected components of seeds, refined with distance-transform
                    peaks so that a single blob with two bulges still splits
    4. labels     = watershed(-distance, markers, mask=foreground)
                    -> grows each seed back out to fill the full nucleus
    5. drop components smaller than min_nucleus_area

If the model only has one output channel (plain binary U-Net), step 2 is skipped
and we rely on distance-transform peaks alone — the classic approach, and the
ablation you can quote in the write-up.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label as cc_label
from skimage.segmentation import watershed


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components with fewer than ``min_size`` pixels (version-agnostic)."""
    if min_size <= 1 or not mask.any():
        return mask
    cc = cc_label(mask)
    sizes = np.bincount(cc.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[cc]


def probs_to_instances(
    probs: np.ndarray,
    fg_threshold: float = 0.5,
    boundary_threshold: float = 0.5,
    min_nucleus_area: int = 20,
    min_distance: int = 5,
) -> np.ndarray:
    """(C, H, W) probabilities -> int32 instance label map (0 = background)."""
    fg = probs[0] > fg_threshold
    fg = remove_small_objects(fg, max(min_nucleus_area, 1))
    fg = ndi.binary_fill_holes(fg)
    if not fg.any():
        return np.zeros(fg.shape, dtype=np.int32)

    distance = ndi.distance_transform_edt(fg)

    if probs.shape[0] >= 2:
        seeds = fg & ~(probs[1] > boundary_threshold)
        seeds = remove_small_objects(seeds, max(min_nucleus_area // 4, 1))
        markers = cc_label(seeds)
    else:
        markers = np.zeros(fg.shape, dtype=np.int32)

    # Add distance peaks as extra markers for blobs that still lack a seed.
    peaks = peak_local_max(distance, min_distance=min_distance, labels=cc_label(fg), exclude_border=False)
    next_id = int(markers.max()) + 1
    for y, x in peaks:
        if markers[y, x] == 0:
            markers[y, x] = next_id
            next_id += 1

    labels = watershed(-distance, markers, mask=fg)
    labels = _drop_small(labels, min_nucleus_area)
    return labels.astype(np.int32)


def _drop_small(labels: np.ndarray, min_area: int) -> np.ndarray:
    if labels.max() == 0:
        return labels
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False
    remap = np.zeros_like(areas)
    remap[keep] = np.arange(1, keep.sum() + 1)
    return remap[labels]
