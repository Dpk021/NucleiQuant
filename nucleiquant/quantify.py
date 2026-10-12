"""Per-nucleus morphometrics and image-level summary statistics.

Everything here is classical image processing on the instance label map —
no learning involved — which is exactly what a pathology QA pipeline wants:
deterministic, explainable numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

PROPS = [
    "label", "area", "perimeter", "eccentricity", "solidity",
    "equivalent_diameter_area", "major_axis_length", "minor_axis_length",
    "centroid", "bbox", "orientation",
]


def measure_nuclei(labels: np.ndarray, microns_per_pixel: float | None = None,
                   intensity_image: np.ndarray | None = None) -> pd.DataFrame:
    """One row per nucleus with shape descriptors (pixels, plus µm if calibrated)."""
    if labels.max() == 0:
        return pd.DataFrame(columns=["label", "area_px"])

    props = list(PROPS)
    kwargs = {}
    if intensity_image is not None:
        gray = intensity_image.mean(-1) if intensity_image.ndim == 3 else intensity_image
        kwargs["intensity_image"] = gray.astype(np.float32)
        props += ["intensity_mean"]

    df = pd.DataFrame(regionprops_table(labels, properties=props, **kwargs))
    df = df.rename(columns={
        "area": "area_px",
        "perimeter": "perimeter_px",
        "equivalent_diameter_area": "equiv_diameter_px",
        "major_axis_length": "major_axis_px",
        "minor_axis_length": "minor_axis_px",
        "centroid-0": "centroid_y",
        "centroid-1": "centroid_x",
        "bbox-0": "bbox_y0", "bbox-1": "bbox_x0", "bbox-2": "bbox_y1", "bbox-3": "bbox_x1",
        "intensity_mean": "mean_intensity",
    })
    # Circularity: 1.0 for a perfect disc, lower for elongated/irregular nuclei.
    df["circularity"] = (4 * np.pi * df["area_px"] / (df["perimeter_px"] ** 2 + 1e-6)).clip(upper=1.0)
    df["aspect_ratio"] = df["major_axis_px"] / (df["minor_axis_px"] + 1e-6)

    if microns_per_pixel:
        mpp = float(microns_per_pixel)
        df["area_um2"] = df["area_px"] * mpp**2
        df["perimeter_um"] = df["perimeter_px"] * mpp
        df["equiv_diameter_um"] = df["equiv_diameter_px"] * mpp
    return df


def summarize(df: pd.DataFrame, image_shape: tuple[int, int],
              microns_per_pixel: float | None = None) -> dict:
    """Image-level statistics for the report / app header."""
    h, w = image_shape[:2]
    n = len(df)
    out = {
        "nuclei_count": int(n),
        "image_height_px": int(h),
        "image_width_px": int(w),
    }
    if n == 0:
        return out

    out.update({
        "mean_area_px": float(df["area_px"].mean()),
        "median_area_px": float(df["area_px"].median()),
        "std_area_px": float(df["area_px"].std(ddof=0)),
        "mean_circularity": float(df["circularity"].mean()),
        "mean_eccentricity": float(df["eccentricity"].mean()),
        "mean_solidity": float(df["solidity"].mean()),
        "nuclear_area_fraction": float(df["area_px"].sum() / (h * w)),
        "density_per_1000px2": float(n / (h * w) * 1000),
    })
    if microns_per_pixel:
        mpp = float(microns_per_pixel)
        area_mm2 = h * w * mpp**2 / 1e6
        out["mean_area_um2"] = float(df["area_um2"].mean())
        out["density_per_mm2"] = float(n / area_mm2) if area_mm2 > 0 else float("nan")
    return out
