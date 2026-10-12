"""Convert raw MoNuSeg downloads into a uniform processed layout.

Expected raw layout (what you get after unzipping the Kaggle/official archives):

    data/raw/monuseg/
      MoNuSeg 2018 Training Data/
        Tissue Images/*.tif
        Annotations/*.xml
      MoNuSegTestData/
        *.tif
        *.xml

The folder names vary slightly between mirrors, so this script just looks for
any directory containing both images and XMLs and asks you to map them to a
split name. Output:

    data/processed/monuseg/
      train/images/<stem>.png
      train/instances/<stem>.png      (uint16 label map)
      test/images/<stem>.png
      test/instances/<stem>.png
      meta.csv                         (stem, split, height, width, n_nuclei)

Usage:
    python -m nucleiquant.data.prepare_monuseg \
        --train "data/raw/monuseg/MoNuSeg 2018 Training Data" \
        --test  "data/raw/monuseg/MoNuSegTestData" \
        --out   data/processed/monuseg
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .monuseg import find_pairs, xml_to_instance_mask


def _locate(root: Path) -> tuple[Path, Path]:
    """Find the image and XML directories inside a MoNuSeg split folder."""
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    img_dir = xml_dir = None
    for d in candidates:
        has_img = any(p.suffix.lower() in {".tif", ".tiff", ".png"} for p in d.iterdir())
        has_xml = any(p.suffix.lower() == ".xml" for p in d.iterdir())
        if has_img and img_dir is None:
            img_dir = d
        if has_xml and xml_dir is None:
            xml_dir = d
    if img_dir is None or xml_dir is None:
        raise FileNotFoundError(f"Could not find images + XMLs under {root}")
    return img_dir, xml_dir


def read_image(path: Path) -> np.ndarray:
    """Read RGB uint8. OpenCV handles LZW-compressed MoNuSeg TIFFs natively;
    tifffile (+imagecodecs) is only a fallback for exotic TIFF variants."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is not None:
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        import tifffile
        img = tifffile.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    return np.ascontiguousarray(img[..., :3]).astype(np.uint8)


def process_split(root: Path, split: str, out: Path, rows: list[dict]) -> None:
    img_dir, xml_dir = _locate(root)
    pairs = find_pairs(img_dir, xml_dir)
    if not pairs:
        raise RuntimeError(f"No image/xml pairs found in {root}")
    (out / split / "images").mkdir(parents=True, exist_ok=True)
    (out / split / "instances").mkdir(parents=True, exist_ok=True)

    for img_path, xml_path in tqdm(pairs, desc=f"{split:5s}"):
        img = read_image(img_path)
        h, w = img.shape[:2]
        inst = xml_to_instance_mask(xml_path, h, w)
        cv2.imwrite(str(out / split / "images" / f"{img_path.stem}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / split / "instances" / f"{img_path.stem}.png"), inst)
        rows.append(dict(stem=img_path.stem, split=split, height=h, width=w,
                         n_nuclei=int(len(np.unique(inst)) - 1)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/processed/monuseg"))
    args = ap.parse_args()

    rows: list[dict] = []
    process_split(args.train, "train", args.out, rows)
    process_split(args.test, "test", args.out, rows)

    with open(args.out / "meta.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    n_train = sum(r["split"] == "train" for r in rows)
    n_test = len(rows) - n_train
    total = sum(r["n_nuclei"] for r in rows)
    print(f"Done: {n_train} train, {n_test} test images, {total:,} nuclei -> {args.out}")


if __name__ == "__main__":
    main()
