"""MoNuSeg annotation parsing.

MoNuSeg ships Aperio ImageScope XML files. Each nucleus is a ``<Region>`` whose
``<Vertices>`` list the polygon outline in pixel coordinates:

    <Annotations>
      <Annotation>
        <Regions>
          <Region Id="1">
            <Vertices>
              <Vertex X="12.3" Y="45.6"/>
              ...

We rasterise every polygon into an instance label map (uint16: 0 = background,
1..N = nucleus id). Everything downstream (binary mask, boundary mask, metrics)
is derived from that single map.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


def parse_aperio_xml(xml_path: str | Path) -> list[np.ndarray]:
    """Return a list of (N_i, 2) float arrays of polygon vertices (x, y)."""
    tree = ET.parse(xml_path)
    polygons: list[np.ndarray] = []
    for region in tree.getroot().iter("Region"):
        pts = [
            (float(v.attrib["X"]), float(v.attrib["Y"]))
            for v in region.iter("Vertex")
        ]
        if len(pts) >= 3:
            polygons.append(np.asarray(pts, dtype=np.float32))
    return polygons


def xml_to_instance_mask(
    xml_path: str | Path, height: int, width: int
) -> np.ndarray:
    """Rasterise all polygons in an Aperio XML into a uint16 instance label map.

    Later polygons overwrite earlier ones where they overlap, which matches the
    convention used by most MoNuSeg baselines.
    """
    labels = np.zeros((height, width), dtype=np.uint16)
    for idx, poly in enumerate(parse_aperio_xml(xml_path), start=1):
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(labels, [pts], color=int(idx))
    return labels


def find_pairs(image_dir: Path, xml_dir: Path) -> list[tuple[Path, Path]]:
    """Match image files with their XML annotation by stem."""
    xmls = {p.stem: p for p in xml_dir.glob("*.xml")}
    pairs = []
    for img in sorted(image_dir.iterdir()):
        if img.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"} and img.stem in xmls:
            pairs.append((img, xmls[img.stem]))
    return pairs
