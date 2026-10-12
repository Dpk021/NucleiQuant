"""End-to-end smoke test on synthetic data (no dataset download needed).

    pytest -q tests/

Also doubles as a tutorial: generate blobs -> targets -> U-Net forward/backward
-> post-process -> metrics -> quantify. Runs in < 30 s on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nucleiquant.data.dataset import NucleiDataset, make_targets, normalize  # noqa: E402
from nucleiquant.data.monuseg import xml_to_instance_mask  # noqa: E402
from nucleiquant.inference import predict_probs  # noqa: E402
from nucleiquant.losses import BCEDiceLoss  # noqa: E402
from nucleiquant.metrics import aggregate, aggregated_jaccard_index, evaluate_pair, instance_prf  # noqa: E402
from nucleiquant.models import UNet  # noqa: E402
from nucleiquant.postprocess import probs_to_instances  # noqa: E402
from nucleiquant.quantify import measure_nuclei, summarize  # noqa: E402
from nucleiquant.utils import overlay_instances  # noqa: E402


# --------------------------------------------------------------------------- #
def synthetic_tissue(h=256, w=256, n=40, seed=0):
    """Random ellipses on a pinkish background, returning (rgb uint8, instance labels)."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((h, w), np.int32)
    img = np.full((h, w, 3), (235, 200, 215), np.uint8)
    for i in range(1, n + 1):
        cy, cx = rng.integers(10, h - 10), rng.integers(10, w - 10)
        ax, ay = rng.integers(5, 12), rng.integers(5, 12)
        ang = rng.uniform(0, 180)
        cv2.ellipse(labels, (int(cx), int(cy)), (int(ax), int(ay)), ang, 0, 360, int(i), -1)
    for i in range(1, n + 1):
        m = labels == i
        img[m] = (rng.integers(60, 110), rng.integers(30, 70), rng.integers(110, 160))
    noise = rng.normal(0, 6, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, labels


@pytest.fixture(scope="module")
def processed_dir(tmp_path_factory):
    """Create a tiny processed dataset in the layout prepare_monuseg.py produces."""
    root = tmp_path_factory.mktemp("processed")
    for split, n_imgs in [("train", 3), ("test", 2)]:
        (root / split / "images").mkdir(parents=True)
        (root / split / "instances").mkdir(parents=True)
        for k in range(n_imgs):
            img, inst = synthetic_tissue(seed=hash((split, k)) % 1000)
            cv2.imwrite(str(root / split / "images" / f"{split}_{k}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(root / split / "instances" / f"{split}_{k}.png"), inst.astype(np.uint16))
    return root


# --------------------------------------------------------------------------- #
def test_targets_shape_and_values():
    _, inst = synthetic_tissue()
    t = make_targets(inst, boundary_width=2)
    assert t.shape == (2, 256, 256)
    assert set(np.unique(t)) <= {0.0, 1.0}
    assert t[0].sum() > t[1].sum() > 0  # boundary is a thin subset of foreground region


def test_dataset_and_augment(processed_dir):
    ds = NucleiDataset(processed_dir, "train", patch_size=128, patches_per_image=4)
    assert len(ds) == 12
    s = ds[0]
    assert s["image"].shape == (3, 128, 128) and s["image"].dtype == torch.float32
    assert s["target"].shape == (2, 128, 128)
    full = NucleiDataset(processed_dir, "test", patch_size=None, augment=False)
    assert full[0]["image"].shape == (3, 256, 256)


def test_unet_forward_backward():
    model = UNet(3, 2, base_channels=8, depth=3)
    x = torch.randn(2, 3, 96, 96)
    y = (torch.rand(2, 2, 96, 96) > 0.5).float()
    out = model(x)
    assert out.shape == (2, 2, 96, 96)
    loss = BCEDiceLoss(channel_weights=[1.0, 2.0])(out, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.divisor == 8


def test_predict_probs_handles_odd_sizes():
    model = UNet(3, 2, base_channels=8, depth=3).eval()
    img = np.random.randint(0, 255, (101, 77, 3), np.uint8)
    p = predict_probs(model, img, torch.device("cpu"))
    assert p.shape == (2, 101, 77) and 0 <= p.min() and p.max() <= 1
    p_tiled = predict_probs(model, img, torch.device("cpu"), tile=64, overlap=16)
    assert p_tiled.shape == (2, 101, 77)


def test_postprocess_recovers_instances_from_perfect_probs():
    _, inst = synthetic_tissue(n=25, seed=3)
    probs = make_targets(inst, boundary_width=2)  # oracle prediction
    pred = probs_to_instances(probs, min_nucleus_area=10)
    m = evaluate_pair(pred, inst)
    assert m["dice"] > 0.95
    assert m["f1"] > 0.85, m
    assert m["count_abs_error"] <= 3, m


def test_metrics_edge_cases():
    a = np.zeros((20, 20), np.int32)
    assert instance_prf(a, a)["f1"] == 1.0
    assert aggregated_jaccard_index(a, a) == 1.0
    b = a.copy(); b[2:8, 2:8] = 1
    assert instance_prf(a, b)["recall"] == 0.0
    assert aggregated_jaccard_index(b, b) == 1.0
    agg = aggregate([evaluate_pair(b, b), evaluate_pair(a, a)])
    assert agg["count_mae"] == 0.0 and agg["n_images"] == 2


def test_quantify_and_overlay():
    img, inst = synthetic_tissue(n=15, seed=7)
    df = measure_nuclei(inst, microns_per_pixel=0.25, intensity_image=img)
    assert len(df) == 15
    assert {"area_px", "area_um2", "circularity", "centroid_x", "mean_intensity"} <= set(df.columns)
    assert (df["circularity"] <= 1.0).all()
    s = summarize(df, img.shape, 0.25)
    assert s["nuclei_count"] == 15 and s["density_per_mm2"] > 0
    ov = overlay_instances(img, inst)
    assert ov.shape == img.shape and ov.dtype == np.uint8


def test_xml_parsing(tmp_path):
    xml = """<Annotations><Annotation><Regions>
      <Region Id="1"><Vertices><Vertex X="2" Y="2"/><Vertex X="10" Y="2"/><Vertex X="10" Y="10"/><Vertex X="2" Y="10"/></Vertices></Region>
      <Region Id="2"><Vertices><Vertex X="15.4" Y="15.2"/><Vertex X="19" Y="15"/><Vertex X="19" Y="19"/></Vertices></Region>
    </Regions></Annotation></Annotations>"""
    p = tmp_path / "a.xml"
    p.write_text(xml)
    m = xml_to_instance_mask(p, 24, 24)
    assert m.dtype == np.uint16 and m.max() == 2
    assert (m == 1).sum() == 81  # 9x9 inclusive square


def test_tiny_training_loop_improves(processed_dir):
    """Three epochs on synthetic blobs should visibly reduce loss."""
    torch.manual_seed(0)
    ds = NucleiDataset(processed_dir, "train", patch_size=96, patches_per_image=8)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    model = UNet(3, 2, base_channels=8, depth=3)
    crit = BCEDiceLoss(channel_weights=[1.0, 2.0])
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    losses = []
    for _ in range(3):
        tot = 0.0
        for b in dl:
            opt.zero_grad()
            loss = crit(model(b["image"]), b["target"])
            loss.backward()
            opt.step()
            tot += loss.item()
        losses.append(tot / len(dl))
    assert losses[-1] < losses[0], losses
