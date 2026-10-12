"""NucleiQuant — Streamlit app.

    streamlit run app/app.py

Upload an H&E image, get per-nucleus segmentation, counts, morphometrics and a
CSV. Optionally upload the MoNuSeg-style XML ground truth to see live metrics.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nucleiquant.data.monuseg import xml_to_instance_mask  # noqa: E402
from nucleiquant.inference import load_checkpoint, segment_image  # noqa: E402
from nucleiquant.metrics import evaluate_pair  # noqa: E402
from nucleiquant.quantify import measure_nuclei, summarize  # noqa: E402
from nucleiquant.utils import count_parameters, get_device, overlay_instances  # noqa: E402

st.set_page_config(page_title="NucleiQuant", page_icon="🔬", layout="wide")


# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def get_model(ckpt_path: str, device_name: str):
    device = get_device(device_name)
    model, cfg = load_checkpoint(ckpt_path, device)
    return model, cfg, device


def find_checkpoints() -> list[Path]:
    return sorted((ROOT / "outputs" / "checkpoints").glob("**/*.pth"))


def to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
st.title("🔬 NucleiQuant")
st.caption("Nuclei segmentation, counting and morphometry for H&E histology · research use only, not a diagnostic device")

with st.sidebar:
    st.header("Model")
    ckpts = find_checkpoints()
    if not ckpts:
        st.error("No checkpoint found under `outputs/checkpoints/`. Train one with `scripts/train.py` first.")
        st.stop()
    ckpt = st.selectbox("Checkpoint", ckpts, format_func=lambda p: str(p.relative_to(ROOT)),
                        index=next((i for i, p in enumerate(ckpts) if p.name == "best.pth"), 0))
    device_name = st.selectbox("Device", ["auto", "cpu", "mps", "cuda"], index=0)
    tta = st.toggle("Test-time augmentation (flips)", value=False, help="≈4× slower, slightly better masks")

    model, cfg, device = get_model(str(ckpt), device_name)
    st.caption(f"{count_parameters(model)/1e6:.1f}M params · running on **{device.type}**")

    st.header("Post-processing")
    pp = dict(cfg.get("postprocess", {}))
    pp["fg_threshold"] = st.slider("Foreground threshold", 0.1, 0.9, float(pp.get("fg_threshold", 0.5)), 0.05)
    pp["boundary_threshold"] = st.slider("Boundary threshold", 0.1, 0.9, float(pp.get("boundary_threshold", 0.5)), 0.05)
    pp["min_nucleus_area"] = st.slider("Min nucleus area (px)", 0, 200, int(pp.get("min_nucleus_area", 20)), 5)
    cfg = {**cfg, "postprocess": pp}

    st.header("Calibration")
    mpp_default = cfg.get("quantify", {}).get("microns_per_pixel") or 0.0
    mpp = st.number_input("Microns per pixel (0 = unknown)", 0.0, 10.0, float(mpp_default), 0.01, format="%.3f")
    mpp = mpp or None

uploaded = st.file_uploader("Upload an H&E tissue image", type=["png", "jpg", "jpeg", "tif", "tiff"])
gt_xml = st.file_uploader("Optional: MoNuSeg XML ground truth for this image", type=["xml"])

if uploaded is None:
    st.info("Upload an image to begin. MoNuSeg test images (1000×1000) take ~1–3 s on an M-series Mac.")
    st.stop()

# --- Read image ----------------------------------------------------------- #
raw = np.frombuffer(uploaded.read(), np.uint8)
bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
if bgr is not None:
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
else:  # exotic TIFF variants OpenCV can't decode
    import tifffile
    image = tifffile.imread(io.BytesIO(raw.tobytes()))
if image.ndim == 2:
    image = np.stack([image] * 3, -1)
image = np.ascontiguousarray(image[..., :3]).astype(np.uint8)

# --- Run ------------------------------------------------------------------- #
with st.spinner("Segmenting…"):
    labels, probs, secs = segment_image(model, image, device, cfg, tta=tta)
    df = measure_nuclei(labels, mpp, intensity_image=image)
    summary = summarize(df, image.shape, mpp)
    overlay = overlay_instances(image, labels)

# --- Header metrics -------------------------------------------------------- #
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Nuclei detected", f"{summary['nuclei_count']:,}")
if summary["nuclei_count"]:
    area_label = f"{summary['mean_area_um2']:.1f} µm²" if mpp else f"{summary['mean_area_px']:.0f} px"
    c2.metric("Mean nucleus area", area_label)
    c3.metric("Nuclear area fraction", f"{summary['nuclear_area_fraction']*100:.1f} %")
    dens = f"{summary['density_per_mm2']:,.0f} / mm²" if mpp else f"{summary['density_per_1000px2']:.2f} / 1000 px²"
    c4.metric("Density", dens)
c5.metric("Inference time", f"{secs*1000:.0f} ms", help=f"model + post-processing on {device.type}")

# --- Images ---------------------------------------------------------------- #
tab_img, tab_prob, tab_stats, tab_table = st.tabs(["Segmentation", "Probability maps", "Statistics", "Per-nucleus table"])

with tab_img:
    a, b = st.columns(2)
    a.image(image, caption="Original", use_container_width=True)
    b.image(overlay, caption=f"{summary['nuclei_count']} nuclei (random colour per instance, yellow = outline)", use_container_width=True)

with tab_prob:
    a, b = st.columns(2)
    a.image((probs[0] * 255).astype(np.uint8), caption="P(nucleus)", use_container_width=True, clamp=True)
    if probs.shape[0] > 1:
        b.image((probs[1] * 255).astype(np.uint8), caption="P(boundary) — used to split touching nuclei", use_container_width=True, clamp=True)

with tab_stats:
    if df.empty:
        st.warning("No nuclei detected — try lowering the foreground threshold.")
    else:
        a, b = st.columns(2)
        area_col = "area_um2" if mpp else "area_px"
        a.plotly_chart(px.histogram(df, x=area_col, nbins=40, title="Nucleus area distribution",
                                    labels={area_col: "area (µm²)" if mpp else "area (px)"}), use_container_width=True)
        b.plotly_chart(px.scatter(df, x="circularity", y="eccentricity", hover_data=["label", "area_px"],
                                  title="Shape: circularity vs eccentricity", opacity=0.6), use_container_width=True)
        st.dataframe(pd.DataFrame([summary]).T.rename(columns={0: "value"}), use_container_width=True)

with tab_table:
    st.dataframe(df, use_container_width=True, height=400)

# --- Ground truth comparison ---------------------------------------------- #
if gt_xml is not None:
    st.subheader("Comparison with ground truth")
    tmp = ROOT / "outputs" / "_tmp_gt.xml"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_bytes(gt_xml.read())
    gt = xml_to_instance_mask(tmp, *image.shape[:2]).astype(np.int32)
    m = evaluate_pair(labels, gt)
    g1, g2, g3, g4, g5, g6 = st.columns(6)
    g1.metric("Dice", f"{m['dice']:.3f}")
    g2.metric("IoU", f"{m['iou']:.3f}")
    g3.metric("AJI", f"{m['aji']:.3f}")
    g4.metric("F1 @ IoU 0.5", f"{m['f1']:.3f}")
    g5.metric("Precision / Recall", f"{m['precision']:.2f} / {m['recall']:.2f}")
    g6.metric("Count error", f"{m['count_error']:+d}", help=f"pred {m['n_pred']} vs GT {m['n_gt']}")
    a, b = st.columns(2)
    a.image(overlay_instances(image, gt), caption=f"Ground truth: {m['n_gt']} nuclei", use_container_width=True)
    b.image(overlay, caption=f"Prediction: {m['n_pred']} nuclei", use_container_width=True)

# --- Downloads ------------------------------------------------------------- #
st.subheader("Export")
d1, d2, d3 = st.columns(3)
stem = Path(uploaded.name).stem
d1.download_button("⬇ Per-nucleus CSV", df.to_csv(index=False).encode(), f"{stem}_nuclei.csv", "text/csv")
d2.download_button("⬇ Annotated image (PNG)", to_png_bytes(overlay), f"{stem}_overlay.png", "image/png")
d3.download_button("⬇ Instance mask (16-bit PNG)", to_png_bytes(labels.astype(np.uint16)), f"{stem}_labels.png", "image/png")
