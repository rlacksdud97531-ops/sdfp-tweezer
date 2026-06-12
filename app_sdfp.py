"""
app_sdfp.py — Surrogate-Distilled Fast Predictor (Streamlit)

POH generator demo for Er / SU(8) optical-tweezer arrays.
  desired array → U-Net → POH (a few ms)  vs  GS / WGS (iterative)
All three POHs are propagated through a *simulated real optical system*
(aberrations + SLM non-linearity) and compared on reconstruction
intensity, uniformity, efficiency and speed.

⚠️ No experimental data yet, so the "real optics" is a simulator (SimOptics).
   Once camera data is available, only the Surrogate needs to be retrained —
   the same pipeline then works unchanged.

Run:
    source venv/bin/activate
    streamlit run app_sdfp.py
"""

import os
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from sdfp.optics_sim import SimOptics
from sdfp import data as D
from sdfp import infer as INF

st.set_page_config(page_title="SDFP — Er Tweezer POH", page_icon="⚛️", layout="wide")

st.title("⚛️ Surrogate-Distilled Fast Predictor")
st.markdown(
    "**POH generation for Er / SU(8) tweezer arrays** — U-Net (fast) vs GS/WGS "
    "(iterative), compared on an aberrated optical system"
)

st.warning(
    "No **experimental data** yet — the \"real optical system\" is replaced by a "
    "**simulator (SimOptics)** with injected aberrations and SLM non-linearity. "
    "Once camera data is available, only the Surrogate needs retraining; "
    "the rest of the pipeline works unchanged.",
    icon="⚠️",
)


# ─────────────────────────────────────────────────────
@st.cache_resource
def load_model(ckpt):
    return INF.load(ckpt)


@st.cache_resource
def get_optics(shape, device):
    # Same optics parameters as used in training (seed=0)
    return SimOptics(shape=shape, device=device, read_noise=0.0, seed=0)


def fig_bytes(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig); buf.seek(0); return buf.read()


# ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ckpt = st.text_input("Checkpoint", "sdfp/checkpoint_256.pt")
    if not os.path.exists(ckpt):
        st.error(f"Checkpoint not found: `{ckpt}`\n\n"
                 "Train first:  `python -m sdfp.train --quick`")
        st.stop()
    net, sur, shape, device = load_model(ckpt)
    sc = max(1, shape[0] // 128)   # layout scale vs the 128² reference

    st.subheader("Array")
    array_type = st.selectbox("Type", ["Square", "Triangular", "Honeycomb"])
    spacing = st.slider("Spacing (px)", 11 * sc, 24 * sc, 12 * sc,
                        help=f"Training covered {11*sc}–{24*sc} px at {shape[0]}².")
    if array_type in ("Square", "Triangular"):
        rows = st.slider("Rows", 2, 5, 4, help="Training covered 2–5 rows/cols.")
        cols = st.slider("Cols", 2, 5, 4)
        rings = 2
    else:
        rings = st.slider("Rings", 1, 2, 2)
        rows = cols = 4

    gs_iter = st.slider("GS/WGS iterations", 20, 150, 60, step=10)
    refine_steps = st.slider(
        "U-Net refine steps (test-time)", 0, 100, 40, step=10,
        help="Polishes the U-Net output through the differentiable Surrogate to "
             "equalize spot peaks (per-instance). "
             "0 = raw U-Net · 40 ≈ balanced · 100 ≈ best quality "
             "(nu≈0.2%, eff≈83%, ~0.4 s on GPU; slower on CPU). "
             "More Adam steps beat L-BFGS in our benchmark.")
    run = st.button("▶ Run", type="primary", use_container_width=True)


optics = get_optics(shape, device)


@st.cache_resource
def _warmup(_net, _sur, _optics, shape, device):
    """Pay the one-time JIT/CUDA init cost upfront so displayed timings are steady-state."""
    d = torch.zeros(shape, device=device)
    d[shape[0] // 2, shape[1] // 2] = 1.0
    pk = np.array([[shape[1] // 2], [shape[0] // 2]])
    INF.compare(_net, _sur, _optics, d, pk, device, gs_iter=5, refine_steps=3)
    return True


_warmup(net, sur, optics, shape, device)
H, W = shape
st.caption(f"device = `{device}`  ·  SLM = {W}×{H}  ·  trained model loaded")


if not run:
    with st.expander("🧠 How it works (Surrogate-Distilled Fast Predictor)", expanded=True):
        st.markdown(
            """
**Problem**: Er has a short coherence window → no time to re-run iterative GS
for every POH. At the same time, SU(8) physics demands highly uniform traps.

**Idea**: combine speed (U-Net) with quality (differentiable Surrogate).

1. **Offline — train the Surrogate** (no coherence constraint)
   Learn a *differentiable model of the real optics* from measured
   (phase φ → intensity I) pairs. The FFT is fixed physics; only the unknown
   parts (SLM gamma, illumination profile, static aberrations) are learned →
   the optical calibration is inverted from data.

2. **Offline — distill the Predictor**
   desired array → U-Net → φ → [frozen Surrogate] → I.
   The U-Net is trained with a spot geometric-mean loss (efficiency +
   uniformity) → it learns phases that *pre-compensate* the aberrations.

3. **Online — inference** (coherence-limited)
   desired array → U-Net → POH in a few ms, uniform on the real optics —
   without GS's uncompensated-aberration problem.

Phases are *never supervised directly* (phase-retrieval solutions are
non-unique / speckled). Only the reconstructed intensity is optimized
through differentiable propagation.
            """
        )
    st.info("👈 Configure settings, then press **▶ Run**")
    st.stop()


# ── Build target ──────────────────────────────────────
pos = D.spot_positions(array_type, shape, rows=rows, cols=cols,
                       spacing=spacing, rings=rings)
margin = 6
if (pos[0].min() < margin or pos[0].max() > W - margin or
        pos[1].min() < margin or pos[1].max() > H - margin):
    st.error("⚠️ The array exceeds the SLM area. Reduce spacing or array size.")
    st.stop()

target = D.target_intensity(pos, shape, device=device)
peaks = np.stack([np.round(pos[0]).astype(int), np.round(pos[1]).astype(int)])
n_spots = pos.shape[1]

st.subheader(f"📊 {array_type} — {n_spots} spots")

with st.spinner("Generating GS / WGS / U-Net POHs + propagating through the optics..."):
    res = INF.compare(net, sur, optics, target, peaks, device,
                      gs_iter=gs_iter, refine_steps=refine_steps)

# ── Three result columns ──────────────────────────────
cols_ui = st.columns(3)
sdfp_title = f"U-Net + Refine×{refine_steps}" if refine_steps > 0 else "U-Net (SDFP)"
for col, key, name in zip(cols_ui,
                          ["GS", "WGS", "U-Net (SDFP)"],
                          ["GS", "WGS", sdfp_title]):
    r = res[key]
    with col:
        st.markdown(f"### {name}")
        m1, m2 = st.columns(2)
        m1.metric("Non-uniformity", f"{r['nonunif']*100:.1f}%")
        m2.metric("Time", f"{r['time']*1000:.0f} ms")
        st.metric("Spot efficiency", f"{r['eff']*100:.1f}%")

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(r["phase"], cmap="gray"); ax[0].axis("off")
        ax[0].set_title("POH (phase)", fontsize=10)
        ax[1].imshow(r["recon"], cmap="hot")
        ax[1].scatter(peaks[0], peaks[1], s=18, c="cyan", marker="+", linewidths=0.6)
        ax[1].axis("off"); ax[1].set_title("Reconstruction (real optics)", fontsize=10)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        poh8 = (r["phase"] / (2 * np.pi) * 255).astype(np.uint8)
        buf = io.BytesIO(); Image.fromarray(poh8, "L").save(buf, format="PNG")
        st.download_button(f"⬇️ POH ({name})", buf.getvalue(),
                           file_name=f"poh_{name.split()[0].lower()}.png",
                           mime="image/png", use_container_width=True)

# ── Summary ───────────────────────────────────────────
st.divider()
st.subheader("📈 Summary")

c1, c2 = st.columns([1, 1])
with c1:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    names = list(res.keys())
    x = np.arange(n_spots)
    w_ = 0.27
    colors = ["#A8C8FF", "#FFD27F", "#7FE3A0"]
    for i, nm in enumerate(names):
        ax.bar(x + i * w_, res[nm]["vals"], w_,
               label=f"{nm} (σ/μ={res[nm]['nonunif']*100:.1f}%)", color=colors[i])
    ax.set_xlabel("Spot index"); ax.set_ylabel("Intensity")
    ax.set_title("Per-spot intensity @ real optics"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True); plt.close(fig)

with c2:
    gs_nu = res["GS"]["nonunif"] * 100
    un_nu = res["U-Net (SDFP)"]["nonunif"] * 100
    imp = (gs_nu - un_nu) / gs_nu * 100 if gs_nu > 0 else 0
    sdfp_t = res["U-Net (SDFP)"]["time"] * 1000
    st.metric("GS non-uniformity", f"{gs_nu:.1f}%")
    st.metric("SDFP non-uniformity", f"{un_nu:.1f}%",
              delta=f"{un_nu - gs_nu:.1f}%", delta_color="inverse")
    st.metric("Uniformity improvement", f"{imp:.0f}%")
    st.metric("SDFP time", f"{sdfp_t:.0f} ms")
    st.caption("SDFP = U-Net (high-efficiency warm start) + per-instance Surrogate "
               "refinement (peak equalization). The calibrated Surrogate can be "
               "swapped for camera feedback in a real experiment.")
