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
import json
import time
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
from sdfp.gs import gs as gs_fn
from sdfp.metrics import nonuniformity, efficiency

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

tab_demo, tab_conv = st.tabs(["🔬 Live demo", "⏱️ GS/WGS convergence study"])


# ═════════════════════════════════════════════════════
# Tab 1 — live demo
# ═════════════════════════════════════════════════════
with tab_demo:
    if not run:
        with st.expander("🧠 How it works (Surrogate-Distilled Fast Predictor)",
                         expanded=True):
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
    else:
        pos = D.spot_positions(array_type, shape, rows=rows, cols=cols,
                               spacing=spacing, rings=rings)
        margin = 6
        if (pos[0].min() < margin or pos[0].max() > W - margin or
                pos[1].min() < margin or pos[1].max() > H - margin):
            st.error("⚠️ The array exceeds the SLM area. Reduce spacing or array size.")
        else:
            target = D.target_intensity(pos, shape, device=device)
            peaks = np.stack([np.round(pos[0]).astype(int),
                              np.round(pos[1]).astype(int)])
            n_spots = pos.shape[1]

            st.subheader(f"📊 {array_type} — {n_spots} spots")

            with st.spinner("Generating GS / WGS / U-Net POHs + propagating "
                            "through the optics..."):
                res = INF.compare(net, sur, optics, target, peaks, device,
                                  gs_iter=gs_iter, refine_steps=refine_steps)

            cols_ui = st.columns(3)
            sdfp_title = (f"U-Net + Refine×{refine_steps}"
                          if refine_steps > 0 else "U-Net (SDFP)")
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
                    ax[1].scatter(peaks[0], peaks[1], s=18, c="cyan",
                                  marker="+", linewidths=0.6)
                    ax[1].axis("off")
                    ax[1].set_title("Reconstruction (real optics)", fontsize=10)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    poh8 = (r["phase"] / (2 * np.pi) * 255).astype(np.uint8)
                    buf = io.BytesIO()
                    Image.fromarray(poh8, "L").save(buf, format="PNG")
                    st.download_button(f"⬇️ POH ({name})", buf.getvalue(),
                                       file_name=f"poh_{name.split()[0].lower()}.png",
                                       mime="image/png", use_container_width=True)

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
                           label=f"{nm} (σ/μ={res[nm]['nonunif']*100:.1f}%)",
                           color=colors[i])
                ax.set_xlabel("Spot index"); ax.set_ylabel("Intensity")
                ax.set_title("Per-spot intensity @ real optics")
                ax.legend(fontsize=8)
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
                st.caption("SDFP = U-Net (high-efficiency warm start) + per-instance "
                           "Surrogate refinement (peak equalization). The calibrated "
                           "Surrogate can be swapped for camera feedback in a real "
                           "experiment.")


# ═════════════════════════════════════════════════════
# Tab 2 — GS/WGS convergence study (precomputed offline)
# ═════════════════════════════════════════════════════
with tab_conv:
    st.markdown(
        "**Question**: how many GS/WGS iterations does it take to match the trained "
        "model's uniformity on the aberrated optics?\n\n"
        "**Answer: no number of iterations gets there.** GS/WGS optimize an *ideal* "
        "FFT model — they don't know the aberrations — so their realized uniformity "
        "is limited by **information, not compute**."
    )

    SWEEP = "gs_sweep.json"
    if not os.path.exists(SWEEP):
        st.info("Precomputed sweep file `gs_sweep.json` not found. "
                "Use the live run below instead.")
        sw = None
    else:
        with open(SWEEP) as f:
            sw = json.load(f)

    if sw:
        g, ref, its = sw["geometry"], sw["sdfp_ref"], sw["iters"]
        st.caption(f"Geometry: {g['array']} {g['rows']}×{g['cols']}, "
                   f"spacing {g['spacing']} px, {g['size']}² — identical to the "
                   f"reference benchmark · mean over {sw['meta']['seeds']} random "
                   f"seeds (band = min–max) · precomputed offline on "
                   f"`{sw['meta']['device']}`")

        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for nm, c in [("GS", "#5B8DEF"), ("WGS", "#F0A35E")]:
            d = sw[nm]
            ax[0].fill_between(its, d["nu_min_pct"], d["nu_max_pct"],
                               color=c, alpha=0.18)
            ax[0].plot(its, d["nu_mean_pct"], "o-", color=c, label=nm, ms=4)
            ax[1].plot(its, d["eff_mean_pct"], "o-", color=c, label=nm, ms=4)
        ax[0].axhline(ref["nu_pct"], color="#2BA84A", ls="--", lw=1.6)
        ax[0].annotate(f"SDFP {ref['nu_pct']}%  (~{ref['time_s']} s)",
                       xy=(its[0], ref["nu_pct"]), xytext=(its[0], ref["nu_pct"] * 1.8),
                       color="#2BA84A", fontsize=9, fontweight="bold")
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_xlabel("GS/WGS iterations"); ax[0].set_ylabel("Non-uniformity σ/μ (%)")
        ax[0].set_title("Uniformity never converges to the model", fontsize=11)
        ax[0].grid(True, alpha=0.3, which="both"); ax[0].legend(fontsize=9)

        ax[1].axhline(ref["eff_pct"], color="#2BA84A", ls="--", lw=1.6,
                      label=f"SDFP {ref['eff_pct']}%")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("GS/WGS iterations"); ax[1].set_ylabel("Spot efficiency (%)")
        ax[1].set_title("Efficiency: comparable (≈3–4 pp above SDFP)", fontsize=11)
        ax[1].grid(True, alpha=0.3, which="both"); ax[1].legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        gs_last = sw["GS"]["nu_mean_pct"][-1]
        wgs_last = sw["WGS"]["nu_mean_pct"][-1]
        st.success(
            f"**Takeaways** — over 20 → 4,800 iterations (240×):\n"
            f"- **GS gets *worse*** ({sw['GS']['nu_mean_pct'][0]}% → {gs_last}%): more "
            f"iterations converge harder onto the *wrong* (ideal-FFT) optimum.\n"
            f"- **WGS stays flat** around {wgs_last}% — its weighted feedback equalizes "
            f"spots of the ideal model, not of the real optics.\n"
            f"- Both sit **~200× above** the model's {ref['nu_pct']}%, while efficiency "
            f"is similar (≈87% vs {ref['eff_pct']}%).\n"
            f"- The model's advantage is **not speed but calibration**: the Surrogate "
            f"has learned the aberrations and pre-compensates them.",
            icon="🎯")
        st.caption("Note: this is *single-shot computational* GS/WGS (no camera in the "
                   "loop). Experimental WGS reaches ~1% only via per-array camera "
                   "feedback — exactly the measurement loop that does not fit in the "
                   "coherence window. SDFP performs that calibration **once, offline**.")

        with st.expander("Raw data"):
            rows_tbl = []
            for nm in ["GS", "WGS"]:
                d = sw[nm]
                for i, itn in enumerate(its):
                    rows_tbl.append({
                        "method": nm, "iterations": itn,
                        "nu mean %": d["nu_mean_pct"][i],
                        "nu min %": d["nu_min_pct"][i],
                        "nu max %": d["nu_max_pct"][i],
                        "eff mean %": d["eff_mean_pct"][i],
                        "time (s)": d["time_med_s"][i],
                    })
            st.dataframe(rows_tbl, use_container_width=True, height=320)

    with st.expander("🔁 Re-run a reduced sweep live on this server"):
        st.caption("Runs GS/WGS at 20–600 iterations (seed 0) through the same "
                   "aberrated optics. Takes seconds on GPU, ~½–1 min on CPU.")
        if st.button("Run live sweep", key="b_sweep"):
            gg = sw["geometry"] if sw else {"size": 256, "rows": 4, "cols": 4,
                                            "spacing": 24, "radius": 6,
                                            "array": "Square"}
            sshape = (gg["size"], gg["size"])
            sw_optics = SimOptics(shape=sshape, device=device,
                                  read_noise=0.0, seed=0)
            spos = D.spot_positions("Square", sshape, rows=gg["rows"],
                                    cols=gg["cols"], spacing=gg["spacing"], rings=2)
            stgt = D.target_intensity(spos, sshape, device=device)
            spk = np.stack([np.round(spos[0]).astype(int),
                            np.round(spos[1]).astype(int)])
            subset = [20, 60, 150, 300, 600]
            out_rows = []
            prog = st.progress(0.0)
            for k, (nm, wgt) in enumerate([("GS", False), ("WGS", True)]):
                for j, itn in enumerate(subset):
                    t0 = time.time()
                    ph = gs_fn(stgt, n_iter=itn, weighted=wgt, device=device, seed=0)
                    dt = time.time() - t0
                    I = sw_optics.forward(ph).detach()
                    nu, _ = nonuniformity(I, spk, radius=gg["radius"])
                    ef = efficiency(I, spk, radius=gg["radius"])
                    out_rows.append({"method": nm, "iterations": itn,
                                     "nu %": round(nu * 100, 1),
                                     "eff %": round(ef * 100, 1),
                                     "time (s)": round(dt, 2)})
                    prog.progress((k * len(subset) + j + 1) / (2 * len(subset)))
            prog.empty()
            st.dataframe(out_rows, use_container_width=True)
            st.caption(f"Measured just now on `{device}`.")
