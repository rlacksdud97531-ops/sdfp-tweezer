"""
app_lab.py — Lab data-collection guide (Streamlit)

Step-by-step guide for collecting the training data needed to apply
SDFP for Er/SU(8) tweezers to a real optical system, with one-click
generation/download of all required SLM patterns.

Run:
    source venv/bin/activate
    streamlit run app_lab.py
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
from PIL import Image

from sdfp import patterns as P

st.set_page_config(page_title="SDFP Lab Data Collection Guide", page_icon="🔬",
                   layout="wide")

st.title("🔬 SDFP Lab Data Collection Guide")
st.markdown("**No atoms needed** — build Surrogate training data with just an "
            "SLM + laser + camera.")


# ─────────────────────────────────────────────────────
# Sidebar: common settings
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Common settings")
    size = st.selectbox("Compute grid (px)", [128, 256], index=0,
                        help="Resolution of the trained model. Pattern generation "
                             "and training both use this size")
    shape = (size, size)

    st.subheader("0th-order avoidance (blazed grating)")
    use_blaze = st.checkbox("Add blaze to all patterns", value=True)
    blaze_px = st.slider("Blaze period x (px)", 4, 32, 8, disabled=not use_blaze,
                         help="Smaller period moves the array farther from the "
                              "0th order (at slightly lower efficiency)")
    bx = blaze_px if use_blaze else 0

    st.subheader("SLM canvas")
    embed = st.checkbox("Embed at center of PLUTO 1080×1920", value=False,
                        help="If enabled, downloaded PNGs become full-resolution "
                             "SLM canvases")
    canvas = (1080, 1920) if embed else None

    st.divider()
    st.caption(f"Pattern format: 8-bit PNG (0 = 0 rad, 255 ≈ 2π)\nGrid {size}×{size}"
               + (", embedded in PLUTO canvas" if embed else ""))


def dl_zip(named_phases, label, fname, manifest=None, key=None):
    """Phase list → apply blaze/embedding → zip download button."""
    items = P.finalize(named_phases, shape, blaze_x=bx, canvas=canvas)
    data = P.make_zip(items, manifest)
    st.download_button(f"⬇️ {label} ({len(items)} files, zip)", data,
                       file_name=fname, mime="application/zip",
                       use_container_width=True, key=key)


def show_preview(named_phases, n=4):
    items = P.finalize(named_phases[:n], shape, blaze_x=bx)
    cols = st.columns(len(items))
    for c, (name, u8) in zip(cols, items):
        c.image(u8, caption=name, use_container_width=True)


# ─────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Overview", "Step 0 — Setup", "Step 1 — Pattern generation",
    "Step 2 — Acquisition", "Step 3 — Training & DAgger"])


# ═════════════════════════════════════════════════════
with tab0:
    st.header("Workflow")
    st.markdown("""
```
[Step 0] Setup (half a day)              ← skip this and ALL data is wasted
   warm-up · 0th-order avoidance · camera↔SLM registration

[Step 1] Pattern generation (this site, just clicks)
   gamma calibration + aberration probes + GS array POHs + random + reference

[Step 2] Automated acquisition (~overnight, unattended)
   collect_data.py: HDR + averaging + dark subtraction + drift monitoring
   → 2,000–5,000 (φ, I) pairs

[Step 3] Training (~1 h, GPU)
   train_real_surrogate.py: Surrogate ← measured data
   → U-Net distill → checkpoint_real.pt
   → (optional) DAgger: measure U-Net outputs → retrain → repeat
```
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Equipment", "SLM+laser+camera", help="No atoms or vacuum needed")
    c2.metric("Measured pairs", "2,000–5,000",
              help="The literature (Ren 2024) reports 2,000 is sufficient")
    c3.metric("Total time", "~1 day",
              help="half-day setup + overnight unattended acquisition + 1 h training")

    st.info("**Key principle**: the only thing that needs real measurements is the "
            "(φ, I) pairs for the Surrogate. Target arrays for U-Net training are "
            "generated in code — unlimited and free.", icon="💡")

    st.warning("**Expectation management**: the 0.6% uniformity seen in simulation "
               "will not happen on real hardware. A realistic target is "
               "**few-percent uniformity** — still well ahead of GS (10–36%) — plus "
               "inference within the coherence window (~ms). Quantization, flicker "
               "and other unmodeled effects set the floor.",
               icon="⚠️")


# ═════════════════════════════════════════════════════
with tab1:
    st.header("Step 0 — Setup (half a day)")
    st.markdown("Skip this step and **all collected data will be thrown away.** "
                "In order:")

    st.subheader("☑️ Checklist")
    st.checkbox("Turn on the SLM & laser and **warm up 30–60 min** "
                "(LCOS phase response is temperature-dependent)")
    st.checkbox("Connect a laser power monitor (photodiode) — logs power during "
                "acquisition")
    st.checkbox("Check each spot spans **≥3–5 pixels** at the camera exposure "
                "(adjust magnification)")
    st.checkbox("Reserve space for a 0th-order beam block (locate it with the "
                "blaze pattern below)")

    st.divider()
    st.subheader("1) 0th-order avoidance — standalone blazed grating")
    st.markdown("""
Display this pattern on the SLM and **all light deflects toward the 1st order.**
- Identify the **undeflected spot (0th order)** and the **deflected beam (1st order)** on the camera
- Install a beam block at the 0th-order position → blocks 0th-order contamination in all later measurements
- Keep "Add blaze to all patterns" enabled in the sidebar so all later patterns share the same offset
""")
    if st.button("Generate blaze pattern", key="b_blaze"):
        ph = [("blaze_only", P.blazed(shape, period_x=blaze_px))]
        show_preview(ph, 1)
        items = P.finalize(ph, shape, canvas=canvas)   # blaze itself — don't blaze twice
        buf = io.BytesIO()
        Image.fromarray(items[0][1], "L").save(buf, format="PNG")
        st.download_button("⬇️ blaze_only.png", buf.getvalue(),
                           file_name="blaze_only.png", mime="image/png")

    st.divider()
    st.subheader("2) Camera↔SLM registration — probe spots")
    st.markdown("""
Blazed gratings of different frequencies = **the 1st-order spot moves across the field.**
- Record the spot centroid on the camera for each pattern
- Fit an **affine transform** from the (SLM frequency) ↔ (camera pixel) pairs
- Step 3 uses this matrix to resample camera images onto the compute grid
- Bonus: spot shape vs position → **prior information about field aberrations**
""")
    pg = st.slider("Probe grid (n×n)", 3, 5, 3)
    if st.button("Generate probe patterns", key="b_probe"):
        ph = P.probe_patterns(shape, grid=pg)
        show_preview(ph)
        dl_zip(ph, "Probe patterns", "probes.zip",
               manifest={"type": "probe", "grid": pg, "size": size}, key="d_probe")


# ═════════════════════════════════════════════════════
with tab2:
    st.header("Step 1 — Training pattern generation")
    st.markdown("""
The key is mixing patterns that **directly excite the unknowns** the Surrogate must learn (gamma, aberrations, illumination).

| Kind | Suggested share | Role |
|---|---|---|
| GS array POHs | ~60% | same distribution as the U-Net's operating region |
| Pure random phases | ~20% | diversity, prevents overfitting |
| Calibration (gamma + probes) | ~20% | directly identifies the unknowns |
""")

    st.subheader("1) Gamma calibration patterns")
    st.markdown("Binary gratings (0 ↔ gray level g) sweeping g = 0→255. Recording "
                "the **1st-order diffracted power P(g)** of each pattern inverts the "
                "SLM phase response (gamma curve) via P ∝ sin²(Δφ(g)/2).")
    nlev = st.slider("Number of gray levels", 8, 32, 16)
    if st.button("Generate gamma patterns", key="b_gamma"):
        ph = P.gamma_calib_patterns(shape, n_levels=nlev)
        show_preview(ph)
        dl_zip(ph, "Gamma calibration", "calib_gamma.zip",
               manifest={"type": "gamma", "n_levels": nlev, "size": size},
               key="d_gamma")

    st.divider()
    st.subheader("2) GS array POHs (main data)")
    st.markdown("GS POHs of random tweezer arrays (Square/Tri/Honeycomb/Random). "
                "**The browser generates them in batches** — change the seed between "
                "downloads and filenames won't collide. (For a full 2,000+ set: "
                "e.g. seeds 0–9 × 200 each.)")
    c1, c2 = st.columns(2)
    n_gs = c1.slider("Batch size", 10, 200, 50, step=10)
    seed_gs = c2.number_input("Seed (change per batch)", 0, 999, 0)
    if st.button("Generate GS POHs (tens of seconds – minutes)", key="b_gs"):
        bar = st.progress(0.0, "Running GS...")
        ph = P.gs_array_patterns(shape, n_gs, seed=int(seed_gs),
                                 progress=lambda k, n: bar.progress(k / n))
        bar.empty()
        show_preview(ph)
        dl_zip(ph, f"GS POHs (seed={seed_gs})", f"gs_seed{seed_gs}.zip",
               manifest={"type": "gs", "n": n_gs, "seed": int(seed_gs),
                         "size": size}, key="d_gs")

    st.divider()
    st.subheader("3) Pure random phases")
    n_rand = st.slider("Batch size ", 10, 500, 100, step=10)
    seed_rand = st.number_input("Seed  (change per batch)", 0, 999, 0)
    if st.button("Generate random phases", key="b_rand"):
        ph = P.random_phase_patterns(shape, n_rand, seed=int(seed_rand))
        show_preview(ph)
        dl_zip(ph, f"Random phases (seed={seed_rand})", f"rand_seed{seed_rand}.zip",
               manifest={"type": "random", "n": n_rand, "seed": int(seed_rand),
                         "size": size}, key="d_rand")

    st.divider()
    st.subheader("4) Reference pattern — exactly one, required")
    st.markdown("**The same pattern is re-measured every 50 shots** to monitor "
                "optical drift. The acquisition script looks for the literal name "
                "`reference.png` — do not rename it.")
    if st.button("Generate reference pattern", key="b_ref"):
        ph = P.reference_pattern(shape)
        show_preview(ph, 1)
        items = P.finalize(ph, shape, blaze_x=bx, canvas=canvas)
        buf = io.BytesIO()
        Image.fromarray(items[0][1], "L").save(buf, format="PNG")
        st.download_button("⬇️ reference.png", buf.getvalue(),
                           file_name="reference.png", mime="image/png",
                           key="d_ref")

    st.success("Unzip everything into **one folder (`patterns/`)**, including "
               "reference.png. → Step 2", icon="📁")


# ═════════════════════════════════════════════════════
with tab3:
    st.header("Step 2 — Automated acquisition (unattended, overnight OK)")
    st.markdown("""
`lab/collect_data.py` automates the whole run. **Only two functions need lab-specific code:**

| Function | Job | Reference |
|---|---|---|
| `display_phase(u8)` | show an 8-bit image on the SLM | HOLOEYE SLM Display SDK `showData()` |
| `grab_frame(exp_ms)` | capture one frame at a given exposure | pylablib / pypylon / vendor SDK |

What the script does automatically:
1. Captures a **dark frame** (block the beam once at start)
2. Per pattern: display on SLM → **200 ms settling wait** → **HDR, 3 exposures × 4-frame averaging**
   (short exposures cover the spots, long ones the background — saturated pixels are excluded automatically)
3. Inserts the **reference pattern every 50 shots** → drift log
4. Saves one `.npz` per pattern (sent phase + processed intensity + metadata)

**Time**: ~1.5 s per pattern → 2,000 ≈ 1 h, 5,000 ≈ 2.5 h.
""")
    with open("lab/collect_data.py", encoding="utf-8") as f:
        src = f.read()
    st.download_button("⬇️ Download collect_data.py", src,
                       file_name="collect_data.py", mime="text/x-python",
                       use_container_width=True)
    with st.expander("Script preview"):
        st.code(src, language="python")

    st.warning("**During acquisition, do not**: touch the optical table, change the "
               "lighting, or alter SLM settings. If the reference pattern drifts "
               "more than 5%, re-measure that span.", icon="🚫")


# ═════════════════════════════════════════════════════
with tab4:
    st.header("Step 3 — Training & the DAgger loop")
    st.markdown("""
`lab/train_real_surrogate.py` does it all:

```bash
PYTHONPATH=. python lab/train_real_surrogate.py --data measured --size {size}
```

What it does:
1. **Drift check** — prints how the reference patterns changed over time (warns above 5%)
2. **Surrogate ← measured (φ, I)** — real data replaces the simulator (`SimOptics`).
   Gamma, illumination and aberrations are learned at *their real-hardware values*
3. **U-Net distill** — target arrays are synthetic (free); the loss flows through the
   measurement-trained Surrogate
4. Saves `checkpoint_real.pt` → point the existing app (`app_sdfp.py`) at it
5. **Auto-exports 200 DAgger patterns** (`patterns_dagger/`)

### 🔁 DAgger loop (where the real gains come from)
```
train U-Net → display its 200 output phases on the SLM and measure (reuse collect_data.py)
            → add that data to measured/ → rerun train_real_surrogate.py
            → repeat (converges in 2–3 rounds)
```
The Surrogate becomes accurate **exactly where the U-Net operates**, closing the
"great in sim, different on hardware" gap (model exploitation).

### Final validation
- Display U-Net POHs vs GS POHs on the real SLM and compare **measured uniformity (σ/μ)**
- Compare per-spot **2D-Gaussian-fit amplitudes** (not raw peaks) for publication-grade analysis
""".replace("{size}", str(size)))

    with open("lab/train_real_surrogate.py", encoding="utf-8") as f:
        src2 = f.read()
    st.download_button("⬇️ Download train_real_surrogate.py", src2,
                       file_name="train_real_surrogate.py", mime="text/x-python",
                       use_container_width=True)
    with st.expander("Script preview"):
        st.code(src2, language="python")

    st.success("Once this is done: change the checkpoint path in `app_sdfp.py` to "
               "`sdfp/checkpoint_real.pt` to see the comparison backed by real "
               "measurements.",
               icon="🏁")
