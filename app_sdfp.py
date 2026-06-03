"""
app_sdfp.py — Surrogate-Distilled Fast Predictor (Streamlit)

Er / SU(8) 트위저용 POH 생성기 데모.
  desired array → U-Net → POH (수 ms)  vs  GS / WGS (반복 최적화)
세 방법의 POH 를 *수차가 있는 진짜 광학계 시뮬레이터* 에 통과시켜
재구성 강도·균일도·효율·속도를 비교한다.

⚠️ 아직 실험 데이터가 없으므로 "진짜 광학계" 는 시뮬레이터(SimOptics)로 대체.
   실험 데이터가 생기면 Surrogate 만 그 데이터로 다시 학습하면 동일 구조로 작동.

실행:
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
    "**Er / SU(8) 트위저용 POH 생성** — U-Net(빠름) vs GS/WGS(반복), "
    "수차 있는 광학계에서의 균일도 비교"
)

st.warning(
    "아직 **실험 데이터가 없어서**, '진짜 광학계' 는 수차·SLM 비선형성을 주입한 "
    "**시뮬레이터(SimOptics)** 로 대체했습니다. 실험 데이터가 생기면 Surrogate 만 "
    "카메라 데이터로 다시 학습하면 동일한 파이프라인이 그대로 작동합니다.",
    icon="⚠️",
)


# ─────────────────────────────────────────────────────
@st.cache_resource
def load_model(ckpt):
    return INF.load(ckpt)


@st.cache_resource
def get_optics(shape, device):
    # 학습에 사용한 광학계와 동일 파라미터 (seed=0)
    return SimOptics(shape=shape, device=device, read_noise=0.0, seed=0)


def fig_bytes(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig); buf.seek(0); return buf.read()


# ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ckpt = st.text_input("Checkpoint", "sdfp/checkpoint.pt")

    st.subheader("Array")
    array_type = st.selectbox("Type", ["Square", "Triangular", "Honeycomb"])
    spacing = st.slider("Spacing (px)", 11, 22, 15)
    if array_type in ("Square", "Triangular"):
        rows = st.slider("Rows", 2, 6, 4)
        cols = st.slider("Cols", 2, 6, 4)
        rings = 2
    else:
        rings = st.slider("Rings", 1, 3, 2)
        rows = cols = 4

    gs_iter = st.slider("GS/WGS iterations", 20, 150, 60, step=10)
    run = st.button("▶ Run", type="primary", use_container_width=True)


if not os.path.exists(ckpt):
    st.error(f"체크포인트가 없습니다: `{ckpt}`\n\n"
             "먼저 학습하세요:  `python -m sdfp.train --quick`")
    st.stop()

net, sur, shape, device = load_model(ckpt)
optics = get_optics(shape, device)
H, W = shape
st.caption(f"device = `{device}`  ·  SLM = {W}×{H}  ·  학습된 모델 로드 완료")


if not run:
    with st.expander("🧠 구조 설명 (Surrogate-Distilled Fast Predictor)", expanded=True):
        st.markdown(
            """
**문제**: Er 은 결맞음 시간이 짧다 → POH 를 매번 GS 로 반복 최적화할 시간이 없다.
동시에 SU(8) 물리는 trap 균일도가 매우 중요하다.

**아이디어**: 속도(U-Net)와 품질(미분가능 Surrogate)을 결합.

1. **오프라인 — Surrogate 학습** (결맞음 제약 없음)
   카메라 피드백으로 측정한 (위상 φ → 강도 I) 쌍으로 *진짜 광학계의 미분가능
   모델* 을 학습. FFT 는 물리로 고정하고, 모르는 부분(SLM 감마, 조명 프로파일,
   정적 수차)만 학습 → 광학계 캘리브레이션을 데이터로 역산.

2. **오프라인 — Predictor distill**
   desired array → U-Net → φ → [frozen Surrogate] → I.
   스팟 기하평균 최대화(효율+균일도) 손실로 U-Net 학습.
   → U-Net 이 수차를 *미리 보정한* 위상을 예측하도록 distill.

3. **온라인 — 추론** (결맞음 제약 O)
   desired array → U-Net → POH, 수 ms. GS 의 수차-미보정 문제 없이 균일.

위상은 *직접 지도하지 않는다* (phase-retrieval 해는 비유일/스펙클).
오직 재구성 강도만 미분가능 전파를 통해 최적화한다.
            """
        )
    st.info("👈 설정 후 **▶ Run**")
    st.stop()


# ── 타깃 생성 ──────────────────────────────────────────
pos = D.spot_positions(array_type, shape, rows=rows, cols=cols,
                       spacing=spacing, rings=rings)
margin = 6
if (pos[0].min() < margin or pos[0].max() > W - margin or
        pos[1].min() < margin or pos[1].max() > H - margin):
    st.error("⚠️ 배열이 SLM 범위를 벗어납니다. Spacing/크기를 줄이세요.")
    st.stop()

target = D.target_intensity(pos, shape, device=device)
peaks = np.stack([np.round(pos[0]).astype(int), np.round(pos[1]).astype(int)])
n_spots = pos.shape[1]

st.subheader(f"📊 {array_type} — {n_spots} spots")

with st.spinner("GS / WGS / U-Net POH 생성 + 광학계 통과..."):
    res = INF.compare(net, optics, target, peaks, device, gs_iter=gs_iter)

# ── 결과 3열 ───────────────────────────────────────────
cols_ui = st.columns(3)
for col, name in zip(cols_ui, ["GS", "WGS", "U-Net (SDFP)"]):
    r = res[name]
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

# ── 요약 비교 ──────────────────────────────────────────
st.divider()
st.subheader("📈 비교 요약")

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
    sp = res["GS"]["time"] / max(res["U-Net (SDFP)"]["time"], 1e-6)
    st.metric("GS non-uniformity", f"{gs_nu:.1f}%")
    st.metric("U-Net non-uniformity", f"{un_nu:.1f}%",
              delta=f"{un_nu - gs_nu:.1f}%", delta_color="inverse")
    st.metric("Uniformity improvement", f"{imp:.0f}%")
    st.metric("Speed-up vs GS", f"{sp:.0f}×")
    st.caption("U-Net 은 학습된 Surrogate 를 통해 수차를 미리 보정 → "
               "GS 대비 더 균일하고 추론은 수 ms.")
