"""
wgs_app.py  —  WGS Algorithm with SLMSuite (Streamlit)

실행 방법:
    source venv/bin/activate
    streamlit run wgs_app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import io

from slmsuite.holography.algorithms import SpotHologram

# ─────────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="WGS Algorithm — SLMSuite",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Weighted GS Algorithm (SLMSuite)")
st.markdown("**GS vs WGS** 비교 | 트위저 배열 균일도 개선")
st.divider()


# ─────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def get_spot_positions(array_type, rows, cols, spacing, rings, cx, cy):
    """배열 종류에 따라 (2, N) spot 좌표 반환 (knm basis, absolute pixel)"""
    positions = []

    if array_type == "Square":
        for i in range(rows):
            for j in range(cols):
                x = cx + (j - (cols-1)/2) * spacing
                y = cy + (i - (rows-1)/2) * spacing
                positions.append([x, y])

    elif array_type == "Triangular":
        row_h = int(spacing * np.sqrt(3) / 2)
        for i in range(rows):
            offset = spacing / 2 if i % 2 == 1 else 0
            for j in range(cols):
                x = cx + (j - (cols-1)/2) * spacing + offset
                y = cy + (i - (rows-1)/2) * row_h
                positions.append([x, y])

    elif array_type == "Honeycomb":
        positions.append([cx, cy])
        for ring in range(1, rings + 1):
            for direction in range(6):
                angle = np.radians(direction * 60)
                dx = np.cos(angle) * spacing * ring
                dy = np.sin(angle) * spacing * ring
                for step in range(ring):
                    move_angle = np.radians((direction + 2) * 60)
                    sx = np.cos(move_angle) * spacing * step
                    sy = np.sin(move_angle) * spacing * step
                    positions.append([cx + dx + sx, cy + dy + sy])

    return np.array(positions).T.astype(float)  # (2, N)


def run_hologram(spot_vectors, shape, method, maxiter):
    """SpotHologram 생성 + optimize"""
    holo = SpotHologram(
        shape=shape,
        spot_vectors=spot_vectors,
        basis="knm",
    )
    holo.optimize(method=method, maxiter=maxiter, verbose=False)
    return holo


def get_reconstruction(holo):
    """홀로그램 → 복원 강도 이미지"""
    phase = holo.phase                       # (H, W)
    field = np.fft.fftshift(np.fft.fft2(np.exp(1j * phase)))
    intensity = np.abs(field) ** 2
    return intensity / intensity.max()


def compute_uniformity(recon, spot_vectors, radius=6):
    """각 spot의 최대 강도 추출 → 표준편차/평균 = 비균일도"""
    vals = []
    h, w = recon.shape
    xs = np.clip(spot_vectors[0].astype(int), radius, w - radius)
    ys = np.clip(spot_vectors[1].astype(int), radius, h - radius)
    for x, y in zip(xs, ys):
        patch = recon[y-radius:y+radius, x-radius:x+radius]
        vals.append(float(patch.max()))
    vals = np.array(vals)
    nonuniformity = vals.std() / (vals.mean() + 1e-12)
    return vals, nonuniformity


# ─────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("1. SLM / Canvas")
    slm_h = st.selectbox("Height (px)", [256, 512, 1080], index=1)
    slm_w = st.selectbox("Width (px)",  [256, 512, 1920], index=1)
    shape  = (slm_h, slm_w)
    cx, cy = slm_w // 2, slm_h // 2

    st.subheader("2. Array Type")
    array_type = st.selectbox("Type", ["Square", "Triangular", "Honeycomb"])

    spacing = st.slider("Spacing (px)", 20, 100, 50)

    if array_type == "Square":
        rows = st.slider("Rows", 2, 8, 4)
        cols = st.slider("Cols", 2, 8, 4)
        rings = 2
    elif array_type == "Triangular":
        rows = st.slider("Rows", 2, 8, 4)
        cols = st.slider("Cols", 2, 8, 4)
        rings = 2
    else:
        rings = st.slider("Rings", 1, 4, 2)
        rows = cols = 4

    st.subheader("3. Algorithm")
    method = st.selectbox(
        "Method",
        ["GS", "WGS-Leonardo", "WGS-Kim"],
        index=0,
        help=(
            "GS: 기본 Gerchberg-Saxton\n"
            "WGS-Leonardo: 원조 WGS (Leonardo et al.)\n"
            "WGS-Kim: Kim et al. 개선 버전"
        ),
    )
    maxiter = st.slider("Iterations", 10, 300, 50, step=10)

    st.subheader("4. Compare")
    compare = st.checkbox("GS vs WGS 동시 비교", value=True)

    st.divider()
    run_btn = st.button("▶ Run", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────
if not run_btn:
    st.info("👈 사이드바에서 설정 후 **▶ Run** 을 눌러주세요.", icon="ℹ️")
    st.markdown("""
    ### GS vs WGS 차이

    | | GS | WGS |
    |--|----|----|
    | **균일도** | 낮음 (스펙클 노이즈) | 높음 ✅ |
    | **속도** | 빠름 | 약간 느림 |
    | **방법** | 단순 반복 | 가중치 피드백 반복 |

    ### WGS 방법 종류
    - **WGS-Leonardo** : 각 spot 강도에 따라 가중치 조절 (원조)
    - **WGS-Kim** : 수렴 속도 개선 버전
    """)

else:
    # ── Spot 좌표 생성 ──────────────────────────────
    spot_vectors = get_spot_positions(array_type, rows, cols, spacing, rings, cx, cy)
    n_spots = spot_vectors.shape[1]

    # 범위 체크
    margin = 10
    if (spot_vectors[0].min() < margin or spot_vectors[0].max() > slm_w - margin or
        spot_vectors[1].min() < margin or spot_vectors[1].max() > slm_h - margin):
        st.error("⚠️ 트위저 좌표가 SLM 범위를 벗어납니다. Spacing을 줄이거나 배열 크기를 줄여주세요.")
        st.stop()

    st.subheader(f"📊 Results — {array_type} ({n_spots} spots)")

    # ── 실행 ────────────────────────────────────────
    if compare and method != "GS":
        methods_to_run = ["GS", method]
        cols_ui = st.columns(2)
    else:
        methods_to_run = [method]
        cols_ui = [st.columns(1)[0]]

    results = {}

    with st.spinner(f"Running {' & '.join(methods_to_run)}..."):
        for m in methods_to_run:
            holo  = run_hologram(spot_vectors, shape, m, maxiter)
            recon = get_reconstruction(holo)
            vals, nonunif = compute_uniformity(recon, spot_vectors)
            results[m] = {"holo": holo, "recon": recon,
                          "vals": vals, "nonunif": nonunif}

    # ── 결과 표시 ────────────────────────────────────
    for col_ui, m in zip(cols_ui, methods_to_run):
        res = results[m]
        recon = res["recon"]
        holo  = res["holo"]
        vals  = res["vals"]

        with col_ui:
            st.markdown(f"### `{m}`")

            # 메트릭
            mc1, mc2 = st.columns(2)
            mc1.metric("Non-uniformity", f"{res['nonunif']*100:.2f}%",
                       help="낮을수록 균일 (0% = 완벽)")
            mc2.metric("Spots", n_spots)

            # Phase + Reconstruction 나란히
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(holo.phase, cmap="gray")
            axes[0].set_title(f"Phase (POH)  — {m}", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(recon, cmap="hot", vmin=0, vmax=recon.max())
            axes[1].scatter(spot_vectors[0], spot_vectors[1],
                            s=30, c="cyan", marker="+", linewidths=1)
            axes[1].set_title("Reconstruction", fontsize=11)
            axes[1].axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Spot 강도 분포
            fig2, ax2 = plt.subplots(figsize=(6, 2.5))
            ax2.bar(range(n_spots), vals, color="#4A90D9", alpha=0.8)
            ax2.axhline(vals.mean(), color="red", linestyle="--",
                        linewidth=1.5, label=f"Mean = {vals.mean():.3f}")
            ax2.set_xlabel("Spot index")
            ax2.set_ylabel("Intensity")
            ax2.set_title(f"Spot Intensity Distribution  [{m}]")
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            # 다운로드
            poh_8bit = ((holo.phase % (2*np.pi)) / (2*np.pi) * 255).astype(np.uint8)
            poh_img  = Image.fromarray(poh_8bit, mode="L")
            buf = io.BytesIO()
            poh_img.save(buf, format="PNG")
            st.download_button(
                f"⬇️ Download POH ({m})",
                data=buf.getvalue(),
                file_name=f"poh_{m.lower()}.png",
                mime="image/png",
                use_container_width=True,
            )

    # ── GS vs WGS 균일도 비교 (compare 모드) ────────
    if compare and len(results) == 2:
        st.divider()
        st.subheader("📈 GS vs WGS Uniformity Comparison")

        fig3, ax3 = plt.subplots(figsize=(10, 3))
        x = np.arange(n_spots)
        width = 0.35
        colors = ["#A8C8FF", "#FF8C69"]
        for idx, (m, res) in enumerate(results.items()):
            ax3.bar(x + idx*width, res["vals"], width,
                    label=f"{m}  (σ/μ = {res['nonunif']*100:.2f}%)",
                    color=colors[idx], alpha=0.85)
        ax3.set_xlabel("Spot index")
        ax3.set_ylabel("Intensity")
        ax3.set_title("Spot Intensity: GS vs WGS")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        # 요약 메트릭
        gs_nu  = results["GS"]["nonunif"]  * 100
        wgs_nu = results[method]["nonunif"] * 100
        improvement = (gs_nu - wgs_nu) / gs_nu * 100 if gs_nu > 0 else 0

        s1, s2, s3 = st.columns(3)
        s1.metric("GS Non-uniformity",  f"{gs_nu:.2f}%")
        s2.metric(f"{method} Non-uniformity", f"{wgs_nu:.2f}%",
                  delta=f"{wgs_nu - gs_nu:.2f}%",
                  delta_color="inverse")
        s3.metric("Improvement", f"{improvement:.1f}%")
