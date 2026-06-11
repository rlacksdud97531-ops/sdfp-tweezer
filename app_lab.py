"""
app_lab.py — 실험 데이터 수집 가이드 (Streamlit)

Er/SU(8) 트위저용 SDFP 를 실제 광학계에 적용하기 위한
훈련 데이터 수집 절차를 step-by-step 으로 안내하고,
필요한 SLM 패턴들을 버튼 클릭으로 생성/다운로드한다.

실행:
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

st.set_page_config(page_title="SDFP 실험 데이터 수집 가이드", page_icon="🔬",
                   layout="wide")

st.title("🔬 SDFP 실험 데이터 수집 가이드")
st.markdown("**원자 불필요** — SLM + 레이저 + 카메라만으로 Surrogate 훈련 데이터를 만든다.")


# ─────────────────────────────────────────────────────
# 사이드바: 공통 설정
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 공통 설정")
    size = st.selectbox("계산 격자 (px)", [128, 256], index=0,
                        help="학습 모델 해상도. 패턴 생성·학습 모두 이 크기 기준")
    shape = (size, size)

    st.subheader("0차 회피 (blazed grating)")
    use_blaze = st.checkbox("모든 패턴에 blaze 합성", value=True)
    blaze_px = st.slider("Blaze 주기 x (px)", 4, 32, 8, disabled=not use_blaze,
                         help="작을수록 배열이 0차에서 멀어짐 (단, 효율 약간↓)")
    bx = blaze_px if use_blaze else 0

    st.subheader("SLM 캔버스")
    embed = st.checkbox("PLUTO 1080×1920 중앙에 임베드", value=False,
                        help="켜면 다운로드 PNG 가 SLM 풀해상도 캔버스가 됨")
    canvas = (1080, 1920) if embed else None

    st.divider()
    st.caption(f"패턴 포맷: 8-bit PNG (0=0 rad, 255≈2π)\n격자 {size}×{size}"
               + (", PLUTO 캔버스 임베드" if embed else ""))


def dl_zip(named_phases, label, fname, manifest=None, key=None):
    """위상 목록 → blaze/임베드 적용 → zip 다운로드 버튼."""
    items = P.finalize(named_phases, shape, blaze_x=bx, canvas=canvas)
    data = P.make_zip(items, manifest)
    st.download_button(f"⬇️ {label} ({len(items)}장, zip)", data,
                       file_name=fname, mime="application/zip",
                       use_container_width=True, key=key)


def show_preview(named_phases, n=4):
    items = P.finalize(named_phases[:n], shape, blaze_x=bx)
    cols = st.columns(len(items))
    for c, (name, u8) in zip(cols, items):
        c.image(u8, caption=name, use_container_width=True)


# ─────────────────────────────────────────────────────
# 탭 구조
# ─────────────────────────────────────────────────────
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📋 개요", "Step 0 — 사전 준비", "Step 1 — 패턴 생성",
    "Step 2 — 촬영", "Step 3 — 학습 & DAgger"])


# ═════════════════════════════════════════════════════
with tab0:
    st.header("전체 흐름")
    st.markdown("""
```
[Step 0] 사전 준비 (반나절)          ← 안 하면 데이터 전부 버림
   워밍업 · 0차 회피 · 카메라↔SLM 좌표 정합

[Step 1] 패턴 생성 (이 사이트, 클릭)
   감마 캘리브레이션 + 수차 프로브 + GS 배열 POH + 랜덤 + 기준패턴

[Step 2] 자동 촬영 (~밤새, 무인)
   collect_data.py: HDR + 평균 + 다크차감 + 드리프트 감시
   → (φ, I) 쌍 2,000~5,000개

[Step 3] 학습 (~1시간, GPU)
   train_real_surrogate.py: Surrogate ← 실측 데이터
   → U-Net distill → checkpoint_real.pt
   → (선택) DAgger: U-Net 출력을 다시 측정 → 재학습 → 반복
```
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("필요 장비", "SLM+레이저+카메라", help="원자·진공 불필요")
    c2.metric("측정 쌍", "2,000~5,000", help="논문(Ren 2024)은 2,000으로 충분")
    c3.metric("총 소요", "~하루", help="준비 반나절 + 촬영 밤샘(무인) + 학습 1h")

    st.info("**핵심 원칙**: 실측이 필요한 건 Surrogate 용 (φ, I) 쌍뿐. "
            "U-Net 훈련용 타깃 배열은 코드로 무한 생성(공짜)된다.", icon="💡")

    st.warning("**기대치 관리**: 시뮬레이션의 0.6% 균일도는 실제에선 안 나온다. "
               "현실적 목표는 GS(10~36%) 대비 우위인 **수 %대 균일도**, 그리고 "
               "결맞음 시간 내 추론(~ms). 양자화·flicker 등 모델 밖 효과가 하한을 만든다.",
               icon="⚠️")


# ═════════════════════════════════════════════════════
with tab1:
    st.header("Step 0 — 사전 준비 (반나절)")
    st.markdown("이 단계를 건너뛰면 **수집한 데이터를 전부 버리게 된다.** 순서대로:")

    st.subheader("☑️ 체크리스트")
    st.checkbox("SLM·레이저 켜고 **30분~1시간 워밍업** (LCOS 위상응답은 온도 의존)")
    st.checkbox("레이저 파워 모니터(포토다이오드) 연결 — 촬영 중 파워 기록용")
    st.checkbox("카메라 노출에서 스팟이 **3~5픽셀 이상**에 걸치는지 확인 (배율 조정)")
    st.checkbox("0차 빔블록 설치 자리 확보 (아래 blaze 패턴으로 위치 확인)")

    st.divider()
    st.subheader("1) 0차 회피 — blazed grating 단독 패턴")
    st.markdown("""
이 패턴을 SLM 에 띄우면 **빛 전체가 1차 방향으로 꺾인다.**
- 카메라에서 **안 꺾인 점(0차)** 과 **꺾인 빔(1차)** 위치를 확인
- 0차 자리에 빔블록 설치 → 이후 모든 측정에서 0차 오염 차단
- 사이드바의 "모든 패턴에 blaze 합성"을 켜두면 이후 패턴들도 같은 오프셋을 가짐
""")
    if st.button("blaze 패턴 생성", key="b_blaze"):
        ph = [("blaze_only", P.blazed(shape, period_x=blaze_px))]
        show_preview(ph, 1)
        items = P.finalize(ph, shape, canvas=canvas)   # blaze 자체 — 중복합성 X
        buf = io.BytesIO()
        Image.fromarray(items[0][1], "L").save(buf, format="PNG")
        st.download_button("⬇️ blaze_only.png", buf.getvalue(),
                           file_name="blaze_only.png", mime="image/png")

    st.divider()
    st.subheader("2) 카메라↔SLM 좌표 정합 — 프로브 스팟")
    st.markdown("""
주파수가 다른 blazed grating 들 = **1차 스팟이 화면 곳곳으로 이동.**
- 각 패턴마다 카메라에서 스팟 중심좌표 기록
- (SLM 주파수) ↔ (카메라 픽셀) 쌍들로 **affine 변환 행렬** 피팅
- 이 행렬은 Step 3 에서 카메라 이미지를 계산 격자로 리샘플할 때 사용
- 부수 효과: 위치별 스팟 모양 → **시야 전반 수차의 사전 정보**
""")
    pg = st.slider("프로브 격자 (n×n)", 3, 5, 3)
    if st.button("프로브 패턴 생성", key="b_probe"):
        ph = P.probe_patterns(shape, grid=pg)
        show_preview(ph)
        dl_zip(ph, "프로브 패턴", "probes.zip",
               manifest={"type": "probe", "grid": pg, "size": size}, key="d_probe")


# ═════════════════════════════════════════════════════
with tab2:
    st.header("Step 1 — 훈련 패턴 생성")
    st.markdown("""
Surrogate 가 배워야 할 미지수(감마·수차·조명)를 **직접 자극하는 패턴**을 섞는 게 핵심.

| 종류 | 권장 비율 | 역할 |
|---|---|---|
| GS 배열 POH | ~60% | U-Net 작전 지역과 같은 분포 |
| 순수 랜덤 위상 | ~20% | 다양성, 과적합 방지 |
| 캘리브레이션 (감마+프로브) | ~20% | 미지수 직접 식별 |
""")

    st.subheader("1) 감마 캘리브레이션 패턴")
    st.markdown("이진 격자(0 ↔ 회색값 g)를 g=0→255 스윕. 각 패턴의 **1차 회절 파워 P(g)** 를 "
                "기록하면 P ∝ sin²(Δφ(g)/2) 로 SLM 위상응답(감마곡선)이 역산된다.")
    nlev = st.slider("회색값 단계 수", 8, 32, 16)
    if st.button("감마 패턴 생성", key="b_gamma"):
        ph = P.gamma_calib_patterns(shape, n_levels=nlev)
        show_preview(ph)
        dl_zip(ph, "감마 캘리브레이션", "calib_gamma.zip",
               manifest={"type": "gamma", "n_levels": nlev, "size": size},
               key="d_gamma")

    st.divider()
    st.subheader("2) GS 배열 POH (메인 데이터)")
    st.markdown("무작위 트위저 배열(Square/Tri/Honeycomb/Random)의 GS POH. "
                "**브라우저에선 배치 단위로 생성** — 시드를 바꿔 여러 번 받으면 "
                "이름이 겹치지 않게 누적된다. (풀세트 2,000+장은 시드 0~9 × 200장 등)")
    c1, c2 = st.columns(2)
    n_gs = c1.slider("생성 장수", 10, 200, 50, step=10)
    seed_gs = c2.number_input("시드 (배치마다 바꿀 것)", 0, 999, 0)
    if st.button("GS POH 생성 (수십 초~수 분)", key="b_gs"):
        bar = st.progress(0.0, "GS 계산 중...")
        ph = P.gs_array_patterns(shape, n_gs, seed=int(seed_gs),
                                 progress=lambda k, n: bar.progress(k / n))
        bar.empty()
        show_preview(ph)
        dl_zip(ph, f"GS POH (seed={seed_gs})", f"gs_seed{seed_gs}.zip",
               manifest={"type": "gs", "n": n_gs, "seed": int(seed_gs),
                         "size": size}, key="d_gs")

    st.divider()
    st.subheader("3) 순수 랜덤 위상")
    n_rand = st.slider("생성 장수 ", 10, 500, 100, step=10)
    seed_rand = st.number_input("시드  (배치마다 바꿀 것)", 0, 999, 0)
    if st.button("랜덤 위상 생성", key="b_rand"):
        ph = P.random_phase_patterns(shape, n_rand, seed=int(seed_rand))
        show_preview(ph)
        dl_zip(ph, f"랜덤 위상 (seed={seed_rand})", f"rand_seed{seed_rand}.zip",
               manifest={"type": "random", "n": n_rand, "seed": int(seed_rand),
                         "size": size}, key="d_rand")

    st.divider()
    st.subheader("4) 기준(reference) 패턴 — 필수 1장")
    st.markdown("**같은 패턴을 50장마다 반복 측정**해 광학계 드리프트를 감시한다. "
                "촬영 스크립트가 `reference.png` 라는 이름을 찾으므로 이름 변경 금지.")
    if st.button("기준 패턴 생성", key="b_ref"):
        ph = P.reference_pattern(shape)
        show_preview(ph, 1)
        items = P.finalize(ph, shape, blaze_x=bx, canvas=canvas)
        buf = io.BytesIO()
        Image.fromarray(items[0][1], "L").save(buf, format="PNG")
        st.download_button("⬇️ reference.png", buf.getvalue(),
                           file_name="reference.png", mime="image/png",
                           key="d_ref")

    st.success("생성한 zip 들을 전부 풀어 **한 폴더(`patterns/`)** 에 모으세요. "
               "reference.png 포함. → Step 2 로", icon="📁")


# ═════════════════════════════════════════════════════
with tab3:
    st.header("Step 2 — 자동 촬영 (무인, 밤샘 가능)")
    st.markdown("""
`lab/collect_data.py` 가 전 과정을 자동화한다. **실험실에서 채울 건 함수 2개뿐:**

| 함수 | 할 일 | 참고 |
|---|---|---|
| `display_phase(u8)` | SLM 에 8-bit 이미지 띄우기 | HOLOEYE SLM Display SDK `showData()` |
| `grab_frame(exp_ms)` | 노출 지정해 한 프레임 캡처 | pylablib / pypylon / 제조사 SDK |

스크립트가 자동으로 하는 일:
1. **다크 프레임** 촬영 (시작 시 빔 차단 1회)
2. 패턴마다: SLM 표시 → **200ms 정착 대기** → **HDR 3노출 × 4프레임 평균**
   (스팟은 짧은 노출, 배경은 긴 노출이 담당 — 포화 픽셀 자동 제외)
3. **50장마다 기준 패턴** 삽입 → 드리프트 기록
4. 패턴당 `.npz` 저장 (보낸 위상 + 처리된 강도 + 메타데이터)

**소요 시간**: 패턴당 ~1.5초 → 2,000장 ≈ 1시간, 5,000장 ≈ 2.5시간.
""")
    with open("lab/collect_data.py", encoding="utf-8") as f:
        src = f.read()
    st.download_button("⬇️ collect_data.py 다운로드", src,
                       file_name="collect_data.py", mime="text/x-python",
                       use_container_width=True)
    with st.expander("스크립트 미리보기"):
        st.code(src, language="python")

    st.warning("**촬영 중 금지**: 광학 테이블 접촉, 조명 변화, SLM 설정 변경. "
               "드리프트 기준 패턴이 5% 넘게 변하면 그 구간은 재측정.", icon="🚫")


# ═════════════════════════════════════════════════════
with tab4:
    st.header("Step 3 — 학습 & DAgger 루프")
    st.markdown("""
`lab/train_real_surrogate.py` 하나로 끝난다:

```bash
PYTHONPATH=. python lab/train_real_surrogate.py --data measured --size {size}
```

하는 일:
1. **드리프트 점검** — 기준 패턴들의 시간 변화 출력 (5% 초과 구간 경고)
2. **Surrogate ← 실측 (φ,I)** — 시뮬레이터(`SimOptics`) 자리를 진짜 데이터가 대체.
   감마·조명·수차가 *실제 광학계 값*으로 학습됨
3. **U-Net distill** — 타깃 배열은 합성(공짜), 손실은 실측-학습된 Surrogate 통과
4. `checkpoint_real.pt` 저장 → 기존 앱(`app_sdfp.py`)에서 경로만 바꿔 사용
5. **DAgger 패턴 200장 자동 내보내기** (`patterns_dagger/`)

### 🔁 DAgger 루프 (성능을 진짜로 끌어올리는 단계)
```
U-Net 학습 → U-Net 출력 위상 200장을 SLM 에 띄워 측정 (collect_data.py 재사용)
          → 그 데이터를 measured/ 에 추가 → train_real_surrogate.py 재실행
          → 반복 (2~3 라운드면 수렴)
```
Surrogate 가 **U-Net 이 실제로 내놓는 위상 영역에서** 정확해져서,
"시뮬에선 좋은데 실제에선 다른" 격차(model exploitation)가 닫힌다.

### 최종 검증
- U-Net POH vs GS POH 를 실제 SLM 에 띄워 **실측 균일도(σ/μ)** 비교
- 스팟별 피크가 아니라 **2D 가우시안 피팅 진폭**으로 비교하면 논문급 분석
""".replace("{size}", str(size)))

    with open("lab/train_real_surrogate.py", encoding="utf-8") as f:
        src2 = f.read()
    st.download_button("⬇️ train_real_surrogate.py 다운로드", src2,
                       file_name="train_real_surrogate.py", mime="text/x-python",
                       use_container_width=True)
    with st.expander("스크립트 미리보기"):
        st.code(src2, language="python")

    st.success("여기까지 완료되면: `app_sdfp.py` 에서 checkpoint 경로를 "
               "`sdfp/checkpoint_real.pt` 로 바꿔 실측 기반 비교를 볼 수 있다.",
               icon="🏁")
