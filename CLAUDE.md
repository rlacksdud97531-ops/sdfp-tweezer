# SDFP Project — Claude Code Context

Er/SU(8) 광학 트위저 POH 생성기. U-Net + physics-informed Surrogate로 GS 대비 200배 낮은 비균일도 달성.

---

## 현재 상태 (2026-06-16 기준)

| 항목 | 내용 |
|---|---|
| **배포 모델** | `sdfp/checkpoint_256.pt` (Streamlit Cloud, refine=False) |
| **최고 모델** | `sdfp/checkpoint_best.pt` (로컬만, refine=True — 아직 push 안됨) |
| **Streamlit** | https://sdfp-tweezer.streamlit.app (또는 GitHub repo 확인) |
| **GPU** | RTX 4070, cuda:0, torch cu124 |

### checkpoint_best.pt 성능 (Square 4×4, spacing=24, 256²)
- Feedforward: nu=17.7%, eff=75.8%
- Refine×100: **nu=0.11%, eff=74.3%, 시간≈0.4s**
- vs GS: nu=35-54%, eff=85% (수천 iter 해도 SDFP 수준 불가)

---

## 핵심 파일

```
inner/
├── app_sdfp.py          # Streamlit 데모 (GS vs WGS vs SDFP 비교)
├── app_lab.py           # 실험 데이터 수집 가이드 (HoloEye + SpinView)
├── gs_sweep.json        # GS/WGS 수렴 스윕 사전계산 데이터
├── sdfp/
│   ├── checkpoint_256.pt    # 배포중인 모델 (refine=False)
│   ├── checkpoint_best.pt   # 찐최종 모델 (refine=True, 최고 성능)
│   ├── infer.py         # load(), refine_phase(), compare() — 추론 진입점
│   ├── models.py        # UNet (2ch→atan2→phase), Surrogate (MonotoneCurve+Zernike+CNN)
│   ├── train.py         # train_surrogate() / train_predictor() — 실험 데이터로 교체할 지점
│   ├── optics_sim.py    # SimOptics: Gaussian beam + Zernike aberration + gamma (시뮬만)
│   ├── data.py          # spot_positions(), target_intensity(), random_array()
│   ├── gs.py            # Gerchberg-Saxton / WGS (baseline)
│   └── metrics.py       # nonuniformity(), efficiency(), snr()
```

---

## 모델 구조

```
target array
    ↓
  UNet (2ch output → atan2 → phase)     ← base=64, 256²
    ↓
refine_phase() — Surrogate 통해 Adam 100스텝
    ↓
  POH (Phase-Only Hologram)

Surrogate: MonotoneCurve(γ) + A_raw(조명) + Wab(Zernike 수차) + CNN residual(refine=True)
```

**체크포인트 로드 방법** (`infer.py:load()`):
```python
from sdfp.infer import load
net, sur, shape, device = load("sdfp/checkpoint_best.pt")
# shape=(256,256), refine=True, base=64 자동 감지
```

---

## 훈련 분포 (현재 모델 기준)

| 파라미터 | 범위 |
|---|---|
| 해상도 | 256×256 |
| Array type | Square / Tri / Honeycomb / Random |
| Rows, Cols | 2–5 |
| Spacing | 22–49 px |
| UNet base | 64 |
| DAgger rounds | 3 |

**Sliders in app**: spacing 22–48px, rows/cols 2–5 (이 범위 바깥은 out-of-distribution)

---

## 실험 데이터 수집 (진짜 실험 시작됨!)

### 필요한 데이터
`(POH phase pattern, object-plane camera image)` 쌍

### 수집 파이프라인
```
HoloEye SLM ← POH 전송
    ↓
대물렌즈 → Object plane
    ↓
FLIR 카메라 (SpinView/PySpin) → 이미지 저장
```

### 코드에서 교체할 지점 (딱 2곳만!)
`sdfp/train.py` 안에서:
- **Line 135**: `optics.forward(phase)` → 실제 카메라 이미지
- **Line 261**: `optics.forward(phase)` → 실제 카메라 이미지

나머지 DAgger 루프, 손실함수, 모델 구조는 그대로 사용 가능.

### SpinView(PySpin) + HoloEye Python 연동 예시
```python
import PySpin          # FLIR 카메라 SDK
import holoeye         # HoloEye SLM SDK (또는 각 모델별 SDK 확인)

def real_forward(phase_np):
    """phase (256,256 numpy) → object-plane intensity (256,256 numpy)"""
    holoeye.display_phase(phase_np)   # SLM에 패턴 업로드
    time.sleep(0.05)                  # SLM 안정화 대기
    image = camera.grab_frame()       # 카메라 촬영
    return image
```

---

## 알려진 이슈

### spacing=32 efficiency 낮음
- 현상: nu=0.1% (훌륭), eff=48% (낮음)
- 원인: refine_phase optimizer가 낮은-피크 local minimum에 수렴
- 완화: `eff_w=0.001` 항 추가됨 (42%→48% 개선), 근본 해결은 아님
- 실제 SimOptics는 spacing=32에서 85% eff 가능 — surrogate 모델 한계

### refine_phase 손실함수
```python
# infer.py:refine_phase()
loss = (-log(peaks).mean()
        + uni_w * peaks.std() / peaks.mean()   # 균일도
        - eff_w * peaks.mean())                 # 피크 절댓값 (local min 탈출)
```
`uni_w=0.5`, `eff_w=0.001` (기본값)

---

## checkpoint_best.pt push 대기 중

`checkpoint_best.pt`는 아직 로컬에만 있음 (`git status`에 `??`로 표시). Streamlit Cloud 업데이트하려면:
```bash
# sdfp/checkpoint_256.pt를 checkpoint_best.pt로 교체 후 push
cp sdfp/checkpoint_best.pt sdfp/checkpoint_256.pt
git add sdfp/checkpoint_256.pt
git commit -m "Update to 찐최종 model (surrogate_refine=True)"
git push
```

---

## 환경

```bash
# 로컬 실행 (GPU)
cd inner/
streamlit run app_sdfp.py

# 패키지
pip install -r requirements.txt   # torch cu124 별도 설치 필요
# torch: pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Streamlit Cloud는 CPU only, Python 3.14 — `weights_only=True`, plain-text tick labels 필수 (이미 적용됨).

---

## 연락처 / 레포
- GitHub: `rlacksdud97531-ops/sdfp-tweezer`
- 담당자: rlacksdud97531@gmail.com
