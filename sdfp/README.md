# Surrogate-Distilled Fast Predictor (SDFP)

Er(¹⁶⁷Er) / SU(8) 트위저 실험용 **POH(Phase-Only Hologram) 생성기**.
짧은 결맞음 시간(속도)과 SU(N) 물리에 필요한 trap 균일도(품질)를 동시에 잡기 위해
**U-Net(빠른 추론)** 과 **미분가능 Surrogate(광학계 보정)** 를 결합했다.

> ⚠️ **실험 데이터가 아직 없음.** 그래서 "진짜 광학계" 자리에 수차·SLM 비선형성을
> 주입한 시뮬레이터(`SimOptics`)를 둔다. 실험 데이터가 생기면 `SimOptics` 대신
> 카메라로 측정한 `(φ, I)` 쌍으로 **Surrogate 만** 다시 학습하면 동일 구조로 작동한다.

## 파이프라인

```
[STAGE 1] 오프라인 — Surrogate 학습  (결맞음 제약 없음)
   무작위/GS 위상 φ → (진짜 광학계) → 강도 I  를 측정
   Surrogate 가 φ→I 를 흉내 → 광학계 캘리브레이션(감마/조명/수차)을 데이터로 역산
   · FFT 는 물리로 고정, 모르는 부분만 학습 (physics-informed)

[STAGE 2] 오프라인 — Predictor(U-Net) distill
   desired array → U-Net → φ → [frozen Surrogate] → I
   손실 = 스팟 기하평균 최대화 (효율+균일도) , 위상은 직접 지도하지 않음
   → 수차를 미리 보정한 위상을 예측하도록 학습

[ONLINE] 추론  (결맞음 제약 O)
   desired array → U-Net → POH (수 ms)
```

## 핵심 설계 결정

- **위상 비지도(label-free)**: phase-retrieval 해는 비유일/스펙클이라 픽셀단위 회귀가
  불가능 → 위상 라벨 대신 미분가능 전파를 통과한 *강도* 만 최적화.
- **기하평균 목적함수**: 에너지가 보존되므로 스팟 피크들의 기하평균을 키우면
  효율↑ 과 균일도↑ 를 동시에 얻는다 (어두운 스팟이 강하게 벌점받음).
- **2채널 atan2 위상 출력**: 2π wrapping 불연속 회피.
- **physics-informed Surrogate**: FFT 고정 + (감마 lookup, 동공 진폭, 정적 수차)만 학습.

## 파일

| 파일 | 역할 |
|---|---|
| `optics_sim.py` | 진짜 광학계 대역 시뮬레이터 (Zernike 수차 + 가우시안 조명 + SLM 감마) |
| `models.py` | `Surrogate`(미분가능 광학 모델) + `UNet`(강도→위상) |
| `gs.py` | GS / Weighted-GS (warmup·베이스라인) |
| `data.py` | 타깃 배열(Square/Triangular/Honeycomb/Random) 생성 |
| `metrics.py` | 비균일도(σ/μ), 효율 |
| `train.py` | 2-stage 학습 파이프라인 (CLI) |
| `infer.py` | 체크포인트 로드 + GS/WGS/U-Net 비교 |

## 사용법

```bash
# 학습 (빠른 데모 / 전체)
python -m sdfp.train --quick
python -m sdfp.train

# Streamlit 데모
streamlit run app_sdfp.py
```

## 실험 데이터가 생기면

`train.py` 의 STAGE 1 에서 `SimOptics.forward` 호출을 **카메라 측정 `(φ, I)`** 으로
교체하면 된다. STAGE 2 와 추론은 코드 변경 없이 그대로 사용.
