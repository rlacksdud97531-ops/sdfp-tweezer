# SDFP — Surrogate-Distilled Fast Predictor for Optical Tweezer Arrays

Er/SU(8) 트위저 실험용 **POH(Phase-Only Hologram) 생성기**.
U-Net(수 ms 추론)과 physics-informed 미분가능 Surrogate(광학계 수차 보정)를 결합해,
짧은 결맞음 시간과 SU(N) 물리의 trap 균일도 요구를 동시에 잡는다.

## 앱

| 파일 | 내용 | 실행 |
|---|---|---|
| `app_sdfp.py` | GS vs WGS vs U-Net 비교 데모 (학습된 체크포인트 동봉) | `streamlit run app_sdfp.py` |
| `app_lab.py` | **실험 데이터 수집 step-by-step 가이드** + 패턴 생성/다운로드 | `streamlit run app_lab.py` |

## 구조

- `sdfp/` — 핵심 패키지 (시뮬레이터, 모델, 학습, 패턴 생성). 상세: [sdfp/README.md](sdfp/README.md)
- `lab/` — 실험실용 스크립트 (데이터 수집 골격, 실측 데이터 학습)

## 빠른 시작

```bash
pip install -r requirements.txt
streamlit run app_sdfp.py          # 데모 (재학습 불필요)
python -m sdfp.train --quick       # 직접 학습해보기
```

⚠️ 현재 결과는 시뮬레이션 광학계 기준 proof-of-concept. 실험 데이터 수집 절차는 `app_lab.py` 참고.
