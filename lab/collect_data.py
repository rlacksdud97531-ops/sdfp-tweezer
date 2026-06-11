"""
collect_data.py — 실험실 데이터 수집 자동화 스크립트 (골격).

⚠️ 실험실에서 채워야 할 곳은 딱 2개:
    display_phase()  : SLM 에 8-bit 위상 이미지 띄우기 (HOLOEYE SDK)
    grab_frame()     : 카메라에서 노출시간 지정해 한 프레임 캡처

나머지(HDR 합성, 프레임 평균, 다크 빼기, 드리프트 기준 패턴 삽입, 저장)는
모두 구현돼 있음.

사용법:
    1. Streamlit 가이드(또는 sdfp.patterns)로 만든 패턴 PNG 들을 PATTERN_DIR 에 풀기
       (reference.png 포함)
    2. 아래 CONFIG 확인 → 두 드라이버 함수 채우기
    3. python lab/collect_data.py
    4. OUT_DIR 에 패턴당 .npz 1개씩 생성 → lab/train_real_surrogate.py 로 학습

출력 npz 포맷:
    phase_u8  : (H,W) uint8   — SLM 에 보낸 이미지
    intensity : (H,W) float32 — HDR 합성·평균·다크차감 후 강도 (상대 단위)
    meta      : dict (json)   — 시각, 노출, 기준패턴 여부 등
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────
# CONFIG — 실험에 맞게 수정
# ─────────────────────────────────────────────────────
PATTERN_DIR = Path("patterns")        # 패턴 PNG 폴더
OUT_DIR = Path("measured")            # 출력 폴더
SETTLE_MS = 200                       # SLM 액정 정착 대기 (ms)
EXPOSURES_MS = [0.5, 2.0, 8.0]        # HDR 노출 (짧은→긴; 카메라에 맞게)
N_AVG = 4                             # 노출당 평균 프레임 수 (샷노이즈 ↓)
REF_EVERY = 50                        # N 장마다 기준 패턴 삽입 (드리프트 감시)
SAT_LEVEL = 0.95                      # 이 비율 이상이면 포화로 간주 (max 대비)


# ─────────────────────────────────────────────────────
# ⚠️ 드라이버 — 실험실에서 채울 것
# ─────────────────────────────────────────────────────
def display_phase(u8_image: np.ndarray) -> None:
    """SLM 에 8-bit 위상 이미지를 띄운다.

    HOLOEYE PLUTO 예시 (SLM Display SDK):
        from holoeye import slmdisplaysdk
        slm = slmdisplaysdk.SLMInstance()      # 시작 시 1회
        slm.showData(u8_image)
    """
    raise NotImplementedError("SLM 드라이버를 여기에 연결하세요")


def grab_frame(exposure_ms: float) -> np.ndarray:
    """노출시간(ms)으로 카메라 한 프레임 캡처 → float64 (H,W).

    예시 (pylablib / pypylon / 제조사 SDK):
        cam.set_exposure(exposure_ms * 1e-3)
        return cam.snap().astype(np.float64)
    """
    raise NotImplementedError("카메라 드라이버를 여기에 연결하세요")


# ─────────────────────────────────────────────────────
# HDR 합성: 짧은 노출은 밝은 스팟, 긴 노출은 어두운 배경 담당
# ─────────────────────────────────────────────────────
def capture_hdr(dark_frames: dict) -> np.ndarray:
    """노출 스택 → 노출시간 정규화 후 비포화 픽셀만 합성."""
    acc = None
    wsum = None
    for exp in EXPOSURES_MS:
        frames = [grab_frame(exp) for _ in range(N_AVG)]
        f = np.mean(frames, axis=0) - dark_frames[exp]      # 다크 차감
        sat = np.max(frames) * SAT_LEVEL
        valid = (np.mean(frames, axis=0) < sat).astype(np.float64)
        rate = np.clip(f, 0, None) / exp                    # counts/ms
        if acc is None:
            acc = np.zeros_like(rate)
            wsum = np.zeros_like(rate)
        # 긴 노출일수록 SNR 좋음 → 노출시간 가중
        acc += valid * exp * rate
        wsum += valid * exp
    return acc / np.maximum(wsum, 1e-9)


def capture_darks() -> dict:
    """셔터 닫고(또는 빔 차단) 노출별 다크 프레임."""
    input("▶ 빔을 차단하고 Enter (다크 프레임 촬영)...")
    darks = {exp: np.mean([grab_frame(exp) for _ in range(N_AVG)], axis=0)
             for exp in EXPOSURES_MS}
    input("▶ 빔을 다시 열고 Enter...")
    return darks


# ─────────────────────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(exist_ok=True)
    files = sorted(p for p in PATTERN_DIR.glob("*.png") if p.stem != "reference")
    ref_path = PATTERN_DIR / "reference.png"
    assert ref_path.exists(), "reference.png 이 패턴 폴더에 필요합니다 (드리프트 감시용)"
    ref_u8 = np.array(Image.open(ref_path))

    print(f"패턴 {len(files)}장 + 기준패턴 {len(files)//REF_EVERY + 1}회 측정 예정")
    darks = capture_darks()

    def measure(u8, name, is_ref):
        display_phase(u8)
        time.sleep(SETTLE_MS / 1000)
        inten = capture_hdr(darks)
        meta = {"name": name, "t": time.time(), "is_reference": is_ref,
                "exposures_ms": EXPOSURES_MS, "n_avg": N_AVG}
        np.savez_compressed(OUT_DIR / f"{name}.npz",
                            phase_u8=u8.astype(np.uint8),
                            intensity=inten.astype(np.float32),
                            meta=json.dumps(meta))

    t0 = time.time()
    for k, f in enumerate(files):
        if k % REF_EVERY == 0:                       # 드리프트 기준 패턴
            measure(ref_u8, f"reference_{k:05d}", True)
        measure(np.array(Image.open(f)), f.stem, False)
        if k % 20 == 0:
            el = time.time() - t0
            eta = el / max(k, 1) * (len(files) - k)
            print(f"  {k}/{len(files)}  경과 {el/60:.1f}분  남음 ~{eta/60:.1f}분")

    measure(ref_u8, f"reference_{len(files):05d}", True)
    print(f"완료: {OUT_DIR} 에 저장됨. 다음: python lab/train_real_surrogate.py")


if __name__ == "__main__":
    main()
