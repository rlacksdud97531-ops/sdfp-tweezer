"""
collect_data.py — 실험실 데이터 수집 자동화 스크립트.

⚠️ 실험실에서 채워야 할 곳은 딱 2개:
    display_phase()  : SLM 에 8-bit 위상 이미지 띄우기 (HOLOEYE SDK)
    grab_frame()     : 카메라에서 노출시간 지정해 한 프레임 캡처

나머지(SLM 임베딩, 카메라 crop, HDR 합성, 프레임 평균, 다크 빼기,
드리프트 기준 패턴 삽입, 저장)는 모두 구현돼 있음.

사용법:
    1. sdfp.patterns 또는 app_lab.py 로 512² 패턴 PNG 생성 → PATTERN_DIR 에 저장
       (reference.png 포함)
    2. 아래 CONFIG 확인 → 두 드라이버 함수 채우기
    3. python lab/collect_data.py
    4. OUT_DIR 에 패턴당 .npz 1개씩 생성 → lab/train_real_surrogate.py 로 학습

출력 npz 포맷:
    phase_u8  : (MODEL_SIZE, MODEL_SIZE) uint8   — 512² 위상 패턴 (SLM 임베딩 전)
    intensity : (MODEL_SIZE, MODEL_SIZE) float32 — 카메라 crop·resize 후 강도
    meta      : dict (json)
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
MODEL_SIZE  = 512              # 모델 해상도 (훈련 데이터 크기)
SLM_W, SLM_H = 1920, 1080     # HoloEye PLUTO 2.1 전체 해상도
CAM_W, CAM_H = 1288, 964      # 카메라 해상도

# 0차광 회피: SLM 중앙에서 옆으로 몇 픽셀 이동할지 (x, y)
# 지금처럼 옆에 올리고 있다면 실험적으로 맞게 조정
SLM_OFFSET_X = 0              # 양수 = 오른쪽
SLM_OFFSET_Y = 0              # 양수 = 아래쪽

PATTERN_DIR = Path("patterns")
OUT_DIR     = Path("measured")
SETTLE_MS   = 200
EXPOSURES_MS = [0.5, 2.0, 8.0]
N_AVG       = 4
REF_EVERY   = 50
SAT_LEVEL   = 0.95


# ─────────────────────────────────────────────────────
# ⚠️ 드라이버 — 실험실에서 채울 것
# ─────────────────────────────────────────────────────
def display_phase(slm_u8: np.ndarray) -> None:
    """SLM 전체(1920×1080) 8-bit 위상 이미지를 띄운다.

    HOLOEYE PLUTO 2.1 예시 (SLM Display SDK):
        from holoeye import slmdisplaysdk
        slm = slmdisplaysdk.SLMInstance()          # 시작 시 1회
        slm.showData(slm_u8)                       # (H, W) uint8
    """
    raise NotImplementedError("SLM 드라이버를 여기에 연결하세요")


def grab_frame(exposure_ms: float) -> np.ndarray:
    """노출시간(ms)으로 카메라 한 프레임 캡처 → float64 (CAM_H, CAM_W).

    FLIR PySpin 예시:
        cam.ExposureTime.SetValue(exposure_ms * 1e3)   # μs 단위
        cam.BeginAcquisition()
        img = cam.GetNextImage()
        frame = img.GetNDArray().astype(np.float64)
        img.Release()
        cam.EndAcquisition()
        return frame
    """
    raise NotImplementedError("카메라 드라이버를 여기에 연결하세요")


# ─────────────────────────────────────────────────────
# 해상도 변환 헬퍼
# ─────────────────────────────────────────────────────
def embed_in_slm(phase_u8: np.ndarray) -> np.ndarray:
    """512² 위상 패턴 → 1920×1080 SLM 이미지.

    패턴을 SLM 중앙(+offset)에 놓고 나머지는 0(DC, 빔 차단 역할).
    SLM_OFFSET_X/Y 를 조정해 0차광 위치에서 벗어나게 한다.
    """
    slm = np.zeros((SLM_H, SLM_W), dtype=np.uint8)
    h, w = phase_u8.shape[:2]

    # 중앙 + offset 위치 계산
    cy = (SLM_H - h) // 2 + SLM_OFFSET_Y
    cx = (SLM_W - w) // 2 + SLM_OFFSET_X
    cy = np.clip(cy, 0, SLM_H - h)
    cx = np.clip(cx, 0, SLM_W - w)

    slm[cy:cy + h, cx:cx + w] = phase_u8
    return slm


def crop_camera(frame: np.ndarray) -> np.ndarray:
    """카메라 1288×964 → 중앙 964² crop → 512² resize.

    카메라가 스팟 배열 전체를 담고 있다고 가정.
    스팟이 한쪽으로 치우쳐 있으면 cy/cx 를 조정할 것.
    """
    h, w = frame.shape[:2]
    side = min(h, w)           # 964
    cy, cx = h // 2, w // 2
    cropped = frame[cy - side//2 : cy + side//2,
                    cx - side//2 : cx + side//2]
    img = Image.fromarray(cropped.astype(np.float32), mode='F')
    resized = img.resize((MODEL_SIZE, MODEL_SIZE), Image.BILINEAR)
    return np.array(resized, dtype=np.float32)


# ─────────────────────────────────────────────────────
# HDR 합성
# ─────────────────────────────────────────────────────
def capture_hdr(dark_frames: dict) -> np.ndarray:
    """노출 스택 → 비포화 픽셀 가중합성 → (MODEL_SIZE, MODEL_SIZE) float32."""
    acc = None
    wsum = None
    for exp in EXPOSURES_MS:
        raw_frames = [grab_frame(exp) for _ in range(N_AVG)]
        f = np.mean(raw_frames, axis=0) - dark_frames[exp]
        sat = np.max(raw_frames) * SAT_LEVEL
        valid = (np.mean(raw_frames, axis=0) < sat).astype(np.float64)
        rate = np.clip(f, 0, None) / exp
        if acc is None:
            acc  = np.zeros_like(rate)
            wsum = np.zeros_like(rate)
        acc  += valid * exp * rate
        wsum += valid * exp
    raw = (acc / np.maximum(wsum, 1e-9)).astype(np.float32)
    return crop_camera(raw)   # → 512²


def capture_darks() -> dict:
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
    assert ref_path.exists(), "reference.png 이 패턴 폴더에 필요합니다"
    ref_u8 = np.array(Image.open(ref_path).resize((MODEL_SIZE, MODEL_SIZE)))

    print(f"해상도: 모델={MODEL_SIZE}²  SLM={SLM_W}×{SLM_H}  카메라={CAM_W}×{CAM_H}")
    print(f"패턴 {len(files)}장 + 기준패턴 {len(files)//REF_EVERY + 1}회 측정 예정")
    darks = capture_darks()

    def measure(phase_u8: np.ndarray, name: str, is_ref: bool):
        slm_img = embed_in_slm(phase_u8)      # 512² → 1920×1080
        display_phase(slm_img)
        time.sleep(SETTLE_MS / 1000)
        inten = capture_hdr(darks)             # 카메라 → 512²
        meta = {"name": name, "t": time.time(), "is_reference": is_ref,
                "model_size": MODEL_SIZE, "slm": [SLM_W, SLM_H],
                "exposures_ms": EXPOSURES_MS, "n_avg": N_AVG}
        np.savez_compressed(OUT_DIR / f"{name}.npz",
                            phase_u8=phase_u8.astype(np.uint8),   # 512² 원본
                            intensity=inten,                        # 512² crop
                            meta=json.dumps(meta))

    t0 = time.time()
    for k, f in enumerate(files):
        if k % REF_EVERY == 0:
            measure(ref_u8, f"reference_{k:05d}", True)
        u8 = np.array(Image.open(f).resize((MODEL_SIZE, MODEL_SIZE)))
        measure(u8, f.stem, False)
        if k % 20 == 0:
            el = time.time() - t0
            eta = el / max(k, 1) * (len(files) - k)
            print(f"  {k}/{len(files)}  경과 {el/60:.1f}분  남음 ~{eta/60:.1f}분")

    measure(ref_u8, f"reference_{len(files):05d}", True)
    print(f"완료: {OUT_DIR} 에 저장됨. 다음: python lab/train_real_surrogate.py")


if __name__ == "__main__":
    main()
