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

import os
import sys
import json
import time
import atexit
from pathlib import Path

import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────
# HoloEye HEDS SDK 경로 설정 (v4.1) — collect_data_real.py 에서 검증된 설정
# ─────────────────────────────────────────────────────
_HEDS_ROOT = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.1.0"
sys.path.insert(0, os.path.join(_HEDS_ROOT, "api", "python"))
sys.path.insert(0, os.path.join(_HEDS_ROOT, "examples"))
os.environ.setdefault("HEDS_4_1_PYTHON", _HEDS_ROOT)
os.environ.setdefault("HEDS_4_0_PYTHON", _HEDS_ROOT)

import HEDS
from hedslib.heds_types import (
    HEDSERR_NoError, HEDSDTFMT_INT_U8, HEDSSHF_PresentAutomatic,
)
import PySpin

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
MODEL_SIZE  = 512              # 모델 해상도 (훈련 데이터 크기)
SLM_W, SLM_H = 1920, 1080     # HoloEye PLUTO 2.1 전체 해상도
CAM_W, CAM_H = 1288, 964      # 카메라 해상도
SLM_PRESELECT = "name:PLUTO"   # SLM 디스플레이 지정 (monitor 1 = HOLOEYE PLUTO-2.1)
GAIN_DB     = 0.0              # 카메라 게인 (노이즈 최소)
# 512² 패턴을 SLM 에 표시할 때 확대 배율. 2 = 1024px 슈퍼픽셀
# (재구성을 회절주기 중앙에 모아 복제차수를 화면 밖으로 밀어냄 → 깨끗한 단일 배열).
# 구경이 커져 빛이 세지므로 카메라 앞 ND OD1.0(10×) 필요.
SLM_UPSCALE = 2

# 0차광 회피: SLM 중앙에서 옆으로 몇 픽셀 이동할지 (x, y)
# 지금처럼 옆에 올리고 있다면 실험적으로 맞게 조정
SLM_OFFSET_X = 0              # 양수 = 오른쪽
SLM_OFFSET_Y = 0              # 양수 = 아래쪽

PATTERN_DIR = Path(os.environ.get("PATTERN_DIR", r"C:\tweezer_data\patterns512"))
OUT_DIR     = Path(os.environ.get("OUT_DIR", r"C:\tweezer_data\measured512"))
# crop: blaze 로 좌측 이동된 배열 중심에 맞춤 (파일럿에서 측정). 정사각 crop → 512² resize.
CROP_CX     = 258             # 배열 중심 x (카메라 px, blaze=5 파일럿 측정)
CROP_CY     = 410             # 배열 중심 y (카메라 px)
CROP_SIZE   = 520             # crop 한 변 (px). 가장 큰 배열 + 여유 포함, 0차광 줄 제외
SETTLE_MS   = 200
# 카메라 최소노출 ≈46µs. 밝은(점 적은) 패턴은 0.5ms 에서도 포화되므로 0.1ms 추가.
# 12ms 는 점 많은(어두운) 패턴/약신호 dynamic range 보강. HDR 가 노출별 비포화 픽셀만 가중합성.
EXPOSURES_MS = [0.1, 0.5, 2.0, 12.0]
N_AVG       = 4
REF_EVERY   = 50
SAT_LEVEL   = 0.95


# ─────────────────────────────────────────────────────
# ⚠️ 드라이버 — 실험실에서 채울 것
# ─────────────────────────────────────────────────────
# 하드웨어 핸들 (지연 초기화, collect_data_real.py 에서 검증된 호출)
_slm = None
_cam = None
_system = None
_cam_emin = 46.0
_cam_emax = 32754.0
_last_exposure_us = None


def _ensure_slm():
    global _slm
    if _slm is not None:
        return _slm
    err = HEDS.SDK.Init(4, 1)
    assert err == HEDSERR_NoError, f"HEDS init: {HEDS.SDK.ErrorString(err)}"
    # preview 창 끔(주 모니터에 GUI 안 뜨게), PLUTO 디스플레이 명시 지정
    _slm = HEDS.SLM.Init(SLM_PRESELECT, False, 0.0)
    assert _slm.errorCode() == HEDSERR_NoError, \
        f"SLM init: {HEDS.SDK.ErrorString(_slm.errorCode())}"
    print(f"[SLM] 초기화 완료 (preselect='{SLM_PRESELECT}')")
    return _slm


def _ensure_cam():
    global _cam, _system, _cam_emin, _cam_emax
    if _cam is not None:
        return _cam
    _system = PySpin.System.GetInstance()
    cam_list = _system.GetCameras()
    assert cam_list.GetSize() > 0, "카메라가 감지되지 않았습니다 (SpinView 종료했나요?)"
    _cam = cam_list[0]
    _cam.Init()
    cam_list.Clear()
    _cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    _cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    _cam.Gain.SetValue(GAIN_DB)
    _cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    _cam_emin = float(_cam.ExposureTime.GetMin())
    _cam_emax = float(_cam.ExposureTime.GetMax())
    _cam.BeginAcquisition()
    print(f"[CAM] 초기화 완료  게인={GAIN_DB}dB  노출범위=[{_cam_emin:.0f},{_cam_emax:.0f}]µs")
    return _cam


def _shutdown():
    global _cam, _system, _slm
    try:
        if _cam is not None:
            _cam.EndAcquisition(); _cam.DeInit(); del _cam; _cam = None
        if _system is not None:
            _system.ReleaseInstance(); _system = None
    except Exception:
        pass
    try:
        if _slm is not None:
            HEDS.SDK.Close(); _slm = None
    except Exception:
        pass

atexit.register(_shutdown)


def display_phase(slm_u8: np.ndarray) -> None:
    """SLM 전체(1920×1080) 8-bit 위상 이미지를 띄운다 (HOLOEYE HEDS SDK)."""
    slm = _ensure_slm()
    canvas = np.ascontiguousarray(slm_u8, dtype=np.uint8)
    data = HEDS.SLMDataField(SLM_W, SLM_H, HEDSDTFMT_INT_U8,
                             HEDSSHF_PresentAutomatic)
    data._values[:, :, 0] = canvas          # (H, W, 1) numpy ubyte
    err = slm.showImageData(data)
    assert err == HEDSERR_NoError, f"SLM show: {HEDS.SDK.ErrorString(err)}"


def grab_frame(exposure_ms: float) -> np.ndarray:
    """노출시간(ms)으로 카메라 한 프레임 캡처 → float64 (CAM_H, CAM_W)."""
    global _last_exposure_us
    cam = _ensure_cam()
    us = float(np.clip(exposure_ms * 1e3, _cam_emin, _cam_emax))
    if _last_exposure_us != us:
        cam.ExposureTime.SetValue(us)
        _last_exposure_us = us
        # 노출 변경 직후 한 프레임은 이전 노출일 수 있으니 버림
        stale = cam.GetNextImage(2000); stale.Release()
    img = cam.GetNextImage(2000)
    if img.IsIncomplete():
        img.Release()
        raise RuntimeError("불완전한 이미지 수신")
    arr = img.GetNDArray().astype(np.float64)
    img.Release()
    return arr


# ─────────────────────────────────────────────────────
# 해상도 변환 헬퍼
# ─────────────────────────────────────────────────────
def embed_in_slm(phase_u8: np.ndarray) -> np.ndarray:
    """512² 위상 패턴 → 1920×1080 SLM 이미지.

    패턴을 SLM_UPSCALE 배 확대(nearest, 위상 wrap 보존) 후 SLM 중앙(+offset)에 배치.
    나머지는 0. 0차광 회피는 패턴에 합성된 blaze 로 한다(SLM_OFFSET 아님).
    """
    import cv2
    if SLM_UPSCALE and SLM_UPSCALE != 1:
        h0, w0 = phase_u8.shape[:2]
        phase_u8 = cv2.resize(phase_u8, (w0 * SLM_UPSCALE, h0 * SLM_UPSCALE),
                              interpolation=cv2.INTER_NEAREST)
    slm = np.zeros((SLM_H, SLM_W), dtype=np.uint8)
    h, w = phase_u8.shape[:2]
    h = min(h, SLM_H); w = min(w, SLM_W)
    phase_u8 = phase_u8[:h, :w]

    cy = (SLM_H - h) // 2 + SLM_OFFSET_Y
    cx = (SLM_W - w) // 2 + SLM_OFFSET_X
    cy = int(np.clip(cy, 0, SLM_H - h))
    cx = int(np.clip(cx, 0, SLM_W - w))

    slm[cy:cy + h, cx:cx + w] = phase_u8
    return slm


def crop_camera(frame: np.ndarray) -> np.ndarray:
    """blaze 로 좌측 이동된 배열 중심(CROP_CX,CY)을 기준으로 CROP_SIZE 정사각 crop
    → 512² resize. 프레임 밖은 0 패딩. 정사각→정사각이라 aspect 왜곡 없음."""
    h, w = frame.shape[:2]
    half = CROP_SIZE // 2
    box = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float64)
    y0, x0 = CROP_CY - half, CROP_CX - half           # 원하는 crop 좌상단 (음수 가능)
    sy0, sx0 = max(0, y0), max(0, x0)                  # 실제 프레임 내 영역
    sy1, sx1 = min(h, y0 + CROP_SIZE), min(w, x0 + CROP_SIZE)
    box[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    img = Image.fromarray(box.astype(np.float32), mode='F')
    return np.array(img.resize((MODEL_SIZE, MODEL_SIZE), Image.BILINEAR), dtype=np.float32)


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in PATTERN_DIR.glob("*.png") if p.stem != "reference")
    ref_path = PATTERN_DIR / "reference.png"
    assert ref_path.exists(), "reference.png 이 패턴 폴더에 필요합니다"
    ref_u8 = np.array(Image.open(ref_path).resize((MODEL_SIZE, MODEL_SIZE)))

    # 패턴별 기하 정보 (좌표/타입/spacing) — 생성 시 만든 manifest.json
    mpath = PATTERN_DIR / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    if not manifest:
        print("⚠️ manifest.json 없음 — meta 에 좌표가 안 들어갑니다")

    print(f"해상도: 모델={MODEL_SIZE}²  SLM={SLM_W}×{SLM_H}  카메라={CAM_W}×{CAM_H}")
    print(f"패턴 {len(files)}장 + 기준패턴 {len(files)//REF_EVERY + 1}회 측정 예정")
    print(f"HDR 노출 {EXPOSURES_MS}ms  crop=({CROP_CX},{CROP_CY})/{CROP_SIZE}px  upscale={SLM_UPSCALE}×")
    darks = capture_darks()

    def measure(phase_u8: np.ndarray, name: str, is_ref: bool, geo_key: str):
        slm_img = embed_in_slm(phase_u8)      # 512² → 1920×1080
        display_phase(slm_img)
        time.sleep(SETTLE_MS / 1000)
        inten = capture_hdr(darks)             # 카메라 → 512²
        meta = {"name": name, "t": time.time(), "is_reference": is_ref,
                "model_size": MODEL_SIZE, "slm": [SLM_W, SLM_H],
                "exposures_ms": EXPOSURES_MS, "n_avg": N_AVG,
                "slm_upscale": SLM_UPSCALE,
                "crop": {"cx": CROP_CX, "cy": CROP_CY, "size": CROP_SIZE}}
        meta.update(manifest.get(geo_key, {}))   # array_type, spacing_actual, spots_xy
        np.savez_compressed(OUT_DIR / f"{name}.npz",
                            phase_u8=phase_u8.astype(np.uint8),   # 512² 원본
                            intensity=inten,                        # 512² crop
                            meta=json.dumps(meta))

    t0 = time.time()
    for k, f in enumerate(files):
        if k % REF_EVERY == 0:
            measure(ref_u8, f"reference_{k:05d}", True, "reference")
        u8 = np.array(Image.open(f).resize((MODEL_SIZE, MODEL_SIZE)))
        measure(u8, f.stem, False, f.stem)
        if k % 20 == 0:
            el = time.time() - t0
            eta = el / max(k, 1) * (len(files) - k)
            print(f"  {k}/{len(files)}  경과 {el/60:.1f}분  남음 ~{eta/60:.1f}분")

    measure(ref_u8, f"reference_{len(files):05d}", True, "reference")
    print(f"완료: {OUT_DIR} 에 저장됨. 다음: measured/ 를 GPU 컴퓨터로 옮기세요.")


if __name__ == "__main__":
    main()
