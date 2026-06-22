"""
collect_data_real.py — 실험실 자동 데이터 수집 (5000쌍)
실행 환경: miniconda slmsuite env
    C:\\Users\\jolab\\miniconda3\\envs\\slmsuite\\python.exe lab/collect_data_real.py

HW 구성:
    SLM  : HOLOEYE PLUTO (1920×1080), HEDS Python SDK v4.1
    카메라: Teledyne (Spinnaker/PySpin)

출력:
    measured/XXXXXX.npz  — phase_u8(256,256), intensity(H_cam,W_cam), meta(json)
    measured/darks.npz   — 다크 프레임 캐시
"""

import sys, os, json, time, argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────
# HoloEye HEDS SDK 경로 설정 (v4.1)
# ──────────────────────────────────────────────────────────────────
_HEDS_ROOT = r"C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.1.0"
sys.path.insert(0, os.path.join(_HEDS_ROOT, "api", "python"))
sys.path.insert(0, os.path.join(_HEDS_ROOT, "examples"))
os.environ.setdefault("HEDS_4_1_PYTHON", _HEDS_ROOT)
# slmsuite holoeye.py 가 HEDS_4_0_PYTHON 을 참조하므로 aliasing
os.environ.setdefault("HEDS_4_0_PYTHON", _HEDS_ROOT)

import HEDS
from hedslib.heds_types import (
    HEDSERR_NoError, HEDSDTFMT_INT_U8, HEDSSHF_PresentAutomatic,
)
import PySpin

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────
SLM_H, SLM_W    = 1080, 1920      # HOLOEYE PLUTO 해상도
SLM_PRESELECT   = "name:PLUTO"     # SLM 디스플레이 지정 (monitor 1 = HOLOEYE PLUTO-2.1)
HOL_SIZE        = 256              # 홀로그램 크기 (모델 입력 해상도)
HOL_FILL        = 1080             # SLM 에 표시할 때 확대 크기 (회절 효율↑). 0=확대안함(중앙패치)
OUT_DIR         = Path(r"C:\tweezer_data\measured")  # OneDrive 밖 (동기화/용량 회피)
N_TOTAL         = 5000             # 수집 목표 쌍 수
SETTLE_S        = 0.08             # SLM 액정 정착 대기 (s)
N_AVG           = 3                # 카메라 평균 프레임 수
REF_EVERY       = 100              # 기준 패턴 삽입 주기 (드리프트 감시)
EXPOSURE_US     = None             # None → 패턴마다 자동노출, 값 지정 시 고정 (μs)
AUTO_EXPOSURE   = True             # 패턴마다 노출 자동 조절 (점 개수별 밝기 차이 보정)
AE_TARGET       = 0.75             # 트랩 최댓값 목표 (×255 ≈ 191)
AE_START_US     = 150.0            # 자동노출 시작점 (μs)
AE_MAX_US       = 20000.0          # 자동노출 상한 (μs). 점 많은(어두운) 패턴 수용
ZEROTH_GUARD    = 120              # 중앙 0차광 좌측 경계 여유 (px) — 이 우측은 노출평가 제외
GAIN_DB         = 0.0              # 카메라 게인 (dB)
GS_ITER         = 30               # GS 반복 횟수 (충분한 다양성 + 속도)
RANDOM_FRAC     = 0.3              # 완전 랜덤 위상 비율 (GS 아닌 것)
# Blazed grating carrier — 트랩 배열(1차광)을 0차광/세로줄 십자에서 옆으로 밀어냄.
# 단위: 홀로그램(256px) 전체에 걸친 위상 주기 수. x=좌우 이동, y=상하 이동.
# 세로줄이 세로 방향이므로 x 로 밀어 좌우로 분리. --preview 로 값 튜닝.
CARRIER_CYC_X   = 70.0             # x 방향 캐리어 주기 수 (배열을 0차광 옆으로 분리)
CARRIER_CYC_Y   = 0.0              # y 방향 캐리어 주기 수
S_MAX_FFT       = 85               # 배열 최대 폭 (256-px 평면 기준). 초과 시 fit_extent 로 축소

# ──────────────────────────────────────────────────────────────────
# 위상 생성 (numpy only, torch 불필요)
# ──────────────────────────────────────────────────────────────────
def spot_positions(array_type, rows, cols, spacing, rng):
    """(2,N) 스팟 좌표, 중심 = (HOL_SIZE//2, HOL_SIZE//2)."""
    H = W = HOL_SIZE
    cx, cy = W // 2, H // 2
    pts = []
    if array_type == "Square":
        for i in range(rows):
            for j in range(cols):
                pts.append([cx + (j - (cols-1)/2)*spacing,
                             cy + (i - (rows-1)/2)*spacing])
    elif array_type == "Triangular":
        rh = spacing * np.sqrt(3) / 2
        for i in range(rows):
            off = spacing/2 if i % 2 else 0
            for j in range(cols):
                pts.append([cx + (j - (cols-1)/2)*spacing + off,
                             cy + (i - (rows-1)/2)*rh])
    elif array_type == "Honeycomb":
        pts.append([cx, cy])
        rings = int(rng.integers(1, 3))
        for ring in range(1, rings+1):
            for d in range(6):
                a = np.radians(d*60)
                dx, dy = np.cos(a)*spacing*ring, np.sin(a)*spacing*ring
                for step in range(ring):
                    ma = np.radians((d+2)*60)
                    pts.append([cx + dx + np.cos(ma)*spacing*step,
                                 cy + dy + np.sin(ma)*spacing*step])
    elif array_type == "Random":
        margin = 18
        n = int(rng.integers(4, 16))
        chosen = []
        for _ in range(2000):
            if len(chosen) >= n:
                break
            x = float(rng.integers(margin, W-margin))
            y = float(rng.integers(margin, H-margin))
            if all((x-px)**2+(y-py)**2 >= (spacing*0.7)**2 for px,py in chosen):
                chosen.append((x, y))
        pts = list(chosen)
    return np.array(pts, dtype=np.float64).T  # (2,N)


def target_intensity(positions):
    """가우시안 스팟 목표 강도 (H,W) float64, 합=1."""
    H = W = HOL_SIZE
    sig = 1.2 * H / 128.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    img = np.zeros((H, W))
    for x, y in zip(positions[0], positions[1]):
        img += np.exp(-((xx-x)**2+(yy-y)**2)/(2*sig**2))
    img /= img.sum() + 1e-12
    return img


def gs_phase(target, n_iter=GS_ITER, rng=None):
    """Gerchberg-Saxton (numpy FFT). target: (H,W) float, 반환: (H,W) uint8 [0,255]."""
    if rng is None:
        rng = np.random.default_rng()
    H, W = target.shape
    amp_t = np.sqrt(np.clip(target, 0, None))
    amp_s = np.ones((H, W))   # 균일 조명 가정
    phase = rng.uniform(0, 2*np.pi, (H, W))
    field = amp_s * np.exp(1j * phase)
    for _ in range(n_iter):
        F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
        F = amp_t * np.exp(1j * np.angle(F))
        field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F)))
        field = amp_s * np.exp(1j * np.angle(field))
    phase_rad = np.angle(field) % (2 * np.pi)
    return (phase_rad / (2 * np.pi) * 255).astype(np.uint8)


_CARRIER_CACHE = {}

def add_carrier(phase_u8, cyc_x=None, cyc_y=None):
    """Blazed grating 캐리어를 더해 1차광(트랩 배열)을 0차광에서 분리.
    phase_u8 [0,255] 에 선형 위상 경사를 더하고 256 으로 wrap.
    cyc_x/cyc_y: 홀로그램 전폭에 걸친 위상 주기 수 (None 이면 전역 설정 사용)."""
    cx = CARRIER_CYC_X if cyc_x is None else cyc_x
    cy = CARRIER_CYC_Y if cyc_y is None else cyc_y
    if cx == 0 and cy == 0:
        return phase_u8
    key = (phase_u8.shape, cx, cy)
    carrier = _CARRIER_CACHE.get(key)
    if carrier is None:
        H, W = phase_u8.shape
        yy, xx = np.mgrid[0:H, 0:W]
        ramp = cx * xx / W + cy * yy / H        # cycles
        carrier = np.mod(ramp * 256.0, 256.0)   # u8 단위 위상
        _CARRIER_CACHE[key] = carrier
    return np.mod(phase_u8.astype(np.int32) + carrier, 256).astype(np.uint8)


def random_phase(rng):
    """완전 랜덤 위상 패턴 (Surrogate 훈련 다양성 ↑)."""
    return rng.integers(0, 256, (HOL_SIZE, HOL_SIZE), dtype=np.uint8)


def fit_extent(pos):
    """배열이 너무 크면 모양 유지한 채 중심 기준으로 축소해 최대 폭 ≤ S_MAX_FFT.
    carrier 로 0차광 옆으로 밀었을 때 프레임 안에 들어오고 십자와 안 겹치게 함."""
    if pos.shape[1] == 0:
        return pos
    cx, cy = HOL_SIZE / 2.0, HOL_SIZE / 2.0
    ext = max(pos[0].max() - pos[0].min(), pos[1].max() - pos[1].min())
    if ext > S_MAX_FFT:
        scale = S_MAX_FFT / ext
        pos = pos.copy()
        pos[0] = (pos[0] - cx) * scale + cx
        pos[1] = (pos[1] - cy) * scale + cy
    return pos


def make_phase(rng):
    """5000장 분량의 다양한 위상 패턴 1장 생성."""
    if rng.random() < RANDOM_FRAC:
        return random_phase(rng), None

    atype = rng.choice(["Square", "Triangular", "Honeycomb", "Random"])
    s = HOL_SIZE / 128.0
    spacing = int(rng.integers(round(11*s), round(25*s)))
    if atype in ("Square", "Triangular"):
        rows = int(rng.integers(2, 6))
        cols = int(rng.integers(2, 6))
    else:
        rows, cols = 3, 3  # Honeycomb/Random 은 내부에서 무시
    pos = spot_positions(atype, rows, cols, spacing, rng)
    if pos.shape[1] == 0:
        return random_phase(rng), None
    pos = fit_extent(pos)
    tgt = target_intensity(pos)
    phase_u8 = add_carrier(gs_phase(tgt, rng=rng))
    meta_extra = {"array_type": atype, "spacing": spacing,
                  "rows": rows, "cols": cols,
                  "carrier_x": CARRIER_CYC_X, "carrier_y": CARRIER_CYC_Y}
    return phase_u8, meta_extra


def make_reference_phase(rng):
    """고정 기준 패턴 (4×4 Square, spacing=24)."""
    pos = fit_extent(spot_positions("Square", 4, 4, 24, rng))
    tgt = target_intensity(pos)
    return add_carrier(gs_phase(tgt, n_iter=60, rng=rng))


def make_sparse_phase(rng):
    """최악(가장 밝은) 케이스: 2×2 = 4점. 점당 빛이 가장 몰려 포화 위험 최대."""
    pos = fit_extent(spot_positions("Square", 2, 2, int(24 * HOL_SIZE / 128.0), rng))
    tgt = target_intensity(pos)
    return add_carrier(gs_phase(tgt, n_iter=60, rng=rng))


def make_widest_phase(rng):
    """최대 크기 케이스: 5×5, spacing 49 → fit_extent 로 축소됨. 프레임 수용 확인용."""
    pos = fit_extent(spot_positions("Square", 5, 5, int(24.5 * HOL_SIZE / 128.0), rng))
    tgt = target_intensity(pos)
    return add_carrier(gs_phase(tgt, n_iter=60, rng=rng))


# ──────────────────────────────────────────────────────────────────
# SLM 드라이버
# ──────────────────────────────────────────────────────────────────
class SLMController:
    def __init__(self):
        err = HEDS.SDK.Init(4, 1)
        assert err == HEDSERR_NoError, f"HEDS init: {HEDS.SDK.ErrorString(err)}"
        # preselect=SLM_PRESELECT 로 PLUTO 디스플레이를 명시 지정, preview 창은 끔
        # (preview 창이 켜지면 주 모니터/작업 화면에 미리보기 GUI 가 뜸).
        self.slm = HEDS.SLM.Init(SLM_PRESELECT, False, 0.0)
        assert self.slm.errorCode() == HEDSERR_NoError, \
            f"SLM init: {HEDS.SDK.ErrorString(self.slm.errorCode())}"
        print(f"[SLM] 초기화 완료 (preselect='{SLM_PRESELECT}')")

    def show(self, phase_u8: np.ndarray):
        """256×256 uint8 위상을 SLM 에 표시.
        HOL_FILL>0 이면 그 크기로 확대해 SLM 조명영역을 채움(회절 효율↑).
        nearest-neighbor 확대로 위상 wrap(0/255) 경계를 보존."""
        import cv2
        if HOL_FILL and HOL_FILL > 0:
            fill = min(HOL_FILL, SLM_H, SLM_W)
            patt = cv2.resize(phase_u8, (fill, fill),
                              interpolation=cv2.INTER_NEAREST)
        else:
            fill = HOL_SIZE
            patt = phase_u8
        canvas = np.zeros((SLM_H, SLM_W), dtype=np.uint8)
        r0 = (SLM_H - fill) // 2
        c0 = (SLM_W - fill) // 2
        canvas[r0:r0+fill, c0:c0+fill] = patt

        data = HEDS.SLMDataField(SLM_W, SLM_H, HEDSDTFMT_INT_U8,
                                  HEDSSHF_PresentAutomatic)
        # _values shape: (H, W, 1) numpy ubyte
        data._values[:, :, 0] = canvas
        err = self.slm.showImageData(data)
        assert err == HEDSERR_NoError, f"SLM show: {HEDS.SDK.ErrorString(err)}"

    def close(self):
        HEDS.SDK.Close()


# ──────────────────────────────────────────────────────────────────
# 카메라 드라이버 (PySpin / Teledyne Spinnaker)
# ──────────────────────────────────────────────────────────────────
class CameraController:
    def __init__(self, exposure_us=None, gain_db=0.0):
        self.system = PySpin.System.GetInstance()
        cam_list = self.system.GetCameras()
        assert cam_list.GetSize() > 0, "카메라가 감지되지 않았습니다"
        self.cam = cam_list[0]
        self.cam.Init()
        cam_list.Clear()

        # 획득 모드 설정
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

        # 게인 설정
        self.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        self.cam.Gain.SetValue(gain_db)

        # 노출 범위 파악
        self._emin = float(self.cam.ExposureTime.GetMin())
        self._emax = min(float(self.cam.ExposureTime.GetMax()), AE_MAX_US)

        # 노출 설정
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        if exposure_us is not None:
            self._exposure_us = float(np.clip(exposure_us, self._emin, self._emax))
        else:
            self._exposure_us = float(np.clip(AE_START_US, self._emin, self._emax))
        self.cam.ExposureTime.SetValue(self._exposure_us)

        self.cam.BeginAcquisition()
        print(f"[CAM] 초기화 완료  노출={self._exposure_us:.1f}μs  "
              f"게인={gain_db}dB  노출범위=[{self._emin:.0f},{self._emax:.0f}]μs")

    @property
    def exposure_us(self):
        return self._exposure_us

    def set_exposure(self, us):
        self._exposure_us = float(np.clip(us, self._emin, self._emax))
        self.cam.ExposureTime.SetValue(self._exposure_us)

    def grab(self) -> np.ndarray:
        """단일 프레임 캡처 → float64 (H,W)."""
        img_raw = self.cam.GetNextImage(2000)
        if img_raw.IsIncomplete():
            img_raw.Release()
            raise RuntimeError("불완전한 이미지 수신")
        arr = img_raw.GetNDArray().astype(np.float64)
        img_raw.Release()
        return arr

    def grab_avg(self, n=N_AVG) -> np.ndarray:
        """n 프레임 평균."""
        frames = [self.grab() for _ in range(n)]
        return np.mean(frames, axis=0)

    def auto_expose(self, target_frac=AE_TARGET, n_iter=7):
        """트랩 영역(0차광 제외)의 최댓값이 target_frac*255 가 되도록 노출 조절.
        carrier 로 배열이 좌측에 있으므로 중앙 0차광 우측을 제외한 좌측 영역만 평가.
        반환: 최종 노출(μs)."""
        for _ in range(n_iter):
            f = self.grab()
            H, W = f.shape
            col = max(1, W // 2 - ZEROTH_GUARD)   # 0차광 좌측 경계
            region = f[:, :col]
            mx = float(region.max())
            if mx < 1.0:
                mx = 1.0
            target = target_frac * 255.0
            new = self._exposure_us * target / mx
            # 한 스텝 변화 제한 (진동 방지). 어두운 패턴이 빨리 오르도록 5× 까지 허용.
            new = float(np.clip(new, self._exposure_us * 0.2, self._exposure_us * 5.0))
            new = float(np.clip(new, self._emin, self._emax))
            converged = abs(new - self._exposure_us) / max(self._exposure_us, 1) < 0.06
            self.set_exposure(new)
            if converged:
                break
        return self._exposure_us

    def close(self):
        self.cam.EndAcquisition()
        self.cam.DeInit()
        del self.cam
        self.system.ReleaseInstance()


# ──────────────────────────────────────────────────────────────────
# 다크 프레임
# ──────────────────────────────────────────────────────────────────
def capture_dark(cam: CameraController) -> np.ndarray:
    input("\n▶ 빔을 차단(빔 블로커/셔터)하고 Enter를 누르세요 (다크 프레임 촬영)...")
    dark = cam.grab_avg(n=8)
    input("▶ 빔을 다시 열고 Enter를 누르세요...")
    print(f"  다크 프레임: mean={dark.mean():.2f}  max={dark.max():.2f}")
    return dark


# ──────────────────────────────────────────────────────────────────
# 메인 수집 루프
# ──────────────────────────────────────────────────────────────────
def preview(args):
    """레퍼런스 패턴(캐리어 적용)을 SLM 에 띄우고 카메라 1프레임 저장.
    0차광/세로줄 십자에서 트랩 배열이 분리됐는지 확인 + 캐리어/노출 튜닝용."""
    from PIL import Image
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    if getattr(args, "sparse", False):
        ref_phase = make_sparse_phase(rng)
        print("[PREVIEW] 최악(가장 밝은) 2×2(4점) 패턴으로 테스트")
    elif getattr(args, "large", False):
        ref_phase = make_widest_phase(rng)
        print("[PREVIEW] 최대 크기 5×5 spacing49 패턴으로 테스트 (프레임 수용 확인)")
    else:
        ref_phase = make_reference_phase(rng)

    slm = SLMController()
    slm.show(ref_phase)
    time.sleep(SETTLE_S * 3)
    cam = CameraController(exposure_us=args.exposure, gain_db=GAIN_DB)
    time.sleep(SETTLE_S * 2)
    if AUTO_EXPOSURE and args.exposure is None:
        cam.auto_expose()
        print(f"[PREVIEW] 자동노출 수렴: {cam.exposure_us:.1f}us")
    img = cam.grab_avg(n=4)

    out_png = OUT_DIR / "preview.png"
    disp = np.clip(img / max(img.max(), 1) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(disp).save(out_png)
    sat = (img >= 254).mean() * 100
    # 트랩 영역(왼쪽, 0차광 제외) vs 중앙 0차광 영역 분리 분석
    H, W = img.shape
    col = max(1, W // 2 - ZEROTH_GUARD)
    trap = img[:, :col]
    trap_sat = (trap >= 254).mean() * 100
    print(f"\n[PREVIEW] HOL_FILL={HOL_FILL}  carrier=({CARRIER_CYC_X},{CARRIER_CYC_Y})  "
          f"노출={cam.exposure_us:.1f}us  (범위 [{cam._emin:.0f},{cam._emax:.0f}])")
    print(f"  전체프레임: max={img.max():.0f}/255  mean={img.mean():.2f}  "
          f"포화={sat:.3f}%")
    print(f"  트랩영역(좌측): max={trap.max():.0f}/255  mean={trap.mean():.2f}  "
          f"포화={trap_sat:.3f}%   ← 이게 핵심")
    if cam.exposure_us <= cam._emin * 1.1 and trap.max() >= 254:
        print("  ⚠️  최소노출인데 트랩 포화 → 빛이 너무 셈. ND필터/레이저감쇠 또는 HOL_FILL↓ 필요")
    elif trap.max() >= 254:
        print("  ⚠️  트랩 일부 포화 → 노출 더 낮추거나 빛 감쇠 필요")
    else:
        print(f"  ✓ 트랩 비포화. 자동노출 정상 작동.")
    print(f"  저장: {out_png.resolve()}")
    slm.close()
    cam.close()


def collect(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    ref_phase = make_reference_phase(rng)

    # 이미 수집된 파일 수 확인 (재시작 지원)
    done = set(p.stem for p in OUT_DIR.glob("*.npz")
               if not p.stem.startswith("ref_") and p.stem != "darks")
    print(f"[INFO] 기존 수집 파일: {len(done)}장 / 목표 {N_TOTAL}장")

    slm = SLMController()

    # SLM에 뭔가 띄운 뒤 카메라 자동 노출 결정
    slm.show(ref_phase)
    time.sleep(SETTLE_S * 3)
    cam = CameraController(exposure_us=args.exposure, gain_db=GAIN_DB)

    # 다크 프레임
    dark_path = OUT_DIR / "darks.npz"
    if dark_path.exists() and not args.redo_dark:
        dark = np.load(dark_path)["dark"].astype(np.float64)
        print(f"[INFO] 기존 다크 프레임 로드: mean={dark.mean():.2f}")
    else:
        dark = capture_dark(cam)
        np.savez_compressed(dark_path, dark=dark.astype(np.float32),
                            exposure_us=cam.exposure_us)
        print(f"[INFO] 다크 프레임 저장 완료")

    # 고정노출(--exposure)을 주면 자동노출 끔, 아니면 패턴마다 자동노출
    use_auto_exposure = AUTO_EXPOSURE and (args.exposure is None)
    print(f"[INFO] 자동노출: {'ON (패턴마다)' if use_auto_exposure else f'OFF (고정 {cam.exposure_us:.0f}μs)'}")

    remaining = N_TOTAL - len(done)
    print(f"[INFO] 수집 시작: 남은 {remaining}장")

    t0 = time.time()
    collected = 0

    for idx in tqdm(range(N_TOTAL), desc="수집", unit="쌍"):
        name = f"{idx:06d}"
        if name in done:
            continue

        # 기준 패턴 삽입 (드리프트 감시)
        if idx % REF_EVERY == 0:
            slm.show(ref_phase)
            time.sleep(SETTLE_S)
            if use_auto_exposure:
                cam.auto_expose()
            ref_img = cam.grab_avg() - dark
            np.savez_compressed(
                OUT_DIR / f"ref_{idx:06d}.npz",
                phase_u8=ref_phase,
                intensity=np.clip(ref_img, 0, None).astype(np.float32),
                meta=json.dumps({"idx": idx, "is_reference": True,
                                  "t": time.time(), "exposure_us": cam.exposure_us}),
            )

        # 데이터 패턴
        phase_u8, meta_extra = make_phase(rng)

        slm.show(phase_u8)
        time.sleep(SETTLE_S)

        # 패턴마다 노출 자동 조절 (점 개수별 밝기 차이 보정). 고정노출이면 건너뜀.
        if use_auto_exposure:
            cam.auto_expose()

        intensity = np.clip(cam.grab_avg() - dark, 0, None).astype(np.float32)
        sat_frac = float((intensity >= 254).mean())

        meta = {
            "idx": idx, "is_reference": False,
            "t": time.time(),
            "exposure_us": cam.exposure_us,
            "n_avg": N_AVG,
            "saturated_frac": sat_frac,
        }
        if meta_extra:
            meta.update(meta_extra)

        np.savez_compressed(
            OUT_DIR / f"{name}.npz",
            phase_u8=phase_u8,
            intensity=intensity,
            meta=json.dumps(meta),
        )
        collected += 1

        if collected % 200 == 0:
            elapsed = time.time() - t0
            rate = collected / elapsed
            eta = (remaining - collected) / rate
            tqdm.write(f"  [{collected}/{remaining}] "
                       f"경과={elapsed/60:.1f}분  남음≈{eta/60:.1f}분  "
                       f"속도={rate:.2f}쌍/s")

    slm.close()
    cam.close()
    total_elapsed = time.time() - t0
    print(f"\n완료: {collected}쌍 수집 ({total_elapsed/60:.1f}분)")
    print(f"저장 위치: {OUT_DIR.resolve()}")
    print(f"다음 단계: python lab/train_real_surrogate.py")


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실험 데이터 수집 (5000쌍)")
    parser.add_argument("--n", type=int, default=N_TOTAL,
                        help=f"수집 목표 쌍 수 (기본: {N_TOTAL})")
    parser.add_argument("--exposure", type=float, default=EXPOSURE_US,
                        help="카메라 노출시간 μs (기본: 자동)")
    parser.add_argument("--redo-dark", action="store_true",
                        help="기존 다크 프레임 무시하고 재촬영")
    parser.add_argument("--settle", type=float, default=SETTLE_S,
                        help=f"SLM 정착 대기시간 초 (기본: {SETTLE_S})")
    parser.add_argument("--out", type=str, default=str(OUT_DIR),
                        help=f"저장 폴더 (기본: {OUT_DIR})")
    parser.add_argument("--carrier-x", type=float, default=CARRIER_CYC_X,
                        help=f"blazed grating x 주기수, 0차광 좌우 분리 (기본: {CARRIER_CYC_X})")
    parser.add_argument("--carrier-y", type=float, default=CARRIER_CYC_Y,
                        help=f"blazed grating y 주기수, 상하 분리 (기본: {CARRIER_CYC_Y})")
    parser.add_argument("--preview", action="store_true",
                        help="레퍼런스 패턴 1장만 띄우고 카메라 1프레임 저장 후 종료 (튜닝용)")
    parser.add_argument("--sparse", action="store_true",
                        help="preview 시 최악(가장 밝은) 2×2 패턴으로 테스트")
    parser.add_argument("--large", action="store_true",
                        help="preview 시 최대 크기 5×5 패턴으로 테스트 (프레임 수용 확인)")
    parser.add_argument("--hol-fill", type=int, default=HOL_FILL,
                        help=f"SLM 확대 크기 px. 작을수록 트랩 어둡고 더 퍼짐 (기본: {HOL_FILL})")
    args = parser.parse_args()
    N_TOTAL = args.n
    SETTLE_S = args.settle
    OUT_DIR = Path(args.out)
    CARRIER_CYC_X = args.carrier_x
    CARRIER_CYC_Y = args.carrier_y
    HOL_FILL = args.hol_fill
    if args.preview:
        preview(args)
    else:
        collect(args)
