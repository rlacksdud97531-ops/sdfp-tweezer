"""
_pilot_capture.py — 파일럿 진단: 각 512² 패턴을 SLM에 띄우고 카메라 전체 프레임 저장.
실행 환경: slmsuite (PySpin)  — torch 불필요 (PNG 읽기만)
    C:\\Users\\jolab\\miniconda3\\envs\\slmsuite\\python.exe lab/_pilot_capture.py

목적: blaze로 이동된 스팟이 카메라 어디에 맺히는지 + 0차광 분리 + 포화 확인.
출력: lab/pilot_out/<name>_cam.png (전체 카메라 프레임, 정규화)
"""
import sys, os, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_data as C   # display_phase, grab_frame, embed_in_slm

PAT_DIR = Path(os.environ.get("PILOT_PAT_DIR", "lab/patterns_pilot"))
OUT_DIR = Path(os.environ.get("PILOT_OUT_DIR", "lab/pilot_out"))
GUARD = 150   # 중앙 0차광 제외 폭(px) — off-center 평가용


def auto_grab(target=180.0, exp0=1.0):
    """off-center 최댓값이 target 되게 노출(ms) 맞춰 전체 프레임 반환."""
    exp = exp0
    f = None
    for _ in range(6):
        f = C.grab_frame(exp)
        H, W = f.shape
        m = f.copy()
        m[:, W // 2 - GUARD: W // 2 + GUARD] = 0   # 0차광 세로띠 제외
        mx = float(m.max())
        if mx < 1:
            mx = 1.0
        new = exp * target / mx
        new = float(np.clip(new, 0.05, 25.0))      # ms (카메라 46µs~32ms)
        if abs(new - exp) / exp < 0.1:
            exp = new
            f = C.grab_frame(exp)
            break
        exp = new
    return f, exp


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PAT_DIR.glob("*.png"))
    assert files, f"{PAT_DIR} 에 패턴 PNG 없음 (cgslm2 로 make_patterns_512.py --pilot 먼저)"
    print(f"파일럿 {len(files)}장 캡처 시작\n")

    for f in files:
        u8 = np.array(Image.open(f).convert("L"))          # 512²
        slm_img = C.embed_in_slm(u8)                        # 1920×1080 (중앙, offset 0)
        C.display_phase(slm_img)
        time.sleep(C.SETTLE_MS / 1000)
        frame, exp = auto_grab()

        H, W = frame.shape
        # off-center 분석 (0차광 세로띠 제외)
        m = frame.copy()
        m[:, W // 2 - GUARD: W // 2 + GUARD] = 0
        oc_max = float(m.max())
        ys, xs = np.where(m > oc_max * 0.5)
        if len(xs):
            cx, cy = xs.mean(), ys.mean()
            bbox = f"x[{xs.min()}-{xs.max()}] y[{ys.min()}-{ys.max()}]"
        else:
            cx = cy = -1
            bbox = "none"
        sat = float((frame >= 254).mean()) * 100

        # 정규화 저장 (전체 프레임)
        disp = np.clip(frame / max(frame.max(), 1) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(disp).save(OUT_DIR / f"{f.stem}_cam.png")
        # 실제 저장될 512² crop (crop_camera 결과)
        cropped = C.crop_camera(frame)
        cdisp = np.clip(cropped / max(cropped.max(), 1) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(cdisp).save(OUT_DIR / f"{f.stem}_crop.png")

        print(f"{f.stem:18s} 노출={exp:.2f}ms  off중심=({cx:.0f},{cy:.0f}) "
              f"범위 {bbox}  offmax={oc_max:.0f}  포화={sat:.3f}%")

    print(f"\n저장: {OUT_DIR.resolve()}  (frame center = {W//2},{H//2})")


if __name__ == "__main__":
    main()
