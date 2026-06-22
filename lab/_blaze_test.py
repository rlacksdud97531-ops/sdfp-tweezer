"""
_blaze_test.py — SLM 위상 캘리브레이션 실측 점검.
순수 blazed grating 을 그레이레벨 깊이(gmax) 별로 띄워 1차 회절 효율 측정.
  - 1차 스팟이 또렷+밝으면 → 2π 변조 정상
  - 1차 효율이 특정 gmax 에서 피크 → 그 gmax 가 2π (255 가 아니면 LUT 미스매치)
실행: slmsuite env
"""
import sys, os, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_data as C

OUT = Path("lab/blaze_test"); OUT.mkdir(parents=True, exist_ok=True)
PERIOD = 5            # 512 패턴 기준 blaze period (2× 업스케일 → 1024 에서 10px)
EXP_MS = 0.5          # 고정 노출 (효율 비교 위해)
FIRST = (258, 410)    # 1차 스팟 예상 위치 (blaze=5)
ZERO  = (644, 482)    # 0차광 위치
BOX = 70


def blaze_u8(period, gmax):
    x = np.arange(512)
    row = (((x / period) % 1.0) * gmax).astype(np.uint8)
    return np.tile(row, (512, 1))


def boxsum(f, cx, cy, b=BOX):
    return float(f[cy-b:cy+b, cx-b:cx+b].sum())


def main():
    print(f"blaze period={PERIOD}(512)/{PERIOD*C.SLM_UPSCALE}(SLM)  노출={EXP_MS}ms 고정\n")
    print(f"{'gmax':>5} {'1차합':>12} {'0차합':>12} {'1차/0차':>8} {'1차peak':>8}")
    best = (None, -1)
    for g in [120, 150, 180, 200, 215, 235, 255]:
        slm_img = C.embed_in_slm(blaze_u8(PERIOD, g))
        C.display_phase(slm_img)
        time.sleep(C.SETTLE_MS / 1000)
        # 노출 변경 직후 stale 버림 위해 두 번 grab
        C.grab_frame(EXP_MS)
        f = C.grab_frame(EXP_MS)
        p1 = boxsum(f, *FIRST); p0 = boxsum(f, *ZERO)
        peak = float(f[FIRST[1]-BOX:FIRST[1]+BOX, FIRST[0]-BOX:FIRST[0]+BOX].max())
        ratio = p1 / max(p0, 1)
        print(f"{g:>5} {p1:>12.0f} {p0:>12.0f} {ratio:>8.3f} {peak:>8.0f}")
        if p1 > best[1]:
            best = (g, p1)
        # 1차 스팟 zoom 저장
        crop = f[FIRST[1]-BOX:FIRST[1]+BOX, FIRST[0]-BOX:FIRST[0]+BOX]
        cd = np.clip(crop / max(crop.max(), 1) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(cd).save(OUT / f"first_g{g}.png")
        # 전체 프레임도
        fd = np.clip(f / max(f.max(), 1) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(fd).save(OUT / f"full_g{g}.png")

    print(f"\n→ 1차 효율 최대 gmax = {best[0]} (이게 2π graylevel)")
    print(f"  255 이면 캘리브 정상. 더 작으면 그 값을 2π 로 써야 함.")
    print(f"  저장: {OUT.resolve()}")


if __name__ == "__main__":
    main()
