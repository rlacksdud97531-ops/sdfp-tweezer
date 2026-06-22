"""
_qc.py — 수집 데이터 품질 검사. (slmsuite env: numpy/scipy/PIL)
점검: 무결성 / 밝기·포화 / 드리프트(reference) / 좌표 정합(affine) / 스팟 분해능.
"""
import sys, os, json, glob
import numpy as np
from PIL import Image
from scipy import ndimage

D = r"C:\tweezer_data\measured512"
OUT = "lab/qc_out"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)


def load(p):
    z = np.load(p, allow_pickle=True)
    return z["phase_u8"], z["intensity"], json.loads(str(z["meta"]))


def detect_spots(img, n):
    """상위 n개 스팟 centroid (intensity px). 임계+라벨링."""
    g = ndimage.gaussian_filter(img, 3)
    thr = g.max() * 0.25
    lbl, m = ndimage.label(g > thr)
    if m == 0:
        return np.zeros((0, 2))
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, m + 1))
    sums = ndimage.sum(g, lbl, range(1, m + 1))
    order = np.argsort(sums)[::-1][:n]
    cents = ndimage.center_of_mass(g, lbl, [o + 1 for o in order])  # (y,x)
    return np.array([[c[1], c[0]] for c in cents])  # (x,y)


def fit_affine(src, dst):
    """src(model xy) → dst(camera xy) affine. 반환 (resid_px)."""
    n = min(len(src), len(dst))
    if n < 4:
        return None
    # 둘 다 (y, x) 순으로 정렬해 대응
    s = src[np.lexsort((src[:, 0], src[:, 1]))][:n]
    d = dst[np.lexsort((dst[:, 0], dst[:, 1]))][:n]
    A = np.hstack([s, np.ones((n, 1))])
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    resid = d - A @ coef
    return float(np.sqrt((resid ** 2).sum(1)).mean())


print("=== [1] 무결성 + 밝기/포화 (gs 300장 샘플) ===")
gs = glob.glob(os.path.join(D, "gs_*.npz"))
samp = rng.choice(gs, min(300, len(gs)), replace=False)
maxes, means, sat, bad = [], [], [], 0
ph_shape = inten_shape = None
for p in samp:
    ph, it, m = load(p)
    ph_shape, inten_shape = ph.shape, it.shape
    if ph.dtype != np.uint8 or not np.isfinite(it).all() or it.min() < -1:
        bad += 1
    maxes.append(float(it.max())); means.append(float(it.mean()))
    sat.append(float((it >= it.max() * 0.99).mean()) * 100)
maxes, means = np.array(maxes), np.array(means)
print(f"  phase shape={ph_shape} dtype=uint8, intensity shape={inten_shape}")
print(f"  이상 파일: {bad}/{len(samp)}")
print(f"  intensity max: 중앙값 {np.median(maxes):.1f}  범위 [{maxes.min():.1f}, {maxes.max():.1f}]")
print(f"  intensity mean: 중앙값 {np.median(means):.2f}")
print(f"  최상위 1% 영역(스팟피크 근처) 비율: 중앙값 {np.median(sat):.3f}%  (포화 아님=HDR)")

print("\n=== [2] 드리프트 (reference 전체) ===")
refs = sorted(glob.glob(os.path.join(D, "reference_*.npz")))
tot, idxs = [], []
for p in refs:
    _, it, _ = load(p)
    tot.append(float(it.sum())); idxs.append(int(os.path.basename(p)[10:15]))
tot = np.array(tot)
print(f"  reference {len(refs)}장")
print(f"  총강도 변동: 평균 {tot.mean():.3e}, RMS변동 {100*tot.std()/tot.mean():.2f}%")
print(f"  처음→끝: {tot[0]:.3e} → {tot[-1]:.3e}  ({100*(tot[-1]-tot[0])/tot[0]:+.1f}%)")

print("\n=== [3] 좌표 정합 (reference affine) ===")
_, it0, m0 = load(refs[0])
spots_model = np.array(m0.get("spots_xy", []))
det = detect_spots(it0, 16)
print(f"  reference 모델좌표 {len(spots_model)}개, 검출 스팟 {len(det)}개")
if len(spots_model) >= 4 and len(det) >= 4:
    r = fit_affine(spots_model, det)
    print(f"  affine 피팅 잔차(RMS): {r:.2f} px  ({'양호 <3px' if r and r < 3 else '확인필요'})")

print("\n=== [4] 스팟 분해능 (reference) ===")
det_all = detect_spots(it0, 16)
if len(det_all) >= 2:
    dd = np.sqrt(((det_all[:, None] - det_all[None]) ** 2).sum(-1))
    np.fill_diagonal(dd, np.inf)
    print(f"  검출 스팟 최근접 간격: 중앙값 {np.median(dd.min(1)):.1f} px (512² 기준)")

print("\n=== [5] 시각 검사용 몽타주 저장 ===")
def norm(it): return np.clip(it / max(it.max(), 1e-9) * 255, 0, 255).astype(np.uint8)
picks = [refs[0]] + list(rng.choice(gs, 3, replace=False))
tiles = []
for p in picks:
    ph, it, _ = load(p)
    tiles.append(np.hstack([ph.astype(np.uint8), norm(it)]))
mont = np.vstack(tiles)
Image.fromarray(mont).save(os.path.join(OUT, "montage.png"))
print(f"  저장: {os.path.join(OUT,'montage.png')} (각 행: 왼=phase, 오=intensity)")
print("\n완료.")
