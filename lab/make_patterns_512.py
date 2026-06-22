"""
make_patterns_512.py — 512² SLM 위상 패턴 PNG 생성 (sdfp.patterns 사용).
실행 환경: cgslm2 (torch 필요)
    C:\\Users\\jolab\\miniconda3\\envs\\cgslm2\\python.exe lab/make_patterns_512.py --pilot
    C:\\Users\\jolab\\miniconda3\\envs\\cgslm2\\python.exe lab/make_patterns_512.py --n 10000 --blaze-x 6

canvas=None: SLM 임베드는 collect_data.py 가 함. blaze 는 여기서 위상에 합성.
"""
import sys, os, argparse
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sdfp import patterns as P
from sdfp import data as D
from sdfp.gs import gs as _gs
import torch

SHAPE = (512, 512)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
S_MAX = 90    # 배열 최대 폭(512 격자 px). 초과 시 모양 유지하며 축소 → crop 안에 콤팩트하게


def fit_extent(pos, shape=SHAPE, s_max=S_MAX):
    """배열이 너무 크면 중심 기준 축소해 최대 폭 ≤ s_max."""
    if pos.shape[1] == 0:
        return pos
    cx, cy = shape[1] / 2.0, shape[0] / 2.0
    ext = max(pos[0].max() - pos[0].min(), pos[1].max() - pos[1].min())
    if ext > s_max:
        sc = s_max / ext
        pos = pos.copy()
        pos[0] = (pos[0] - cx) * sc + cx
        pos[1] = (pos[1] - cy) * sc + cy
    return pos


def _spacing_actual(pos):
    """최근접 이웃 평균 거리 (실제 spacing 추정). 좌표가 1개면 None."""
    if pos.shape[1] < 2:
        return None
    P = pos.T  # (N,2)
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def gs_capped(n, seed=0, n_iter=40, manifest=None):
    """gs_array_patterns 와 동일하나 fit_extent 로 크기 제한 + 좌표 기록."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        pos, atype = D.random_array(SHAPE, rng)
        pos = fit_extent(pos)
        tgt = D.target_intensity(pos, SHAPE, device=DEV)
        ph = _gs(tgt, n_iter=n_iter, device=DEV).cpu().numpy()
        name = f"gs_{seed}_{k:04d}"
        out.append((name, ph))
        if manifest is not None:
            manifest[name] = {"array_type": atype, "spacing_actual": _spacing_actual(pos),
                              "spots_xy": pos.T.round(2).tolist()}
    return out


def square_phase(rows, cols, spacing, manifest=None, name=None):
    pos = fit_extent(D.spot_positions("Square", SHAPE, rows=rows, cols=cols, spacing=spacing))
    ph = _gs(D.target_intensity(pos, SHAPE, device=DEV), n_iter=60, device=DEV).cpu().numpy()
    if manifest is not None and name is not None:
        manifest[name] = {"array_type": "Square", "spacing_actual": _spacing_actual(pos),
                          "spots_xy": pos.T.round(2).tolist()}
    return ph


def save(named_u8, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for name, u8 in named_u8:
        Image.fromarray(u8, "L").save(outdir / f"{name}.png")


def gen_sizesweep(outdir, blaze_x=3, blaze_y=0):
    """최종 확정 파일럿: 크기별 Square(2×2~5×5, 큰 spacing) + reference, 고정 blaze.
    fit_extent 로 큰 건 축소됨 → 전부 프레임 수용 + 0차광 분리 확인용."""
    out = []
    for (r, c, sp) in [(2, 2, 120), (3, 3, 80), (4, 4, 61), (5, 5, 60), (5, 5, 99)]:
        ph = square_phase(r, c, sp)
        out.append((f"sq_{r}x{c}_sp{sp}", ph))
    out.append(("reference", square_phase(4, 4, 61)))   # fit_extent 적용된 기준패턴
    fin = P.finalize(out, SHAPE, blaze_x=blaze_x, blaze_y=blaze_y, canvas=None)
    save(fin, outdir)
    print(f"[sizesweep] {len(fin)}장 (blaze_x={blaze_x}) → {outdir.resolve()}")


def gen_pilot(outdir, n_iter=40):
    """파일럿: reference blaze 스윕 + GS 배열 몇 개 + 랜덤위상. blaze/crop/FOV 진단용."""
    ref = P.reference_pattern(SHAPE)                  # [("reference", ph)]
    out = []
    # blaze 스윕 (0차광 분리 정도 + 카메라 FOV 확인)
    for px in [0, 4, 6, 8, 12]:
        u8 = P.finalize(ref, SHAPE, blaze_x=px, blaze_y=0, canvas=None)[0][1]
        out.append((f"refblaze_px{px:02d}", u8))
    # GS 배열 (다양한 크기) — blaze_x=8 고정, 프레임 수용/오버플로 확인
    gs = P.gs_array_patterns(SHAPE, 4, seed=1, n_iter=n_iter)
    out += P.finalize(gs, SHAPE, blaze_x=8, blaze_y=0, canvas=None)
    # 랜덤 위상
    rp = P.random_phase_patterns(SHAPE, 2, seed=2)
    out += P.finalize(rp, SHAPE, blaze_x=8, blaze_y=0, canvas=None)
    save(out, outdir)
    print(f"[pilot] {len(out)}장 저장 → {outdir.resolve()}")


def gen_full(outdir, n, blaze_x, blaze_y, n_iter=40, rand_frac=0.3):
    """본 수집용: GS 배열(메인) + 랜덤위상 + reference.png."""
    import json
    manifest = {}
    n_rand = int(n * rand_frac)
    n_gs = n - n_rand
    print(f"[full] GS {n_gs}장 + 랜덤 {n_rand}장 생성 중...")
    gs = gs_capped(n_gs, seed=0, n_iter=n_iter, manifest=manifest)   # fit_extent + 좌표기록
    out = P.finalize(gs, SHAPE, blaze_x=blaze_x, blaze_y=blaze_y, canvas=None)
    rp = P.random_phase_patterns(SHAPE, n_rand, seed=99)
    for name, _ in rp:
        manifest[name] = {"array_type": "random_phase", "spacing_actual": None, "spots_xy": None}
    out += P.finalize(rp, SHAPE, blaze_x=blaze_x, blaze_y=blaze_y, canvas=None)
    # 드리프트/정합 기준 패턴: 4×4 정사각 spacing=30 정확히 고정 (fit_extent 경계 → 그대로 통과)
    ref = [("reference", square_phase(4, 4, 30, manifest=manifest, name="reference"))]
    out += P.finalize(ref, SHAPE, blaze_x=blaze_x, blaze_y=blaze_y, canvas=None)
    manifest["_meta"] = {"shape": SHAPE, "S_MAX": S_MAX, "blaze_x": blaze_x,
                         "blaze_y": blaze_y, "slm_upscale_note": "collect_data.py 가 표시",
                         "reference": "4x4 Square spacing=30, x,y in {256+-15,256+-45}"}
    save(out, outdir)
    with open(outdir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False)
    print(f"[full] {len(out)}장 + manifest.json 저장 → {outdir.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="파일럿 소량 (진단용)")
    ap.add_argument("--sizesweep", action="store_true", help="크기별 확정 파일럿")
    ap.add_argument("--n", type=int, default=10000, help="본 수집 패턴 수")
    ap.add_argument("--blaze-x", type=int, default=6, help="blaze period_x (px)")
    ap.add_argument("--blaze-y", type=int, default=0, help="blaze period_y (px)")
    ap.add_argument("--out", type=str, default=None, help="출력 폴더")
    ap.add_argument("--n-iter", type=int, default=40, help="GS 반복")
    args = ap.parse_args()
    if args.sizesweep:
        outdir = Path(args.out or "patterns_pilot")
        gen_sizesweep(outdir, blaze_x=args.blaze_x, blaze_y=args.blaze_y)
    elif args.pilot:
        outdir = Path(args.out or "patterns_pilot")
        gen_pilot(outdir, n_iter=args.n_iter)
    else:
        outdir = Path(args.out or "patterns")
        gen_full(outdir, args.n, args.blaze_x, args.blaze_y, n_iter=args.n_iter)
