"""benchmark.py — 종합 벤치마크: GS / WGS / U-Net / U-Net+Refine 비교.

배열타입(Square·Triangular·Honeycomb) × 무작위 인스턴스를 실제 광학계(SimOptics)에
통과시켜 비균일도(σ/μ, peak)·효율·시간을 평균±표준편차로 보고.

실행:  python -m sdfp.benchmark [--ckpt sdfp/checkpoint.pt] [--n 5]
"""
import argparse
import time

import numpy as np
import torch

from . import data as D
from . import infer as INF
from .optics_sim import SimOptics
from .gs import gs
from .metrics import nonuniformity, efficiency


def _rand_of_type(shape, typ, rng):
    """해상도 비례 spacing 으로 해당 타입의 무작위 배열 1개 생성."""
    s = shape[0] / 128.0
    spacing = int(rng.integers(round(12 * s), round(20 * s)))
    if typ in ("Square", "Triangular"):
        r, c = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        return D.spot_positions(typ, shape, rows=r, cols=c, spacing=spacing, rng=rng)
    rings = int(rng.integers(1, 3))
    return D.spot_positions(typ, shape, spacing=spacing, rings=rings, rng=rng)


def _stats(optics, peaks, phase):
    I = optics.forward(phase).detach()
    r = max(3, round(3 * I.shape[-1] / 128))   # 해상도 비례 측정 반경
    return nonuniformity(I, peaks, r)[0] * 100, efficiency(I, peaks, r) * 100


def benchmark(ckpt="sdfp/checkpoint.pt", n=5, refine_steps=(0, 40, 100), seed=123):
    net, sur, shape, device = INF.load(ckpt)
    optics = SimOptics(shape=shape, device=device, read_noise=0.0, seed=0)
    rng = np.random.default_rng(seed)
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)

    methods = ["GS", "WGS"] + [("U-Net" if s == 0 else f"U-Net+R{s}") for s in refine_steps]
    acc = {m: {"nu": [], "eff": [], "t": []} for m in methods}

    for _ in range(n):
        for typ in ("Square", "Triangular", "Honeycomb"):
            pos = _rand_of_type(shape, typ, rng)
            tgt = D.target_intensity(pos, shape, device=device)
            peaks = np.stack([np.round(pos[0]).astype(int), np.round(pos[1]).astype(int)])

            for nm, w in (("GS", False), ("WGS", True)):
                t = time.time(); ph = gs(tgt, n_iter=60, weighted=w, device=device); sync()
                nu, ef = _stats(optics, peaks, ph)
                acc[nm]["nu"].append(nu); acc[nm]["eff"].append(ef)
                acc[nm]["t"].append((time.time() - t) * 1000)

            for s in refine_steps:
                nm = "U-Net" if s == 0 else f"U-Net+R{s}"
                t = time.time(); ph, _ = INF.refine_phase(net, sur, tgt, peaks, device, steps=s); sync()
                nu, ef = _stats(optics, peaks, ph)
                acc[nm]["nu"].append(nu); acc[nm]["eff"].append(ef)
                acc[nm]["t"].append((time.time() - t) * 1000)

    print(f"\n체크포인트: {ckpt}  shape={shape}  device={device}  (배열타입 3 × {n}회)")
    print(f"{'method':12s} {'nu % (낮을수록)':>16s} {'eff % (높을수록)':>16s} {'time ms':>10s}")
    print("-" * 58)
    for m in methods:
        nu, ef, tt = (np.array(acc[m][k]) for k in ("nu", "eff", "t"))
        print(f"{m:12s} {nu.mean():7.2f} ± {nu.std():5.2f}  {ef.mean():7.1f} ± {ef.std():4.1f}  {tt.mean():8.0f}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="sdfp/checkpoint.pt")
    ap.add_argument("--n", type=int, default=5, help="배열타입당 무작위 인스턴스 수")
    args = ap.parse_args()
    benchmark(args.ckpt, n=args.n)


if __name__ == "__main__":
    main()
