"""
train.py — Surrogate-Distilled Fast Predictor 학습 파이프라인.

2단계:
  [STAGE 1] Surrogate 학습  (오프라인, 결맞음 제약 없음)
     무작위/GS 위상 φ 를 진짜 광학계(SimOptics)에 통과 → I 측정.
     Surrogate 가 (φ→I) 를 흉내내도록 학습. = 진짜 광학계의 캘리브레이션 학습.
     실험 데이터가 생기면 SimOptics 대신 카메라 데이터를 쓰면 됨.

  [STAGE 2] Predictor(U-Net) 학습  (오프라인 distill)
     desired array → U-Net → φ_pred → [frozen Surrogate] → I_pred.
     Loss = 스팟 강도 매칭 + 균일도(σ/μ) + (초반) GS warmup 복소장 손실.
     → 진짜 광학계의 수차/비선형성을 미리 상쇄하는 φ 를 예측하도록 학습.

  [ONLINE]  desired array → U-Net → POH (0.1초급).  ← 실험에서 이것만 사용.

CLI:  python -m sdfp.train --quick     (빠른 데모)
      python -m sdfp.train             (기본 설정)
"""

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F

from .optics_sim import SimOptics
from .models import Surrogate, UNet
from .gs import gs
from . import data as D
from .metrics import nonuniformity


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ─────────────────────────────────────────────────────
# 데이터 생성
# ─────────────────────────────────────────────────────
def make_samples(shape, n, device, seed=0):
    """n 개의 (target, peaks) 샘플 생성. (위상 라벨은 사용하지 않음 — 비지도 전파손실)"""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        pos, _ = D.random_array(shape, rng)
        tgt = D.target_intensity(pos, shape, device=device)
        px = np.clip(np.round(pos[0]).astype(int), 1, shape[1] - 2)
        py = np.clip(np.round(pos[1]).astype(int), 1, shape[0] - 2)
        # 스팟 영역 마스크 (각 피크 둘레 ±r) — 효율(배경 억제)용
        r = 2
        mask = np.zeros(shape, dtype=np.float32)
        for x, y in zip(px, py):
            mask[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = 1.0
        samples.append({
            "target": tgt,
            "peaks": np.stack([px, py]),                       # (2,N) np
            "px": torch.tensor(px, dtype=torch.long, device=device),
            "py": torch.tensor(py, dtype=torch.long, device=device),
            "mask": torch.tensor(mask, device=device),         # (H,W) 0/1
        })
    return samples


# ─────────────────────────────────────────────────────
# STAGE 1 — Surrogate
# ─────────────────────────────────────────────────────
def sample_phases(shape, n_pairs, device, rng):
    """학습용 위상 분포: GS-유사 POH + 순수 랜덤 (실제 POH 분포 근사)."""
    phases = []
    for _ in range(n_pairs):
        if rng.random() < 0.7:
            pos, _ = D.random_array(shape, rng)
            tgt = D.target_intensity(pos, shape, device=device)
            phases.append(gs(tgt, n_iter=25, device=device))
        else:
            phases.append(2 * np.pi * torch.rand(*shape, device=device))
    return torch.stack(phases)


def train_surrogate(optics, shape, device, n_pairs=400, epochs=8, bs=8, seed=1,
                    model=None, extra_phases=None, lr=3e-3):
    """Surrogate 학습/미세조정.

    extra_phases: (M,H,W) — U-Net 이 실제로 내놓는 위상들. 이걸 데이터에 섞으면
                  Surrogate 가 U-Net 작동영역에서 정확해져 model-exploitation 을 막는다.
    model: 주어지면 이어서 학습(refinement).
    """
    rng = np.random.default_rng(seed)
    sur = model if model is not None else Surrogate(shape).to(device)
    for p in sur.parameters():       # 이전 단계에서 freeze 됐을 수 있음
        p.requires_grad_(True)
    sur.train()
    opt = torch.optim.Adam(sur.parameters(), lr=lr)

    phases = sample_phases(shape, n_pairs, device, rng)
    if extra_phases is not None and len(extra_phases) > 0:
        phases = torch.cat([phases, extra_phases.to(device)], 0)
    with torch.no_grad():
        targets = torch.stack([optics.forward(p, add_noise=True) for p in phases])

    n = phases.shape[0]
    last = 0.0
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            p, I = phases[idx], targets[idx]
            Ip = sur(p)
            loss = F.l1_loss(Ip, I) + F.mse_loss(torch.sqrt(Ip + 1e-9),
                                                 torch.sqrt(I + 1e-9))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        last = tot / n
        print(f"  [surrogate] epoch {ep+1}/{epochs}  loss={last:.3e}")
    return sur


# ─────────────────────────────────────────────────────
# STAGE 2 — Predictor (U-Net), surrogate-in-the-loop
# ─────────────────────────────────────────────────────
def train_predictor(sur, samples, shape, device, epochs=20, bs=8,
                    warmup_ep=0, uni_w=0.5, eff_w=0.0, net=None, lr=1e-3):
    """비지도(label-free) 전파손실 학습.

    위상을 직접 지도하지 않는다 — phase-retrieval 해는 비유일/스펙클이라
    픽셀단위 회귀가 불가능. 대신 학습된 Surrogate 를 통과시킨 *강도*만 본다:
      desired array → U-Net → φ → [frozen Surrogate] → I_pred

    목적함수 = 스팟별 *패치 에너지*(주변 5×5 합)의 기하평균 최대화.
      에너지는 보존(ΣI=HW)되므로 모든 스팟 에너지를 키우면
      (효율↑: 빛이 스팟으로 모임) 동시에 기하평균이라
      (균일도↑: 가장 어두운 스팟이 가장 큰 벌점). 한 항으로 둘 다 달성.
    + 명시적 균일도 항(σ/μ) 약간.
    피크가 아니라 '에너지'를 봐야 효율이 보장됨 (피크만 보면 어두운 해 가능).

    Surrogate 가 수차/비선형성을 알고 있으므로 U-Net 은 그것을 미리
    보정한 위상을 학습 → 진짜 광학계에서 GS 보다 균일.
    """
    for p in sur.parameters():
        p.requires_grad_(False)
    sur.eval()

    if net is None:
        net = UNet().to(device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(samples)

    for ep in range(epochs):
        perm = np.random.permutation(n)
        tot = tot_eff = tot_u = 0.0
        u_w = uni_w if ep >= warmup_ep else 0.0
        for i in range(0, n, bs):
            batch = [samples[j] for j in perm[i:i + bs]]
            tgt = torch.stack([b["target"] for b in batch])
            phase = net(tgt)
            Ipred = sur(phase)
            # 스팟별 패치(5x5) 에너지 — 벡터화 (avg_pool×면적)
            k = 5
            patch = torch.nn.functional.avg_pool2d(
                Ipred.unsqueeze(1), k, 1, k // 2).squeeze(1) * (k * k)

            mask = torch.stack([bb["mask"] for bb in batch])    # (B,H,W)

            l_geo = torch.tensor(0.0, device=device)
            l_uni = torch.tensor(0.0, device=device)
            for b_i, bb in enumerate(batch):
                vals = patch[b_i, bb["py"], bb["px"]]           # (N,) 스팟 에너지
                l_geo = l_geo - torch.log(vals + 1e-6).mean()   # -log 기하평균
                if u_w > 0:
                    l_uni = l_uni + vals.std() / (vals.mean() + 1e-9)
            l_geo = l_geo / len(batch)
            l_uni = l_uni / len(batch)
            # 효율: 스팟영역 에너지 비율 (-log 으로 강하게 최대화)
            eff = (Ipred * mask).sum() / (Ipred.sum() + 1e-9)
            l_eff = -torch.log(eff + 1e-3)
            eff_frac = float(eff.detach())

            loss = l_geo + u_w * l_uni + eff_w * l_eff
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(batch)
            tot_eff += eff_frac * len(batch)
            tot_u += float(l_uni) * len(batch)
        sched.step()
        print(f"  [predictor] epoch {ep+1}/{epochs}  "
              f"loss={tot/n:.4f}  eff={tot_eff/n*100:.1f}%  uni={tot_u/n:.4f}")
    return net


# ─────────────────────────────────────────────────────
# 평가: GS / WGS / U-Net 을 진짜 광학계에 통과 → 균일도 비교
# ─────────────────────────────────────────────────────
def evaluate(net, optics, shape, device, n=8, seed=99):
    from .gs import gs as gsfun
    rng = np.random.default_rng(seed)
    rows = {"GS": [], "WGS": [], "U-Net": []}
    for _ in range(n):
        pos, _ = D.random_array(shape, rng)
        tgt = D.target_intensity(pos, shape, device=device)
        peaks = np.stack([np.round(pos[0]).astype(int),
                          np.round(pos[1]).astype(int)])

        for name, phase in [
            ("GS", gsfun(tgt, n_iter=60, device=device)),
            ("WGS", gsfun(tgt, n_iter=60, weighted=True, device=device)),
            ("U-Net", net(tgt.unsqueeze(0)).squeeze(0).detach()),
        ]:
            I = optics.forward(phase).detach()   # 진짜 광학계 통과
            nu, _ = nonuniformity(I, peaks, radius=3)
            rows[name].append(nu * 100)
    return {k: float(np.mean(v)) for k, v in rows.items()}


def collect_unet_phases(net, samples, device, max_n=200):
    """U-Net 이 현재 내놓는 위상들 (Surrogate refinement 용)."""
    net.eval()
    phases = []
    with torch.no_grad():
        for s in samples[:max_n]:
            phases.append(net(s["target"].unsqueeze(0)).squeeze(0))
    return torch.stack(phases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="빠른 데모 설정")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", default="sdfp/checkpoint.pt")
    args = ap.parse_args()

    device = pick_device()
    shape = (args.size, args.size)
    print(f"device={device}  shape={shape}")

    if args.quick:
        n_pairs, sur_ep, n_train, pred_ep, n_refine = 200, 25, 120, 24, 2
    else:
        n_pairs, sur_ep, n_train, pred_ep, n_refine = 700, 40, 900, 70, 3

    optics = SimOptics(shape=shape, device=device, read_noise=0.01, seed=0)

    t0 = time.time()
    print("STAGE 1: Surrogate 학습 (진짜 광학계 캘리브레이션 학습)")
    sur = train_surrogate(optics, shape, device, n_pairs=n_pairs, epochs=sur_ep)

    print("샘플 생성 + STAGE 2: Predictor distill")
    samples = make_samples(shape, n_train, device, seed=2)
    net = train_predictor(sur, samples, shape, device,
                          epochs=pred_ep, warmup_ep=pred_ep // 3)
    print(f"  [eval] {evaluate(net, optics, shape, device)}")

    # ── STAGE 3: DAgger refinement — surrogate-reality gap 닫기 ──
    for k in range(n_refine):
        print(f"STAGE 3.{k+1}: refinement (U-Net 출력을 실제 광학계로 측정→surrogate 보강)")
        extra = collect_unet_phases(net, samples, device)
        sur = train_surrogate(optics, shape, device, n_pairs=n_pairs // 2,
                              epochs=max(8, sur_ep // 2), model=sur,
                              extra_phases=extra, lr=1e-3)
        net = train_predictor(sur, samples, shape, device,
                              epochs=max(10, pred_ep // 2), warmup_ep=0,
                              net=net, lr=5e-4)
        print(f"  [eval] {evaluate(net, optics, shape, device)}")
    print(f"학습 완료 ({time.time()-t0:.1f}s)")

    print("최종 평가: 진짜 광학계에서의 비균일도 (낮을수록 좋음)")
    res = evaluate(net, optics, shape, device)
    for k, v in res.items():
        print(f"  {k:6s} non-uniformity = {v:.2f}%")

    # 추론 속도
    tgt = samples[0]["target"].unsqueeze(0)
    net.eval()
    with torch.no_grad():
        _ = net(tgt)
        t = time.time()
        for _ in range(20):
            _ = net(tgt)
    print(f"  U-Net 추론 속도: {(time.time()-t)/20*1000:.1f} ms / POH")

    torch.save({"surrogate": sur.state_dict(),
                "unet": net.state_dict(),
                "shape": shape}, args.out)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
