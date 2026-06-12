"""
infer.py — 학습된 체크포인트 로드 + POH 생성/비교 유틸 (Streamlit/CLI 공용).
"""

import numpy as np
import torch

from .optics_sim import SimOptics
from .models import Surrogate, UNet
from .gs import gs
from . import data as D
from .metrics import nonuniformity, efficiency


def load(checkpoint="sdfp/checkpoint.pt", device=None):
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = "cuda"
        else:
            device = "cpu"
    # 항상 CPU 로 먼저 로드 후 이동 — cuda 로 저장한 체크포인트도 CPU 전용 머신에서 안전.
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    sur_sd, net_sd = ck["surrogate"], ck["unet"]
    # 구조(해상도/refine 분기/base 폭)는 메타키가 아니라 state dict 자체에서 역산
    # → torch 버전·체크포인트 세대와 무관하게 mismatch 불가능.
    shape = tuple(sur_sd["A_raw"].shape)
    refine = any(k.startswith("cnn.") for k in sur_sd)
    base = int(net_sd["d1.net.0.weight"].shape[0])
    try:
        sur = Surrogate(shape, refine=refine)
        sur.load_state_dict(sur_sd)
        net = UNet(base=base)
        net.load_state_dict(net_sd)
    except RuntimeError as e:
        meta = {k: v for k, v in ck.items() if k not in ("surrogate", "unet")}
        raise RuntimeError(
            f"[SDFP load] state_dict mismatch: ckpt={checkpoint} shape={shape} "
            f"refine={refine} base={base} meta={meta} torch={torch.__version__}"
        ) from e
    sur = sur.to(device).eval()
    net = net.to(device).eval()
    return net, sur, shape, device


def unet_phase(net, target, device):
    with torch.no_grad():
        t0 = _now()
        phase = net(target.unsqueeze(0)).squeeze(0)
        dt = _now() - t0
    return phase % (2 * np.pi), dt


def _now():
    import time
    return time.time()


def refine_phase(net, sur, target, peaks, device, steps=20, uni_w=0.5, lr=0.08):
    """U-Net 출력을 warm-start 로, 미분가능 Surrogate 를 통해 스팟 *피크(중심픽셀)*
    강도를 기하평균 최대화 + 균일화하도록 위상을 직접 미세조정 (per-instance).

    피드포워드 U-Net 은 피크 정밀 균일화를 일반화하지 못하므로(배열마다 다른 해),
    추론 시점에 보정된 Surrogate(≈실제 광학계)로 몇 스텝만 다듬는다.
    U-Net 이 고효율 출발점을 주므로 효율을 유지한 채 균일도만 끌어올린다.
    반환: (phase in [0,2π), time)."""
    t0 = _now()
    with torch.no_grad():
        phase = (net(target.unsqueeze(0)).squeeze(0) % (2 * np.pi)).clone()
    if steps <= 0:
        return phase, _now() - t0
    px = torch.as_tensor(peaks[0], device=device, dtype=torch.long)
    py = torch.as_tensor(peaks[1], device=device, dtype=torch.long)
    for p in sur.parameters():
        p.requires_grad_(False)
    sur.eval()
    phase = phase.detach().requires_grad_(True)
    opt = torch.optim.Adam([phase], lr=lr)
    for _ in range(steps):
        I = sur(phase)
        vals = I[py, px]                       # 중심픽셀 = 피크(트랩 깊이), 지표와 정합
        loss = -torch.log(vals + 1e-6).mean() + uni_w * vals.std() / (vals.mean() + 1e-9)
        opt.zero_grad(); loss.backward(); opt.step()
    return (phase.detach() % (2 * np.pi)), _now() - t0


def compare(net, sur, optics, target, peaks, device, gs_iter=60, refine_steps=0):
    """GS / WGS / U-Net POH 를 진짜 광학계(optics)에 통과시켜 비교.

    refine_steps>0 이면 U-Net(SDFP) 결과를 Surrogate 로 per-instance 미세조정.
    반환: dict[name] = {phase, recon, nonunif, eff, time}
    """
    out = {}
    import time

    t = time.time()
    p_gs = gs(target, n_iter=gs_iter, device=device)
    t_gs = time.time() - t

    t = time.time()
    p_wgs = gs(target, n_iter=gs_iter, weighted=True, device=device)
    t_wgs = time.time() - t

    if refine_steps > 0:
        p_un, t_un = refine_phase(net, sur, target, peaks, device, steps=refine_steps)
    else:
        p_un, t_un = unet_phase(net, target, device)

    radius = max(3, round(3 * target.shape[-1] / 128))   # 해상도 비례 측정 반경
    for name, phase, dt in [("GS", p_gs, t_gs),
                            ("WGS", p_wgs, t_wgs),
                            ("U-Net (SDFP)", p_un, t_un)]:
        recon = optics.forward(phase).detach()
        nu, vals = nonuniformity(recon, peaks, radius=radius)
        eff = efficiency(recon, peaks, radius=radius)
        out[name] = {
            "phase": (phase % (2 * np.pi)).detach().cpu().numpy(),
            "recon": recon.cpu().numpy(),
            "nonunif": nu, "vals": vals, "eff": eff, "time": dt,
        }
    return out
