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
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint, map_location=device)
    shape = tuple(ck["shape"])
    sur = Surrogate(shape).to(device); sur.load_state_dict(ck["surrogate"]); sur.eval()
    net = UNet().to(device); net.load_state_dict(ck["unet"]); net.eval()
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


def compare(net, optics, target, peaks, device, gs_iter=60):
    """GS / WGS / U-Net POH 를 진짜 광학계(optics)에 통과시켜 비교.

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

    p_un, t_un = unet_phase(net, target, device)

    for name, phase, dt in [("GS", p_gs, t_gs),
                            ("WGS", p_wgs, t_wgs),
                            ("U-Net (SDFP)", p_un, t_un)]:
        recon = optics.forward(phase).detach()
        nu, vals = nonuniformity(recon, peaks, radius=3)
        eff = efficiency(recon, peaks, radius=3)
        out[name] = {
            "phase": (phase % (2 * np.pi)).detach().cpu().numpy(),
            "recon": recon.cpu().numpy(),
            "nonunif": nu, "vals": vals, "eff": eff, "time": dt,
        }
    return out
