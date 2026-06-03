"""
gs.py — Gerchberg-Saxton / Weighted-GS (이상적 FFT 모델 기반).

용도 두 가지:
  1) U-Net 학습 시 *warmup 초기값* 제공 (local minima 회피)
  2) 베이스라인 비교 (GS/WGS 가 만든 POH 를 진짜 광학계에 통과시키면
     수차 때문에 균일도가 무너진다 → Surrogate 보정의 필요성을 보여줌)

이상적 FFT 모델만 사용 (수차/감마 모름). torch 로 작성해 GPU/MPS 가능.
"""

import numpy as np
import torch


def gs(target_intensity, n_iter=60, weighted=False, device="cpu", seed=None):
    """target_intensity: (H,W) torch. 반환: phase (H,W) in [0,2π)."""
    if seed is not None:
        torch.manual_seed(seed)
    amp = torch.sqrt(torch.clamp(target_intensity, min=0)).to(device)
    H, W = amp.shape
    phase = 2 * np.pi * torch.rand(H, W, device=device)
    mask = amp > (amp.max() * 1e-3)
    w = torch.ones_like(amp)

    for _ in range(n_iter):
        obj = amp * w * torch.exp(1j * phase)
        slm = torch.fft.ifft2(torch.fft.ifftshift(obj))
        slm = torch.exp(1j * torch.angle(slm))
        out = torch.fft.fftshift(torch.fft.fft2(slm))
        phase = torch.angle(out)
        if weighted:
            cur = torch.abs(out)
            cur = cur / (cur[mask].mean() + 1e-9)
            target_n = amp / (amp[mask].mean() + 1e-9)
            w = torch.where(mask, w * torch.sqrt(target_n / (cur + 1e-9)), w)
            w = torch.clamp(w, 0.1, 10.0)

    slm_phase = torch.angle(
        torch.fft.ifft2(torch.fft.ifftshift(amp * w * torch.exp(1j * phase)))
    )
    return slm_phase % (2 * np.pi)
