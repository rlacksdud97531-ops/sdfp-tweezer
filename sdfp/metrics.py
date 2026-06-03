"""
metrics.py — 트위저 품질 지표.

- spot_intensities: 각 스팟 위치의 피크 강도 추출
- nonuniformity   : σ/μ (낮을수록 균일)  ← SU(N) 실험의 핵심 지표
- efficiency      : 스팟에 들어간 에너지 / 전체 에너지
"""

import numpy as np
import torch


def spot_intensities(intensity, positions, radius=3):
    """intensity: (H,W) torch/np. 반환: (N,) 각 스팟 피크 강도."""
    if isinstance(intensity, torch.Tensor):
        intensity = intensity.detach().cpu().numpy()
    H, W = intensity.shape
    vals = []
    for x, y in zip(positions[0], positions[1]):
        xi = int(np.clip(round(x), radius, W - radius - 1))
        yi = int(np.clip(round(y), radius, H - radius - 1))
        patch = intensity[yi - radius:yi + radius + 1, xi - radius:xi + radius + 1]
        vals.append(float(patch.max()))
    return np.array(vals)


def nonuniformity(intensity, positions, radius=3):
    v = spot_intensities(intensity, positions, radius)
    return float(v.std() / (v.mean() + 1e-12)), v


def efficiency(intensity, positions, radius=3):
    if isinstance(intensity, torch.Tensor):
        intensity = intensity.detach().cpu().numpy()
    H, W = intensity.shape
    total = intensity.sum() + 1e-12
    inside = 0.0
    for x, y in zip(positions[0], positions[1]):
        xi = int(np.clip(round(x), radius, W - radius - 1))
        yi = int(np.clip(round(y), radius, H - radius - 1))
        inside += intensity[yi - radius:yi + radius + 1,
                            xi - radius:xi + radius + 1].sum()
    return float(inside / total)
