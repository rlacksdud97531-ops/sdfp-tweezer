"""
models.py — Surrogate (미분가능 광학계 모델) + U-Net (빠른 위상 예측기).

[Surrogate]  Physics-informed differentiable surrogate.
  진짜 광학계가 대략 FFT 라는 사실을 *구조에 박아넣고*, 모르는 부분만 학습:
    - gamma 곡선 (SLM 비선형성)   : 단조 증가 lookup
    - 동공 진폭 A(x,y) (조명)      : 학습 파라미터
    - 정적 수차 W(x,y)            : 학습 파라미터
    - FFT 자체는 고정 (물리)
  → (φ, I) 쌍만 보고 진짜 광학계의 캘리브레이션을 역으로 알아낸다.
  실험 데이터가 생기면 SimOptics 대신 그 데이터로 이 Surrogate 만 다시 학습하면 됨.

[U-Net]  desired array(강도) → SLM phase. 추론 0.1초급 → 짧은 결맞음 시간 대응.
  학습 시 손실은 (FFT 가 아니라) 학습된 Surrogate 를 통과시켜 계산 →
  진짜 광학계의 수차/비선형성을 *미리 보정한* 위상을 예측하도록 distill.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────
# 단조 증가 곡선 (SLM 감마 lookup)  [0,1] -> [0,1]
# ─────────────────────────────────────────────────────
class MonotoneCurve(nn.Module):
    def __init__(self, n_knots=16):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(n_knots))

    def forward(self, x):
        inc = F.softplus(self.raw) + 1e-3
        cum = torch.cumsum(inc, 0)
        cum = torch.cat([torch.zeros(1, device=x.device), cum])
        cum = cum / cum[-1]                      # (n_knots+1,) 0..1
        pos = torch.clamp(x, 0, 1) * (cum.numel() - 1)
        idx = torch.clamp(pos.floor().long(), 0, cum.numel() - 2)
        frac = pos - idx.float()
        return cum[idx] * (1 - frac) + cum[idx + 1] * frac


# ─────────────────────────────────────────────────────
# Physics-informed Surrogate
# ─────────────────────────────────────────────────────
class Surrogate(nn.Module):
    """refine=False: 순수 물리 모델(FFT+감마+조명+수차). 시뮬레이션 정합·일반화 우수.
    refine=True : + CNN 잔차항. 실제 광학계의 *미모델링* 효과(산란/다중반사 등)
                  를 잡을 때만 사용. 단, 과도하면 U-Net 이 오차를 exploit 할 수 있음.
    """

    def __init__(self, shape=(128, 128), refine=False):
        super().__init__()
        H, W = shape
        self.shape = shape
        self.curve = MonotoneCurve()
        self.A_raw = nn.Parameter(torch.zeros(H, W))   # softplus → 진폭>0
        self.Wab = nn.Parameter(torch.zeros(H, W))     # 수차(rad)
        self.refine = refine
        if refine:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1),
            )

    def forward(self, phase):
        single = phase.dim() == 2
        if single:
            phase = phase.unsqueeze(0)
        x = (phase % (2 * np.pi)) / (2 * np.pi)
        phi = 2 * np.pi * self.curve(x)
        A = F.softplus(self.A_raw).unsqueeze(0)
        field = A * torch.exp(1j * (phi + self.Wab.unsqueeze(0)))
        far = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1))),
            dim=(-2, -1),
        )
        I = far.real ** 2 + far.imag ** 2
        H, W = self.shape
        I = I / (I.sum(dim=(-2, -1), keepdim=True) + 1e-12) * (H * W)
        if self.refine:
            I = F.relu(I + self.cnn(I.unsqueeze(1)).squeeze(1))
            I = I / (I.sum(dim=(-2, -1), keepdim=True) + 1e-12) * (H * W)
        return I.squeeze(0) if single else I


# ─────────────────────────────────────────────────────
# U-Net (강도 → 위상)
# ─────────────────────────────────────────────────────
class _DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """입력 (B,1,H,W) 강도 → 출력 위상 (B,H,W) in (-π,π].

    출력 2채널(cos/sin)로 만들어 atan2 → 2π wrapping 문제 회피.
    """

    def __init__(self, base=32):
        super().__init__()
        self.d1 = _DoubleConv(1, base)
        self.d2 = _DoubleConv(base, base * 2)
        self.d3 = _DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bott = _DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = _DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = _DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = _DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 2, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        b = self.bott(self.pool(c3))
        u = self.u3(torch.cat([self.up3(b), c3], 1))
        u = self.u2(torch.cat([self.up2(u), c2], 1))
        u = self.u1(torch.cat([self.up1(u), c1], 1))
        o = self.out(u)                      # (B,2,H,W)
        phase = torch.atan2(o[:, 0], o[:, 1])  # (B,H,W) in (-π,π]
        return phase
