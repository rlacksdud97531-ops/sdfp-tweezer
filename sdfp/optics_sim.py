"""
optics_sim.py — Simulated "ground-truth" optical system.

이 모듈은 *실제 Er 실험의 대역(stand-in)* 입니다.
아직 실험 데이터가 없기 때문에, 진짜 광학계가 가지는 비이상성(non-ideality)을
일부러 주입한 시뮬레이터를 만들어 둡니다. Surrogate-Distilled Fast Predictor 는
이 "미지의 광학계"를 데이터만 보고 역으로 학습하게 됩니다.

순수 Fraunhofer FFT 모델과 다른 점 (= 실제 실험이 다른 점):
  1) SLM 감마 비선형성     : 명령 위상 φ_cmd → 실제 위상 (룩업이 비선형)
  2) 가우시안 조명 프로파일 : SLM 을 평면파가 아니라 유한한 빔이 비춤
  3) 정적 수차(aberration) : 렌즈/광학계의 고정 위상오차 (Zernike 합)
  4) (옵션) 검출 노이즈     : 카메라 샷노이즈

forward(phase) -> object-plane intensity (에너지 정규화).
이 모든 비이상성은 forward 안에서 고정된 파라미터로 적용됩니다 —
"실험은 한 번 정렬하면 잘 안 바뀐다"는 사용자의 관찰과 일치.
"""

import math
import numpy as np
import torch


# ─────────────────────────────────────────────────────
# Zernike 다항식 (저차) — 정적 수차 생성용
# ─────────────────────────────────────────────────────
def _zernike_basis(shape, n_terms=10):
    """단위원에 내접하는 좌표계에서 저차 Zernike 다항식 스택을 만든다."""
    H, W = shape
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    x = (x - (W - 1) / 2) / ((min(H, W) - 1) / 2)
    y = (y - (H - 1) / 2) / ((min(H, W) - 1) / 2)
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # (n, m) 순서: piston 제외, tilt 부터 spherical 까지
    terms = [
        (1, -1), (1, 1),          # tilt x/y
        (2, 0), (2, -2), (2, 2),  # defocus, astig
        (3, -1), (3, 1),          # coma
        (3, -3), (3, 3),          # trefoil
        (4, 0),                   # spherical
    ][:n_terms]

    def radial(n, m, r):
        m = abs(m)
        R = np.zeros_like(r)
        for k in range((n - m) // 2 + 1):
            c = ((-1) ** k * math.factorial(n - k)) / (
                math.factorial(k)
                * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)
            )
            R += c * r ** (n - 2 * k)
        return R

    basis = []
    for (n, m) in terms:
        R = radial(n, m, rho)
        if m >= 0:
            Z = R * np.cos(m * theta)
        else:
            Z = R * np.sin(-m * theta)
        Z[rho > 1] = 0.0
        basis.append(Z)
    return np.stack(basis, axis=0)  # (n_terms, H, W)


class SimOptics:
    """고정된(미지의) 진짜 광학계 시뮬레이터.

    Surrogate 가 학습으로 흉내내야 하는 대상. 학습 중에는 forward 만
    호출되고 내부 파라미터(감마/조명/수차)는 절대 노출되지 않는다.
    """

    def __init__(
        self,
        shape=(128, 128),
        gamma=1.35,           # SLM 비선형성 (1.0 이면 선형/이상적)
        beam_waist_frac=0.55, # 가우시안 빔 허리 (SLM 크기 대비)
        aberration_rms=0.9,   # 수차 세기 (rad RMS)
        read_noise=0.0,       # 카메라 노이즈 표준편차 (정규화 강도 기준)
        device="cpu",
        seed=0,
    ):
        self.shape = shape
        self.device = device
        self.gamma = float(gamma)
        self.read_noise = float(read_noise)
        H, W = shape
        rng = np.random.default_rng(seed)

        # 1) 가우시안 조명 진폭 A(x,y)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        cx, cy = (W - 1) / 2, (H - 1) / 2
        w0 = beam_waist_frac * min(H, W) / 2
        A = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (w0 ** 2))
        self.A = torch.tensor(A, dtype=torch.float32, device=device)

        # 2) 정적 수차 W(x,y) = Σ c_k Z_k
        basis = _zernike_basis(shape, n_terms=10)
        coeffs = rng.normal(0, 1, size=basis.shape[0])
        Wab = np.tensordot(coeffs, basis, axes=(0, 0))
        # RMS 정규화 (단위원 내부 기준)
        mask = basis[0] != 0
        rms = Wab[mask].std() + 1e-9
        Wab = Wab / rms * aberration_rms
        self.aberration = torch.tensor(Wab, dtype=torch.float32, device=device)

    def apply_gamma(self, phase):
        """명령 위상 → 실제 위상 (SLM 비선형 룩업)."""
        x = (phase % (2 * np.pi)) / (2 * np.pi)      # [0,1)
        x = torch.clamp(x, 0.0, 1.0)
        return 2 * np.pi * x.pow(self.gamma)

    def forward(self, phase, add_noise=False):
        """phase: (B,H,W) 또는 (H,W) 명령 위상(rad). 반환: 에너지 정규화 강도."""
        single = phase.dim() == 2
        if single:
            phase = phase.unsqueeze(0)
        phase = phase.to(self.device)

        phi_actual = self.apply_gamma(phase)
        total = phi_actual + self.aberration.unsqueeze(0)
        field_slm = self.A.unsqueeze(0) * torch.exp(1j * total)

        far = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(field_slm, dim=(-2, -1))),
            dim=(-2, -1),
        )
        I = (far.real ** 2 + far.imag ** 2)
        H, W = self.shape
        # 에너지 정규화 후 평균≈1 로 스케일 (수치 안정 / 그래디언트 보존)
        I = I / (I.sum(dim=(-2, -1), keepdim=True) + 1e-12) * (H * W)

        if add_noise and self.read_noise > 0:
            I = I + self.read_noise * I.mean() * torch.randn_like(I)
            I = torch.clamp(I, min=0.0)

        return I.squeeze(0) if single else I
