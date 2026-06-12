"""test_sdfp.py — 기본 정합성 테스트 (CPU, 소형). 실행: python test_sdfp.py

핵심 불변식: 에너지 정규화, 출력 shape/범위, 위상 범위, 체크포인트 메타데이터 왕복,
미세조정 동작, 해상도 비례 spacing. pytest 없이 plain assert 로 작동.
"""
import os
import numpy as np
import torch

from sdfp.models import UNet, Surrogate
from sdfp.data import target_intensity, spot_positions, random_array
from sdfp.gs import gs
from sdfp import infer as INF


def test_target_energy_normalized():
    pos = spot_positions("Square", (64, 64), rows=3, cols=3, spacing=10)
    t = target_intensity(pos, (64, 64))
    assert abs(float(t.sum()) - 64 * 64) < 1.0, "target 에너지 합 ≈ H*W"


def test_unet_output_shape_range():
    net = UNet(base=8)
    y = net(torch.zeros(2, 1, 64, 64))
    assert y.shape == (2, 64, 64)
    assert y.min() >= -np.pi - 1e-3 and y.max() <= np.pi + 1e-3, "atan2 → (-π, π]"


def test_surrogate_energy_normalized():
    sur = Surrogate((64, 64))
    I = sur(torch.zeros(64, 64))
    assert I.shape == (64, 64)
    assert abs(float(I.sum()) - 64 * 64) < 1.0, "surrogate 출력 에너지 정규화"


def test_surrogate_refine_variant():
    sur = Surrogate((48, 48), refine=True)
    I = sur(torch.zeros(48, 48))
    assert I.shape == (48, 48) and float(I.min()) >= 0.0


def test_gs_phase_range():
    pos = spot_positions("Square", (64, 64), rows=2, cols=2, spacing=10)
    t = target_intensity(pos, (64, 64))
    ph = gs(t, n_iter=5)
    assert ph.shape == (64, 64)
    assert float(ph.min()) >= 0.0 and float(ph.max()) < 2 * np.pi + 1e-3


def test_checkpoint_metadata_roundtrip():
    tmp = "sdfp/_test_ckpt.pt"
    sur = Surrogate((64, 64), refine=True)
    net = UNet(base=16)
    torch.save({"surrogate": sur.state_dict(), "unet": net.state_dict(),
                "shape": (64, 64), "unet_base": 16, "surrogate_refine": True}, tmp)
    try:
        net2, sur2, shape, dev = INF.load(tmp, device="cpu")
        assert shape == (64, 64)
        assert sur2.refine is True, "surrogate_refine 메타데이터 복원"
        assert sum(p.numel() for p in net2.parameters()) == \
            sum(p.numel() for p in UNet(base=16).parameters()), "unet_base 복원"
    finally:
        os.remove(tmp)


def test_refine_phase_runs():
    net = UNet(base=8)
    sur = Surrogate((32, 32))
    pos = spot_positions("Square", (32, 32), rows=2, cols=2, spacing=6)
    t = target_intensity(pos, (32, 32))
    peaks = np.stack([np.round(pos[0]).astype(int), np.round(pos[1]).astype(int)])
    ph, dt = INF.refine_phase(net, sur, t, peaks, "cpu", steps=3)
    assert ph.shape == (32, 32) and float(ph.min()) >= 0.0


def test_spacing_scales_with_resolution():
    # 256² 의 평균 spacing 이 128² 의 ~2배인지 (해상도 비례)
    def mean_spread(size):
        rng = np.random.default_rng(0)
        sp = []
        for _ in range(40):
            pos = random_array((size, size), rng)[0]
            sp.append(pos[0].max() - pos[0].min())
        return np.mean(sp)
    r = mean_spread(256) / mean_spread(128)
    assert 1.6 < r < 3.0, f"256² 배열 폭이 128² 의 ~2배여야 (실제 {r:.2f})"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}"); passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
