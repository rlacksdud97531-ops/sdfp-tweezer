"""
train_real_surrogate.py — 측정 데이터로 Surrogate 재학습 + U-Net distill.

collect_data.py 가 만든 measured/*.npz 를 읽어:
  [1] 드리프트 점검 (기준 패턴들의 시간 변화 출력)
  [2] Surrogate 를 실측 (φ, I) 로 학습   ← 시뮬레이션 대신 진짜 광학계
  [3] U-Net 을 그 Surrogate 로 distill   (타깃 배열은 합성 — 측정 불필요)
  [4] checkpoint_real.pt 저장 → app_sdfp.py 등에서 그대로 사용

사용법:
    PYTHONPATH=. python lab/train_real_surrogate.py --data measured --size 128
(--size 는 패턴 생성 때 쓴 계산 격자 크기와 같아야 함)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdfp.models import Surrogate, UNet              # noqa: E402
from sdfp.train import (train_predictor, make_samples,  # noqa: E402
                        collect_unet_phases, pick_device)
import torch.nn.functional as F                       # noqa: E402


def load_measured(data_dir, shape, device):
    """npz 폴더 → (phases[rad], intensities[평균≈1 스케일]) 텐서."""
    phases, intens, refs = [], [], []
    for f in sorted(Path(data_dir).glob("*.npz")):
        if "darks" in f.stem:                 # darks.npz 는 학습데이터 아님 (meta 없음)
            continue
        d = np.load(f, allow_pickle=True)
        if "meta" not in d.files:             # 안전장치: meta 없는 npz 건너뜀
            continue
        meta = json.loads(str(d["meta"]))
        u8 = d["phase_u8"].astype(np.float64)
        # 캔버스 임베드본이면 중앙 crop
        if u8.shape != shape:
            H, W = u8.shape
            h, w = shape
            u8 = u8[(H - h) // 2:(H - h) // 2 + h, (W - w) // 2:(W - w) // 2 + w]
        phase = u8 / 255 * 2 * np.pi
        I = d["intensity"].astype(np.float64)
        # ⚠️ 카메라 이미지는 카메라 해상도 — 좌표 정합 후 계산 격자로 리샘플 필요.
        #    (affine 정합 행렬을 구했다면 여기서 cv2.warpAffine 적용)
        if I.shape != shape:
            from PIL import Image
            I = np.array(Image.fromarray(I).resize(shape[::-1], Image.BILINEAR))
        I = I / (I.sum() + 1e-12) * (shape[0] * shape[1])   # 평균≈1 (코드 규약)
        if meta.get("is_reference"):
            refs.append((meta["t"], I))
        else:
            phases.append(phase)
            intens.append(I)
    print(f"학습쌍 {len(phases)}개, 기준패턴 {len(refs)}개 로드")

    # [1] 드리프트 점검: 기준 패턴 강도의 시간 변화 (첫 기준 대비 RMS)
    if len(refs) >= 2:
        t0, I0 = refs[0]
        for t, I in refs[1:]:
            drift = np.sqrt(np.mean((I - I0) ** 2)) / I0.mean()
            print(f"  드리프트 t+{(t - t0)/60:5.1f}분: RMS {drift*100:.2f}%"
                  + ("  ⚠️ 5% 초과 — 해당 구간 재측정 권장" if drift > 0.05 else ""))

    ph = torch.tensor(np.stack(phases), dtype=torch.float32, device=device)
    It = torch.tensor(np.stack(intens), dtype=torch.float32, device=device)
    return ph, It


def fit_surrogate(phases, intens, shape, device, epochs=40, bs=8, lr=3e-3,
                  model=None):
    """실측 (φ,I) 로 Surrogate 학습 (sdfp.train 과 동일 손실)."""
    sur = model if model is not None else Surrogate(shape).to(device)
    for p in sur.parameters():
        p.requires_grad_(True)
    sur.train()
    opt = torch.optim.Adam(sur.parameters(), lr=lr)
    n = phases.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            Ip = sur(phases[idx])
            I = intens[idx]
            loss = F.l1_loss(Ip, I) + F.mse_loss(torch.sqrt(Ip + 1e-9),
                                                 torch.sqrt(I + 1e-9))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f"  [surrogate-real] epoch {ep+1}/{epochs}  loss={tot/n:.3e}")
    return sur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="measured")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", default="sdfp/checkpoint_real.pt")
    ap.add_argument("--pred-epochs", type=int, default=60)
    args = ap.parse_args()

    device = pick_device()
    shape = (args.size, args.size)
    print(f"device={device}  shape={shape}")

    # [2] Surrogate ← 실측 데이터
    phases, intens = load_measured(args.data, shape, device)
    sur = fit_surrogate(phases, intens, shape, device)

    # [3] U-Net distill (타깃은 합성 — 측정 불필요)
    samples = make_samples(shape, 1600, device, seed=2)
    net = train_predictor(sur, samples, shape, device,
                          epochs=args.pred_epochs, warmup_ep=0)

    # 다음 DAgger 라운드용: U-Net 출력 위상들을 패턴 PNG 로 내보내기
    #   → 이 PNG 들을 다시 SLM 에 띄워 측정(collect_data.py) → 본 스크립트 재실행
    from sdfp.patterns import phase_to_u8
    from PIL import Image
    dagger_dir = Path("patterns_dagger"); dagger_dir.mkdir(exist_ok=True)
    for k, ph in enumerate(collect_unet_phases(net, samples, device, max_n=200)):
        Image.fromarray(phase_to_u8(ph.cpu().numpy()), "L").save(
            dagger_dir / f"dagger_{k:04d}.png")
    print(f"DAgger 패턴 200장 → {dagger_dir}/ (다음 측정 라운드에 사용)")

    torch.save({"surrogate": sur.state_dict(), "unet": net.state_dict(),
                "shape": shape}, args.out)
    print(f"저장: {args.out}  ← app_sdfp.py 에서 이 체크포인트 지정하면 됨")


if __name__ == "__main__":
    main()
