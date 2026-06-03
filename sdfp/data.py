"""
data.py — 타깃 트위저 배열 생성 + 데이터셋 유틸.

- spot_positions: Square / Triangular / Honeycomb / Random 배열의 (2,N) 좌표
- target_intensity: 좌표 → 가우시안 스팟이 찍힌 목표 강도맵 (정규화)
- random_array: 학습용으로 다양한 배열을 무작위 생성
"""

import numpy as np
import torch


def spot_positions(array_type, shape, rows=4, cols=4, spacing=14,
                   rings=2, n_random=16, rng=None):
    """배열 좌표 (2,N) [x;y] (절대 픽셀). knm/FFT 격자 기준 중심 = shape//2."""
    if rng is None:
        rng = np.random.default_rng()
    H, W = shape
    cx, cy = W // 2, H // 2
    pts = []

    if array_type == "Square":
        for i in range(rows):
            for j in range(cols):
                pts.append([cx + (j - (cols - 1) / 2) * spacing,
                            cy + (i - (rows - 1) / 2) * spacing])

    elif array_type == "Triangular":
        row_h = spacing * np.sqrt(3) / 2
        for i in range(rows):
            off = spacing / 2 if i % 2 else 0
            for j in range(cols):
                pts.append([cx + (j - (cols - 1) / 2) * spacing + off,
                            cy + (i - (rows - 1) / 2) * row_h])

    elif array_type == "Honeycomb":
        pts.append([cx, cy])
        for ring in range(1, rings + 1):
            for d in range(6):
                a = np.radians(d * 60)
                dx, dy = np.cos(a) * spacing * ring, np.sin(a) * spacing * ring
                for step in range(ring):
                    ma = np.radians((d + 2) * 60)
                    pts.append([cx + dx + np.cos(ma) * spacing * step,
                                cy + dy + np.sin(ma) * spacing * step])

    elif array_type == "Random":
        margin = 18
        chosen = []
        tries = 0
        while len(chosen) < n_random and tries < 2000:
            tries += 1
            x = rng.integers(margin, W - margin)
            y = rng.integers(margin, H - margin)
            if all((x - px) ** 2 + (y - py) ** 2 >= (spacing * 0.7) ** 2
                   for px, py in chosen):
                chosen.append((x, y))
        pts = [[x, y] for x, y in chosen]

    return np.array(pts, dtype=np.float64).T  # (2,N)


def target_intensity(positions, shape, spot_sigma=1.2, device="cpu"):
    """좌표 → 가우시안 스팟 강도맵 (에너지 합 = 1)."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    img = np.zeros((H, W), dtype=np.float64)
    for x, y in zip(positions[0], positions[1]):
        img += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * spot_sigma ** 2))
    img = img / (img.sum() + 1e-12) * (H * W)   # 평균≈1 로 스케일 (광학계 강도와 일치)
    return torch.tensor(img, dtype=torch.float32, device=device)


def random_array(shape, rng):
    """학습용 무작위 배열 1개 (타입/크기/간격 랜덤)."""
    t = rng.choice(["Square", "Triangular", "Honeycomb", "Random"])
    spacing = int(rng.integers(11, 20))
    if t in ("Square", "Triangular"):
        rows = int(rng.integers(2, 6)); cols = int(rng.integers(2, 6))
        pos = spot_positions(t, shape, rows=rows, cols=cols, spacing=spacing, rng=rng)
    elif t == "Honeycomb":
        rings = int(rng.integers(1, 3))
        pos = spot_positions(t, shape, spacing=spacing, rings=rings, rng=rng)
    else:
        n = int(rng.integers(6, 20))
        pos = spot_positions(t, shape, spacing=spacing, n_random=n, rng=rng)
    return pos, t
