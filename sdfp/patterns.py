"""
patterns.py — 실험 데이터 수집용 SLM 위상 패턴 생성 (앱/스크립트 공용 순수 함수).

생성 가능한 패턴:
  - blazed grating        : 0차 회피용 오프셋 (모든 패턴에 합성 가능)
  - gamma 캘리브레이션     : 이진 격자(0↔회색값 g) — g별 1차 회절 파워 → 위상 응답 곡선
  - 수차 프로브            : 기울기 다른 blazed grating — 1차 스팟이 화면 곳곳 이동
  - GS 배열 POH           : U-Net 작전 지역과 같은 분포 (Surrogate 학습 메인 데이터)
  - 순수 랜덤 위상         : 다양성 확보
  - 기준(reference) 패턴   : 드리프트 감시용 고정 패턴

모든 위상은 [0, 2π) rad. SLM 에는 8-bit (0=0, 255≈2π) PNG 로 전송.
"""

import io
import json
import zipfile

import numpy as np
from PIL import Image

from .gs import gs
from . import data as D


# ─────────────────────────────────────────────────────
# 기본 유틸
# ─────────────────────────────────────────────────────
def phase_to_u8(phase):
    """[0,2π) 위상 → 8-bit (SLM 업로드 포맷)."""
    p = np.asarray(phase, dtype=np.float64) % (2 * np.pi)
    return (p / (2 * np.pi) * 255).round().astype(np.uint8)


def blazed(shape, period_x=0, period_y=0):
    """0차 회피용 blazed grating 위상. period(px)=0 이면 그 방향 생략.

    period_x=P 이면 1차 회절이 x 방향으로 λf/(P·pitch) 만큼 이동.
    """
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    p = np.zeros((H, W))
    if period_x:
        p += xx / float(period_x)
    if period_y:
        p += yy / float(period_y)
    return (2 * np.pi * p) % (2 * np.pi)


def add_blaze(phase, shape, period_x=0, period_y=0):
    if not period_x and not period_y:
        return np.asarray(phase) % (2 * np.pi)
    return (np.asarray(phase) + blazed(shape, period_x, period_y)) % (2 * np.pi)


def embed_canvas(u8, canvas=(1080, 1920)):
    """계산 격자 POH 를 실제 SLM 캔버스(기본 PLUTO 1080×1920) 중앙에 임베드."""
    H, W = canvas
    h, w = u8.shape
    out = np.zeros((H, W), np.uint8)
    out[(H - h) // 2:(H - h) // 2 + h, (W - w) // 2:(W - w) // 2 + w] = u8
    return out


def make_zip(named_u8s, manifest=None):
    """[(이름, uint8 이미지)] → zip bytes (+manifest.json)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, u8 in named_u8s:
            ib = io.BytesIO()
            Image.fromarray(u8, "L").save(ib, format="PNG")
            z.writestr(f"{name}.png", ib.getvalue())
        if manifest is not None:
            z.writestr("manifest.json",
                       json.dumps(manifest, indent=2, ensure_ascii=False))
    return buf.getvalue()


# ─────────────────────────────────────────────────────
# 패턴 종류별 생성기 — 모두 [(name, phase_float)] 반환
# ─────────────────────────────────────────────────────
def gamma_calib_patterns(shape, n_levels=16, period=8):
    """이진 세로 격자: 열마다 0 ↔ 회색값 g. g 를 0→255 스윕.

    측정법: 각 패턴에서 1차 회절 스팟 파워 P(g) 기록
    → P ∝ sin²(Δφ(g)/2) 로 위상 응답 Δφ(g) 역산 (감마 곡선).
    """
    H, W = shape
    cols = ((np.arange(W) // period) % 2).astype(bool)
    out = []
    for g in np.linspace(0, 255, n_levels).round().astype(int):
        u8 = np.zeros((H, W), np.uint8)
        u8[:, cols] = g
        # u8 자체가 보낼 이미지지만, 일관성 위해 위상으로 환산해 보관
        out.append((f"calib_gamma_g{g:03d}", u8.astype(np.float64) / 255 * 2 * np.pi))
    return out


def probe_patterns(shape, grid=3, max_freq=0.08):
    """수차 프로브: 주파수 (fx,fy) 격자의 blazed grating.

    각 패턴은 1차 스팟 하나를 화면의 서로 다른 위치에 만든다
    → 위치별 스팟 모양/세기에서 시야 전반의 수차 정보가 나온다.
    """
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    fs = np.linspace(-max_freq, max_freq, grid)
    out = []
    for i, fy in enumerate(fs):
        for j, fx in enumerate(fs):
            if abs(fx) < 1e-9 and abs(fy) < 1e-9:
                continue                      # (0,0)=0차 자리 — 제외
            ph = (2 * np.pi * (fx * xx + fy * yy)) % (2 * np.pi)
            out.append((f"probe_fy{i}_fx{j}", ph))
    return out


def gs_array_patterns(shape, n, seed=0, n_iter=40, device="cpu", progress=None):
    """무작위 트위저 배열 → GS POH. Surrogate 학습 메인 데이터 (U-Net 작전 지역)."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        pos, _ = D.random_array(shape, rng)
        tgt = D.target_intensity(pos, shape, device=device)
        ph = gs(tgt, n_iter=n_iter, device=device).cpu().numpy()
        out.append((f"gs_{seed}_{k:04d}", ph))
        if progress:
            progress(k + 1, n)
    return out


def random_phase_patterns(shape, n, seed=0):
    """순수 랜덤 위상 — 다양성/과적합 방지."""
    rng = np.random.default_rng(seed)
    return [(f"rand_{seed}_{k:04d}", rng.uniform(0, 2 * np.pi, shape))
            for k in range(n)]


def reference_pattern(shape, device="cpu"):
    """드리프트 감시용 고정 기준 패턴 (4×4 정사각 배열, 시드 고정)."""
    spacing = max(8, int(min(shape) * 0.12))
    pos = D.spot_positions("Square", shape, rows=4, cols=4, spacing=spacing)
    tgt = D.target_intensity(pos, shape, device=device)
    import torch
    torch.manual_seed(12345)
    ph = gs(tgt, n_iter=60, device=device, seed=12345).cpu().numpy()
    return [("reference", ph)]


def finalize(named_phases, shape, blaze_x=0, blaze_y=0, canvas=None):
    """위상 목록에 blaze 합성 + 8-bit 변환 + (옵션) SLM 캔버스 임베드."""
    out = []
    for name, ph in named_phases:
        u8 = phase_to_u8(add_blaze(ph, shape, blaze_x, blaze_y))
        if canvas is not None:
            u8 = embed_canvas(u8, canvas)
        out.append((name, u8))
    return out
