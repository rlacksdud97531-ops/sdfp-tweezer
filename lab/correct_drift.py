"""
correct_drift.py — 드리프트 보정 계수 산출 (reference 기반).

수집 중 레이저/열 드리프트로 전역 강도가 변함. 50장마다 찍은 reference 의 총강도로
시간축 드리프트 곡선을 만들고, 각 데이터 패턴을 그 시점 레벨로 정규화하는 계수를 계산.

데이터 11GB 를 복제하지 않고 **사이드카 JSON** 만 생성:
    drift_correction.json = { name: {t, ref_level, drift_factor, cold_start}, "_meta": {...} }

GPU 학습 로더 적용법:
    corr = json.load(open("drift_correction.json"))
    intensity = npz["intensity"] * corr[name]["drift_factor"]   # 전역 드리프트 제거
    if corr[name]["cold_start"]: down-weight or skip            # 워밍업 구간

사용:
    python lab/correct_drift.py --data C:\tweezer_data\measured512
    (numpy 만 있으면 어디서든 실행 가능 — GPU 컴퓨터에서 돌려도 됨)
"""
import os, json, glob, argparse
import numpy as np


def meta_of(p):
    z = np.load(p, allow_pickle=True)
    return json.loads(str(z["meta"])), float(z["intensity"].sum())


def ref_total(p):
    z = np.load(p, allow_pickle=True)
    m = json.loads(str(z["meta"]))
    return float(m["t"]), float(z["intensity"].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\tweezer_data\measured512")
    ap.add_argument("--out", default=None, help="사이드카 경로 (기본: data/drift_correction.json)")
    ap.add_argument("--cold-thresh", type=float, default=0.6,
                    help="레벨이 중앙값×이값 미만이면 cold_start 표시 (기본 0.6)")
    args = ap.parse_args()
    D = args.data
    out = args.out or os.path.join(D, "drift_correction.json")

    # 1) reference 드리프트 곡선 (t, total) — 시간순 정렬
    refs = sorted(glob.glob(os.path.join(D, "reference_*.npz")))
    assert refs, "reference_*.npz 없음"
    rt = np.array([ref_total(p) for p in refs])          # (R,2): t, total
    rt = rt[np.argsort(rt[:, 0])]
    t_ref, lvl_ref = rt[:, 0], rt[:, 1]
    median_lvl = float(np.median(lvl_ref))
    print(f"reference {len(refs)}장  레벨 중앙값={median_lvl:.3e}  "
          f"범위 [{lvl_ref.min():.3e}, {lvl_ref.max():.3e}]")

    def level_at(t):
        """시점 t 의 드리프트 레벨 (reference 선형보간, 범위밖은 끝값)."""
        return float(np.interp(t, t_ref, lvl_ref))

    # 2) 모든 데이터 패턴에 계수 부여
    data = sorted(glob.glob(os.path.join(D, "gs_*.npz")) +
                  glob.glob(os.path.join(D, "rand_*.npz")))
    corr = {}
    n_cold = 0
    for i, p in enumerate(data):
        name = os.path.basename(p)[:-4]
        m, _ = meta_of(p)
        t = float(m["t"])
        lvl = level_at(t)
        factor = median_lvl / max(lvl, 1e-9)             # 이 시점 강도를 중앙값 레벨로
        cold = lvl < args.cold_thresh * median_lvl
        n_cold += int(cold)
        corr[name] = {"t": t, "ref_level": lvl,
                      "drift_factor": round(factor, 5), "cold_start": bool(cold)}
        if i % 1000 == 0:
            print(f"  {i}/{len(data)}")

    corr["_meta"] = {
        "median_ref_level": median_lvl,
        "method": "intensity *= drift_factor (전역 reference 총강도 기준 정규화)",
        "cold_thresh": args.cold_thresh,
        "n_cold_start": n_cold,
        "n_patterns": len(data),
        "ref_curve": {"t": t_ref.tolist(), "level": lvl_ref.tolist()},
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(corr, fh, ensure_ascii=False)
    print(f"\n저장: {out}")
    print(f"cold_start 표시: {n_cold}/{len(data)}장 (워밍업 구간 — 학습 시 down-weight 권장)")
    print(f"drift_factor 범위: "
          f"{min(v['drift_factor'] for k,v in corr.items() if k!='_meta'):.3f} ~ "
          f"{max(v['drift_factor'] for k,v in corr.items() if k!='_meta'):.3f}")


if __name__ == "__main__":
    main()
