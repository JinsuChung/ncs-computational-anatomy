"""T0 · H3 병합 정밀도 검증 2차(R2) — 약한 다리 하나만 겨냥한 소규모 표본.

1차(200쌍, 8인)에서 크기 21 이상 군집의 전이 정밀도는 문항 약 7개에 기댔다. 이 표본은 그 구간만 채운다:
  · 전이 동반 쌍 60 — cos≥.90 Louvain 군집 중 크기 21–100 에서 30, 100 초과에서 30 (직접 엣지 없는 쌍, 1차와 중복 제외)
  · 보정 앵커 10 — 1차 앵커 50 중 대역 층화로 10 (직접 대역 2×3 + 전이 4). 새 판정자의 엄격도를 1차 8인과 대조한다.

배정: 새 판정자 4인(EV-6·EV-7 학생, EV-8·EV-9 직원). 앵커 10은 전원, 새 문항 60은 각 3인(4개 leave-one-out 패턴 × 15,
층 균형) → 문항당 3판정 다수결. 1인 55문항, 1차 속도(중앙값 6–10초/문항)면 7–9분.

출력: docs/t0/t0_h3_validation_sample_r2.json (UI 가 임베드; 비추적)
사용: .venv/bin/python scripts/t0_build_validation_sample_r2.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t0_build_validation_sample import KTA_NAME, load  # noqa: E402

SEED = 20260911
RATERS = ["EV-6", "EV-7", "EV-8", "EV-9"]
STRATA = {"21-100": (21, 100), "100+": (101, 10**9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("DATABASE_URL", "postgresql://ncs:ncs@localhost:55432/ncs"))
    ap.add_argument("--r1", default="docs/t0/t0_h3_validation_sample.json")
    ap.add_argument("--cache", default="docs/t0/h3_vecs_326592.npz")
    ap.add_argument("--pairs", default="docs/t0/h3_vecs_326592_pairs_0.85.npz")
    ap.add_argument("--per-stratum", type=int, default=30)
    ap.add_argument("--n-anchor", type=int, default=10)
    ap.add_argument("--out", default="docs/t0/t0_h3_validation_sample_r2.json")
    args = ap.parse_args()
    rng = random.Random(SEED)

    r1 = json.loads(Path(args.r1).read_text(encoding="utf-8"))
    r1_ids = {it["id"] for it in r1["items"]}
    r1_anchor = [it for it in r1["items"] if it["block"] == "anchor"]

    print("[r2] 키 메타 적재 …", flush=True)
    key_surfs, key_jobs, key_types, key_majors = load(args.dsn)
    z = np.load(args.cache, allow_pickle=True)
    keys = list(z["keys"]); vecs = z["vecs"]
    zp = np.load(args.pairs)
    pi, pj, ps = zp["pi"], zp["pj"], zp["ps"]

    import networkx as nx
    m = ps >= 0.90
    g = nx.Graph()
    for a, b, s in zip(pi[m], pj[m], ps[m]):
        g.add_edge(int(a), int(b), weight=float(s))
    comms = [sorted(c) for c in nx.community.louvain_communities(g, weight="weight", seed=20260909, resolution=1.0)]
    by_stratum = {name: [c for c in comms if lo <= len(c) <= hi] for name, (lo, hi) in STRATA.items()}
    print("[r2] 군집 수:", {k: len(v) for k, v in by_stratum.items()}, flush=True)

    def pid_of(a, b):
        return hashlib.md5(f"{keys[a]}||{keys[b]}".encode()).hexdigest()[:12]

    new_items = []
    for name, cl in by_stratum.items():
        seen = set(); tries = 0
        while sum(1 for x in new_items if x["_hidden"]["stratum"] == name) < args.per_stratum and tries < 20000:
            tries += 1
            c = cl[rng.randrange(len(cl))]         # 군집 단위 균등 → 거대 군집 한 개가 표본을 독점하지 않음
            a, b = rng.sample(c, 2)
            if g.has_edge(a, b) or (a, b) in seen or (b, a) in seen:
                continue
            pid = pid_of(a, b)
            if pid in r1_ids:
                continue
            seen.add((a, b))
            new_items.append({"id": pid, "a": a, "b": b, "_hidden": {"arm": "B", "cosine": round(float(vecs[a] @ vecs[b]), 4),
                                                                   "band": "transitive", "stratum": name, "cluster_size": len(c)}})
    print("[r2] 새 전이 쌍:", Counter(x["_hidden"]["stratum"] for x in new_items), flush=True)

    def side(i):
        k = keys[i]
        t = key_types[k].most_common(1)[0][0]
        return {"key": k, "surfaces": sorted(key_surfs[k], key=lambda s: (len(s), s))[:4], "kta": KTA_NAME.get(t, "기타"),
                "n_jobs": len(key_jobs[k]), "majors": [mj for mj, _ in key_majors[k].most_common(3)]}

    # 배정: 4 leave-one-out 패턴, 층별 라운드로빈
    patterns = [[r for r in RATERS if r != drop] for drop in RATERS]
    items = []
    for name in STRATA:
        grp = [x for x in new_items if x["_hidden"]["stratum"] == name]
        rng.shuffle(grp)
        for i, x in enumerate(grp):
            items.append({"id": x["id"], "left": side(x["a"]), "right": side(x["b"]), "_hidden": x["_hidden"],
                          "block": "r2", "raters": patterns[i % 4]})
    # 보정 앵커: 대역 층화 (직접 2×3 + 전이 4)
    quota = {"0.85-0.90": 2, "0.90-0.95": 2, "0.95-1.01": 2, "transitive": args.n_anchor - 6}
    for band, q in quota.items():
        pool = [it for it in r1_anchor if it["_hidden"]["band"] == band]
        for it in rng.sample(pool, min(q, len(pool))):
            items.append({"id": it["id"], "left": it["left"], "right": it["right"],
                          "_hidden": {**it["_hidden"], "calibration": True, "stratum": "anchor-r1"},
                          "block": "anchor", "raters": list(RATERS)})
    rng.shuffle(items)

    per_rater = {r: sum(1 for it in items if r in it["raters"]) for r in RATERS}
    payload = {"meta": {"built_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "seed": SEED,
                        "round": 2, "n_items": len(items), "raters": RATERS, "per_rater_items": per_rater,
                        "design": "전이 쌍 60(크기 21–100 / 100+ 각 30, 각 3인 판정) + 1차 앵커 10(전원, 엄격도 보정)",
                        "strata": {k: sum(1 for it in items if it["_hidden"].get("stratum") == k) for k in list(STRATA) + ["anchor-r1"]},
                        "anchor_hidden": ["arm", "cosine", "band", "cluster_size", "stratum", "calibration"],
                        "source": "docs/t0/h3_vecs_326592_pairs_0.85.npz · cos≥.90 Louvain(seed 20260909)"},
               "items": items}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[r2] 완료 → {args.out} · 문항 {len(items)} · 판정자별 {per_rater}")


if __name__ == "__main__":
    main()
