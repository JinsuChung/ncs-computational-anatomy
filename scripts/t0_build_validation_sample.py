"""T0 · H3 병합 정밀도 수동 검증용 표본 추출.

H3 2단계는 "임베딩이 규칙 너머로 28.1%를 더 회수한다"고 말하지만, 그 병합이 **옳은지**는
사람이 봐야 안다. 이 스크립트는 판정용 쌍 표본을 뽑는다.

두 팔(arm) — 서로 다른 질문에 답한다:
  A  직접 엣지 쌍 : 코사인 임계값 이상으로 직접 연결된 쌍. 대역별(.85–.90 / .90–.95 / .95–1.0)
                    로 층화 → "유사도 신호 자체가 옳은가", 그리고 임계값 선택의 실증 근거.
  B  전이 동반 쌍 : cos≥.90 Louvain 군집의 같은 클러스터에 속하되 **직접 엣지가 없는** 쌍.
                    → "전이적으로 묶인 결정이 옳은가" = 연쇄 병합 피해의 직접 측정.

앵커링 차단(프로젝트 표준): 판정 화면에 코사인·클러스터 크기·팔(arm)을 노출하지 않는다.
이 스크립트는 그 값들을 별도 키(`_hidden`)에 담아 두고, UI 는 판정 완료 후에만 쓴다.

출력: docs/t0/t0_h3_validation_sample.json  (UI 가 그대로 임베드)

사용:
  .venv/bin/python scripts/t0_build_validation_sample.py --out docs/t0 --n-edge 120 --n-transitive 80
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ontology_keys import normalize_key  # noqa: E402

SEED = 20260909
BANDS = [(0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
KTA_NAME = {"01": "지식", "02": "기술", "03": "태도"}

KEY_SQL = """
SELECT DISTINCT cu.detail_category_full_code AS job, ki.kta_type_code AS ktype, ki.description,
       cp.major_category_name AS major
FROM ncs.kta_items ki
JOIN ncs.performance_criteria p ON p.id = ki.performance_criterion_id
JOIN ncs.competency_unit_elements e ON e.code = p.competency_unit_element_code
JOIN ncs.competency_units cu ON cu.code = e.competency_unit_code
JOIN ncs.classification_paths cp ON cp.detail_category_full_code = cu.detail_category_full_code
"""


def load(dsn):
    key_surfs = defaultdict(set)
    key_jobs = defaultdict(set)
    key_types = defaultdict(Counter)
    key_majors = defaultdict(Counter)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as c0:
            c0.execute("SET work_mem = '1GB'")
        with conn.cursor(name="c") as cur:
            cur.itersize = 50000
            cur.execute(KEY_SQL)
            for job, ktype, desc, major in cur:
                d = unicodedata.normalize("NFKC", (desc or "").strip())
                if not d:
                    continue
                k = normalize_key(d)
                if not k:
                    continue
                key_surfs[k].add(d)
                key_jobs[k].add(job)
                key_types[k][ktype] += 1
                key_majors[k][major] += 1
    return key_surfs, key_jobs, key_types, key_majors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("DATABASE_URL", "postgresql://ncs:ncs@localhost:55432/ncs"))
    ap.add_argument("--out", default="docs/t0")
    ap.add_argument("--cache", default="docs/t0/h3_vecs_326592.npz")
    ap.add_argument("--pairs", default="docs/t0/h3_vecs_326592_pairs_0.85.npz")
    ap.add_argument("--n-edge", type=int, default=120)
    ap.add_argument("--n-transitive", type=int, default=80)
    ap.add_argument("--cluster-threshold", type=float, default=0.90)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print("[val] 키 메타 적재 …", flush=True)
    key_surfs, key_jobs, key_types, key_majors = load(args.dsn)

    z = np.load(args.cache, allow_pickle=True)
    keys = list(z["keys"])
    idx_of = {k: i for i, k in enumerate(keys)}
    zp = np.load(args.pairs)
    pi, pj, ps = zp["pi"], zp["pj"], zp["ps"]
    print(f"[val] 키 {len(keys):,} · 쌍 {len(ps):,}", flush=True)

    # ── 팔 A: 대역별 직접 엣지
    edge_sample = []
    for lo, hi in BANDS:
        m = np.nonzero((ps >= lo) & (ps < hi))[0]
        take = min(args.n_edge // len(BANDS), len(m))
        for t in rng.sample(list(m), take):
            edge_sample.append((int(pi[t]), int(pj[t]), float(ps[t]), f"{lo:.2f}-{hi:.2f}"))
    print(f"[val] 팔 A 직접 엣지 {len(edge_sample)}쌍", flush=True)

    # ── 팔 B: cos≥threshold Louvain 클러스터의 비인접 동반 쌍
    import networkx as nx
    m = ps >= args.cluster_threshold
    g = nx.Graph()
    for a, b, s in zip(pi[m], pj[m], ps[m]):
        g.add_edge(int(a), int(b), weight=float(s))
    comms = nx.community.louvain_communities(g, weight="weight", seed=SEED, resolution=1.0)
    comms = [sorted(c) for c in comms if len(c) >= 3]
    print(f"[val] cos≥{args.cluster_threshold} 클러스터 {len(comms):,}개(크기 3 이상)", flush=True)

    trans_sample = []
    tries = 0
    while len(trans_sample) < args.n_transitive and tries < args.n_transitive * 400:
        tries += 1
        c = comms[rng.randrange(len(comms))]
        a, b = rng.sample(c, 2)
        if g.has_edge(a, b):          # 직접 연결이면 팔 A 와 겹친다 — 제외
            continue
        if any(x[0] == a and x[1] == b for x in trans_sample):
            continue
        # 이 쌍의 실제 코사인(참고용, 판정 중에는 숨김)
        va, vb = z["vecs"][a], z["vecs"][b]
        cos = float(np.dot(va, vb))
        trans_sample.append((a, b, cos, f"transitive(cluster={len(c)})"))
    print(f"[val] 팔 B 전이 동반 {len(trans_sample)}쌍", flush=True)

    def side(i):
        k = keys[i]
        surfs = sorted(key_surfs[k], key=lambda s: (len(s), s))[:4]
        t = key_types[k].most_common(1)[0][0]
        return {"key": k, "surfaces": surfs, "kta": KTA_NAME.get(t, "기타"),
                "n_jobs": len(key_jobs[k]),
                "majors": [m for m, _ in key_majors[k].most_common(3)]}

    items = []
    for arm, rows in (("A", edge_sample), ("B", trans_sample)):
        for a, b, cos, band in rows:
            pid = hashlib.md5(f"{keys[a]}||{keys[b]}".encode()).hexdigest()[:12]
            items.append({
                "id": pid,
                "left": side(a), "right": side(b),
                "_hidden": {"arm": arm, "cosine": round(cos, 4), "band": band},
            })
    rng.shuffle(items)   # 팔·대역이 순서로 드러나지 않게 섞는다

    payload = {
        "meta": {
            "built_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "seed": SEED, "n_items": len(items),
            "arms": {"A": "직접 엣지(대역 층화) — 유사도 신호의 정밀도",
                     "B": f"cos≥{args.cluster_threshold} Louvain 클러스터의 비인접 동반 쌍 — 전이 병합의 정밀도"},
            "bands": [f"{lo:.2f}-{hi:.2f}" for lo, hi in BANDS],
            "anchor_hidden": ["arm", "cosine", "band", "cluster_size"],
            "source": "docs/t0/t0_h3_embedding.json (cos≥.90 Louvain, 2026-09-09)",
        },
        "items": items,
    }
    p = out / "t0_h3_validation_sample.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[val] 완료 → {p} ({len(items)}쌍)")
    print("  팔 구성:", dict(Counter(i["_hidden"]["arm"] for i in items)))
    print("  대역 구성:", dict(Counter(i["_hidden"]["band"] for i in items)))


if __name__ == "__main__":
    main()
