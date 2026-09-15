"""T0 · H3 2단계 — 임베딩 정렬이 규칙 정규화 너머로 회수하는 파편화의 크기 (전수).

배경: 1단계(규칙 키)는 표면형의 8.2%만 병합했다(t0_analysis.py 실측). 규칙 키는 공백·구두점·역할
접미사만 흡수하므로 '안전사항 준수'와 '안전수칙 준수'처럼 어간이 다른 동의 표기를 합치지 못한다.
이 스크립트는 그 잔여를 임베딩으로 얼마나 회수하는지 잰다.

설계 — 전수. 초기 설계는 롱테일을 표본으로 추정하려 했으나 **편향이 크다**: 표본 내부에서만
클러스터링하면 어떤 키의 진짜 동의어(모집단의 다른 30만 개 중에 있음)를 만날 수 없어 회수율이
구조적으로 0에 수렴한다(스모크 테스트에서 300개 표본 회수 0.0%로 확인). 그래서 326,592개 키를
전부 임베딩하고 전역 클러스터링한다. 층(등장 직무 수)은 표집 단위가 아니라 **보고 차원**이다.

비용: 임베딩 32.7만 건(MPS) + 블록 행렬곱 O(n²). 임계값 3종은 한 번의 패스로 처리한다 —
최저 임계값 이상의 쌍만 모아 두고 임계값별로 union-find 를 다시 돌린다.

지표
  · 전역 키 감소율 = (키 수 - 클러스터 수) / 키 수 — 임베딩이 동의 표기로 합친 비율
  · 결합 회수율 = (표면형 수 - 최종 클러스터 수) / 표면형 수 — 1단계 8.2%와 직접 비교되는 수치
  · 층별 병합 참여율 = 그 층의 키 중 크기 ≥2 클러스터에 속한 비율
  · 층 교차 병합 = 롱테일 키가 빈출 개념에 흡수된 사례 수 (온톨로지 관점에서 가장 값진 회수)

사용:
  .venv/bin/python scripts/t0_embed_recovery.py --out docs/t0
  .venv/bin/python scripts/t0_embed_recovery.py --out docs/t0 --max-keys 30000   # 축소 검증
"""

from __future__ import annotations

import argparse
import csv
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
MODEL_DEFAULT = "BAAI/bge-m3"

# 보고용 층 — 키가 등장하는 직무 수. 표집 단위가 아니다.
def stratum_of(n_jobs: int) -> str:
    if n_jobs >= 10:
        return "jobs>=10"
    if n_jobs >= 3:
        return "jobs 3-9"
    if n_jobs == 2:
        return "jobs=2"
    return "jobs=1"


STRATA_ORDER = ["jobs>=10", "jobs 3-9", "jobs=2", "jobs=1"]

KEY_SQL = """
SELECT DISTINCT cu.detail_category_full_code AS job, ki.kta_type_code AS ktype, ki.description
FROM ncs.kta_items ki
JOIN ncs.performance_criteria p ON p.id = ki.performance_criterion_id
JOIN ncs.competency_unit_elements e ON e.code = p.competency_unit_element_code
JOIN ncs.competency_units cu ON cu.code = e.competency_unit_code
"""

KTA_NAME = {"01": "지식", "02": "기술", "03": "태도"}


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def load_keys(dsn):
    key_surfs = defaultdict(set)
    key_jobs = defaultdict(set)
    key_types = defaultdict(Counter)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as c0:
            c0.execute("SET work_mem = '1GB'")
        with conn.cursor(name="keycur") as cur:
            cur.itersize = 50000
            cur.execute(KEY_SQL)
            for job, ktype, desc in cur:
                d = unicodedata.normalize("NFKC", (desc or "").strip())
                if not d:
                    continue
                k = normalize_key(d)
                if not k:
                    continue
                key_surfs[k].add(d)
                key_jobs[k].add(job)
                key_types[k][ktype] += 1
    n_surface = len({s for v in key_surfs.values() for s in v})
    # 키의 대표 유형 = 최빈 유형
    key_main_type = {k: KTA_NAME.get(c.most_common(1)[0][0], "기타") for k, c in key_types.items()}
    return key_surfs, key_jobs, key_main_type, n_surface


def embed_texts(model_name, texts, device, batch_size):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=True)
    v = np.asarray(vecs, dtype=np.float32)
    bad = ~np.isfinite(v).all(axis=1)
    if bad.any():
        raise RuntimeError(f"임베딩에 비유한 값 {int(bad.sum())}건 — 중단")
    norms = np.linalg.norm(v, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-2):
        raise RuntimeError(f"정규화 실패: norm 범위 [{norms.min():.4f}, {norms.max():.4f}]")
    return v


def collect_pairs(vecs, min_threshold, block, device, max_pairs=80_000_000):
    """코사인 ≥ min_threshold 인 상삼각 쌍을 한 번의 패스로 모은다. torch(MPS) 가능하면 사용."""
    n = len(vecs)
    use_torch = False
    try:
        import torch
        if device in ("mps", "cuda") and (
                (device == "mps" and torch.backends.mps.is_available()) or
                (device == "cuda" and torch.cuda.is_available())):
            use_torch = True
            tv = torch.from_numpy(vecs).to(device)
    except Exception:
        use_torch = False

    I, J, S = [], [], []
    total = 0
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        if use_torch:
            import torch
            sims_t = tv[i0:i1] @ tv.T
            hit = (sims_t >= min_threshold).nonzero(as_tuple=False)
            if hit.numel():
                rows = (hit[:, 0] + i0).to(torch.int64)
                cols = hit[:, 1].to(torch.int64)
                keep = cols > rows
                rows, cols = rows[keep], cols[keep]
                if rows.numel():
                    vals = sims_t[rows - i0, cols]
                    I.append(rows.cpu().numpy().astype(np.int32))
                    J.append(cols.cpu().numpy().astype(np.int32))
                    S.append(vals.cpu().numpy().astype(np.float32))
                    total += int(rows.numel())
            del sims_t
        else:
            sims = vecs[i0:i1] @ vecs.T
            if not np.isfinite(sims).all():
                raise RuntimeError("유사도 행렬에 비유한 값 — 중단")
            r, c = np.nonzero(sims >= min_threshold)
            gr = r + i0
            keep = c > gr
            gr, c = gr[keep], c[keep]
            if len(gr):
                I.append(gr.astype(np.int32))
                J.append(c.astype(np.int32))
                S.append(sims[gr - i0, c].astype(np.float32))
                total += len(gr)
        if total > max_pairs:
            raise RuntimeError(f"쌍이 {total:,}개를 넘었다 — 임계값을 올리거나 max_pairs 를 조정하라")
        if (i0 // block) % 20 == 0:
            print(f"  [pairs] {i1:,}/{n:,} · 누적 쌍 {total:,}", flush=True)
    if not I:
        return (np.array([], np.int32),) * 2 + (np.array([], np.float32),)
    return np.concatenate(I), np.concatenate(J), np.concatenate(S)


def cluster_single_linkage(n, pi, pj, ps, threshold):
    """전이 폐포(union-find). 연쇄 병합에 취약 — 상한 추정치로만 쓴다."""
    uf = UnionFind(n)
    m = ps >= threshold
    for a, b in zip(pi[m], pj[m]):
        uf.union(int(a), int(b))
    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    min_sim = {}
    for a, b, s in zip(pi[m], pj[m], ps[m]):
        root = uf.find(int(a))
        v = float(s)
        if v < min_sim.get(root, 1.0):
            min_sim[root] = v
    return groups, min_sim


def cluster_louvain(n, pi, pj, ps, threshold):
    """유사도 그래프의 Louvain 군집 — 사슬을 모듈성으로 끊는다. 주 추정치.

    임계값 이상 쌍만 엣지로 쓰고 가중치는 코사인. 엣지가 없는 키는 단독 클러스터.
    """
    import networkx as nx
    m = ps >= threshold
    g = nx.Graph()
    for a, b, s in zip(pi[m], pj[m], ps[m]):
        g.add_edge(int(a), int(b), weight=float(s))
    comms = nx.community.louvain_communities(g, weight="weight", seed=SEED, resolution=1.0)
    groups = {}
    for i, c in enumerate(comms):
        groups[("c", i)] = sorted(c)
    in_graph = set(g.nodes())
    for i in range(n):
        if i not in in_graph:
            groups[("s", i)] = [i]
    # 클러스터 내 최소 쌍 유사도 — 엣지가 있는 쌍만 대상(같은 군집에 속한 엣지)
    node_comm = {v: k for k, mem in groups.items() for v in mem}
    min_sim = {}
    for a, b, s in zip(pi[m], pj[m], ps[m]):
        ca, cb = node_comm[int(a)], node_comm[int(b)]
        if ca == cb:
            v = float(s)
            if v < min_sim.get(ca, 1.0):
                min_sim[ca] = v
    return groups, min_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("DATABASE_URL", "postgresql://ncs:ncs@localhost:55432/ncs"))
    ap.add_argument("--out", default="docs/t0")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--thresholds", default="0.85,0.90,0.95")
    ap.add_argument("--primary-threshold", type=float, default=0.90)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--max-keys", type=int, default=None, help="축소 검증용 — 무작위 부분집합")
    ap.add_argument("--examples", type=int, default=40)
    ap.add_argument("--cache", default=None, help="임베딩 캐시 경로(.npz). 기본: <out>/h3_vecs_<n>.npz")
    ap.add_argument("--no-cache", action="store_true", help="캐시를 무시하고 재계산")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    thresholds = sorted(float(t) for t in args.thresholds.split(","))

    device = args.device
    if device is None:
        try:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            device = "cpu"

    print("[h3] 규칙 키 적재 …", flush=True)
    key_surfs, key_jobs, key_main_type, n_surface = load_keys(args.dsn)
    keys = sorted(key_surfs)
    if args.max_keys and len(keys) > args.max_keys:
        keys = sorted(random.Random(SEED).sample(keys, args.max_keys))
        print(f"[h3] 축소 검증 모드 — 키 {len(keys):,}개만 사용", flush=True)
    n = len(keys)
    strat = [stratum_of(len(key_jobs[k])) for k in keys]
    ktype_of = [key_main_type.get(k, "기타") for k in keys]
    print(f"[h3] 표면형 {n_surface:,} · 규칙 키 {len(key_surfs):,} (분석 {n:,}) · "
          f"층 분포 {dict(Counter(strat))}", flush=True)

    cache = Path(args.cache) if args.cache else (out / f"h3_vecs_{n}.npz")
    if cache.exists() and not args.no_cache:
        z = np.load(cache, allow_pickle=True)
        if list(z["keys"]) == keys:
            vecs = z["vecs"]
            print(f"[h3] 임베딩 캐시 사용 {cache} {vecs.shape}", flush=True)
        else:
            print("[h3] 캐시의 키 집합이 달라 재계산", flush=True)
            vecs = None
    else:
        vecs = None
    if vecs is None:
        print(f"[h3] 임베딩 — {args.model} · device {device} · batch {args.batch_size}", flush=True)
        vecs = embed_texts(args.model, keys, device, args.batch_size)
        np.savez(cache, vecs=vecs, keys=np.array(keys, dtype=object))
        print(f"[h3] 임베딩 완료 {vecs.shape} · 캐시 저장 {cache}", flush=True)

    pair_cache = cache.with_name(cache.stem + f"_pairs_{thresholds[0]}.npz")
    if pair_cache.exists() and not args.no_cache:
        z = np.load(pair_cache)
        pi, pj, ps = z["pi"], z["pj"], z["ps"]
        print(f"[h3] 쌍 캐시 사용 {pair_cache} · {len(ps):,}개", flush=True)
    else:
        print(f"[h3] 쌍 수집 (cos ≥ {thresholds[0]}) …", flush=True)
        pi, pj, ps = collect_pairs(vecs, thresholds[0], args.block, device)
        np.savez(pair_cache, pi=pi, pj=pj, ps=ps)
        print(f"[h3] 임계값 {thresholds[0]} 이상 쌍 {len(ps):,}개 · 캐시 저장", flush=True)

    surfaces_total = sum(len(key_surfs[k]) for k in keys)

    def summarize(groups, min_sim, th, want_examples):
        n_clusters = len(groups)
        merged_idx = {i for g in groups.values() if len(g) > 1 for i in g}
        per_stratum = {}
        for s in STRATA_ORDER:
            idxs = [i for i in range(n) if strat[i] == s]
            if not idxs:
                continue
            merged = sum(1 for i in idxs if i in merged_idx)
            per_stratum[s] = {"n_keys": len(idxs), "merged_keys": merged,
                              "merge_participation": merged / len(idxs)}
        cross = sum(1 for g in groups.values() if len(g) > 1 and len({strat[i] for i in g}) > 1)
        # KSA 유형별 병합 참여율 — 회수가 '태도' 항목에 쏠렸는지 확인
        per_type = {}
        for t in ("지식", "기술", "태도"):
            idxs = [i for i in range(n) if ktype_of[i] == t]
            if not idxs:
                continue
            merged = sum(1 for i in idxs if i in merged_idx)
            per_type[t] = {"n_keys": len(idxs), "merged_keys": merged,
                           "merge_participation": merged / len(idxs)}
        sizes = sorted((len(g) for g in groups.values()), reverse=True)
        res = {
            "n_keys": n, "n_clusters": n_clusters,
            "key_reduction_rate": (n - n_clusters) / n,
            "n_surface_forms_in_scope": surfaces_total,
            "combined_recovery_vs_surface": (surfaces_total - n_clusters) / surfaces_total,
            "n_merged_clusters": sum(1 for g in groups.values() if len(g) > 1),
            "cross_stratum_clusters": cross,
            "largest_cluster": sizes[0] if sizes else 0,
            "top10_cluster_sizes": sizes[:10],
            "keys_in_largest_cluster_pct": (sizes[0] / n) if sizes else 0,
            "per_stratum": per_stratum,
            "per_kta_type": per_type,
        }
        if want_examples:
            ex = []
            for root, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                if len(members) < 2 or len(ex) >= args.examples:
                    continue
                mk = [keys[i] for i in members]
                ex.append({"n_keys": len(mk), "keys": mk[:10],
                           "n_jobs": len(set().union(*[key_jobs[k] for k in mk])),
                           "strata": sorted({strat[i] for i in members}),
                           "kta_mix": dict(Counter(ktype_of[i] for i in members).most_common()),
                           "min_pair_sim": round(min_sim.get(root, 1.0), 4)})
            res["examples"] = ex
        return res

    results = {}
    for th in thresholds:
        primary = (th == args.primary_threshold)
        sl_groups, sl_min = cluster_single_linkage(n, pi, pj, ps, th)
        sl = summarize(sl_groups, sl_min, th, want_examples=False)
        del sl_groups
        print(f"[h3] cos≥{th} 단일연결(상한): 클러스터 {sl['n_clusters']:,} · "
              f"결합 회수율 {sl['combined_recovery_vs_surface']:.1%} · "
              f"최대 클러스터 {sl['largest_cluster']:,} ({sl['keys_in_largest_cluster_pct']:.1%})", flush=True)
        lv_groups, lv_min = cluster_louvain(n, pi, pj, ps, th)
        lv = summarize(lv_groups, lv_min, th, want_examples=primary)
        del lv_groups
        print(f"[h3] cos≥{th} Louvain(주): 클러스터 {lv['n_clusters']:,} · "
              f"키 감소 {lv['key_reduction_rate']:.1%} · "
              f"결합 회수율 {lv['combined_recovery_vs_surface']:.1%} · "
              f"최대 클러스터 {lv['largest_cluster']:,} · 층 교차 {lv['cross_stratum_clusters']:,}", flush=True)
        results[str(th)] = {"louvain": lv, "single_linkage": sl}

    meta = {"run_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "model": args.model, "device": device, "seed": SEED,
            "thresholds": thresholds, "primary_threshold": args.primary_threshold,
            "design": "전수 클러스터링(표본 추정 아님). 층은 보고 차원. "
                      "주 추정치는 Louvain(연쇄 병합 저항), 단일연결은 상한으로 병기.",
            "n_surface_forms_total": n_surface,
            "n_pairs_above_min_threshold": int(len(ps)),
            "stage1_rule_recovery_vs_surface": 0.08233842937703886,
            "note": "결합 회수율 = (표면형 - 최종 클러스터)/표면형 — 1단계 8.2%와 직접 비교되는 수치. "
                    "단일연결은 전이 폐포라 사슬로 거대 성분을 만든다(실측: cos≥0.90에서 최대 성분이 "
                    "키의 5.2%인 17,118개, cos≥0.85에서 29.9%). 따라서 주 수치는 Louvain."}
    (out / "t0_h3_embedding.json").write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    pt = str(args.primary_threshold)
    with open(out / "t0_h3_embedding_merges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_keys", "n_jobs", "strata", "min_pair_sim", "keys"])
        for e in results[pt]["louvain"].get("examples", []):
            w.writerow([e["n_keys"], e["n_jobs"], "|".join(e["strata"]), e["min_pair_sim"],
                        "|".join(e["keys"])])

    print(f"\n[h3] 완료 → {out}/t0_h3_embedding.json")
    print(f"  1단계(규칙) 결합 회수율 8.2%  →  2단계 포함(Louvain 주 추정):")
    for th in thresholds:
        lv = results[str(th)]["louvain"]; sl = results[str(th)]["single_linkage"]
        print(f"  cos≥{th}: {lv['combined_recovery_vs_surface']:.1%} "
              f"(키 감소 {lv['key_reduction_rate']:.1%}, 최대 클러스터 {lv['largest_cluster']:,}) "
              f"· 단일연결 상한 {sl['combined_recovery_vs_surface']:.1%} "
              f"(최대 {sl['largest_cluster']:,})")


if __name__ == "__main__":
    main()
