# -*- coding: utf-8 -*-
"""다운스트림 시연 (NCS) — 개념층 정책이 실제 과제 결과를 바꾸는가 (심사 지적 ①, 2차 ②③).

같은 KSA 데이터를 개념층 정책으로 표현하고, 두 유사도(비가중 Jaccard / TF-IDF 코사인)로 두 과제를 비교한다.
  개념층: rules(정규화 키만) · size2 · le5 · le20 · all(Louvain@.90 크기 상한) · both_le5 / both_all(bge ∩ e5 엣지)
  유사도: jaccard — 편재 개념(예: '기준·법규 준수' 계열 1,084키·636직무)이 전 쌍을 끌어올릴 수 있음
          tfidf   — idf = log(N/df) 로 편재 개념을 다운웨이팅한 코사인. 붕괴가 가중치 문제인지 정책 문제인지 가른다.
  과제 A  직무 유사 검색: 최근접 5개가 같은 소분류/중분류인 비율(P@5). 기준 = 공식 분류(§4.4 가 부분적이라 논증한 그 분류 —
          따라서 '품질'이 아니라 '공식 분류와의 정합'으로 읽는다). 직무 단위 부트스트랩 95% CI.
  과제 B  분류 복원: 유사도 그래프 → Louvain(1.0) → NMI·AMI. Louvain 시드 10개의 범위와 순열 영분포를 함께 낸다.
출력: docs/t0/t0_downstream_ncs.json · t0_downstream_ncs_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t0_embed_recovery import SEED, cluster_louvain, collect_pairs, load_keys  # noqa: E402

OUT = Path("docs/t0")
N_BOOT = 1000
SEEDS = [SEED + i for i in range(10)]


def assign_from_groups(n, groups, cap):
    cid = list(range(n))
    for gi, mem in enumerate(groups.values()):
        if 1 < len(mem) <= cap:
            for i in mem:
                cid[i] = n + gi
    return cid


def job_matrix(keys, key_jobs, cid, jobs):
    """직무 × 개념 이진 행렬 (개념 = cid)."""
    jidx = {j: i for i, j in enumerate(jobs)}
    cmap = {}
    rows, cols = [], []
    for i, k in enumerate(keys):
        c = cmap.setdefault(cid[i], len(cmap))
        for j in key_jobs[k]:
            if j in jidx:
                rows.append(jidx[j]); cols.append(c)
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(jobs), len(cmap)))
    X.data[:] = 1.0
    X.sum_duplicates(); X.data[:] = 1.0
    return X


def sim_matrix(X, kind):
    n = X.shape[0]
    if kind == "jaccard":
        inter = (X @ X.T).toarray()
        size = np.asarray(X.sum(axis=1)).ravel()
        union = size[:, None] + size[None, :] - inter
        S = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    else:
        df = np.asarray(X.sum(axis=0)).ravel()
        idf = np.log(n / np.maximum(df, 1))
        Xt = X.multiply(idf).tocsr()
        norms = np.sqrt(np.asarray(Xt.multiply(Xt).sum(axis=1)).ravel())
        Xn = sp.diags(1.0 / np.maximum(norms, 1e-12)) @ Xt
        S = (Xn @ Xn.T).toarray()
    np.fill_diagonal(S, 0.0)
    return S


def task_a(S, jobs, minor_of, middle_of, k=5, rng=None):
    n = len(jobs)
    top = np.argsort(-S, axis=1)[:, :k]
    hit_minor = np.array([np.mean([minor_of[jobs[o]] == minor_of[jobs[i]] for o in top[i]]) for i in range(n)])
    hit_middle = np.array([np.mean([middle_of[jobs[o]] == middle_of[jobs[i]] for o in top[i]]) for i in range(n)])
    rng = rng or np.random.default_rng(SEED)
    bm = [hit_minor[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    bd = [hit_middle[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    return {"n_queries": n, "p_at_5_minor": float(hit_minor.mean()), "p_at_5_minor_ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
            "p_at_5_middle": float(hit_middle.mean()), "p_at_5_middle_ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]}


def task_b(S, jobs, major_of, middle_of, permutations=200):
    import networkx as nx
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
    n = len(jobs)
    iu, ju = np.triu_indices(n, 1)
    w = S[iu, ju]
    keep = w > 0
    g = nx.Graph(); g.add_nodes_from(range(n))
    g.add_weighted_edges_from(zip(iu[keep].tolist(), ju[keep].tolist(), w[keep].tolist()))
    majors = [major_of[j] for j in jobs]; middles = [middle_of[j] for j in jobs]
    runs = []
    for sd in SEEDS:
        comms = nx.community.louvain_communities(g, weight="weight", seed=sd, resolution=1.0)
        lab = [0] * n
        for i, c in enumerate(comms):
            for u in c:
                lab[u] = i
        runs.append({"n_clusters": len(comms), "nmi_major": float(normalized_mutual_info_score(majors, lab)),
                     "ami_major": float(adjusted_mutual_info_score(majors, lab)), "ari_major": float(adjusted_rand_score(majors, lab)),
                     "nmi_middle": float(normalized_mutual_info_score(middles, lab)), "ami_middle": float(adjusted_mutual_info_score(middles, lab)), "lab": lab})
    main = runs[0]
    rng = random.Random(SEED); perm = list(majors); null = []
    for _ in range(permutations):
        rng.shuffle(perm); null.append(normalized_mutual_info_score(perm, main["lab"]))
    def rng_of(key): return [min(r[key] for r in runs), max(r[key] for r in runs)]
    return {"n_edges": g.number_of_edges(), "n_clusters": main["n_clusters"], "n_clusters_range": rng_of("n_clusters"),
            "nmi_major": main["nmi_major"], "nmi_major_seed_range": rng_of("nmi_major"),
            "ami_major": main["ami_major"], "ami_major_seed_range": rng_of("ami_major"),
            "ari_major": main["ari_major"], "nmi_middle": main["nmi_middle"], "ami_middle": main["ami_middle"], "ami_middle_seed_range": rng_of("ami_middle"),
            "null_nmi_major": float(np.mean(null)), "null_nmi_major_sd": float(np.std(null))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("DATABASE_URL", "postgresql://ncs:ncs@localhost:55432/ncs"))
    ap.add_argument("--e5-cut", type=float, default=0.967)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(OUT / "t0_metrics.csv", encoding="utf-8")))
    major_of = {r["job"]: r["major"] for r in rows}; middle_of = {r["job"]: r["middle"] for r in rows}
    minor_of = {r["job"]: r["job"][:6] for r in rows}
    print("[ds] 키 적재 …", flush=True)
    key_surfs, key_jobs, _, _ = load_keys(args.dsn)
    keys = sorted(key_surfs); n = len(keys)
    jobs = sorted({j for k in keys for j in key_jobs[k] if j in major_of})
    zp = np.load(OUT / "h3_vecs_326592_pairs_0.85.npz"); pi, pj, ps = zp["pi"], zp["pj"], zp["ps"]
    print("[ds] Louvain@.90 …", flush=True)
    groups, _ = cluster_louvain(n, pi, pj, ps, 0.90)
    m = ps >= 0.90
    layers = {"rules": list(range(n)), "size2": assign_from_groups(n, groups, 2), "le5": assign_from_groups(n, groups, 5),
              "le20": assign_from_groups(n, groups, 20), "all": assign_from_groups(n, groups, 10**9)}
    e5c = OUT / "h3_vecs_multilingual-e5-large_326592.npz"
    if e5c.exists():
        print("[ds] e5 교집합 엣지 …", flush=True)
        import torch
        z = np.load(e5c, allow_pickle=True); assert list(z["keys"]) == keys
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        ei, ej, _ = collect_pairs(z["vecs"], args.e5_cut, 2048, device)
        eset = set(zip(ei.tolist(), ej.tolist()))
        keep = np.array([(int(a), int(b)) in eset or (int(b), int(a)) in eset for a, b in zip(pi[m], pj[m])])
        g2, _ = cluster_louvain(n, pi[m][keep], pj[m][keep], ps[m][keep], 0.0)
        layers["both_le5"] = assign_from_groups(n, g2, 5); layers["both_all"] = assign_from_groups(n, g2, 10**9)
    R = {"meta": {"run_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "n_keys": n, "n_jobs": len(jobs),
                  "e5_cut": args.e5_cut, "n_boot": N_BOOT, "louvain_seeds": len(SEEDS)}, "layers": {}}
    for name, cid in layers.items():
        X = job_matrix(keys, key_jobs, cid, jobs)
        R["layers"][name] = {"n_concepts": len(set(cid))}
        for kind in ("jaccard", "tfidf"):
            print(f"[ds] {name} / {kind} …", flush=True)
            S = sim_matrix(X, kind)
            R["layers"][name][kind] = {"task_a": task_a(S, jobs, minor_of, middle_of), "task_b": task_b(S, jobs, major_of, middle_of)}
            a, b = R["layers"][name][kind]["task_a"], R["layers"][name][kind]["task_b"]
            print(f"    P@5 소분류 {a['p_at_5_minor']*100:.1f} [{a['p_at_5_minor_ci95'][0]*100:.1f},{a['p_at_5_minor_ci95'][1]*100:.1f}] · NMI {b['nmi_major']:.3f} (seed {b['nmi_major_seed_range'][0]:.3f}–{b['nmi_major_seed_range'][1]:.3f}) · 군집 {b['n_clusters']}", flush=True)
    (OUT / "t0_downstream_ncs.json").write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    L = ["# 다운스트림 시연 (NCS)", ""]
    for kind in ("jaccard", "tfidf"):
        L += [f"## 유사도 = {kind}", "", "| 개념층 | 개념 수 | P@5 소분류 [CI] | P@5 중분류 [CI] | 군집(시드 범위) | NMI 대분류 (시드 범위; 영분포) | AMI 대분류 (범위) | ARI | AMI 중분류 |", "|---|---|---|---|---|---|---|---|---|"]
        for name, v in R["layers"].items():
            a, b = v[kind]["task_a"], v[kind]["task_b"]
            L.append(f"| {name} | {v['n_concepts']:,} | {a['p_at_5_minor']*100:.1f}% [{a['p_at_5_minor_ci95'][0]*100:.1f}, {a['p_at_5_minor_ci95'][1]*100:.1f}] | {a['p_at_5_middle']*100:.1f}% [{a['p_at_5_middle_ci95'][0]*100:.1f}, {a['p_at_5_middle_ci95'][1]*100:.1f}] | {b['n_clusters']} ({b['n_clusters_range'][0]}–{b['n_clusters_range'][1]}) | {b['nmi_major']:.3f} ({b['nmi_major_seed_range'][0]:.3f}–{b['nmi_major_seed_range'][1]:.3f}; {b['null_nmi_major']:.3f}) | {b['ami_major']:.3f} ({b['ami_major_seed_range'][0]:.3f}–{b['ami_major_seed_range'][1]:.3f}) | {b['ari_major']:.3f} | {b['ami_middle']:.3f} |")
        L.append("")
    (OUT / "t0_downstream_ncs_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
