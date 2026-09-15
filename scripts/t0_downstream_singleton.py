# -*- coding: utf-8 -*-
"""다운스트림 분해 — 임베딩 단계의 이득이 단일 직무 키 흡수에서 오는가 (심사 4회차 A·B).

상한 5·20 정책을 두 번 돌린다: 그대로 / 단일 직무 키(전체의 92.3%)를 병합에서 제외(le*_nosingle).
차이가 이득 전부이면 임베딩 정렬의 다운스트림 가치는 "희소 표현을 공유 개념으로 끌어올리는 것"이다.
더불어 같은 직무에서의 짝 부트스트랩으로 rules 대비 P@5 차이의 구간을 낸다(CI 겹침보다 정확).
출력: docs/t0/t0_downstream_singleton.json · 사용: .venv/bin/python scripts/t0_downstream_singleton.py
"""
import sys, os, json, csv
sys.path.insert(0, "/Users/suaidev/개발/ontological-ncs/scripts"); os.chdir("/Users/suaidev/개발/ontological-ncs")
import numpy as np, networkx as nx
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score as nmi
from t0_embed_recovery import SEED, cluster_louvain, load_keys
from t0_downstream_ncs import assign_from_groups, job_matrix, sim_matrix, OUT
OUTF = str(OUT / 't0_downstream_singleton.json')
DSN = os.environ.get("DATABASE_URL", "postgresql://ncs:ncs@localhost:55432/ncs")
rows = list(csv.DictReader(open(OUT / "t0_metrics.csv", encoding="utf-8")))
major_of = {r["job"]: r["major"] for r in rows}; middle_of = {r["job"]: r["middle"] for r in rows}
minor_of = {r["job"]: r["job"][:6] for r in rows}
print("[a] 키 적재 …", flush=True)
key_surfs, key_jobs, _, _ = load_keys(DSN)
keys = sorted(key_surfs); n = len(keys)
jobs = sorted({j for k in keys for j in key_jobs[k] if j in major_of})
single = np.array([len(key_jobs[k]) == 1 for k in keys])
print(f"[a] 키 {n:,} · 단일직무 {single.sum():,} ({single.mean()*100:.1f}%) · 직무 {len(jobs):,}", flush=True)
zp = np.load(OUT / "h3_vecs_326592_pairs_0.85.npz"); pi, pj, ps = zp["pi"], zp["pj"], zp["ps"]
print("[a] Louvain@.90 …", flush=True)
groups, _ = cluster_louvain(n, pi, pj, ps, 0.90)

def assign_nosingle(cap):
    """상한 cap 군집을 병합하되 단일 직무 키는 병합에서 제외(자기 개념 유지)."""
    cid = list(range(n))
    for gi, mem in enumerate(groups.values()):
        if 1 < len(mem) <= cap:
            shared = [i for i in mem if not single[i]]
            if len(shared) > 1:
                for i in shared: cid[i] = n + gi
    return cid

# 서술 통계: 상한별로 병합되는 키 중 단일 직무 키 비율, 단일 키가 공유 개념에 붙는 건수
desc = {}
for cap, name in ((2, "size2"), (5, "le5"), (20, "le20"), (10**9, "all")):
    merged_single = merged_shared = single_into_shared = 0
    for mem in groups.values():
        if 1 < len(mem) <= cap:
            s_ = [i for i in mem if single[i]]; sh = [i for i in mem if not single[i]]
            merged_single += len(s_); merged_shared += len(sh)
            if sh: single_into_shared += len(s_)
    desc[name] = {"merged_keys": merged_single + merged_shared, "merged_single": merged_single,
                  "merged_shared": merged_shared, "single_absorbed_into_shared_concept": single_into_shared}
print("[a] 서술:", json.dumps(desc), flush=True)

layers = {"rules": list(range(n)), "le5": assign_from_groups(n, groups, 5), "le5_nosingle": assign_nosingle(5),
          "le20": assign_from_groups(n, groups, 20), "le20_nosingle": assign_nosingle(20)}
def hits(S, k=5):
    top = np.argsort(-S, axis=1)[:, :k]
    return np.array([np.mean([minor_of[jobs[o]] == minor_of[jobs[i]] for o in top[i]]) for i in range(len(jobs))])
def nmi_runs(S):
    iu, ju = np.triu_indices(len(jobs), 1); w = S[iu, ju]; keep = w > 0
    g = nx.Graph(); g.add_nodes_from(range(len(jobs)))
    g.add_weighted_edges_from(zip(iu[keep].tolist(), ju[keep].tolist(), w[keep].tolist()))
    majors = [major_of[j] for j in jobs]; out = []
    for sd in [SEED + i for i in range(10)]:
        comms = nx.community.louvain_communities(g, weight="weight", seed=sd, resolution=1.0)
        lab = [0]*len(jobs)
        for i, c in enumerate(comms):
            for u in c: lab[u] = i
        out.append(nmi(majors, lab))
    return {"main": out[0], "range": [min(out), max(out)]}
R = {"desc": desc, "layers": {}, "hits": {}}
for name, cid in layers.items():
    X = job_matrix(keys, key_jobs, cid, jobs)
    R["layers"][name] = {"n_concepts": len(set(cid))}
    for kind in ("jaccard", "tfidf"):
        print(f"[a] {name}/{kind} …", flush=True)
        S = sim_matrix(X, kind); h = hits(S)
        R["hits"][f"{name}/{kind}"] = h.tolist()
        R["layers"][name][kind] = {"p5_minor": float(h.mean()), "nmi": nmi_runs(S)}
    json.dump(R, open(OUTF, "w"), ensure_ascii=False)
# (B) 짝 부트스트랩 — 같은 직무에서의 차이
rng = np.random.default_rng(20260915)
def paired(a, b, B=4000):
    a = np.array(R["hits"][a]); b = np.array(R["hits"][b]); d = b - a; idx = np.arange(len(d))
    bs = [d[rng.choice(idx, len(idx), True)].mean() for _ in range(B)]
    return {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_le0": float(np.mean(np.array(bs) <= 0))}
R["paired"] = {"tfidf le5-rules": paired("rules/tfidf", "le5/tfidf"), "tfidf le20-rules": paired("rules/tfidf", "le20/tfidf"),
               "tfidf le5-le5_nosingle": paired("le5_nosingle/tfidf", "le5/tfidf"),
               "jaccard le5-rules": paired("rules/jaccard", "le5/jaccard"), "jaccard le5-le5_nosingle": paired("le5_nosingle/jaccard", "le5/jaccard")}
json.dump(R, open(OUTF, "w"), ensure_ascii=False)
for k, v in R["layers"].items():
    print(f"  {k:14s} 개념 {v['n_concepts']:>8,} | jac P@5 {v['jaccard']['p5_minor']*100:5.1f} NMI {v['jaccard']['nmi']['main']:.3f} [{v['jaccard']['nmi']['range'][0]:.3f}–{v['jaccard']['nmi']['range'][1]:.3f}] | tfidf P@5 {v['tfidf']['p5_minor']*100:5.1f} NMI {v['tfidf']['nmi']['main']:.3f} [{v['tfidf']['nmi']['range'][0]:.3f}–{v['tfidf']['nmi']['range'][1]:.3f}]")
for k, v in R["paired"].items(): print(f"  짝 {k}: {v['diff']*100:+.2f}%p CI [{v['ci'][0]*100:+.2f}, {v['ci'][1]*100:+.2f}] P(≤0)={v['p_le0']:.3f}")
print("[a] 완료", flush=True)
