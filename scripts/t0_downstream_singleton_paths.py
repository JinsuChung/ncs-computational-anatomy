# -*- coding: utf-8 -*-
"""단일 직무 키 흡수의 세 경로 분리 (심사 5회차).

병합되는 단일 직무 키의 행선지는 셋이다 — (c) 다른 직무가 이미 가진 공유 개념에 붙음, (a) 다른 직무의
단일 키와 붙어 없던 공유 개념이 새로 생김, (b) 같은 직무 안의 단일 키끼리 붙어 분모만 줆.
경로별로 (1) 키 개수, (2) §4.6 판정 쌍에서 잰 정밀도, (3) le5_nosingle 위에 한 경로씩 더한 절제 실험을 낸다.
출력: docs/t0/t0_downstream_singleton_paths.json · 키→직무 캐시 docs/t0/t0_key_jobs.json(비추적)
"""
import sys, os, json, csv
from collections import defaultdict, Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); os.chdir(Path(__file__).resolve().parent.parent)
import numpy as np, networkx as nx
from sklearn.metrics import normalized_mutual_info_score as nmi
from t0_embed_recovery import SEED, cluster_louvain, load_keys
from t0_downstream_ncs import job_matrix, sim_matrix, OUT
DSN = os.environ.get("DATABASE_URL", "postgresql://ncs:ncs@localhost:55432/ncs")
OUTF = OUT / "t0_downstream_singleton_paths.json"
R1 = {"EV-1","EV-2","EV-3","EV-4","EV-5","EV-3B","EV-4B","EV-5B"}

rows = list(csv.DictReader(open(OUT / "t0_metrics.csv", encoding="utf-8")))
major_of = {r["job"]: r["major"] for r in rows}; minor_of = {r["job"]: r["job"][:6] for r in rows}
cache = OUT / "t0_key_jobs.json"
if cache.exists():
    kj = json.loads(cache.read_text(encoding="utf-8")); keys = sorted(kj); key_jobs = {k: set(v) for k, v in kj.items()}
    print("[p] 키→직무 캐시 사용", flush=True)
else:
    print("[p] 키 적재(DB) …", flush=True)
    key_surfs, key_jobs, _, _ = load_keys(DSN); keys = sorted(key_surfs)
    cache.write_text(json.dumps({k: sorted(key_jobs[k]) for k in keys}, ensure_ascii=False), encoding="utf-8")
n = len(keys); kidx = {k: i for i, k in enumerate(keys)}
jobs = sorted({j for k in keys for j in key_jobs[k] if j in major_of})
single = np.array([len(key_jobs[k]) == 1 for k in keys])
zp = np.load(OUT / "h3_vecs_326592_pairs_0.85.npz"); pi, pj, ps = zp["pi"], zp["pj"], zp["ps"]
print("[p] Louvain@.90 …", flush=True)
groups, _ = cluster_louvain(n, pi, pj, ps, 0.90)
m = ps >= 0.90; edges = set(zip(np.minimum(pi[m], pj[m]).tolist(), np.maximum(pi[m], pj[m]).tolist()))

# (1) 경로별 키 개수 — 상한 5
def classify(mem):
    s_ = [i for i in mem if single[i]]; sh = [i for i in mem if not single[i]]
    if sh: return "c_into_shared"
    occs = {next(iter(key_jobs[keys[i]])) for i in s_}
    return "a_cross_occupation" if len(occs) > 1 else "b_same_occupation"
cnt = Counter(); gcnt = Counter(); groups5 = {}
for gid, mem in groups.items():
    if 1 < len(mem) <= 5:
        cls = classify(mem); groups5[gid] = cls; gcnt[cls] += 1
        cnt[cls] += sum(single[i] for i in mem)
print("[p] 상한5 단일키 행선지:", dict(cnt), "군집:", dict(gcnt), flush=True)

# (2) §4.6 판정 쌍의 경로별 정밀도 (1차 8인, 팔 A 직접 엣지 + 팔 B 전이)
J = json.loads((OUT / "t0_judgments.json").read_text(encoding="utf-8"))["rows"]
S1 = {it["id"]: it for it in json.loads((OUT / "t0_h3_validation_sample.json").read_text(encoding="utf-8"))["items"]}
def pair_class(a, b):
    ja, jb = key_jobs[keys[a]], key_jobs[keys[b]]
    if len(ja) == 1 and len(jb) == 1: return "a_cross_occupation" if ja != jb else "b_same_occupation"
    if len(ja) == 1 or len(jb) == 1: return "c_into_shared"
    return "d_shared_shared"
by = defaultdict(lambda: defaultdict(list))
for r in J:
    if r["rater"].upper() in R1 and r["pair_id"] in S1 and r["left_key"] in kidx and r["right_key"] in kidx:
        a, b = kidx[r["left_key"]], kidx[r["right_key"]]
        arm = S1[r["pair_id"]]["_hidden"]["arm"]
        by[(arm, pair_class(a, b))][r["pair_id"]].append(r["verdict"] == "same")
rng = np.random.default_rng(20260915)
prec = {}
for (arm, cls), items in sorted(by.items()):
    ids = sorted(items); flat = lambda sel: np.mean([x for i in sel for x in items[i]])
    bs = [flat(rng.choice(ids, len(ids), True)) for _ in range(4000)]
    prec[f"{arm}/{cls}"] = {"n_items": len(ids), "n_judgments": sum(len(v) for v in items.values()),
                            "strict": float(flat(ids)), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    print(f"[p] 정밀도 {arm}/{cls}: 문항 {len(ids)} · 엄격 {flat(ids)*100:.1f}% [{np.percentile(bs,2.5)*100:.1f}, {np.percentile(bs,97.5)*100:.1f}]", flush=True)

# (3) 절제 — le5_nosingle 위에 경로 하나씩
def assign(allow):
    cid = list(range(n))
    for gid, mem in groups.items():
        if not (1 < len(mem) <= 5): continue
        cls = groups5[gid]
        if cls == "c_into_shared":
            sel = [i for i in mem if not single[i]] + ([i for i in mem if single[i]] if "c" in allow else [])
        else:
            sel = mem if cls[0] in allow else []
        if len(sel) > 1:
            for i in sel: cid[i] = n + hash(gid) % (10**9)
    return cid
layers = {"nosingle": assign(""), "+c_into_shared": assign("c"), "+a_cross": assign("a"), "+b_same": assign("b"), "le5_all": assign("abc")}
def hits(S, k=5):
    top = np.argsort(-S, axis=1)[:, :k]
    return np.array([np.mean([minor_of[jobs[o]] == minor_of[jobs[i]] for o in top[i]]) for i in range(len(jobs))])
def nmi_runs(S):
    iu, ju = np.triu_indices(len(jobs), 1); w = S[iu, ju]; keep = w > 0
    g = nx.Graph(); g.add_nodes_from(range(len(jobs))); g.add_weighted_edges_from(zip(iu[keep].tolist(), ju[keep].tolist(), w[keep].tolist()))
    majors = [major_of[j] for j in jobs]; out = []
    for sd in [SEED + i for i in range(10)]:
        comms = nx.community.louvain_communities(g, weight="weight", seed=sd, resolution=1.0)
        lab = [0]*len(jobs)
        for i, c in enumerate(comms):
            for u in c: lab[u] = i
        out.append(nmi(majors, lab))
    return {"main": out[0], "range": [min(out), max(out)]}
R = {"singleton_destinations_le5": {"keys": {k: int(v) for k, v in cnt.items()}, "groups": {k: int(v) for k, v in gcnt.items()}}, "judged_pair_precision_by_path": prec, "layers": {}, "hits": {}}
for name, cid in layers.items():
    X = job_matrix(keys, key_jobs, cid, jobs); R["layers"][name] = {"n_concepts": len(set(cid))}
    for kind in ("jaccard", "tfidf"):
        print(f"[p] {name}/{kind} …", flush=True)
        S = sim_matrix(X, kind); h = hits(S); R["hits"][f"{name}/{kind}"] = h.tolist()
        R["layers"][name][kind] = {"p5_minor": float(h.mean()), "nmi": nmi_runs(S)}
def paired(a, b, B=4000):
    d = np.array(R["hits"][b]) - np.array(R["hits"][a]); idx = np.arange(len(d))
    bs = [d[rng.choice(idx, len(idx), True)].mean() for _ in range(B)]
    return {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
R["paired_vs_nosingle"] = {f"{k}/{s}": paired(f"nosingle/{s}", f"{k}/{s}") for k in ("+c_into_shared", "+a_cross", "+b_same", "le5_all") for s in ("jaccard", "tfidf")}
R.pop("hits"); OUTF.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in R["layers"].items():
    print(f"  {k:15s} 개념 {v['n_concepts']:>8,} | jac P@5 {v['jaccard']['p5_minor']*100:5.1f} NMI {v['jaccard']['nmi']['main']:.3f} | tfidf P@5 {v['tfidf']['p5_minor']*100:5.1f} NMI {v['tfidf']['nmi']['main']:.3f}")
for k, v in R["paired_vs_nosingle"].items(): print(f"  짝 {k}: {v['diff']*100:+.2f}%p [{v['ci'][0]*100:+.2f}, {v['ci'][1]*100:+.2f}]")
print("[p] 완료", flush=True)
