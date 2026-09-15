# -*- coding: utf-8 -*-
"""다운스트림 시연 (ESCO, 정답 있음) — 스킬 라벨로 직업 검색 (심사 지적 ①).

과제: 사용자가 색인에 없는 표기(등록된 대체 라벨 하나를 색인에서 빼고 질의로 씀)로 스킬을 검색해 그 스킬을 필수로
      요구하는 직업을 찾는다. 정답 = 그 라벨이 속한 스킬 URI 의 필수 직업 집합.
개념층(색인이 질의를 다른 표기와 잇는 방식):
  exact   질의와 문자열이 같은 표기만 → 색인에서 뺐으므로 항상 실패 (하한)
  rules   R1 규칙 키가 같은 표기
  direct  bge cos≥.90 직접 이웃
  le5 / le20 / all   Louvain@.90 군집 동료 (크기 상한 정책)
  both_*  bge≥.90 ∩ e5 일치 cut 엣지로 만든 군집 (§4.8 지렛대)
지표: 스킬 적중률(정답 스킬이 검색됨), 직업 정밀도·재현율·F1(질의 평균), 응답률(검색 결과 비어 있지 않음).
출력: docs/esco/esco_downstream.json · esco_downstream_report.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esco_replication import build_surfaces, load as esco_load, r1  # noqa: E402
from t0_embed_recovery import cluster_louvain, collect_pairs  # noqa: E402

ESCO = Path("docs/esco")
SEED = 20260914


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--e5-cut", type=float, default=0.958)
    args = ap.parse_args()
    skills, occs = esco_load()
    surf_concepts, concept_surfs = build_surfaces(skills)
    surfs = sorted(surf_concepts); n = len(surfs); idx = {s: i for i, s in enumerate(surfs)}
    pref = {s["uri"]: s["preferredLabel"] for s in skills}
    skill_occ = defaultdict(set)
    for o in occs:
        for u in o.get("essentialSkills", []):
            skill_occ[u].add(o["uri"])
    z = np.load(ESCO / "esco_vecs_99676.npz", allow_pickle=True); assert list(z["keys"]) == surfs
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pi, pj, ps = collect_pairs(z["vecs"], 0.85, 2048, device)
    m = ps >= 0.90
    nbr = defaultdict(set)
    for a, b in zip(pi[m].tolist(), pj[m].tolist()):
        nbr[a].add(b); nbr[b].add(a)
    groups, _ = cluster_louvain(n, pi, pj, ps, 0.90)
    members = {}
    for mem in groups.values():
        for i in mem:
            members[i] = mem
    layers = {}
    rkey = [r1(s) for s in surfs]
    by_rkey = defaultdict(set)
    for i, k in enumerate(rkey):
        by_rkey[k].add(i)
    layers["exact"] = lambda i: set()
    layers["rules"] = lambda i: by_rkey[rkey[i]] - {i}
    # 문자 3-gram TF-IDF 코사인 — 임베딩 없는 실제 어휘 베이스라인
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    tv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), lowercase=True, sublinear_tf=True)
    Xc = tv.fit_transform(surfs)
    nn = NearestNeighbors(n_neighbors=21, metric="cosine").fit(Xc)
    qidx = sorted({idx[s_] for s_, _ in []})  # placeholder, filled below
    cgram_nbr = {}
    def cgram(i, th):
        if i not in cgram_nbr:
            d, nb = nn.kneighbors(Xc[i], n_neighbors=21)
            cgram_nbr[i] = [(int(j), 1 - float(dd)) for dd, j in zip(d[0], nb[0]) if j != i]
        return {j for j, c in cgram_nbr[i] if c >= th}
    layers["char3gram_ge0.5"] = lambda i: cgram(i, 0.5)
    layers["char3gram_ge0.6"] = lambda i: cgram(i, 0.6)
    layers["direct"] = lambda i: nbr[i]
    for name, cap in (("le5", 5), ("le20", 20), ("all", 10**9)):
        layers[name] = (lambda cap: lambda i: (set(members[i]) - {i}) if len(members[i]) <= cap else set())(cap)
    e5c = ESCO / "esco_vecs_multilingual-e5-large_99676.npz"
    if e5c.exists():
        z2 = np.load(e5c, allow_pickle=True); assert list(z2["keys"]) == surfs
        ei, ej, _ = collect_pairs(z2["vecs"], args.e5_cut, 2048, device)
        eset = set(zip(ei.tolist(), ej.tolist()))
        keep = np.array([(int(a), int(b)) in eset or (int(b), int(a)) in eset for a, b in zip(pi[m], pj[m])])
        g2, _ = cluster_louvain(n, pi[m][keep], pj[m][keep], ps[m][keep], 0.0)
        mem2 = {}
        for mem in g2.values():
            for i in mem:
                mem2[i] = mem
        for name, cap in (("both_le5", 5), ("both_all", 10**9)):
            layers[name] = (lambda cap: lambda i: (set(mem2[i]) - {i}) if len(mem2[i]) <= cap else set())(cap)
    # 질의: 대체 라벨(선호 라벨 제외)이면서 그 스킬에 다른 표기가 남는 것, 필수 직업이 1개 이상
    cands = []
    for u, ss in concept_surfs.items():
        if len(ss) < 2 or not skill_occ.get(u):
            continue
        for s in ss:
            if s != pref[u] and len(surf_concepts[s]) == 1:
                cands.append((s, u))
    rng = random.Random(SEED); rng.shuffle(cands); cands = cands[: args.n_queries]
    R = {"meta": {"run_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "n_queries": len(cands), "e5_cut": args.e5_cut}, "layers": {}}
    rng_b = np.random.default_rng(SEED)
    for name, fn in layers.items():
        hit = answered = 0; P = Rc = F = 0.0
        per_hit, per_f, per_p, per_ans = [], [], [], []
        for s, u in cands:
            i = idx[s]
            found = fn(i)
            sk = set()
            for j in found:
                sk |= surf_concepts[surfs[j]]
            sk.discard(u) if False else None
            truth = skill_occ[u]
            got = set()
            for v in sk:
                got |= skill_occ.get(v, set())
            f_ = p_ = 0.0
            if got:
                answered += 1
                p = len(got & truth) / len(got); r = len(got & truth) / len(truth)
                p_ = p; f_ = (2 * p * r / (p + r)) if (p + r) else 0.0
                P += p; Rc += r; F += f_
            hit += u in sk
            per_hit.append(u in sk); per_f.append(f_); per_p.append(p_ if got else np.nan); per_ans.append(bool(got))
        N = len(cands)
        ph, pf, pp = np.array(per_hit, float), np.array(per_f), np.array(per_p)
        bh, bf, bp = [], [], []
        for _ in range(1000):
            ix = rng_b.integers(0, N, N)
            bh.append(ph[ix].mean()); bf.append(pf[ix].mean())
            v = pp[ix]; v = v[~np.isnan(v)]; bp.append(v.mean() if len(v) else np.nan)
        ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
        R["layers"][name] = {"skill_hit_rate": hit / N, "skill_hit_ci95": ci(bh), "answer_rate": answered / N, "occ_precision": P / N, "occ_recall": Rc / N,
                             "occ_f1": F / N, "occ_f1_ci95": ci(bf),
                             "occ_precision_when_answered": (P / answered) if answered else None, "occ_precision_when_answered_ci95": ci(bp) if answered else None}
        print(name, json.dumps(R["layers"][name]), flush=True)
    (ESCO / "esco_downstream.json").write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    L = ["# 다운스트림 시연 (ESCO 스킬 라벨 → 직업 검색, 정답 기준)", "", f"질의 {len(cands):,} (색인에서 뺀 대체 라벨)", "",
         "| 개념층 | 스킬 적중 [CI] | 응답률 | 직업 정밀도 | 직업 재현율 | F1 [CI] | 응답 시 정밀도 [CI] |", "|---|---|---|---|---|---|---|"]
    for name, v in R["layers"].items():
        pw = v['occ_precision_when_answered_ci95'] or [0, 0]
        L.append(f"| {name} | {v['skill_hit_rate']*100:.1f}% [{v['skill_hit_ci95'][0]*100:.1f}, {v['skill_hit_ci95'][1]*100:.1f}] | {v['answer_rate']*100:.1f}% | {v['occ_precision']*100:.1f}% | {v['occ_recall']*100:.1f}% | {v['occ_f1']*100:.1f}% [{v['occ_f1_ci95'][0]*100:.1f}, {v['occ_f1_ci95'][1]*100:.1f}] | {(v['occ_precision_when_answered'] or 0)*100:.1f}% [{pw[0]*100:.1f}, {pw[1]*100:.1f}] |")
    (ESCO / "esco_downstream_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
