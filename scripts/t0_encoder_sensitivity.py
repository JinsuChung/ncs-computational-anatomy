"""임베딩 인코더 민감도 — bge-m3 의 .90 무릎이 모델의 보정값인가, 과제의 성질인가.

해설집 7.2 MID "임베딩 단일 모델" 항목. 다른 계열 다국어 인코더(기본 intfloat/multilingual-e5-large, bge-m3 와
크기가 비슷해 '계열 차이'만 본다)로 같은 텍스트를 다시 임베딩하고 세 가지를 비교한다.

  A. NCS 판정 200쌍 — 같은 쌍의 두 모델 코사인 상관, 새 모델 코사인 3분위별 외부 판정 정밀도(엄격/관대).
  B. ESCO 전수 — 정답(같은 URI) 기준 임계값–정밀도 곡선. 코사인 척도가 모델마다 다르므로 절대 임계값이 아니라
     '쌍 수 일치'(bge 의 ≥.85/.90/.95 쌍 수와 같은 수의 상위 쌍)로 맞춘 임계값에서 정밀도·재현율을 비교하고,
     그 일치 임계값에서 Louvain 군집의 검증 회수·병합 정밀도·전이 정밀도·최대 군집을 낸다.
  C. NCS 전수(326,592 키) — 쌍 수 일치 임계값에서 결합 회수율·최대 군집·유형별 참여율(태도 쏠림).

무릎 판정: 쌍을 코사인 내림차순으로 정렬해 누적 정밀도 곡선을 그리고, 대역(분위) 정밀도가 가장 크게 꺾이는 지점을
'무릎'으로 잡는다. 두 모델의 무릎이 같은 누적 쌍 수(=같은 재현율)에 있으면 무릎은 과제의 성질이다.

사용: .venv/bin/python scripts/t0_encoder_sensitivity.py [--model intfloat/multilingual-e5-large] [--stages A,B,C]
출력: docs/t0/t0_encoder_sensitivity.json · t0_encoder_sensitivity_report.md (캐시 npz 는 비추적)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esco_replication import build_surfaces, load as esco_load, same_concept  # noqa: E402
from t0_analyze_validation import load as load_judgments, lenient, strict  # noqa: E402
from t0_embed_recovery import SEED, cluster_louvain, cluster_single_linkage, collect_pairs, load_keys  # noqa: E402

OUT = Path("docs/t0")
ESCO = Path("docs/esco")
BGE_THRESHOLDS = (0.85, 0.90, 0.95)


def slug(model): return model.split("/")[-1].replace(".", "_")


def prefix_for(model):
    # e5 계열은 대칭 과제에 'query: ' 접두를 권장한다. 그 밖의 모델은 원문 그대로.
    return "query: " if "e5" in model.lower() else ""


def embed(model, texts, device, batch, cache):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["keys"]) == list(texts):
            print(f"  캐시 {cache}", flush=True)
            return z["vecs"]
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model, device=device)
    p = prefix_for(model)
    v = m.encode([p + t for t in texts], batch_size=batch, convert_to_numpy=True,
                 normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    if not np.isfinite(v).all():
        raise RuntimeError("비유한 임베딩")
    np.savez(cache, vecs=v, keys=np.array(list(texts), dtype=object))
    return v


def matched_threshold(ps_sorted_desc, n_target):
    """상위 n_target 쌍을 포함하는 코사인 임계값."""
    if n_target >= len(ps_sorted_desc):
        return float(ps_sorted_desc[-1])
    return float(ps_sorted_desc[n_target - 1])


def knee(ps, truth, n_bins=20):
    """코사인 내림차순 정렬 → 분위 대역별 정밀도 → 인접 대역 정밀도 낙차가 최대인 경계."""
    order = np.argsort(-ps)
    ps_s, tr_s = ps[order], truth[order]
    edges = np.linspace(0, len(ps_s), n_bins + 1).astype(int)
    bands = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            bands.append({"rank_from": int(a), "rank_to": int(b), "cos_hi": float(ps_s[a]), "cos_lo": float(ps_s[b - 1]),
                          "precision": float(tr_s[a:b].mean())})
    drops = [(bands[i]["precision"] - bands[i + 1]["precision"], i) for i in range(len(bands) - 1)]
    d, i = max(drops)
    return {"bands": bands, "knee_after_band": i, "knee_rank": bands[i]["rank_to"],
            "knee_cos": bands[i]["cos_lo"], "precision_drop": float(d)}


def adaptive_pairs(vecs, device, target, start=0.95, floor=0.70, step=0.02, block=2048):
    """쌍 수가 target 이상이 될 때까지 최소 임계값을 내리며 수집(코사인 척도가 모델마다 달라서)."""
    t = start
    while True:
        pi, pj, ps = collect_pairs(vecs, t, block, device)
        print(f"  min cos {t:.2f} → 쌍 {len(ps):,}", flush=True)
        if len(ps) >= target or t - step < floor:
            return pi, pj, ps, t
        t = round(t - step, 2)


# ───────────────────────────────────────────── A. NCS 판정 200쌍

def stage_a(model, device, batch):
    judgs, items, _ = load_judgments(OUT / "t0_judgments.json", OUT / "t0_h3_validation_sample.json")
    judgs = [j for j in judgs if not j["rater"].startswith("AU")]
    keys = sorted({it["left"]["key"] for it in items.values()} | {it["right"]["key"] for it in items.values()})
    vecs = embed(model, keys, device, batch, OUT / f"t0_encsens_{slug(model)}_pairs200.npz")
    ki = {k: i for i, k in enumerate(keys)}
    new_cos = {pid: float(vecs[ki[it["left"]["key"]]] @ vecs[ki[it["right"]["key"]]]) for pid, it in items.items()}
    old_cos = {pid: it["_hidden"]["cosine"] for pid, it in items.items()}
    pids = sorted(items)
    from scipy.stats import spearmanr, pearsonr
    rho = spearmanr([old_cos[p] for p in pids], [new_cos[p] for p in pids])
    r = pearsonr([old_cos[p] for p in pids], [new_cos[p] for p in pids])
    # 새 모델 코사인 3분위 × 판정 정밀도 (직접 쌍만: arm A) + 전이(arm B)
    by_pid = defaultdict(list)
    for j in judgs:
        by_pid[j["pair_id"]].append(j["verdict"])
    def prec(pid_list):
        vs = [v for p in pid_list for v in by_pid.get(p, [])]
        return {"n_items": len(pid_list), "n_judgments": len(vs),
                "strict": float(np.mean([strict(v) for v in vs])) if vs else None,
                "lenient": float(np.mean([lenient(v) for v in vs])) if vs else None}
    direct = [p for p in pids if items[p]["_hidden"]["arm"] == "A"]
    trans = [p for p in pids if items[p]["_hidden"]["arm"] == "B"]
    res = {"model": model, "n_pairs": len(pids), "spearman_old_new": float(rho.correlation), "pearson_old_new": float(r[0]),
           "new_cos_range": [float(min(new_cos.values())), float(max(new_cos.values()))],
           "old_cos_range": [float(min(old_cos.values())), float(max(old_cos.values()))]}
    # bge 대역(사전 등록 .85–.90/.90–.95/.95+)별 새 모델 코사인 분포와 정밀도 — 같은 쌍의 위치 이동
    res["by_old_band"] = {}
    for band in ("0.85-0.90", "0.90-0.95", "0.95-1.01", "transitive"):
        ps_ = [p for p in pids if items[p]["_hidden"]["band"] == band]
        if ps_:
            nc = [new_cos[p] for p in ps_]
            res["by_old_band"][band] = {**prec(ps_), "new_cos_mean": float(np.mean(nc)), "new_cos_sd": float(np.std(nc))}
    # 새 모델 코사인 3분위(직접 쌍) 정밀도 — 단조 상승·무릎 여부
    d_sorted = sorted(direct, key=lambda p: -new_cos[p])
    thirds = np.array_split(np.array(d_sorted, dtype=object), 3)
    res["direct_by_new_tercile"] = [{**prec(list(t)), "cos_hi": new_cos[t[0]], "cos_lo": new_cos[t[-1]]} for t in thirds]
    res["transitive"] = prec(trans)
    # 재정렬: bge 의 .90 경계와 같은 '순위'에 새 모델 경계를 두었을 때 정밀도 (직접 쌍)
    old_sorted = sorted(direct, key=lambda p: -old_cos[p])
    k = sum(1 for p in direct if old_cos[p] >= 0.90)
    res["direct_rank_matched_at_old_090"] = {"n_above": k, "old_model": prec(old_sorted[:k]), "new_model": prec(d_sorted[:k]),
                                             "new_model_cut": new_cos[d_sorted[k - 1]] if k else None,
                                             "overlap_of_top_sets": len(set(old_sorted[:k]) & set(d_sorted[:k])) / k if k else None}
    return res


# ───────────────────────────────────────────── B. ESCO 전수

def stage_b(model, device, batch, block):
    skills, _ = esco_load()
    surf_concepts, concept_surfs = build_surfaces(skills)
    surfs = sorted(surf_concepts)
    n = len(surfs)
    idx = {s: i for i, s in enumerate(surfs)}
    bge = json.loads((ESCO / "esco_replication.json").read_text(encoding="utf-8"))["EH3_embedding"]
    targets = {t: bge["cumulative_precision"][str(t) if str(t) in bge["cumulative_precision"] else str(float(t))]["n_pairs"]
               for t in BGE_THRESHOLDS}
    vecs = embed(model, surfs, device, batch, ESCO / f"esco_vecs_{slug(model)}_{n}.npz")
    pi, pj, ps, tmin = adaptive_pairs(vecs, device, max(targets.values()) * 1.5, block=block)
    truth = np.array([same_concept(surfs[a], surfs[b], surf_concepts) for a, b in zip(pi, pj)], dtype=bool)
    true_pairs = set()
    for c, ss in concept_surfs.items():
        if len(ss) > 1:
            for a, b in combinations(sorted(idx[s] for s in ss), 2):
                true_pairs.add((a, b))
    order = np.argsort(-ps)
    ps_d = ps[order]
    res = {"model": model, "n_surface_forms": n, "min_cos_collected": tmin, "n_pairs_collected": int(len(ps)),
           "n_true_pairs": len(true_pairs), "matched": {}, "curve": []}
    # 누적 정밀도 곡선(절대 임계값 격자)
    for t in np.arange(round(tmin, 2), 1.0001, 0.01):
        m = ps >= t
        if m.any():
            got = {(int(a), int(b)) for a, b in zip(pi[m], pj[m])}
            res["curve"].append({"cos": round(float(t), 2), "n_pairs": int(m.sum()), "precision": float(truth[m].mean()),
                                 "recall": len(got & true_pairs) / len(true_pairs)})
    # 쌍 수 일치 임계값에서 비교 + 대역 정밀도(일치 대역)
    cuts = {}
    for t, n_target in sorted(targets.items()):
        cut = matched_threshold(ps_d, n_target)
        cuts[t] = cut
        m = ps >= cut
        got = {(int(a), int(b)) for a, b in zip(pi[m], pj[m])}
        res["matched"][str(t)] = {"bge_n_pairs": n_target, "new_cut": cut, "n_pairs": int(m.sum()),
                                  "precision": float(truth[m].mean()), "recall": len(got & true_pairs) / len(true_pairs),
                                  "bge_precision": bge["cumulative_precision"][str(t) if str(t) in bge["cumulative_precision"] else str(float(t))]["precision"],
                                  "bge_recall": bge["cumulative_precision"][str(t) if str(t) in bge["cumulative_precision"] else str(float(t))]["recall_of_true_pairs"]}
    bands = [(cuts[0.85], cuts[0.90]), (cuts[0.90], cuts[0.95]), (cuts[0.95], 1.01)]
    res["matched_band_precision"] = {}
    for (lo, hi), name in zip(bands, ("0.85-0.90", "0.90-0.95", "0.95-1.00")):
        m = (ps >= lo) & (ps < hi)
        res["matched_band_precision"][name] = {"new_cos_lo": lo, "new_cos_hi": hi, "n_pairs": int(m.sum()),
                                               "precision": float(truth[m].mean()) if m.any() else None,
                                               "bge_precision": bge["direct_precision_by_band"][{"0.85-0.90": "0.85-0.90", "0.90-0.95": "0.90-0.95", "0.95-1.00": "0.95-1.01"}[name]]["precision"]}
    # 무릎(분위 20대역) — bge 캐시 쌍으로도 같은 절차
    res["knee"] = knee(ps, truth)
    bz = np.load(ESCO / "esco_vecs_99676.npz", allow_pickle=True) if (ESCO / "esco_vecs_99676.npz").exists() else None
    if bz is not None and list(bz["keys"]) == surfs:
        bpi, bpj, bps = collect_pairs(bz["vecs"], 0.85, block, device)
        btruth = np.array([same_concept(surfs[a], surfs[b], surf_concepts) for a, b in zip(bpi, bpj)], dtype=bool)
        # 같은 쌍 수로 잘라 같은 절차
        k = min(len(bps), len(ps))
        res["knee_bge_same_n"] = knee(bps[np.argsort(-bps)[:k]], btruth[np.argsort(-bps)[:k]])
        res["knee"] = knee(ps_d[:k], truth[order][:k])
        res["knee_n_pairs_compared"] = int(k)
    # 군집 — 일치 .90 임계값
    cut90 = cuts[0.90]
    res["clusters_at_matched_090"] = {}
    for method, fn in (("louvain", cluster_louvain), ("single_linkage", cluster_single_linkage)):
        groups, _ = fn(n, pi, pj, ps, cut90)
        n_clusters = len(groups)
        verified = 0; trans_pairs = trans_same = 0
        edges = {(int(a), int(b)) for a, b in zip(pi[ps >= cut90], pj[ps >= cut90])}
        by_size = defaultdict(lambda: [0, 0])
        for g in groups.values():
            if len(g) < 2:
                continue
            distinct = len({frozenset(surf_concepts[surfs[i]]) for i in g})
            verified += len(g) - distinct
            if len(g) >= 3:
                sz = "3-5" if len(g) <= 5 else "6-20" if len(g) <= 20 else "21-100" if len(g) <= 100 else "100+"
                if len(g) <= 400:
                    for a, c in combinations(g, 2):
                        if (a, c) not in edges and (c, a) not in edges:
                            s_ = same_concept(surfs[a], surfs[c], surf_concepts)
                            trans_pairs += 1; trans_same += s_
                            by_size[sz][0] += 1; by_size[sz][1] += s_
        sizes = sorted((len(g) for g in groups.values()), reverse=True)
        res["clusters_at_matched_090"][method] = {
            "n_clusters": n_clusters, "largest": sizes[0], "largest_share": sizes[0] / n,
            "candidate_recovery": (n - n_clusters) / n, "verified_recovery": verified / n,
            "merge_precision": verified / (n - n_clusters) if n > n_clusters else None,
            "transitive_precision": trans_same / trans_pairs if trans_pairs else None, "transitive_pairs": trans_pairs,
            "transitive_by_size": {k: {"n": v[0], "precision": v[1] / v[0] if v[0] else None} for k, v in sorted(by_size.items())}}
    # 단일연결 .85 일치 거대 성분
    groups85, _ = cluster_single_linkage(n, pi, pj, ps, cuts[0.85])
    res["single_linkage_at_matched_085_largest_share"] = max(len(g) for g in groups85.values()) / n
    res["bge_reference"] = {"clusters_louvain_090": bge["clusters"]["louvain@0.9"], "single_linkage_085_largest_share": bge["clusters"]["single_linkage@0.85"]["largest_share"]}
    return res


# ───────────────────────────────────────────── C. NCS 전수

def stage_c(model, device, batch, block, dsn):
    key_surfs, key_jobs, key_main_type, n_surface = load_keys(dsn)
    keys = sorted(key_surfs)
    n = len(keys)
    ktype_of = [key_main_type.get(k, "기타") for k in keys]
    surfaces_total = sum(len(key_surfs[k]) for k in keys)
    bz = np.load(OUT / "h3_vecs_326592_pairs_0.85.npz")
    bps = bz["ps"]
    targets = {t: int((bps >= t).sum()) for t in BGE_THRESHOLDS}
    vecs = embed(model, keys, device, batch, OUT / f"h3_vecs_{slug(model)}_{n}.npz")
    pi, pj, ps, tmin = adaptive_pairs(vecs, device, max(targets.values()) * 1.5, block=block)
    ps_d = np.sort(ps)[::-1]
    cuts = {t: matched_threshold(ps_d, k) for t, k in targets.items()}
    bge = json.loads((OUT / "t0_h3_embedding.json").read_text(encoding="utf-8"))["results"]
    res = {"model": model, "n_keys": n, "min_cos_collected": tmin, "n_pairs_collected": int(len(ps)),
           "matched_cuts": {str(t): {"bge_n_pairs": k, "new_cut": cuts[t]} for t, k in targets.items()}, "at_matched": {}}
    # bge 쌍과의 겹침(같은 쌍 수에서 상위 집합 일치도) — .90 일치
    bset = {(int(a), int(b)) for a, b, s in zip(bz["pi"], bz["pj"], bps) if s >= 0.90}
    m90 = ps >= cuts[0.90]
    nset = {(int(a), int(b)) for a, b in zip(pi[m90], pj[m90])}
    res["pair_set_overlap_at_090"] = {"jaccard": len(bset & nset) / len(bset | nset), "bge": len(bset), "new": len(nset), "both": len(bset & nset)}
    for t in BGE_THRESHOLDS:
        cut = cuts[t]
        groups, _ = cluster_louvain(n, pi, pj, ps, cut)
        merged_idx = {i for g in groups.values() if len(g) > 1 for i in g}
        sizes = sorted((len(g) for g in groups.values()), reverse=True)
        per_type = {}
        for ty in ("지식", "기술", "태도"):
            idxs = [i for i in range(n) if ktype_of[i] == ty]
            per_type[ty] = sum(1 for i in idxs if i in merged_idx) / len(idxs) if idxs else None
        ref = bge[str(t) if str(t) in bge else str(float(t))]["louvain"]
        res["at_matched"][str(t)] = {
            "new_cut": cut, "n_clusters": len(groups),
            "combined_recovery_vs_surface": (surfaces_total - len(groups)) / surfaces_total,
            "largest_cluster": sizes[0], "top5": sizes[:5], "per_kta_type_participation": per_type,
            "bge": {"combined_recovery_vs_surface": ref["combined_recovery_vs_surface"], "largest_cluster": ref["largest_cluster"],
                    "per_kta_type_participation": {k: v["merge_participation"] for k, v in ref["per_kta_type"].items()}}}
        if t == 0.85:
            g85, _ = cluster_single_linkage(n, pi, pj, ps, cut)
            res["single_linkage_at_matched_085_largest_share"] = max(len(g) for g in g85.values()) / n
    return res


# ───────────────────────────────────────────── D. 두 모델의 합의 — 교집합/차집합 정밀도 (ESCO 정답, NCS 판정)

def stage_d(model, device, batch, block):
    """같은 쌍 수(.90 일치)에서 두 모델이 고른 쌍 집합이 얼마나 겹치고, 겹치는 쌍과 한쪽만 고른 쌍의 정밀도가 어떻게 다른가.
    집계가 같아도 개별 쌍이 다르면 교집합 정책(두 모델 모두 임계값 이상)이 정밀도를 올릴 수 있다."""
    skills, _ = esco_load()
    surf_concepts, _ = build_surfaces(skills)
    surfs = sorted(surf_concepts); n = len(surfs)
    bge_bz = np.load(ESCO / "esco_vecs_99676.npz", allow_pickle=True)
    new_z = np.load(ESCO / f"esco_vecs_{slug(model)}_{n}.npz", allow_pickle=True)
    assert list(bge_bz["keys"]) == surfs == list(new_z["keys"])
    bpi, bpj, bps = collect_pairs(bge_bz["vecs"], 0.85, block, device)
    npi, npj, nps = collect_pairs(new_z["vecs"], 0.90, block, device)
    res = {"model": model}
    for t in BGE_THRESHOLDS:
        k = int((bps >= t).sum())
        bset = {(int(a), int(b)) for a, b, s in zip(bpi, bpj, bps) if s >= t}
        order = np.argsort(-nps)[:k]
        nset = {(int(a), int(b)) for a, b in zip(npi[order], npj[order])}
        both, only_b, only_n = bset & nset, bset - nset, nset - bset
        def prec(S): return float(np.mean([same_concept(surfs[a], surfs[b], surf_concepts) for a, b in S])) if S else None
        res[str(t)] = {"n_each": k, "jaccard": len(both) / len(bset | nset),
                       "intersection": {"n": len(both), "precision": prec(both)},
                       "only_bge": {"n": len(only_b), "precision": prec(only_b)},
                       "only_new": {"n": len(only_n), "precision": prec(only_n)},
                       "union": {"n": len(bset | nset), "precision": prec(bset | nset)}}
    # NCS 판정 200쌍: 직접 쌍 80(bge ≥.90) 중 새 모델 상위 80 에도 든 것 vs 못 든 것의 외부 판정 정밀도
    judgs, items, _ = load_judgments(OUT / "t0_judgments.json", OUT / "t0_h3_validation_sample.json")
    judgs = [j for j in judgs if not j["rater"].startswith("AU")]
    z = np.load(OUT / f"t0_encsens_{slug(model)}_pairs200.npz", allow_pickle=True)
    ki = {k: i for i, k in enumerate(z["keys"])}; V = z["vecs"]
    new_cos = {pid: float(V[ki[it["left"]["key"]]] @ V[ki[it["right"]["key"]]]) for pid, it in items.items()}
    direct = [p for p in items if items[p]["_hidden"]["arm"] == "A"]
    top_b = {p for p in direct if items[p]["_hidden"]["cosine"] >= 0.90}
    top_n = set(sorted(direct, key=lambda p: -new_cos[p])[:len(top_b)])
    by_pid = defaultdict(list)
    for j in judgs: by_pid[j["pair_id"]].append(j["verdict"])
    def prec_j(S):
        vs = [v for p in S for v in by_pid.get(p, [])]
        return {"n_items": len(S), "strict": float(np.mean([strict(v) for v in vs])) if vs else None,
                "lenient": float(np.mean([lenient(v) for v in vs])) if vs else None}
    res["ncs_200"] = {"both": prec_j(top_b & top_n), "only_bge": prec_j(top_b - top_n), "only_new": prec_j(top_n - top_b),
                      "neither": prec_j(set(direct) - top_b - top_n)}
    return res


def pct(v): return "—" if v is None else f"{v*100:.1f}%"


def write_report(path, R):
    L = []; A = L.append
    A(f"# 인코더 민감도 — {R['meta']['model']} vs bge-m3\n\n실행 {R['meta']['run_at']}\n")
    if "A" in R:
        a = R["A"]
        A(f"## A. NCS 판정 200쌍\n\n두 모델 코사인 상관: Spearman {a['spearman_old_new']:.3f} · Pearson {a['pearson_old_new']:.3f} · "
          f"새 모델 코사인 범위 [{a['new_cos_range'][0]:.3f}, {a['new_cos_range'][1]:.3f}] (bge [{a['old_cos_range'][0]:.3f}, {a['old_cos_range'][1]:.3f}])\n")
        A("| bge 대역 | 문항 | 새 모델 cos 평균±SD | 정밀도 엄격 | 관대 |\n|---|---|---|---|---|")
        for b, v in a["by_old_band"].items():
            A(f"| {b} | {v['n_items']} | {v['new_cos_mean']:.3f}±{v['new_cos_sd']:.3f} | {pct(v['strict'])} | {pct(v['lenient'])} |")
        A("\n새 모델 코사인 3분위(직접 쌍):\n\n| 분위 | cos 범위 | 문항 | 엄격 | 관대 |\n|---|---|---|---|---|")
        for i, t in enumerate(a["direct_by_new_tercile"]):
            A(f"| {i+1} | {t['cos_lo']:.3f}–{t['cos_hi']:.3f} | {t['n_items']} | {pct(t['strict'])} | {pct(t['lenient'])} |")
        rm = a["direct_rank_matched_at_old_090"]
        A(f"\n순위 일치(bge ≥.90 직접 쌍 {rm['n_above']}개 = 새 모델 상위 {rm['n_above']}개, cut {rm['new_model_cut']:.3f}): "
          f"bge 엄격 {pct(rm['old_model']['strict'])} vs 새 {pct(rm['new_model']['strict'])} · 상위 집합 겹침 {pct(rm['overlap_of_top_sets'])}\n")
    if "B" in R:
        b = R["B"]
        A(f"## B. ESCO 전수 (표면형 {b['n_surface_forms']:,} · 수집 최소 cos {b['min_cos_collected']:.2f} · 쌍 {b['n_pairs_collected']:,})\n")
        A("| bge 임계값 | bge 쌍 수 | 새 모델 일치 cut | 정밀도 bge → 새 | 재현율 bge → 새 |\n|---|---|---|---|---|")
        for t, v in b["matched"].items():
            A(f"| {t} | {v['bge_n_pairs']:,} | {v['new_cut']:.3f} | {pct(v['bge_precision'])} → {pct(v['precision'])} | {pct(v['bge_recall'])} → {pct(v['recall'])} |")
        A("\n일치 대역 정밀도:\n\n| 대역(bge 기준) | 새 모델 cos | 쌍 | bge | 새 |\n|---|---|---|---|---|")
        for k, v in b["matched_band_precision"].items():
            A(f"| {k} | {v['new_cos_lo']:.3f}–{v['new_cos_hi']:.3f} | {v['n_pairs']:,} | {pct(v['bge_precision'])} | {pct(v['precision'])} |")
        kn = b["knee"]
        A(f"\n무릎(20분위, 같은 쌍 수 {b.get('knee_n_pairs_compared', b['n_pairs_collected']):,}): 새 모델 — 대역 {kn['knee_after_band']+1}/{len(kn['bands'])} 뒤, 순위 {kn['knee_rank']:,}, cos {kn['knee_cos']:.3f}, 낙차 {kn['precision_drop']*100:.1f}%p")
        if "knee_bge_same_n" in b:
            kb = b["knee_bge_same_n"]
            A(f"bge-m3 — 대역 {kb['knee_after_band']+1}/{len(kb['bands'])} 뒤, 순위 {kb['knee_rank']:,}, cos {kb['knee_cos']:.3f}, 낙차 {kb['precision_drop']*100:.1f}%p")
        A("\n군집 @ 일치 .90:\n\n| 방법 | 군집 | 최대(비율) | 제안 회수 | 검증 회수 | 병합 정밀도 | 전이 정밀도 |\n|---|---|---|---|---|---|---|")
        for m, v in b["clusters_at_matched_090"].items():
            A(f"| {m} | {v['n_clusters']:,} | {v['largest']:,} ({pct(v['largest_share'])}) | {pct(v['candidate_recovery'])} | {pct(v['verified_recovery'])} | {pct(v['merge_precision'])} | {pct(v['transitive_precision'])} (n={v['transitive_pairs']:,}) |")
        rb = b["bge_reference"]["clusters_louvain_090"]
        A(f"| bge louvain@.90 | {rb['n_clusters']:,} | {rb['largest']:,} ({pct(rb['largest_share'])}) | {pct(rb['candidate_recovery'])} | {pct(rb['verified_recovery'])} | {pct(rb['merge_precision'])} | {pct(rb['transitive_precision'])} |")
        lv = b["clusters_at_matched_090"]["louvain"]["transitive_by_size"]
        A("\n전이 정밀도 크기별(새 모델 Louvain): " + " · ".join(f"{k} {pct(v['precision'])}" for k, v in lv.items()))
        A(f"\n단일연결 @ 일치 .85 최대 성분: 새 {pct(b['single_linkage_at_matched_085_largest_share'])} vs bge {pct(b['bge_reference']['single_linkage_085_largest_share'])}\n")
    if "C" in R:
        c = R["C"]
        A(f"## C. NCS 전수 (키 {c['n_keys']:,} · 수집 최소 cos {c['min_cos_collected']:.2f} · 쌍 {c['n_pairs_collected']:,})\n")
        po = c["pair_set_overlap_at_090"]
        A(f"bge ≥.90 쌍 집합 vs 새 모델 일치 상위 집합: Jaccard {po['jaccard']:.3f} (bge {po['bge']:,} · 새 {po['new']:,} · 교집합 {po['both']:,})\n")
        A("| bge 임계값 | 새 cut | 결합 회수 bge → 새 | 최대 군집 bge → 새 | 태도/기술/지식 참여 bge → 새 |\n|---|---|---|---|---|")
        for t, v in c["at_matched"].items():
            bp, np_ = v["bge"]["per_kta_type_participation"], v["per_kta_type_participation"]
            A(f"| {t} | {v['new_cut']:.3f} | {pct(v['bge']['combined_recovery_vs_surface'])} → {pct(v['combined_recovery_vs_surface'])} | "
              f"{v['bge']['largest_cluster']:,} → {v['largest_cluster']:,} | "
              f"{pct(bp.get('태도'))}/{pct(bp.get('기술'))}/{pct(bp.get('지식'))} → {pct(np_.get('태도'))}/{pct(np_.get('기술'))}/{pct(np_.get('지식'))} |")
        A(f"\n단일연결 @ 일치 .85 최대 성분: 새 {pct(c['single_linkage_at_matched_085_largest_share'])} vs bge 29.9%\n")
    if "D" in R:
        d = R["D"]
        A("## D. 두 모델의 합의 — 같은 쌍 수에서 교집합/차집합 정밀도 (ESCO 정답)\n")
        A("| bge 임계값 | 각 쌍 수 | Jaccard | 교집합 n·정밀도 | bge만 | 새 모델만 | 합집합 |\n|---|---|---|---|---|---|---|")
        for t in ("0.85", "0.9", "0.95"):
            v = d[t]
            A(f"| {t} | {v['n_each']:,} | {v['jaccard']:.3f} | {v['intersection']['n']:,} · {pct(v['intersection']['precision'])} | "
              f"{v['only_bge']['n']:,} · {pct(v['only_bge']['precision'])} | {v['only_new']['n']:,} · {pct(v['only_new']['precision'])} | {v['union']['n']:,} · {pct(v['union']['precision'])} |")
        nc = d["ncs_200"]
        A("\nNCS 판정 직접 쌍(bge ≥.90 80개 vs 새 모델 상위 80개): " + " · ".join(f"{k} n={v['n_items']} 엄격 {pct(v['strict'])} 관대 {pct(v['lenient'])}" for k, v in nc.items()))
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-large")
    ap.add_argument("--stages", default="A,B,C")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--dsn", default=os.environ.get("DATABASE_URL", "postgresql://ncs:ncs@localhost:55432/ncs"))
    args = ap.parse_args()
    device = args.device
    if device is None:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    out_json = OUT / "t0_encoder_sensitivity.json"
    R = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    R["meta"] = {"run_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "model": args.model,
                 "reference_model": "BAAI/bge-m3", "device": device, "seed": SEED, "prefix": prefix_for(args.model)}
    for st in args.stages.split(","):
        print(f"[enc] stage {st} …", flush=True)
        R[st] = {"A": lambda: stage_a(args.model, device, args.batch),
                 "B": lambda: stage_b(args.model, device, args.batch, args.block),
                 "C": lambda: stage_c(args.model, device, args.batch, args.block, args.dsn),
                 "D": lambda: stage_d(args.model, device, args.batch, args.block)}[st]()
        out_json.write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(OUT / "t0_encoder_sensitivity_report.md", R)
        print(f"[enc] stage {st} 완료 → {out_json}", flush=True)


if __name__ == "__main__":
    main()
