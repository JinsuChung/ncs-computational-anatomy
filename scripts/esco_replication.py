"""ESCO 재현 — T0 의 규칙/임베딩 경계(H3)와 분류 회복(H4)이 다른 코퍼스에서도 성립하는가.

ESCO 는 NCS 와 결정적으로 다른 점이 하나 있다: 한 개념의 표기 변형(altLabels)이 **선언되어 있다**.
그래서 NCS 에서 사람 8인이 200쌍을 판정해 얻은 정밀도를, ESCO 에서는 전수 쌍에 대해 정답으로 잰다.
"같은 개념" = 같은 스킬 URI 아래 등록된 표기.

  E-H1  템플릿 규칙성: skill/competence 라벨의 동사 초두 비율, knowledge 라벨의 명사 초두 비율 (spaCy)
  E-H2  재사용 층: 스킬이 필수인 직업 수 (NCS 의 등장 직무 수 층과 대응)
  E-H3  파편화: 규칙 정규화(R0 소문자·구두점 / R1 + 영어 굴절 접미사 제거)의 회수·정밀도,
        임베딩(bge-m3) 직접 쌍 대역별 정밀도 · 누적 정밀도(무릎) · 전이 쌍 정밀도(군집 크기별) ·
        단일연결 vs Louvain 최대 성분 · 검증 회수율(정답 기반)
  E-H4  직업 × 필수 스킬 그래프 → Louvain → ISCO 1자리·2자리와 NMI/ARI, 순열 영분포, 범용 스킬 제외 민감도

입력  docs/esco/esco_skills.json · esco_occupations.json (scripts/esco_fetch.py)
출력  docs/esco/esco_replication.json · esco_replication_report.md · esco_vecs_<n>.npz(캐시, 비추적)

사용: .venv/bin/python scripts/esco_replication.py [--permutations 1000]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t0_embed_recovery import cluster_louvain, cluster_single_linkage, collect_pairs, embed_texts  # noqa: E402

SEED = 20260909
IN = Path("docs/esco")
BANDS = [(0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
_PUNCT = re.compile(r"[\s\-–—_/\\(){}\[\]<>,.:;!?'\"“”‘’`~^*+=|&]+")


# ───────────────────────────────────────── 규칙 정규화 (영어 아날로그)

def r0(text):
    """소문자 + NFKC + 공백·구두점 제거 — 한국어 키에서 접미사 규칙을 뺀 것."""
    return _PUNCT.sub("", unicodedata.normalize("NFKC", text).lower())


def strip_inflection(w):
    # 한국어 역할 접미사 제거의 영어 아날로그 — 굴절 접미사만 뗀다(어간이 다른 동의어는 못 합침).
    if len(w) >= 5 and w.endswith("ies"):
        return w[:-3] + "y"
    for suf in ("ing", "ed", "es", "s"):
        if len(w) >= 4 + len(suf) and w.endswith(suf):
            return w[: -len(suf)]
    return w


def r1(text):
    toks = re.split(r"[\s\-–—_/]+", unicodedata.normalize("NFKC", text).lower())
    return _PUNCT.sub("", " ".join(strip_inflection(t) for t in toks if t))


# ───────────────────────────────────────── 데이터

def load():
    skills = json.loads((IN / "esco_skills.json").read_text(encoding="utf-8"))
    occs = json.loads((IN / "esco_occupations.json").read_text(encoding="utf-8"))
    return skills, occs


def build_surfaces(skills):
    """표면형 → 개념 집합. 정답 = 같은 스킬 URI."""
    surf_concepts = defaultdict(set)
    concept_surfs = defaultdict(set)
    for s in skills:
        labels = [s.get("preferredLabel") or ""] + list(s.get("altLabels") or [])
        for lab in labels:
            t = unicodedata.normalize("NFKC", (lab or "").strip())
            if not t:
                continue
            surf_concepts[t].add(s["uri"])
            concept_surfs[s["uri"]].add(t)
    return surf_concepts, concept_surfs


def same_concept(a, b, surf_concepts):
    return bool(surf_concepts[a] & surf_concepts[b])


# ───────────────────────────────────────── E-H1

def eh1_template(skills):
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    except Exception as e:  # noqa: BLE001
        return {"skipped": f"spaCy 불가: {e}"}
    out = {}
    for kind, want in (("skill/competence", "VERB"), ("knowledge", "NOUN")):
        labs = [s["preferredLabel"] for s in skills if s.get("preferredLabel") and any(kind in t for t in s.get("skillType", []))]
        hit = 0
        for doc in nlp.pipe(labs, batch_size=512):
            toks = [t for t in doc if not t.is_space]
            if not toks:
                continue
            p = toks[0].pos_
            if want == "VERB" and p in ("VERB", "AUX"):
                hit += 1
            elif want == "NOUN" and p in ("NOUN", "PROPN", "ADJ"):
                hit += 1
        out[kind] = {"n": len(labs), "template": want + "-initial", "rate": hit / len(labs) if labs else None}
    return out


# ───────────────────────────────────────── E-H2

def eh2_reuse(skills, occs):
    ess = Counter()
    for o in occs:
        for u in o.get("essentialSkills", []):
            ess[u] += 1
    counts = [ess.get(s["uri"], 0) for s in skills]
    strata = Counter(("jobs>=10" if c >= 10 else "jobs 3-9" if c >= 3 else "jobs=2" if c == 2 else "jobs=1" if c == 1 else "jobs=0") for c in counts)
    return {"n_skills": len(skills), "n_essential_links": sum(ess.values()),
            "strata": dict(strata), "mean_occupations_per_skill": float(np.mean(counts)),
            "share_used_by_ge10": strata["jobs>=10"] / len(skills)}


# ───────────────────────────────────────── E-H3

def eh3_rules(surf_concepts):
    surfs = sorted(surf_concepts)
    n_surf = len(surfs)
    out = {"n_surface_forms": n_surf, "n_concepts": len({c for v in surf_concepts.values() for c in v}),
           "ambiguous_surfaces": sum(1 for v in surf_concepts.values() if len(v) > 1)}
    for name, fn in (("R0_lower_punct", r0), ("R1_plus_inflection", r1)):
        key_surfs = defaultdict(set)
        for s in surfs:
            key_surfs[fn(s)].add(s)
        n_keys = len(key_surfs)
        # 규칙이 합친 쌍의 정밀도(같은 개념인가)
        merged_pairs = same = 0
        for ss in key_surfs.values():
            if len(ss) > 1:
                for a, b in combinations(sorted(ss), 2):
                    merged_pairs += 1
                    same += same_concept(a, b, surf_concepts)
        out[name] = {"n_keys": n_keys, "surface_reduction": (n_surf - n_keys) / n_surf,
                     "merged_pairs": merged_pairs, "pair_precision": same / merged_pairs if merged_pairs else None}
    return out


def eh3_embedding(surf_concepts, concept_surfs, device, thresholds=(0.85, 0.90, 0.95), block=2048):
    surfs = sorted(surf_concepts)
    n = len(surfs)
    idx = {s: i for i, s in enumerate(surfs)}
    cache = IN / f"esco_vecs_{n}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        vecs = z["vecs"] if list(z["keys"]) == surfs else None
    else:
        vecs = None
    if vecs is None:
        vecs = embed_texts("BAAI/bge-m3", surfs, device, 128)
        np.savez(cache, vecs=vecs, keys=np.array(surfs, dtype=object))
    pi, pj, ps = collect_pairs(vecs, min(thresholds), block, device)
    truth = np.array([same_concept(surfs[a], surfs[b], surf_concepts) for a, b in zip(pi, pj)], dtype=bool)

    # 같은 개념 쌍 전체(재현율 분모)
    true_pairs = set()
    for c, ss in concept_surfs.items():
        if len(ss) > 1:
            for a, b in combinations(sorted(idx[s] for s in ss), 2):
                true_pairs.add((a, b))
    n_true_pairs = len(true_pairs)

    res = {"n_surface_forms": n, "n_pairs_ge_min": int(len(ps)), "n_true_same_concept_pairs": n_true_pairs}
    # 직접 쌍 대역별 정밀도 (정답 기반, 전수)
    res["direct_precision_by_band"] = {}
    for lo, hi in BANDS:
        m = (ps >= lo) & (ps < hi)
        res["direct_precision_by_band"][f"{lo:.2f}-{hi:.2f}"] = {"n_pairs": int(m.sum()),
                                                                  "precision": float(truth[m].mean()) if m.any() else None}
    res["cumulative_precision"] = {}
    for t in thresholds:
        m = ps >= t
        got = {(int(a), int(b)) for a, b in zip(pi[m], pj[m])}
        res["cumulative_precision"][str(t)] = {"n_pairs": int(m.sum()), "precision": float(truth[m].mean()) if m.any() else None,
                                               "recall_of_true_pairs": len(got & true_pairs) / n_true_pairs if n_true_pairs else None}

    # 군집 (Louvain 주, 단일연결 상한) — 전이 정밀도·검증 회수율
    res["clusters"] = {}
    direct = {(int(a), int(b)) for a, b, s in zip(pi, pj, ps) if s >= 0.0}  # 채움용
    for t in thresholds:
        m = ps >= t
        edges_t = {(int(a), int(b)) for a, b in zip(pi[m], pj[m])}
        for method, fn in (("louvain", cluster_louvain), ("single_linkage", cluster_single_linkage)):
            groups, _ = fn(n, pi, pj, ps, t)
            sizes = sorted((len(g) for g in groups.values()), reverse=True)
            n_clusters = len(groups)
            # 검증 회수: 군집 안에서 개념을 넘지 않는 병합만 인정 = Σ(|c| − 군집 내 개념 수)
            verified = 0
            trans_pairs = trans_same = 0
            by_size = defaultdict(lambda: [0, 0])
            for g in groups.values():
                if len(g) < 2:
                    continue
                cons = set()
                for i in g:
                    cons |= surf_concepts[surfs[i]]
                # 보수적: 군집 내 개념 수 = 서로 다른 개념 집합의 수 (겹치는 다의 표면은 하나로 셈)
                distinct = len({frozenset(surf_concepts[surfs[i]]) for i in g})
                verified += len(g) - distinct
                b = "2" if len(g) == 2 else "3-5" if len(g) <= 5 else "6-20" if len(g) <= 20 else "21-100" if len(g) <= 100 else "100+"
                for a, c in combinations(sorted(g), 2):
                    if (a, c) not in edges_t:
                        trans_pairs += 1
                        s_ = same_concept(surfs[a], surfs[c], surf_concepts)
                        trans_same += s_
                        by_size[b][0] += 1; by_size[b][1] += s_
            res["clusters"][f"{method}@{t}"] = {
                "n_clusters": n_clusters, "largest": sizes[0] if sizes else 0,
                "largest_share": sizes[0] / n if sizes else 0,
                "candidate_recovery": (n - n_clusters) / n,
                "verified_recovery": verified / n,
                "merge_precision": verified / (n - n_clusters) if n > n_clusters else None,
                "transitive_pairs": trans_pairs,
                "transitive_precision": trans_same / trans_pairs if trans_pairs else None,
                "transitive_precision_by_size": {k: {"n_pairs": v[0], "precision": v[1] / v[0] if v[0] else None} for k, v in sorted(by_size.items())},
            }
    return res


# ───────────────────────────────────────── E-H4

def eh4_taxonomy(occs, permutations=1000, drop_top_pct=None):
    import networkx as nx
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    skill_occ = defaultdict(set)
    for o in occs:
        for u in o.get("essentialSkills", []):
            skill_occ[u].add(o["uri"])
    n_occ = len(occs)
    cutoff = int(n_occ * drop_top_pct) if drop_top_pct else None
    inter, size = Counter(), Counter()
    n_keys = 0
    for u, os_ in skill_occ.items():
        if len(os_) < 2 or (cutoff and len(os_) > cutoff):
            continue
        n_keys += 1
        js = sorted(os_)
        for j in js:
            size[j] += 1
        for i in range(len(js)):
            for k in range(i + 1, len(js)):
                inter[(js[i], js[k])] += 1
    g = nx.Graph(); g.add_nodes_from(size)
    for (a, b), c in inter.items():
        d = size[a] + size[b] - c
        if d > 0:
            g.add_edge(a, b, weight=c / d)
    comms = nx.community.louvain_communities(g, weight="weight", seed=SEED, resolution=1.0)
    assign = {u: i for i, c in enumerate(comms) for u in c}
    code_of = {o["uri"]: (o.get("code") or "") for o in occs}
    nodes = sorted(assign)
    labels = [assign[u] for u in nodes]
    isco1 = [code_of[u][:1] for u in nodes]
    isco2 = [code_of[u][:2] for u in nodes]
    nmi1 = float(normalized_mutual_info_score(isco1, labels)); ari1 = float(adjusted_rand_score(isco1, labels))
    nmi2 = float(normalized_mutual_info_score(isco2, labels))
    rng = random.Random(SEED); null = []; perm = list(isco1)
    for _ in range(permutations):
        rng.shuffle(perm); null.append(normalized_mutual_info_score(perm, labels))
    null = np.asarray(null)
    return {"n_nodes": g.number_of_nodes(), "n_edges": g.number_of_edges(), "n_skills_used": n_keys,
            "n_clusters": len(comms), "cluster_sizes_top10": sorted((len(c) for c in comms), reverse=True)[:10],
            "n_isco1": len(set(isco1)), "n_isco2": len(set(isco2)),
            "nmi_isco1": nmi1, "ari_isco1": ari1, "nmi_isco2": nmi2,
            "permutation_null": {"mean": float(null.mean()), "sd": float(null.std()),
                                 "p": float(((null >= nmi1).sum() + 1) / (permutations + 1))},
            "drop_top_pct": drop_top_pct}


# ───────────────────────────────────────── main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device
    if device is None:
        try:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            device = "cpu"

    skills, occs = load()
    print(f"[esco] 스킬 {len(skills):,} · 직업 {len(occs):,}", flush=True)
    R = {"meta": {"run_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "seed": SEED,
                  "n_skills": len(skills), "n_occupations": len(occs), "device": device,
                  "ground_truth": "같은 개념 = 같은 ESCO 스킬 URI 아래 등록된 영문 preferredLabel/altLabels"}}
    print("[esco] E-H1 …", flush=True); R["EH1_template"] = eh1_template(skills)
    print("[esco] E-H2 …", flush=True); R["EH2_reuse"] = eh2_reuse(skills, occs)
    surf_concepts, concept_surfs = build_surfaces(skills)
    print(f"[esco] 표면형 {len(surf_concepts):,} · 개념 {len(concept_surfs):,}", flush=True)
    print("[esco] E-H3 규칙 …", flush=True); R["EH3_rules"] = eh3_rules(surf_concepts)
    print("[esco] E-H3 임베딩 …", flush=True); R["EH3_embedding"] = eh3_embedding(surf_concepts, concept_surfs, device)
    print("[esco] E-H4 …", flush=True)
    R["EH4_taxonomy"] = eh4_taxonomy(occs, args.permutations)
    R["EH4_taxonomy_idf"] = eh4_taxonomy(occs, 200, drop_top_pct=0.10)

    (IN / "esco_replication.json").write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(IN / "esco_replication_report.md", R)
    print(f"[esco] 완료 → {IN}/esco_replication.json · esco_replication_report.md")


def pct(v): return "—" if v is None else f"{v*100:.1f}%"


def write_report(path, R):
    L = []; A = L.append
    m = R["meta"]
    A(f"# ESCO 재현 결과\n\n실행 {m['run_at']} · 스킬 {m['n_skills']:,} · 직업 {m['n_occupations']:,} · 정답 = {m['ground_truth']}\n")
    A("## E-H1 템플릿 규칙성 (라벨 초두 품사)\n")
    for k, v in R["EH1_template"].items():
        if isinstance(v, dict): A(f"- {k}: {v['template']} {pct(v['rate'])} (n={v['n']:,})")
        else: A(f"- {k}: {v}")
    e2 = R["EH2_reuse"]
    A(f"\n## E-H2 재사용 층 (필수 스킬로 쓰는 직업 수)\n\n{e2['strata']} · 스킬당 평균 {e2['mean_occupations_per_skill']:.2f} · ≥10 직업 {pct(e2['share_used_by_ge10'])}\n")
    e3 = R["EH3_rules"]
    A(f"## E-H3 파편화 — 규칙\n\n표면형 {e3['n_surface_forms']:,} · 개념 {e3['n_concepts']:,} · 다의 표면형 {e3['ambiguous_surfaces']:,}\n")
    A("| 규칙 | 키 | 표면형 감소(회수) | 병합 쌍 | 병합 정밀도 |"); A("|---|---|---|---|---|")
    for k in ("R0_lower_punct", "R1_plus_inflection"):
        v = e3[k]; A(f"| {k} | {v['n_keys']:,} | {pct(v['surface_reduction'])} | {v['merged_pairs']:,} | {pct(v['pair_precision'])} |")
    e = R["EH3_embedding"]
    A(f"\n## E-H3 파편화 — 임베딩 (전수 쌍 {e['n_pairs_ge_min']:,}, 정답 같은개념 쌍 {e['n_true_same_concept_pairs']:,})\n")
    A("| 직접 쌍 대역 | 쌍 | 정밀도(정답) |"); A("|---|---|---|")
    for k, v in e["direct_precision_by_band"].items(): A(f"| {k} | {v['n_pairs']:,} | {pct(v['precision'])} |")
    A("\n| 누적 cos≥t | 쌍 | 정밀도 | 재현율(정답 쌍) |"); A("|---|---|---|---|")
    for k, v in e["cumulative_precision"].items(): A(f"| {k} | {v['n_pairs']:,} | {pct(v['precision'])} | {pct(v['recall_of_true_pairs'])} |")
    A("\n| 군집 | 클러스터 | 최대(비율) | 제안 회수 | 검증 회수 | 병합 정밀도 | 전이 쌍 정밀도 |"); A("|---|---|---|---|---|---|---|")
    for k, v in e["clusters"].items():
        A(f"| {k} | {v['n_clusters']:,} | {v['largest']:,} ({pct(v['largest_share'])}) | {pct(v['candidate_recovery'])} | {pct(v['verified_recovery'])} | {pct(v['merge_precision'])} | {pct(v['transitive_precision'])} (n={v['transitive_pairs']:,}) |")
    lv = e["clusters"].get("louvain@0.9", {})
    if lv:
        A("\n전이 정밀도 — 군집 크기별 (Louvain@.90):");
        for k, v in lv["transitive_precision_by_size"].items(): A(f"- 크기 {k}: {pct(v['precision'])} (쌍 {v['n_pairs']:,})")
    for key in ("EH4_taxonomy", "EH4_taxonomy_idf"):
        t = R[key]
        A(f"\n## {key}\n\n노드 {t['n_nodes']:,} · 엣지 {t['n_edges']:,} · 스킬 {t['n_skills_used']:,} · 군집 {t['n_clusters']} · "
          f"ISCO1({t['n_isco1']}) NMI **{t['nmi_isco1']:.3f}** ARI {t['ari_isco1']:.3f} · ISCO2({t['n_isco2']}) NMI {t['nmi_isco2']:.3f} · "
          f"영분포 {t['permutation_null']['mean']:.3f}±{t['permutation_null']['sd']:.3f} p={t['permutation_null']['p']:.3f}"
          + (f" · 범용 제외 {t['drop_top_pct']}" if t.get('drop_top_pct') else ""))
    Path(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
