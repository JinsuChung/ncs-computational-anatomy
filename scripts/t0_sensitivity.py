"""T0 민감도 분석 — 해설집 7.3 의 심사자 요구 3·4번.

M9  규칙 정규화 접미사 목록 민감도: 접미사 목록을 빼고/더하고/태도 접미 제거를 추가했을 때 H3 규칙 회수율(8.2%)이
    어디까지 움직이는가. 규칙 병합 정밀도는 NCS 에서 미측정이므로 ESCO 정답 기준(R0 97.8% · R1 98.9%)을 참조로 병기한다.
M-T0-5  H4 군집 민감도: Louvain 해상도 {0.5, 1.0, 2.0} × 유사도 {Jaccard, TF-IDF 코사인} 2×3 표.
        불일치 직무의 대안 정의(군집 수에 덜 의존): 자기 대분류 직무들과의 평균 유사도보다 다른 대분류와의 평균 유사도가
        더 큰 직무 = '끌려간' 직무.

사용: .venv/bin/python scripts/t0_sensitivity.py [--dsn ...] [--out docs/t0]
출력: docs/t0/t0_sensitivity.json · t0_sensitivity_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ontology_keys import ROLE_SUFFIX_PATTERNS, _PUNCT, normalize_key  # noqa: E402
from t0_analysis import SEED, load_ksa, load_metrics, spearman, tune_session  # noqa: E402

# ── M9 접미사 변형. 기본(base)이 사전 등록 규칙. 나머지는 결과를 본 뒤 정한 민감도 변형이며 주 수치를 바꾸지 않는다.
PLUS_PATTERNS = ROLE_SUFFIX_PATTERNS + [
    r"에\s*(?:대한|관한)\s*(?:이해도|숙지|파악|활용|적용|고려)$",
    r"(?:관련\s*)?(?:이해도|숙지|파악|활용\s*능력|적용\s*능력|관련\s*지식|관련\s*기술|관련\s*법규)$",
]
ATTITUDE_PATTERNS = ROLE_SUFFIX_PATTERNS + [
    r"(?:하려는|하고자\s*하는|하는|려는|는)\s*(?:자세|의지|노력|마음가짐|마음|태도)$",
    r"(?:자세|의지|노력|마음가짐|마음)$",
]
VARIANTS = {
    "none(공백·구두점만)": [],
    "phrase_only(에 대한 X만)": ROLE_SUFFIX_PATTERNS[:1],
    "base(사전 등록)": ROLE_SUFFIX_PATTERNS,
    "plus(이해도·숙지·파악·활용능력 추가)": PLUS_PATTERNS,
    "attitude(자세·의지·노력 접미 추가)": ATTITUDE_PATTERNS,
}


def make_normalizer(patterns):
    def strip(text):
        out = text.strip()
        for p in patterns:
            s = re.sub(p, "", out).strip()
            if s and s != out:
                out = s
        return out

    def norm(text):
        base = unicodedata.normalize("NFKC", str(text or "")).strip()
        key = _PUNCT.sub("", strip(base)).lower()
        return key or _PUNCT.sub("", base).lower()
    return norm


def m9_suffix_sensitivity(ksa):
    surfaces = {}
    for job, ktype, desc in ksa:
        d = unicodedata.normalize("NFKC", (desc or "").strip())
        if d:
            surfaces.setdefault(d, [set(), set()])
            surfaces[d][0].add(job); surfaces[d][1].add(ktype)
    n_surf = len(surfaces)
    out = {"n_surface_forms": n_surf, "variants": {}}
    base_groups = None
    for name, pats in VARIANTS.items():
        norm = make_normalizer(pats)
        key_surfs, key_jobs, key_types = defaultdict(set), defaultdict(set), defaultdict(set)
        for d, (jobs, types) in surfaces.items():
            k = norm(d)
            if not k:
                continue
            key_surfs[k].add(d); key_jobs[k] |= jobs; key_types[k] |= types
        merged = sum(len(v) - 1 for v in key_surfs.values())
        freqs = [len(key_jobs[k]) for k in key_surfs]
        frags = [len(key_surfs[k]) for k in key_surfs]
        rho = spearman(freqs, frags)
        mismatch = sum(1 for t in key_types.values() if len(t) > 1)
        rec = {"n_rule_keys": len(key_surfs), "surface_forms_merged": merged,
               "pct_surface_merged": merged / n_surf, "type_mismatch_keys": mismatch,
               "type_mismatch_rate": mismatch / len(key_surfs), "freq_frag_rho": rho["rho"] if rho else None}
        if "base" in name:
            base_groups = key_surfs
        out["variants"][name] = (rec, key_surfs)
    # 변형이 base 에 비해 새로 합친 그룹의 예시 — 오병합 여부를 눈으로 볼 수 있게
    base_key_of = {s: k for k, ss in base_groups.items() for s in ss}
    for name, (rec, key_surfs) in out["variants"].items():
        if "base" in name or not key_surfs:
            continue
        new_groups = []
        for k, ss in key_surfs.items():
            if len({base_key_of.get(s) for s in ss}) > 1:
                new_groups.append(sorted(ss)[:5])
        rec["groups_merged_beyond_base"] = len(new_groups)
        rec["examples_beyond_base"] = sorted(new_groups, key=len, reverse=True)[:12]
    out["variants"] = {name: rec for name, (rec, _) in out["variants"].items()}
    out["esco_reference_precision"] = {"R0_lower_punct": 0.978, "R1_plus_inflection": 0.989,
                                       "note": "ESCO v1.2.1 정답(같은 URI) 기준 규칙 병합 정밀도 — NCS 규칙 정밀도는 미측정"}
    return out


def h4_sensitivity(ksa, major_of, middle_of, resolutions=(0.5, 1.0, 2.0), min_jobs_for_key=2):
    import networkx as nx
    import numpy as np
    import scipy.sparse as sp
    import random
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
    from sklearn.preprocessing import normalize

    key_jobs = defaultdict(set)
    for job, _t, desc in ksa:
        k = normalize_key(desc or "")
        if k:
            key_jobs[k].add(job)
    keys = [k for k, js in key_jobs.items() if len(js) >= min_jobs_for_key]
    jobs = sorted({j for k in keys for j in key_jobs[k]})
    jidx = {j: i for i, j in enumerate(jobs)}
    n = len(jobs)
    rows, cols = [], []
    for ci, k in enumerate(keys):
        for j in key_jobs[k]:
            rows.append(jidx[j]); cols.append(ci)
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(keys)))

    # Jaccard: |A∩B| / (|A|+|B|-|A∩B|)
    inter = (X @ X.T).tocoo()
    size = np.asarray(X.sum(axis=1)).ravel()
    sims = {}
    ju, jv, jw = [], [], []
    for u, v, c in zip(inter.row, inter.col, inter.data):
        if u < v and c > 0:
            ju.append(u); jv.append(v); jw.append(c / (size[u] + size[v] - c))
    sims["jaccard"] = (ju, jv, jw)
    # TF-IDF 코사인 (tf 이진, idf=log(N/df))
    df = np.asarray(X.sum(axis=0)).ravel()
    idf = np.log(n / df)
    Xt = normalize(X.multiply(idf).tocsr(), norm="l2")
    cos = (Xt @ Xt.T).tocoo()
    cu, cv, cw = [], [], []
    for u, v, c in zip(cos.row, cos.col, cos.data):
        if u < v and c > 0:
            cu.append(u); cv.append(v); cw.append(float(c))
    sims["tfidf_cosine"] = (cu, cv, cw)

    majors = [major_of.get(j, "?") for j in jobs]
    middles = [middle_of.get(j, "?") for j in jobs]
    grid = {}
    for sname, (eu, ev, ew) in sims.items():
        g = nx.Graph(); g.add_nodes_from(range(n))
        g.add_weighted_edges_from(zip(eu, ev, ew))
        for res in resolutions:
            cs = nx.community.louvain_communities(g, weight="weight", seed=SEED, resolution=res)
            lab = [0] * n
            for i, c in enumerate(cs):
                for u in c:
                    lab[u] = i
            # 순열 영분포: NMI 는 군집 수가 늘수록 우연 기대치도 오르므로 셀마다 따로 낸다(AMI 는 우연 보정판)
            rng = random.Random(SEED); perm = list(majors); null = []
            for _ in range(200):
                rng.shuffle(perm); null.append(normalized_mutual_info_score(perm, lab))
            grid[f"{sname}@{res}"] = {
                "similarity": sname, "resolution": res, "n_edges": g.number_of_edges(),
                "n_clusters": len(cs), "largest": max(len(c) for c in cs),
                "nmi_major": float(normalized_mutual_info_score(majors, lab)),
                "ami_major": float(adjusted_mutual_info_score(majors, lab)),
                "ari_major": float(adjusted_rand_score(majors, lab)),
                "nmi_middle": float(normalized_mutual_info_score(middles, lab)),
                "ami_middle": float(adjusted_mutual_info_score(middles, lab)),
                "null_nmi_major_mean": float(np.mean(null)), "null_nmi_major_sd": float(np.std(null)),
            }
    # 대안 불일치 정의(Jaccard): 자기 대분류 평균 유사도 < 타 대분류 최대 평균 유사도
    S = sp.coo_matrix((jw + jw, (ju + jv, jv + ju)), shape=(n, n)).tocsr()
    maj_idx = defaultdict(list)
    for i, m in enumerate(majors):
        maj_idx[m].append(i)
    pulled = []
    for i in range(n):
        row = S.getrow(i).toarray().ravel()
        own = majors[i]
        own_members = [k for k in maj_idx[own] if k != i]
        own_mean = row[own_members].mean() if own_members else 0.0
        best_other, best_val = None, -1.0
        for m, members in maj_idx.items():
            if m == own:
                continue
            v = row[members].mean()
            if v > best_val:
                best_val, best_other = v, m
        if best_val > own_mean:
            pulled.append({"job": jobs[i], "major": own, "own_mean": float(own_mean),
                           "pulled_to": best_other, "other_mean": float(best_val)})
    pulled.sort(key=lambda r: -(r["other_mean"] - r["own_mean"]))
    return {"n_nodes": n, "n_keys_used": len(keys), "grid": grid,
            "pulled_jobs": {"definition": "자기 대분류 직무들과의 평균 Jaccard < 타 대분류 직무들과의 최대 평균 Jaccard",
                            "n": len(pulled), "rate": len(pulled) / n, "top": pulled[:15]}}


def pct(v): return "—" if v is None else f"{v*100:.1f}%"


def write_report(path, R, name_of):
    L = []
    A = L.append
    A(f"# T0 민감도 분석\n\n실행 {R['meta']['run_at']} · 시드 {SEED}\n")
    m9 = R["M9_suffix"]
    A(f"## M9 규칙 접미사 목록 민감도 (표면형 {m9['n_surface_forms']:,})\n")
    A("| 변형 | 규칙 키 | 회수(표면형 병합 비율) | 유형 불일치 키(비율) | ρ(빈도–파편화) | base 대비 추가 병합 그룹 |")
    A("|---|---|---|---|---|---|")
    for name, v in m9["variants"].items():
        A(f"| {name} | {v['n_rule_keys']:,} | {pct(v['pct_surface_merged'])} | {v['type_mismatch_keys']:,} ({pct(v['type_mismatch_rate'])}) | "
          f"{v['freq_frag_rho']:.3f} | {v.get('groups_merged_beyond_base', '—')} |")
    A(f"\nESCO 정답 기준 규칙 정밀도 참조: R0 {pct(m9['esco_reference_precision']['R0_lower_punct'])} · R1 {pct(m9['esco_reference_precision']['R1_plus_inflection'])}\n")
    for name, v in m9["variants"].items():
        if v.get("examples_beyond_base"):
            A(f"\n{name} — base 를 넘어 합쳐진 그룹 예시:")
            for ex in v["examples_beyond_base"][:6]:
                A("- " + " | ".join(ex))
    h4 = R["H4_sensitivity"]
    A(f"\n## H4 군집 민감도 (노드 {h4['n_nodes']:,} · 키 {h4['n_keys_used']:,})\n")
    A("| 유사도 @ 해상도 | 군집 | 최대 | NMI 대분류 (영분포) | AMI 대분류 | ARI | NMI 중분류 | AMI 중분류 |")
    A("|---|---|---|---|---|---|---|---|")
    for k, v in h4["grid"].items():
        A(f"| {k} | {v['n_clusters']} | {v['largest']} | {v['nmi_major']:.3f} ({v['null_nmi_major_mean']:.3f}±{v['null_nmi_major_sd']:.3f}) | "
          f"{v['ami_major']:.3f} | {v['ari_major']:.3f} | {v['nmi_middle']:.3f} | {v['ami_middle']:.3f} |")
    p = h4["pulled_jobs"]
    A(f"\n'끌려간' 직무(대안 정의): {p['n']:,} / {h4['n_nodes']:,} ({pct(p['rate'])}) — 기존 정의(군집 최빈 대분류 불일치) 736 (66.5%)\n")
    for r in p["top"][:10]:
        A(f"- {name_of.get(r['job'], r['job'])} ({r['major']}) → {r['pulled_to']} : 자기 {r['own_mean']:.3f} vs 타 {r['other_mean']:.3f}")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("DATABASE_URL", "postgresql://ncs:ncs@localhost:55432/ncs"))
    ap.add_argument("--out", default="docs/t0")
    args = ap.parse_args()
    out = Path(args.out)
    with psycopg.connect(args.dsn) as conn:
        tune_session(conn)
        print("[sens] 지표 적재 …", flush=True)
        metrics = load_metrics(conn)
        major_of = {r["job"]: r["major"] for r in metrics}
        middle_of = {r["job"]: r["middle"] for r in metrics}
        name_of = {r["job"]: r["job_name"] for r in metrics}
        print("[sens] KSA 적재 …", flush=True)
        ksa = load_ksa(conn)
    print(f"[sens] KSA {len(ksa):,} · M9 …", flush=True)
    R = {"meta": {"run_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "seed": SEED}}
    R["M9_suffix"] = m9_suffix_sensitivity(ksa)
    print("[sens] H4 2×3 …", flush=True)
    R["H4_sensitivity"] = h4_sensitivity(ksa, major_of, middle_of)
    for r in R["H4_sensitivity"]["pulled_jobs"]["top"]:
        r["job_name"] = name_of.get(r["job"], r["job"])
    (out / "t0_sensitivity.json").write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out / "t0_sensitivity_report.md", R, name_of)
    print("[sens] 완료 → docs/t0/t0_sensitivity.json · t0_sensitivity_report.md")


if __name__ == "__main__":
    main()
