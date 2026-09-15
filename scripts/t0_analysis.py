"""T0 확증 분석 — NCS의 계산적 해부 (docs/t0-preregistration.html 구현).

사전 등록의 규칙을 코드로 강제한다:
  · 분석 단위는 세분류(n=1,109). 탐색(EA11)이 본 대분류 집계는 보고용일 뿐 검정 단위가 아니다.
  · 반분(split-half): 대분류 안에서 세분류를 시드 고정 무작위로 A/B로 나눠 두 번 검정하고,
    부호와 유의성이 일치할 때만 'supported'로 기록한다.
  · 다중 비교: 전 가설의 주 p값 하나씩을 모아 Benjamini-Hochberg FDR 5%.
  · 효과 크기 우선: eps^2 / Cliff's delta / rho 를 항상 함께 낸다.

산출:
  <out>/t0_metrics.csv        D3 — 세분류 1,109 × 전 지표
  <out>/t0_results.json       전 검정의 통계량·p·효과크기·반분 결과·FDR 판정
  <out>/t0_report.md          사람이 읽는 요약
  <out>/t0_alias_dictionary.csv   D1 — KSA 별칭 사전(규칙 키 단계)
  <out>/t0_h4_clusters.csv    H4 군집 배정과 공식 분류 대조

사용:
  .venv/bin/python scripts/t0_analysis.py --out docs/t0
  .venv/bin/python scripts/t0_analysis.py --out docs/t0 --only H1 H5     # 일부만
  .venv/bin/python scripts/t0_analysis.py --out docs/t0 --permutations 200  # H4 빠르게

주의: 임베딩 기반 H3 2단계(bge-m3 클러스터)는 이 스크립트에 없다 — 규칙 키 단계까지만 수행하고,
임베딩 단계는 scripts/embed_align_extractions.py 계열을 재사용해 별도로 붙인다(사전 등록 §4 H3).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ontology_keys import normalize_key  # noqa: E402

# ─────────────────────────────────────────────────────────────── 사전 선언 상수

SEED = 20260909
FDR_Q = 0.05

# H1: 수행준거 종결 템플릿. 공백·마침표 변형 허용.
TEMPLATE_RE = re.compile(r"수\s*있다\s*\.?\s*$")

# H5: 능력단위 코드 접미 _YYvN
CODE_SUFFIX_RE = re.compile(r"_(\d{2})v(\d+)$")

# H6: 디지털 어휘 두 집합(A=주, B=민감도). 사전 선언, 실행 중 변경 금지.
DIGITAL_A = ["디지털", "데이터", "인공지능", "AI", "소프트웨어", "자동화", "프로그램", "전산"]
DIGITAL_B = DIGITAL_A + ["IoT", "빅데이터", "클라우드", "정보시스템", "온라인", "스마트"]

# S2: 태도 코드북(규칙 코딩). 위에서부터 먼저 맞는 범주로 배정 — 순서가 곧 우선순위.
ATTITUDE_CODEBOOK = [
    ("준수·안전", ["준수", "안전", "규정", "법규", "지침", "매뉴얼", "규격", "기준"]),
    ("윤리·책임", ["윤리", "책임", "청렴", "공정", "보안", "비밀", "정직", "신뢰"]),
    ("대인·소통", ["소통", "커뮤니케이션", "협력", "협업", "친절", "고객", "배려", "존중", "팀"]),
    ("인지·분석", ["분석", "전략", "논리", "판단", "관찰", "탐구", "창의", "통찰", "사고"]),
    ("품질·정확", ["정확", "품질", "꼼꼼", "세밀", "철저", "정밀", "완벽", "치밀"]),
    ("의지·노력", ["의지", "노력", "적극", "능동", "성실", "열정", "자세", "마음", "태도"]),
]


# ─────────────────────────────────────────────────────────────── 통계 유틸

def _lazy_stats():
    import numpy as np
    from scipy import stats
    return np, stats


def kruskal_eps2(groups):
    """Kruskal-Wallis H + eps^2 효과 크기. groups: list[list[float]] (빈 그룹 제거)."""
    np, stats = _lazy_stats()
    gs = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    if len(gs) < 2:
        return None
    h, p = stats.kruskal(*gs)
    n = sum(len(g) for g in gs)
    k = len(gs)
    # eps^2 = (H - k + 1) / (n - k)  (Tomczak & Tomczak 2014)
    eps2 = (h - k + 1) / (n - k) if n > k else float("nan")
    return {"test": "kruskal", "H": float(h), "p": float(p), "eps2": float(eps2),
            "k_groups": k, "n": n}


def spearman(xs, ys):
    np, stats = _lazy_stats()
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 8:
        return None
    rho, p = stats.spearmanr(x[m], y[m])
    return {"test": "spearman", "rho": float(rho), "p": float(p), "n": int(m.sum())}


def gini(values):
    np, _ = _lazy_stats()
    v = np.sort(np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=float))
    if len(v) == 0:
        return float("nan")
    v = v - v.min() + 1e-9  # 음수 방지(연도는 양수지만 안전하게)
    n = len(v)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def benjamini_hochberg(pvals, q=FDR_Q):
    """[(key, p)] → {key: {'p':, 'crit':, 'reject':}}"""
    items = [(k, p) for k, p in pvals if p is not None and not math.isnan(p)]
    m = len(items)
    if m == 0:
        return {}
    items.sort(key=lambda kv: kv[1])
    out = {}
    max_i = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= q * i / m:
            max_i = i
    for i, (k, p) in enumerate(items, start=1):
        out[k] = {"p": p, "rank": i, "crit": q * i / m, "reject": i <= max_i}
    return out


def split_half(units, major_of, seed=SEED):
    """대분류 안에서 세분류를 무작위 반분. units: list[key], major_of: key→대분류."""
    rng = random.Random(seed)
    by_major = defaultdict(list)
    for u in units:
        by_major[major_of[u]].append(u)
    a, b = [], []
    for major in sorted(by_major):
        g = sorted(by_major[major])
        rng.shuffle(g)
        half = len(g) // 2
        a.extend(g[:half])
        b.extend(g[half:])
    return set(a), set(b)


def agreement(res_a, res_b, stat_key):
    """반분 두 결과의 부호·유의성 일치 판정."""
    if not res_a or not res_b:
        return {"agree": False, "reason": "검정 불가(표본 부족)"}
    sa, sb = res_a.get(stat_key), res_b.get(stat_key)
    sig_a, sig_b = res_a["p"] < 0.05, res_b["p"] < 0.05
    same_sign = (sa is None or sb is None) or (sa >= 0) == (sb >= 0)
    return {"agree": bool(sig_a and sig_b and same_sign),
            "half_a": {"stat": sa, "p": res_a["p"]},
            "half_b": {"stat": sb, "p": res_b["p"]},
            "reason": "" if (sig_a and sig_b and same_sign) else "부호 또는 유의성 불일치"}


# ─────────────────────────────────────────────────────────────── 데이터 적재

BASE_SQL = """
WITH unit AS (
  SELECT detail_category_full_code AS job,
         count(*) AS n_units,
         avg(level)::float AS mean_level,
         avg((level >= 5)::int)::float AS rate_l5plus,
         avg(2000 + (regexp_match(code, '_(\\d{2})v(\\d+)$'))[1]::int)::float AS mean_rev_year,
         avg(((regexp_match(code, '_(\\d{2})v(\\d+)$'))[1]::int >= 22)::int)::float AS rate_rev2022,
         avg((regexp_match(code, '_(\\d{2})v(\\d+)$'))[2]::int)::float AS mean_version,
         count(*) FILTER (WHERE code !~ '_\\d{2}v\\d+$') AS n_units_nocode
  FROM ncs.competency_units GROUP BY 1),
pcagg AS (
  SELECT cu.detail_category_full_code AS job,
         count(*) AS n_pc,
         avg((p.description ~ '수\\s*있다\\s*\\.?\\s*$')::int)::float AS template_rate,
         avg(length(p.description))::float AS mean_pc_len
  FROM ncs.performance_criteria p
  JOIN ncs.competency_unit_elements e ON e.code = p.competency_unit_element_code
  JOIN ncs.competency_units cu ON cu.code = e.competency_unit_code
  GROUP BY 1),
kagg AS (
  SELECT cu.detail_category_full_code AS job,
         count(*) AS n_ksa,
         count(DISTINCT ki.description) AS n_ksa_unique,
         avg((ki.description ~ %(digital_a)s)::int)::float AS digital_a,
         avg((ki.description ~ %(digital_b)s)::int)::float AS digital_b
  FROM ncs.kta_items ki
  JOIN ncs.performance_criteria p ON p.id = ki.performance_criterion_id
  JOIN ncs.competency_unit_elements e ON e.code = p.competency_unit_element_code
  JOIN ncs.competency_units cu ON cu.code = e.competency_unit_code
  GROUP BY 1)
SELECT cp.detail_category_full_code AS job,
       cp.detail_category_name AS job_name,
       cp.major_category_name  AS major,
       cp.middle_category_name AS middle,
       u.n_units, u.mean_level, u.rate_l5plus, u.mean_rev_year, u.rate_rev2022,
       u.mean_version, u.n_units_nocode,
       pcagg.n_pc, pcagg.template_rate, pcagg.mean_pc_len,
       kagg.n_ksa, kagg.n_ksa_unique, kagg.digital_a, kagg.digital_b
FROM ncs.classification_paths cp
LEFT JOIN unit  u     ON u.job     = cp.detail_category_full_code
LEFT JOIN pcagg       ON pcagg.job = cp.detail_category_full_code
LEFT JOIN kagg        ON kagg.job  = cp.detail_category_full_code
ORDER BY cp.detail_category_full_code
"""

# H1 이탈 문장 유형화 · H3 별칭 사전 · H4 그래프 · S2 태도 코드북은 별도 질의로 원문을 가져온다.
# 집합 연산만 하므로 (직무, 유형, 문장) 중복은 DB에서 제거한다 — 246만 → 수십만 행.
KSA_SQL = """
SELECT DISTINCT cu.detail_category_full_code AS job, ki.kta_type_code AS ktype, ki.description
FROM ncs.kta_items ki
JOIN ncs.performance_criteria p ON p.id = ki.performance_criterion_id
JOIN ncs.competency_unit_elements e ON e.code = p.competency_unit_element_code
JOIN ncs.competency_units cu ON cu.code = e.competency_unit_code
"""

PC_DEVIANT_SQL = """
SELECT cu.detail_category_full_code AS job, p.description
FROM ncs.performance_criteria p
JOIN ncs.competency_unit_elements e ON e.code = p.competency_unit_element_code
JOIN ncs.competency_units cu ON cu.code = e.competency_unit_code
WHERE p.description !~ '수\\s*있다\\s*\\.?\\s*$'
"""


def tune_session(conn):
    """단발 분석 전용 세션 튜닝 — 큰 집계·정렬을 메모리에서 끝낸다(운영 설정 불변)."""
    with conn.cursor() as cur:
        cur.execute("SET work_mem = '1GB'")
        cur.execute("SET max_parallel_workers_per_gather = 4")


def load_metrics(conn):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(BASE_SQL, {"digital_a": "|".join(DIGITAL_A), "digital_b": "|".join(DIGITAL_B)})
        rows = cur.fetchall()
    for r in rows:
        r["unique_rate"] = (r["n_ksa_unique"] / r["n_ksa"]) if r["n_ksa"] else None
    return rows


def load_ksa(conn):
    """(job, ktype, description) 전량 — 반복 사용하므로 한 번만 읽는다."""
    out = []
    with conn.cursor(name="ksa_cursor") as cur:  # 서버 사이드 커서(2.46M 행)
        cur.itersize = 50000
        cur.execute(KSA_SQL)
        for job, ktype, desc in cur:
            out.append((job, ktype, desc))
    return out


# ─────────────────────────────────────────────────────────────── 가설 구현

def h1_template(metrics, ksa_unused, conn, ctx):
    np, stats = _lazy_stats()
    rows = [r for r in metrics if r["n_pc"]]
    rates = [r["template_rate"] for r in rows]
    n_pc_total = sum(r["n_pc"] for r in rows)
    n_dev_total = sum(round((1 - r["template_rate"]) * r["n_pc"]) for r in rows)

    groups = defaultdict(list)
    for r in rows:
        groups[r["major"]].append(r["template_rate"])
    main = kruskal_eps2(list(groups.values()))

    # 이탈 집중도: 이탈의 50%를 담는 최소 세분류 수
    dev_counts = sorted(((round((1 - r["template_rate"]) * r["n_pc"]), r["job_name"]) for r in rows),
                        key=lambda t: -t[0])
    cum, k50 = 0, 0
    for c, _ in dev_counts:
        cum += c
        k50 += 1
        if cum >= n_dev_total / 2:
            break

    # 이탈 문장 유형화
    with conn.cursor() as cur:
        cur.execute(PC_DEVIANT_SQL)
        deviants = cur.fetchall()
    types = Counter()
    for _, d in deviants:
        d = (d or "").strip()
        if not d:
            types["빈 문장"] += 1
        elif re.search(r"(한다|된다|이다)\s*\.?$", d):
            types["평서형 종결"] += 1
        elif re.search(r"[가-힣]기\s*\.?$|[가-힣]{2,}(?<!다)\s*\.?$", d) and not d.endswith("."):
            types["명사·명사형 종결"] += 1
        elif re.search(r"수\s*있(다|음)", d):
            types["템플릿 변형(문중)"] += 1
        else:
            types["기타"] += 1

    a, b = ctx["halves"]
    ha = kruskal_eps2(list({m: [r["template_rate"] for r in rows if r["major"] == m and r["job"] in a]
                            for m in groups}.values()))
    hb = kruskal_eps2(list({m: [r["template_rate"] for r in rows if r["major"] == m and r["job"] in b]
                            for m in groups}.values()))

    overall = n_pc_total and (n_pc_total - n_dev_total) / n_pc_total
    se = math.sqrt(overall * (1 - overall) / n_pc_total) if n_pc_total else 0
    # 주 검정은 주장에 맞춘다: "전체 준수율이 99% 이상인가"(단측 이항). 대분류 차이는 보조 검정.
    n_ok = n_pc_total - n_dev_total
    binom = stats.binomtest(n_ok, n_pc_total, 0.99, alternative="greater")
    return {
        "hypothesis": "H1", "label": "PEEK",
        "claim": "수행준거는 단일 통사 템플릿에 거의 완전히 순응하고 이탈은 국지적이다",
        "overall_template_rate": overall,
        "ci95": [overall - 1.96 * se, overall + 1.96 * se],
        "n_pc": n_pc_total, "n_deviant": n_dev_total,
        "median_job_rate": float(np.median(rates)),
        "min_job": min(rows, key=lambda r: r["template_rate"])["job_name"],
        "min_rate": min(rates),
        "deviation_concentration_k50": k50,
        "deviation_concentration_pct_of_jobs": k50 / len(rows),
        "deviant_types": dict(types.most_common()),
        "main_test": {"test": "binomial_vs_0.99", "stat": float(overall),
                      "p": float(binom.pvalue), "n": n_pc_total},
        "major_variation_test": main,
        "split_half": agreement(ha, hb, "eps2"),
        "prereg_expectation": "전체 ≥ 99% 그리고 이탈 절반이 세분류 5% 이내에 집중",
        "expectation_met": bool(overall >= 0.99 and k50 / len(rows) <= 0.05),
    }


def h2_duplication(metrics, ksa_unused, conn, ctx):
    """고유 비율의 대분류 효과 — 규모(KSA 수·능력단위 수) 통제 전/후."""
    np, stats = _lazy_stats()
    rows = [r for r in metrics if r["n_ksa"] and r["n_ksa"] >= 30 and r["n_units"]]
    raw_groups = defaultdict(list)
    for r in rows:
        raw_groups[r["major"]].append(r["unique_rate"])
    raw = kruskal_eps2(list(raw_groups.values()))

    # OLS: unique_rate ~ log(n_ksa) + log(n_units) → 잔차의 대분류 효과
    X = np.column_stack([np.ones(len(rows)),
                         np.log([r["n_ksa"] for r in rows]),
                         np.log([r["n_units"] for r in rows])])
    y = np.asarray([r["unique_rate"] for r in rows], dtype=float)
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if not finite.all():  # log(0)·결측이 남았다면 조용히 넘기지 않는다
        dropped = int((~finite).sum())
        rows = [r for r, ok in zip(rows, finite) if ok]
        X, y = X[finite], y[finite]
        print(f"[t0] H2 경고: 비유한 값 {dropped}건 제외", file=sys.stderr)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()

    res_groups = defaultdict(list)
    for r, e in zip(rows, resid):
        res_groups[r["major"]].append(float(e))
    adjusted = kruskal_eps2(list(res_groups.values()))

    a, b = ctx["halves"]
    idx_a = [i for i, r in enumerate(rows) if r["job"] in a]
    idx_b = [i for i, r in enumerate(rows) if r["job"] in b]

    def half_test(idx):
        g = defaultdict(list)
        for i in idx:
            g[rows[i]["major"]].append(float(resid[i]))
        return kruskal_eps2(list(g.values()))

    major_mean = sorted(((m, float(np.mean(v))) for m, v in res_groups.items()), key=lambda t: t[1])
    return {
        "hypothesis": "H2", "label": "PEEK",
        "claim": "KSA 중복률은 산업에 따라 다르며 규모를 통제해도 남는다",
        "n_jobs": len(rows),
        "unique_rate_overall": float(sum(r["n_ksa_unique"] for r in rows) / sum(r["n_ksa"] for r in rows)),
        "raw_test": raw,
        "covariate_model": {"formula": "unique_rate ~ 1 + log(n_ksa) + log(n_units)",
                            "beta": [float(x) for x in beta], "r2": float(r2)},
        "main_test": adjusted,
        "split_half": agreement(half_test(idx_a), half_test(idx_b), "eps2"),
        "residual_major_lowest3": major_mean[:3],
        "residual_major_highest3": major_mean[-3:],
        "prereg_expectation": "규모 통제 후에도 대분류 효과 유의",
        "expectation_met": bool(adjusted and adjusted["p"] < 0.05),
    }


def h3_fragmentation(metrics, ksa, conn, ctx):
    """규칙 정규화 키 단계의 파편화·회수율·유형 불일치. (임베딩 단계는 별도)"""
    np, stats = _lazy_stats()
    surf_jobs = defaultdict(set)      # 표면형 → 직무 집합
    surf_types = defaultdict(set)     # 표면형 → K/S/A
    key_surfs = defaultdict(set)      # 정규화 키 → 표면형들
    key_jobs = defaultdict(set)
    key_types = defaultdict(set)
    for job, ktype, desc in ksa:
        d = unicodedata.normalize("NFKC", (desc or "").strip())
        if not d:
            continue
        surf_jobs[d].add(job)
        surf_types[d].add(ktype)
        k = normalize_key(d)
        if not k:
            continue
        key_surfs[k].add(d)
        key_jobs[k].add(job)
        key_types[k].add(ktype)

    n_surf = len(surf_jobs)
    n_key = len(key_surfs)
    frag_index = n_surf / n_key if n_key else float("nan")
    merged = sum(len(v) - 1 for v in key_surfs.values())          # 규칙 키로 합쳐진 표면형 수
    recovery = merged / max(n_surf - n_key, 1) if n_surf > n_key else 0.0
    # 규칙 키로 회수 가능한 몫 = merged / (표면형 - 이상적 개념 수). 이상적 개념 수는 미지이므로
    # 보수적으로 '규칙 키가 만든 병합 / 규칙 키 기준 잉여'로 정의 → 1.0. 대신 아래 지표를 주 지표로 쓴다:
    pct_surf_merged = merged / n_surf if n_surf else 0.0          # 전체 표면형 중 병합된 비율

    # 빈도-파편화 상관: 키의 등장 직무 수 vs 표면형 수
    freqs = [len(key_jobs[k]) for k in key_surfs]
    frags = [len(key_surfs[k]) for k in key_surfs]
    freq_frag = spearman(freqs, frags)

    # 유형 불일치: 같은 키가 둘 이상의 K/S/A 유형으로
    mismatch_keys = [k for k, t in key_types.items() if len(t) > 1]
    mismatch_rate = len(mismatch_keys) / n_key if n_key else 0.0

    # 세분류별 파편화 지수(반분 검정용)
    job_surf = defaultdict(set)
    job_key = defaultdict(set)
    for surf, jobs in surf_jobs.items():
        k = normalize_key(surf)
        for j in jobs:
            job_surf[j].add(surf)
            if k:
                job_key[j].add(k)
    per_job = {j: len(job_surf[j]) / len(job_key[j]) for j in job_surf if job_key.get(j)}
    major_of = ctx["major_of"]
    groups = defaultdict(list)
    for j, v in per_job.items():
        groups[major_of.get(j, "?")].append(v)
    main = kruskal_eps2(list(groups.values()))
    a, b = ctx["halves"]
    ha = kruskal_eps2(list({m: [v for j, v in per_job.items() if major_of.get(j) == m and j in a]
                            for m in groups}.values()))
    hb = kruskal_eps2(list({m: [v for j, v in per_job.items() if major_of.get(j) == m and j in b]
                            for m in groups}.values()))

    top = sorted(key_surfs.items(), key=lambda kv: -len(kv[1]))[:15]
    ctx["h3_key_surfs"] = key_surfs
    ctx["h3_key_jobs"] = key_jobs
    ctx["h3_key_types"] = key_types
    ctx["h3_surf_jobs"] = surf_jobs
    return {
        "hypothesis": "H3", "label": "CLEAN(지수) / 사례만 관찰",
        "claim": "동일 개념이 다수 표기로 파편화되며 규칙 정규화가 상당 부분 회수한다",
        "n_surface_forms": n_surf, "n_rule_keys": n_key,
        "fragmentation_index": frag_index,
        "surface_forms_merged": merged,
        "pct_surface_merged": pct_surf_merged,
        "type_mismatch_keys": len(mismatch_keys), "type_mismatch_rate": mismatch_rate,
        "freq_fragmentation_corr": freq_frag,
        "main_test": main,
        "split_half": agreement(ha, hb, "eps2"),
        "top_fragmented": [{"key": k, "n_surface": len(v), "n_jobs": len(key_jobs[k]),
                            "examples": sorted(v)[:6]} for k, v in top],
        "prereg_expectation": "고빈도 개념일수록 파편화(rho>0) AND 규칙 회수율 50~70%",
        "expectation_met": bool(freq_frag and freq_frag["rho"] > 0
                                and 0.50 <= pct_surf_merged <= 0.70),
        "expectation_detail": {
            "freq_fragmentation_rho_positive": bool(freq_frag and freq_frag["rho"] > 0),
            "rule_recovery_in_50_70_band": bool(0.50 <= pct_surf_merged <= 0.70),
            "observed_rule_recovery": pct_surf_merged,
        },
        "note": "임베딩 2단계(bge-m3 cos≥0.90) 미포함 — 여기 수치는 규칙 키 단계의 하한이다. "
                "규칙 키는 공백·구두점·역할 접미사만 흡수하므로 '안전사항 준수'와 '안전수칙 준수'처럼 "
                "어간이 다른 동의 표기는 병합하지 못한다.",
    }


def h4_clusters(metrics, ksa, conn, ctx, permutations=1000, min_jobs_for_key=2, drop_top_pct=None):
    """공유 KSA 기반 직무 유사도 → Louvain 군집 → 공식 분류와 NMI/ARI."""
    import networkx as nx
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    np, stats = _lazy_stats()

    key_jobs = ctx.get("h3_key_jobs")
    if key_jobs is None:
        key_jobs = defaultdict(set)
        for job, _t, desc in ksa:
            k = normalize_key(desc or "")
            if k:
                key_jobs[k].add(job)

    n_jobs_total = len({j for js in key_jobs.values() for j in js})

    def build(cut_pct):
        """역인덱스 → Jaccard 그래프 → Louvain. cut_pct: 이 비율 이상의 직무에 등장하는 키 제외(IDF 변형)."""
        cutoff = int(n_jobs_total * cut_pct) if cut_pct else None
        inter, size, n_keys = Counter(), Counter(), 0
        for k, jobs_ in key_jobs.items():
            if len(jobs_) < min_jobs_for_key or (cutoff and len(jobs_) > cutoff):
                continue
            n_keys += 1
            js = sorted(jobs_)
            for j in js:
                size[j] += 1
            for i in range(len(js)):
                for m in range(i + 1, len(js)):
                    inter[(js[i], js[m])] += 1
        g = nx.Graph()
        g.add_nodes_from(size)
        for (u, v), c in inter.items():
            denom = size[u] + size[v] - c
            if denom > 0 and c > 0:
                g.add_edge(u, v, weight=c / denom)
        cs = nx.community.louvain_communities(g, weight="weight", seed=SEED, resolution=1.0)
        return g, cs, n_keys

    G, comms, n_keys_used = build(drop_top_pct)
    assign = {}
    for i, c in enumerate(comms):
        for j in c:
            assign[j] = i

    jobs = sorted(assign)
    major_of, middle_of = ctx["major_of"], ctx["middle_of"]
    labels = [assign[j] for j in jobs]
    majors = [major_of.get(j, "?") for j in jobs]
    middles = [middle_of.get(j, "?") for j in jobs]

    nmi_major = float(normalized_mutual_info_score(majors, labels))
    ari_major = float(adjusted_rand_score(majors, labels))
    nmi_middle = float(normalized_mutual_info_score(middles, labels))

    rng = random.Random(SEED)
    null = []
    perm = list(majors)
    for _ in range(permutations):
        rng.shuffle(perm)
        null.append(normalized_mutual_info_score(perm, labels))
    null = np.asarray(null)
    p_perm = float((null >= nmi_major).sum() + 1) / (permutations + 1)

    # 불일치 직무: 자기 군집의 최빈 대분류가 자기 대분류가 아닌 경우
    comm_major = {}
    for i, c in enumerate(comms):
        cnt = Counter(major_of.get(j, "?") for j in c)
        comm_major[i] = cnt.most_common(1)[0][0]
    mismatched = [{"job": ctx["name_of"].get(j, j), "major": major_of.get(j),
                   "cluster": assign[j], "cluster_major": comm_major[assign[j]]}
                  for j in jobs if comm_major[assign[j]] != major_of.get(j)]

    # 민감도: 범용 키(전체 직무의 10% 이상에 등장) 제외 — '안전사항 준수'류가 산업을 가로질러 잇는 효과 제거
    sens = None
    if drop_top_pct is None:
        g2, comms2, nk2 = build(0.10)
        a2 = {j: i for i, c in enumerate(comms2) for j in c}
        jobs2 = sorted(a2)
        sens = {
            "drop_top_pct": 0.10, "n_keys_used": nk2,
            "n_nodes": g2.number_of_nodes(), "n_edges": g2.number_of_edges(),
            "n_clusters": len(comms2),
            "nmi_major": float(normalized_mutual_info_score(
                [major_of.get(j, "?") for j in jobs2], [a2[j] for j in jobs2])),
            "ari_major": float(adjusted_rand_score(
                [major_of.get(j, "?") for j in jobs2], [a2[j] for j in jobs2])),
        }

    ctx["h4_assign"] = assign
    ctx["h4_comm_major"] = comm_major
    return {
        "hypothesis": "H4", "label": "CLEAN",
        "claim": "데이터 기반 직무 군집은 공식 분류와 부분적으로만 일치한다",
        "graph": {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
                  "n_keys_used": n_keys_used, "drop_top_pct": drop_top_pct},
        "sensitivity_idf": sens,
        "n_clusters": len(comms),
        "cluster_sizes_top10": sorted((len(c) for c in comms), reverse=True)[:10],
        "nmi_major": nmi_major, "ari_major": ari_major, "nmi_middle": nmi_middle,
        "permutation_null": {"mean": float(null.mean()), "sd": float(null.std()),
                             "p": p_perm, "n_perm": permutations},
        "main_test": {"test": "permutation_nmi", "stat": nmi_major, "p": p_perm},
        "split_half": {"agree": None, "reason": "전수 그래프 — 반분 대신 순열 영분포로 검정"},
        "n_mismatched_jobs": len(mismatched),
        "mismatch_rate": len(mismatched) / len(jobs) if jobs else None,
        "mismatch_examples": mismatched[:20],
        "prereg_expectation": "대분류 대비 NMI 0.3~0.6, 무작위보다 유의하게 높음",
        "expectation_met": bool(0.3 <= nmi_major <= 0.6 and p_perm < 0.05),
    }


def h5_revision(metrics, ksa_unused, conn, ctx):
    np, stats = _lazy_stats()
    rows = [r for r in metrics if r["mean_rev_year"] is not None]
    groups = defaultdict(list)
    for r in rows:
        groups[r["major"]].append(r["mean_rev_year"])
    main = kruskal_eps2(list(groups.values()))
    a, b = ctx["halves"]
    ha = kruskal_eps2(list({m: [r["mean_rev_year"] for r in rows if r["major"] == m and r["job"] in a]
                            for m in groups}.values()))
    hb = kruskal_eps2(list({m: [r["mean_rev_year"] for r in rows if r["major"] == m and r["job"] in b]
                            for m in groups}.values()))
    by_major = sorted(((m, float(np.mean(v)), len(v)) for m, v in groups.items()), key=lambda t: t[1])
    return {
        "hypothesis": "H5", "label": "PEEK",
        "claim": "표준의 코드 연도(개정 신선도 대리)는 산업 간 크게 불균등하다",
        "n_jobs": len(rows),
        "n_units_without_code": sum(r["n_units_nocode"] for r in metrics),
        "mean_rev_year_overall": float(np.mean([r["mean_rev_year"] for r in rows])),
        "gini_rev_year": gini([r["mean_rev_year"] for r in rows]),
        "main_test": main,
        "split_half": agreement(ha, hb, "eps2"),
        "oldest3": [{"major": m, "mean_year": y, "n_jobs": n} for m, y, n in by_major[:3]],
        "newest3": [{"major": m, "mean_year": y, "n_jobs": n} for m, y, n in by_major[-3:]],
        "prereg_expectation": "대분류 효과 eps^2 >= 0.3",
        "expectation_met": bool(main and main["eps2"] >= 0.3),
        "caveat": "코드 연도가 '내용 개정'인지 미확인(M-T0-3). 확인 전에는 '코드 연도'로만 해석한다.",
    }


def h6_digital(metrics, ksa_unused, conn, ctx):
    np, stats = _lazy_stats()
    rows = [r for r in metrics if r["n_ksa"] and r["digital_a"] is not None]
    groups = defaultdict(list)
    for r in rows:
        groups[r["major"]].append(r["digital_a"])
    dist = kruskal_eps2(list(groups.values()))

    # 대분류 내 (개정 연도 × 디지털 비율) — 대분류별 중심화 후 전체 Spearman
    sub = [r for r in rows if r["mean_rev_year"] is not None]
    by_major_year = defaultdict(list)
    by_major_dig = defaultdict(list)
    for r in sub:
        by_major_year[r["major"]].append(r["mean_rev_year"])
        by_major_dig[r["major"]].append(r["digital_a"])
    mean_y = {m: float(np.mean(v)) for m, v in by_major_year.items()}
    mean_d = {m: float(np.mean(v)) for m, v in by_major_dig.items()}
    cy = [r["mean_rev_year"] - mean_y[r["major"]] for r in sub]
    cd = [r["digital_a"] - mean_d[r["major"]] for r in sub]
    within = spearman(cy, cd)

    # 민감도: 어휘 B
    groups_b = defaultdict(list)
    for r in rows:
        groups_b[r["major"]].append(r["digital_b"])
    dist_b = kruskal_eps2(list(groups_b.values()))
    by_major_dig_b = defaultdict(list)
    for r in sub:
        by_major_dig_b[r["major"]].append(r["digital_b"])
    mean_d_b = {m: float(np.mean(v)) for m, v in by_major_dig_b.items()}
    cd_b = [r["digital_b"] - mean_d_b[r["major"]] for r in sub]
    within_b = spearman(cy, cd_b)

    a, b = ctx["halves"]
    sa = spearman([c for c, r in zip(cy, sub) if r["job"] in a], [c for c, r in zip(cd, sub) if r["job"] in a])
    sb = spearman([c for c, r in zip(cy, sub) if r["job"] in b], [c for c, r in zip(cd, sub) if r["job"] in b])

    by_major = sorted(((m, float(np.mean(v))) for m, v in groups.items()), key=lambda t: -t[1])
    # 괴리 위험 지수: z(디지털) - z(연도)  — 디지털 요구는 높은데 표준은 오래됨
    ys = np.asarray([mean_y[m] for m in mean_y]); ds = np.asarray([mean_d[m] for m in mean_d])
    risk = sorted(({"major": m,
                    "z_digital": float((mean_d[m] - ds.mean()) / (ds.std() or 1)),
                    "z_year": float((mean_y[m] - ys.mean()) / (ys.std() or 1)),
                    "risk": float((mean_d[m] - ds.mean()) / (ds.std() or 1) - (mean_y[m] - ys.mean()) / (ys.std() or 1))}
                   for m in mean_y), key=lambda d: -d["risk"])
    return {
        "hypothesis": "H6", "label": "PEEK(분포) / CLEAN(연도 교호)",
        "claim": "디지털 어휘 침투는 산업 간 불균등하고, 대분류 안에서는 최근 코드일수록 높다",
        "distribution_test": dist, "distribution_test_vocabB": dist_b,
        "main_test": within, "within_major_corr_vocabB": within_b,
        "split_half": agreement(sa, sb, "rho"),
        "top3": [{"major": m, "rate": v} for m, v in by_major[:3]],
        "bottom3": [{"major": m, "rate": v} for m, v in by_major[-3:]],
        "gap_risk_top5": risk[:5],
        "prereg_expectation": "대분류 내 rho > 0.2",
        "expectation_met": bool(within and within["rho"] > 0.2 and within["p"] < 0.05),
    }


def s1_levels(metrics, ksa_unused, conn, ctx):
    np, _ = _lazy_stats()
    rows = [r for r in metrics if r["mean_level"] is not None]
    groups = defaultdict(list)
    for r in rows:
        groups[r["major"]].append(r["rate_l5plus"])
    main = kruskal_eps2(list(groups.values()))
    a, b = ctx["halves"]
    ha = kruskal_eps2(list({m: [r["rate_l5plus"] for r in rows if r["major"] == m and r["job"] in a] for m in groups}.values()))
    hb = kruskal_eps2(list({m: [r["rate_l5plus"] for r in rows if r["major"] == m and r["job"] in b] for m in groups}.values()))
    by_major = sorted(((m, float(np.mean(v))) for m, v in groups.items()), key=lambda t: -t[1])
    return {
        "hypothesis": "S1", "label": "PEEK(분포) / CLEAN(외부 상관 — 미수행)",
        "claim": "수준 사다리는 산업별로 크게 다르다(경력 천장의 부호화)",
        "main_test": main, "split_half": agreement(ha, hb, "eps2"),
        "top3": [{"major": m, "rate_l5plus": v} for m, v in by_major[:3]],
        "bottom3": [{"major": m, "rate_l5plus": v} for m, v in by_major[-3:]],
        "external_correlation": None,
        "note": "외부 임금·학력 결합은 M-T0-2 확정 후 T0-b에서. 해석은 '반영'까지, '재생산'은 쓰지 않는다.",
        "prereg_expectation": "대분류 효과 큼",
        "expectation_met": bool(main and main["eps2"] >= 0.14),
    }


def s2_attitudes(metrics, ksa, conn, ctx):
    np, stats = _lazy_stats()
    # 태도(A) = kta_type_code '03'
    att = [(job, desc) for job, ktype, desc in ksa if ktype == "03" and desc]
    uniq_by_job = defaultdict(set)
    for job, desc in att:
        uniq_by_job[job].add(unicodedata.normalize("NFKC", desc.strip()))

    def code(text):
        for name, kws in ATTITUDE_CODEBOOK:
            if any(kw in text for kw in kws):
                return name
        return "기타"

    major_of = ctx["major_of"]
    table = defaultdict(Counter)
    overall = Counter()
    for job, descs in uniq_by_job.items():
        m = major_of.get(job, "?")
        for d in descs:
            c = code(d)
            table[m][c] += 1
            overall[c] += 1

    cats = [n for n, _ in ATTITUDE_CODEBOOK] + ["기타"]
    majors = sorted(table)
    obs = np.asarray([[table[m][c] for c in cats] for m in majors], dtype=float)
    obs = obs[:, obs.sum(axis=0) > 0]
    chi2, p, dof, _ = stats.chi2_contingency(obs)
    n = obs.sum()
    cramers_v = float(math.sqrt((chi2 / n) / (min(obs.shape) - 1)))

    top_by_major = {m: table[m].most_common(2) for m in majors}
    return {
        "hypothesis": "S2", "label": "PEEK(상위 어휘) / CLEAN(범주 분포)",
        "claim": "국가가 요구하는 태도는 '준수'가 지배적이며 산업별 구성이 다르다",
        "n_attitude_unique_per_job_sum": int(sum(len(v) for v in uniq_by_job.values())),
        "overall_distribution": {k: v / sum(overall.values()) for k, v in overall.most_common()},
        "main_test": {"test": "chi2", "chi2": float(chi2), "p": float(p), "dof": int(dof),
                      "cramers_v": cramers_v},
        "split_half": {"agree": None, "reason": "범주 분포 — 반분 대신 대분류별 상위 범주 일관성으로 보고"},
        "top_category_by_major": {m: [[c, cnt] for c, cnt in v] for m, v in top_by_major.items()},
        "codebook": {name: kws for name, kws in ATTITUDE_CODEBOOK},
        "prereg_expectation": "전체 1위 = 준수·안전; 서비스 계열은 대인 범주 비중 상승",
        "expectation_met": bool(overall.most_common(1)[0][0] == "준수·안전"),
        "note": "규칙 코딩. 무작위 300건 이중 코딩(kappa)은 논문 작성 시 연구진이 수행 — 미실시.",
    }


def s4_overlap(metrics, ksa_unused, conn, ctx):
    np, stats = _lazy_stats()
    by_major = defaultdict(lambda: {"l5": [], "year": [], "dig": []})
    for r in metrics:
        if r["rate_l5plus"] is None:
            continue
        by_major[r["major"]]["l5"].append(r["rate_l5plus"])
        if r["mean_rev_year"] is not None:
            by_major[r["major"]]["year"].append(r["mean_rev_year"])
        if r["digital_a"] is not None:
            by_major[r["major"]]["dig"].append(r["digital_a"])
    majors = sorted(by_major)
    l5 = np.asarray([np.mean(by_major[m]["l5"]) for m in majors])
    yr = np.asarray([np.mean(by_major[m]["year"]) for m in majors])
    dg = np.asarray([np.mean(by_major[m]["dig"]) for m in majors])

    ranks = np.vstack([stats.rankdata(l5), stats.rankdata(yr), stats.rankdata(dg)])
    k, n = ranks.shape
    R = ranks.sum(axis=0)
    S = ((R - R.mean()) ** 2).sum()
    W = float(12 * S / (k ** 2 * (n ** 3 - n)))
    chi2 = k * (n - 1) * W
    p = float(1 - stats.chi2.cdf(chi2, n - 1))

    z = lambda v: (v - v.mean()) / (v.std() or 1)
    comp = z(l5) + z(yr) + z(dg)
    order = np.argsort(comp)
    return {
        "hypothesis": "S4", "label": "CLEAN",
        "claim": "저수준·저신선·저디지털 산업이 서로 겹친다",
        "main_test": {"test": "kendall_w", "W": W, "chi2": float(chi2), "p": p, "n_majors": int(n)},
        "split_half": {"agree": None, "reason": "대분류 24 순위 일치 — 반분 부적합"},
        "bottom5_composite": [{"major": majors[i], "z_sum": float(comp[i]),
                               "rate_l5plus": float(l5[i]), "mean_year": float(yr[i]),
                               "digital": float(dg[i])} for i in order[:5]],
        "top5_composite": [{"major": majors[i], "z_sum": float(comp[i])} for i in order[-5:][::-1]],
        "prereg_expectation": "W >= 0.5",
        "expectation_met": bool(W >= 0.5 and p < 0.05),
        "note": "기술적 보고. 인과 주장 없음. S3(개정의 정치경제)는 외부 자료 필요 — 미수행.",
    }


HYPOTHESES = {
    "H1": h1_template, "H2": h2_duplication, "H3": h3_fragmentation, "H4": h4_clusters,
    "H5": h5_revision, "H6": h6_digital, "S1": s1_levels, "S2": s2_attitudes, "S4": s4_overlap,
}
NEEDS_KSA = {"H3", "H4", "S2"}


# ─────────────────────────────────────────────────────────────── 산출

def write_metrics_csv(path, metrics):
    cols = ["job", "job_name", "major", "middle", "n_units", "mean_level", "rate_l5plus",
            "mean_rev_year", "rate_rev2022", "mean_version", "n_units_nocode", "n_pc",
            "template_rate", "mean_pc_len", "n_ksa", "n_ksa_unique", "unique_rate",
            "digital_a", "digital_b"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(metrics)


def write_alias_csv(path, ctx, min_surface=2):
    key_surfs = ctx.get("h3_key_surfs")
    if not key_surfs:
        return 0
    key_jobs, key_types = ctx["h3_key_jobs"], ctx["h3_key_types"]
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["normalized_key", "n_surface_forms", "n_jobs", "kta_types",
                    "type_mismatch", "surface_forms"])
        for k, surfs in sorted(key_surfs.items(), key=lambda kv: -len(kv[1])):
            if len(surfs) < min_surface:
                continue
            w.writerow([k, len(surfs), len(key_jobs[k]), "|".join(sorted(key_types[k])),
                        int(len(key_types[k]) > 1), "|".join(sorted(surfs))])
            rows += 1
    return rows


def write_clusters_csv(path, ctx, metrics):
    assign = ctx.get("h4_assign")
    if not assign:
        return 0
    comm_major = ctx["h4_comm_major"]
    by_job = {r["job"]: r for r in metrics}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["job", "job_name", "major", "middle", "cluster", "cluster_major", "mismatch"])
        for j, c in sorted(assign.items()):
            r = by_job.get(j, {})
            w.writerow([j, r.get("job_name"), r.get("major"), r.get("middle"), c,
                        comm_major.get(c), int(comm_major.get(c) != r.get("major"))])
    return len(assign)


def fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_report(path, results, fdr, meta):
    L = []
    A = L.append
    A(f"# T0 확증 분석 결과\n")
    A(f"실행 {meta['run_at']} · 시드 {SEED} · 세분류 n={meta['n_jobs']} · "
      f"사전 등록 docs/t0-preregistration.html\n")
    A("모든 검정은 세분류 단위. 반분은 대분류 내 무작위 분할(시드 고정) 후 부호·유의성 일치 여부. "
      "FDR은 각 가설의 주 검정 p값에 Benjamini-Hochberg 5%.\n")
    A("\n## 요약표\n")
    A("| 가설 | 라벨 | 주 검정 | 통계량 | p | 효과크기 | 반분 일치 | FDR 통과 | 사전 기대 충족 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for hid, r in results.items():
        mt = r.get("main_test") or {}
        stat = mt.get("H", mt.get("rho", mt.get("chi2", mt.get("stat", mt.get("W")))))
        eff = mt.get("eps2", mt.get("cramers_v", mt.get("rho", mt.get("W"))))
        sh = r.get("split_half", {})
        sh_txt = "—" if sh.get("agree") is None else ("일치" if sh["agree"] else "불일치")
        f = fdr.get(hid, {})
        A(f"| {hid} | {r['label']} | {mt.get('test','—')} | {fmt(stat,2)} | "
          f"{fmt(mt.get('p'),4)} | {fmt(eff)} | {sh_txt} | "
          f"{'○' if f.get('reject') else '×' if f else '—'} | "
          f"{'○' if r.get('expectation_met') else '×'} |")
    A("\n## 가설별 상세\n")
    for hid, r in results.items():
        A(f"### {hid} — {r['claim']}\n")
        A(f"라벨 **{r['label']}** · 사전 기대: {r.get('prereg_expectation','—')}\n")
        for k, v in r.items():
            if k in {"hypothesis", "label", "claim", "prereg_expectation"}:
                continue
            if isinstance(v, (dict, list)):
                A(f"- `{k}`: ```{json.dumps(v, ensure_ascii=False)[:600]}```")
            else:
                A(f"- `{k}`: {fmt(v) if isinstance(v, float) else v}")
        A("")
    Path(path).write_text("\n".join(L), encoding="utf-8")


# ─────────────────────────────────────────────────────────────── main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("DATABASE_URL", "postgresql://ncs:ncs@localhost:55432/ncs"))
    ap.add_argument("--out", default="docs/t0")
    ap.add_argument("--only", nargs="+", choices=sorted(HYPOTHESES), help="일부 가설만 실행")
    ap.add_argument("--permutations", type=int, default=1000, help="H4 순열 횟수")
    ap.add_argument("--h4-drop-top-pct", type=float, default=None,
                    help="H4 IDF 변형: 전체 직무의 이 비율 이상에 등장하는 키 제외 (예: 0.1)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selected = args.only or list(HYPOTHESES)

    print(f"[t0] DB 접속 · 지표 적재 …", flush=True)
    with psycopg.connect(args.dsn) as conn:
        tune_session(conn)
        metrics = load_metrics(conn)
        print(f"[t0] 세분류 {len(metrics)}건 지표 계산 완료", flush=True)

        major_of = {r["job"]: r["major"] for r in metrics}
        middle_of = {r["job"]: r["middle"] for r in metrics}
        name_of = {r["job"]: r["job_name"] for r in metrics}
        halves = split_half([r["job"] for r in metrics], major_of)
        ctx = {"major_of": major_of, "middle_of": middle_of, "name_of": name_of, "halves": halves}
        print(f"[t0] 반분 A={len(halves[0])} B={len(halves[1])} (시드 {SEED})", flush=True)

        ksa = None
        if set(selected) & NEEDS_KSA:
            print("[t0] KSA 전량 적재 중 (약 246만 행) …", flush=True)
            ksa = load_ksa(conn)
            print(f"[t0] KSA {len(ksa):,}행 적재", flush=True)

        results = {}
        for hid in HYPOTHESES:
            if hid not in selected:
                continue
            print(f"[t0] {hid} 실행 …", flush=True)
            fn = HYPOTHESES[hid]
            if hid == "H4":
                results[hid] = fn(metrics, ksa, conn, ctx,
                                  permutations=args.permutations,
                                  drop_top_pct=args.h4_drop_top_pct)
            else:
                results[hid] = fn(metrics, ksa, conn, ctx)

    pvals = [(hid, (r.get("main_test") or {}).get("p")) for hid, r in results.items()]
    fdr = benjamini_hochberg(pvals)
    for hid, r in results.items():
        r["fdr"] = fdr.get(hid)

    meta = {"run_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "seed": SEED, "n_jobs": len(metrics), "fdr_q": FDR_Q,
            "hypotheses_run": selected, "permutations": args.permutations,
            "prereg": "docs/t0-preregistration.html",
            "digital_vocab_A": DIGITAL_A, "digital_vocab_B": DIGITAL_B}

    (out / "t0_results.json").write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_metrics_csv(out / "t0_metrics.csv", metrics)
    n_alias = write_alias_csv(out / "t0_alias_dictionary.csv", ctx)
    n_clu = write_clusters_csv(out / "t0_h4_clusters.csv", ctx, metrics)
    write_report(out / "t0_report.md", results, fdr, meta)

    print(f"\n[t0] 완료 → {out}/")
    print(f"  t0_metrics.csv ({len(metrics)}행) · t0_alias_dictionary.csv ({n_alias}행) · "
          f"t0_h4_clusters.csv ({n_clu}행)")
    print(f"  t0_results.json · t0_report.md")
    for hid, r in results.items():
        mt = r.get("main_test") or {}
        print(f"  {hid}: p={fmt(mt.get('p'),4)} "
              f"기대충족={'○' if r.get('expectation_met') else '×'} "
              f"반분={'—' if (r.get('split_half') or {}).get('agree') is None else ('일치' if r['split_half']['agree'] else '불일치')}")


if __name__ == "__main__":
    main()
