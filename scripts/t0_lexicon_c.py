# -*- coding: utf-8 -*-
"""디지털 어휘 강건성 — 세 번째 어휘 C (A 에서 다의어 '프로그램'·'전산' 제거) (심사 지적 ⑦).

B 는 A 의 상위집합이라 A·B 일치는 검사가 아니다. C 는 A 의 부분집합이므로 결론(대분류 분포 ε², 대분류 내 개정 연도와의
교호 ρ)이 다의어에 의존했는지를 실제로 검사한다. 세분류별 비율은 DB 에서 다시 세고, 개정 연도·대분류는 t0_metrics.csv.
출력: docs/t0/t0_lexicon_c.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t0_analysis import kruskal_eps2, spearman  # noqa: E402

LEX_C = ["디지털", "데이터", "인공지능", "AI", "소프트웨어", "자동화"]
SQL = """
SELECT cu.detail_category_full_code AS job,
       count(*) AS n, count(*) FILTER (WHERE ki.description ~* %(pat)s) AS hit
FROM ncs.kta_items ki
JOIN ncs.performance_criteria p ON p.id = ki.performance_criterion_id
JOIN ncs.competency_unit_elements e ON e.code = p.competency_unit_element_code
JOIN ncs.competency_units cu ON cu.code = e.competency_unit_code
GROUP BY cu.detail_category_full_code
"""


def main():
    dsn = os.environ.get("DATABASE_URL", "postgresql://ncs:ncs@localhost:55432/ncs")
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SET work_mem='1GB'")
        cur.execute(SQL, {"pat": "|".join(LEX_C)})
        share = {job: hit / n for job, n, hit in cur.fetchall() if n}
    rows = list(csv.DictReader(open("docs/t0/t0_metrics.csv", encoding="utf-8")))
    major_of = {r["job"]: r["major"] for r in rows}
    year = {r["job"]: float(r["mean_rev_year"]) for r in rows if r["mean_rev_year"]}
    dig_a = {r["job"]: float(r["digital_a"]) for r in rows if r["digital_a"]}
    jobs = [j for j in share if j in major_of and j in year]
    groups = defaultdict(list)
    for j in jobs:
        groups[major_of[j]].append(share[j])
    dist = kruskal_eps2(list(groups.values()))
    # 대분류 내 교호: 대분류 평균 중심화 후 Spearman (H6 절차)
    mean_share = {m: sum(v) / len(v) for m, v in groups.items()}
    ygroups = defaultdict(list)
    for j in jobs:
        ygroups[major_of[j]].append(year[j])
    mean_year = {m: sum(v) / len(v) for m, v in ygroups.items()}
    xs = [share[j] - mean_share[major_of[j]] for j in jobs]
    ys = [year[j] - mean_year[major_of[j]] for j in jobs]
    rho = spearman(xs, ys)
    corr_ac = spearman([share[j] for j in jobs], [dig_a[j] for j in jobs])
    top = sorted(mean_share.items(), key=lambda kv: -kv[1])
    out = {"lexicon_c": LEX_C, "n_jobs": len(jobs), "distribution_eps2": dist, "within_major_rho_vs_year": rho,
           "spearman_c_vs_a": corr_ac, "major_top3": top[:3], "major_bottom3": top[-3:]}
    Path("docs/t0/t0_lexicon_c.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
