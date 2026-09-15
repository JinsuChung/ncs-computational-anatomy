# -*- coding: utf-8 -*-
"""태도 어휘의 집중도 — 코드북 없이 '준수는 복제되고 의지는 새로 쓰인다'를 보이는 분석.

S2 의 범주 배정(6범주·우선순위)은 설계 선택이라 자의성 시비가 붙는다. 이 스크립트는 범주를 아예 쓰지 않고
같은 관찰을 드러낸다: **단일 키워드**별로 (a) 그 말을 담은 서로 다른 문장이 몇 개인지, (b) 그것이 직무×문장 항목
수로 얼마인지, (c) 상위 몇 문장이 그 항목의 몇 %를 가져가는지.

준수·안전 계열은 '소수 문장이 많은 직무에 복제'되므로 문장 수가 적고 집중도가 높다.
의지·노력 계열은 '직무마다 새로 쓰임'이므로 문장 수가 많고 집중도가 낮다.
집중도 차이는 배정 규칙이 아니라 코퍼스의 성질이므로 코드북과 무관하게 관찰된다.

출력: docs/t0/t0_attitude_concentration.json
사용: .venv/bin/python scripts/t0_attitude_concentration.py
"""
from __future__ import annotations

import json
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t0_analysis import KSA_SQL, tune_session  # noqa: E402

# 범주가 아니라 낱말 하나씩. 앞 넷은 준수 계열, 뒤 셋은 의지 계열로 읽히지만 배정은 하지 않는다.
KEYWORDS = ["준수", "안전", "규정", "지침", "의지", "노력", "적극"]
TOP_N = 10


def main():
    dsn = os.environ.get("DATABASE_URL", "postgresql://ncs:ncs@localhost:55432/ncs")
    print("[att] 태도 항목 적재 …", flush=True)
    with psycopg.connect(dsn) as conn:
        tune_session(conn)
        with conn.cursor(name="att") as cur:
            cur.itersize = 50000
            cur.execute(KSA_SQL)
            sent_jobs = defaultdict(set)      # 문장 → 그 문장을 쓰는 직무 집합
            for job, ktype, desc in cur:
                if ktype == "03" and desc:
                    sent_jobs[unicodedata.normalize("NFKC", desc.strip())].add(job)

    jobs_of = {s: len(v) for s, v in sent_jobs.items()}
    total_units = sum(jobs_of.values())       # = 직무별 고유 태도 문장 수의 합 (S2 와 같은 단위)

    by_kw = {}
    for kw in KEYWORDS:
        sents = [(s, jobs_of[s]) for s in jobs_of if kw in s]
        sents.sort(key=lambda x: -x[1])
        units = sum(c for _, c in sents)
        top5 = sum(c for _, c in sents[:5])
        by_kw[kw] = {
            "n_distinct_sentences": len(sents),
            "n_units": units,
            "share_of_all_attitude_units": round(units / total_units, 4),
            "units_per_sentence": round(units / len(sents), 2) if sents else None,
            "top5_concentration": round(top5 / units, 4) if units else None,
            "top_sentence": sents[0][0] if sents else None,
            "top_sentence_jobs": sents[0][1] if sents else None,
            "examples_top3": [{"text": s, "jobs": c} for s, c in sents[:3]],
        }
        print(f"  {kw}: 문장 {len(sents):,} · 항목 {units:,} · 문장당 {by_kw[kw]['units_per_sentence']} · 상위5 집중 {by_kw[kw]['top5_concentration']:.1%}", flush=True)

    top_breadth = sorted(jobs_of.items(), key=lambda kv: -kv[1])[:TOP_N]
    out = {
        "n_distinct_attitude_sentences": len(jobs_of),
        "n_attitude_units": total_units,
        "by_keyword": by_kw,
        "top_sentences_by_breadth": [{"text": s, "jobs": c} for s, c in top_breadth],
        "note": "단위는 S2 와 같다 — 직무별 고유 태도 문장(직무 × 문장). 범주 배정을 쓰지 않는다.",
    }
    Path("docs/t0/t0_attitude_concentration.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[att] 서로 다른 태도 문장 {len(jobs_of):,} · 항목 {total_units:,}")
    print("[att] 직무 폭 상위:")
    for s, c in top_breadth[:6]:
        print(f"   {c:>4}개 직무  {s}")


if __name__ == "__main__":
    main()
