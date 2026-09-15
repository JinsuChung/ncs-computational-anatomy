# -*- coding: utf-8 -*-
"""S2 태도 코드북 민감도 — 우선순위 규칙이 결론을 만들어 낸 것인가 (M8 의 전단계).

규칙 코더는 ATTITUDE_CODEBOOK 순서대로 처음 맞는 범주에 배정한다. 순서가 곧 우선순위이므로,
같은 문장이 여러 범주의 키워드를 가지면 앞선 범주가 가져간다. 사전 등록 순서는 준수·안전이 **첫째**,
의지·노력이 **마지막**이다 — 즉 규칙은 보고된 결론(의지 > 준수)에 <b>불리한</b> 방향으로 편향돼 있다.
이 스크립트는 그 편향의 크기를 잰다.

  1. 다중 해당 비율 — 둘 이상 범주의 키워드를 가진 문장이 얼마나 되는가
  2. 순서 뒤집기 — 우선순위를 역순으로, 그리고 무작위 순열 200회로 돌렸을 때 1·2위가 바뀌는가
  3. 다중 해당 문장을 아예 '복수'로 빼고 단일 해당만 셌을 때의 분포

출력: docs/t0/t0_codebook_sensitivity.json
사용: .venv/bin/python scripts/t0_codebook_sensitivity.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t0_analysis import ATTITUDE_CODEBOOK, KSA_SQL, tune_session  # noqa: E402

SEED = 20260915
CATS = [n for n, _ in ATTITUDE_CODEBOOK]


def matches(text):
    """이 문장이 키워드를 가진 모든 범주."""
    return [name for name, kws in ATTITUDE_CODEBOOK if any(kw in text for kw in kws)]


def code_with_order(hits, order):
    for name in order:
        if name in hits:
            return name
    return "기타"


def main():
    dsn = os.environ.get("DATABASE_URL", "postgresql://ncs:ncs@localhost:55432/ncs")
    print("[cb] 태도 항목 적재 …", flush=True)
    with psycopg.connect(dsn) as conn:
        tune_session(conn)
        with conn.cursor(name="cb") as cur:
            cur.itersize = 50000
            cur.execute(KSA_SQL)
            uniq_by_job = defaultdict(set)
            for job, ktype, desc in cur:
                if ktype == "03" and desc:
                    uniq_by_job[job].add(unicodedata.normalize("NFKC", desc.strip()))

    texts = [d for v in uniq_by_job.values() for d in v]   # S2 와 같은 단위: 직무별 고유 태도 문장
    hits = [matches(t) for t in texts]
    n = len(texts)

    multi = sum(1 for h in hits if len(h) > 1)
    none_ = sum(1 for h in hits if not h)
    dist_pre = Counter(code_with_order(h, CATS) for h in hits)
    dist_rev = Counter(code_with_order(h, list(reversed(CATS))) for h in hits)
    single = Counter(h[0] for h in hits if len(h) == 1)

    rng = random.Random(SEED)
    rank1 = Counter(); volition_beats_compliance = 0
    for _ in range(200):
        order = CATS[:]; rng.shuffle(order)
        d = Counter(code_with_order(h, order) for h in hits)
        top = [c for c, _ in d.most_common() if c != "기타"][0]
        rank1[top] += 1
        volition_beats_compliance += int(d["의지·노력"] > d["준수·안전"])

    pct = lambda c, tot=n: {k: round(v / tot * 100, 1) for k, v in c.most_common()}
    out = {
        "n_attitude_sentences": n,
        "multi_category_rate": round(multi / n, 4),
        "no_category_rate": round(none_ / n, 4),
        "distribution_preregistered_order": pct(dist_pre),
        "distribution_reversed_order": pct(dist_rev),
        "distribution_single_match_only": pct(single, sum(single.values())),
        "n_single_match": sum(single.values()),
        "permutation_200": {"top_category_counts": dict(rank1),
                            "volition_over_compliance_rate": round(volition_beats_compliance / 200, 3)},
        "note": "사전 등록 우선순위는 준수·안전이 첫째, 의지·노력이 마지막이다. 규칙은 보고된 결론(의지 > 준수)에 불리한 방향으로 편향돼 있다.",
    }
    Path("docs/t0/t0_codebook_sensitivity.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
