# -*- coding: utf-8 -*-
"""T0 · H3 병합 정밀도 검증 3차(R3) — 전문가 천장을 2인에서 4인으로.

1차의 "전문가 천장 α = .569"는 엄밀히는 교육공학 박사 *한 쌍*이 서로 맞은 정도다. α 의 분산은 판정자와 판정자 사이에서
나오므로, 같은 사람이 더 판정해도 그 값은 한 쌍의 값으로 남는다. 따라서 **새 박사 2인**이 **1차와 똑같은 앵커 50문항**을
판정하게 해 쌍을 6개로 늘린다 — 그래야 "그 쌍이 전형적이었는가"를 물을 수 있다.

설계 제약
  · 문항은 1차 앵커 50개 그대로 (α 는 동일 문항 위에서만 비교된다)
  · 조건 동일 — 같은 UI, 코사인·군집 크기·팔 비공개, 1차 결과 비공지(앵커링 방지)
  · 판정자 EV-10 · EV-11 (교육공학 박사 2인, 화면 표시명은 익명)

출력: docs/t0/t0_h3_validation_sample_r3.json (비추적)
사용: .venv/bin/python scripts/t0_build_validation_sample_r3.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

RATERS = ["EV-10", "EV-11"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", default="docs/t0/t0_h3_validation_sample.json")
    ap.add_argument("--out", default="docs/t0/t0_h3_validation_sample_r3.json")
    args = ap.parse_args()

    r1 = json.loads(Path(args.r1).read_text(encoding="utf-8"))
    anchors = [it for it in r1["items"] if it.get("block") == "anchor"]
    assert len(anchors) == 50, len(anchors)

    items = [{"id": it["id"], "left": it["left"], "right": it["right"],
              "_hidden": {**it["_hidden"], "round": 3, "source": "r1-anchor"},
              "block": "anchor", "raters": list(RATERS)} for it in anchors]
    # 1차와 같은 순서로 보이면 비교가 쉬워지므로 순서만 새로 섞는다(문항 집합은 동일)
    import random
    random.Random(20260915).shuffle(items)

    payload = {"meta": {"built_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                        "round": 3, "seed": 20260915, "n_items": len(items), "raters": RATERS,
                        "per_rater_items": {r: len(items) for r in RATERS},
                        "design": "1차 앵커 50문항을 새 교육공학 박사 2인이 동일 조건으로 판정 — 전문가 α 를 2인(쌍 1개)에서 4인(쌍 6개)으로",
                        "bands": dict(Counter(it["_hidden"]["band"] for it in items)),
                        "anchor_hidden": ["arm", "cosine", "band", "cluster_size", "round", "source"],
                        "source": "docs/t0/t0_h3_validation_sample.json (block=anchor)"},
               "items": items}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[r3] 완료 → {args.out} · 문항 {len(items)} · 판정자 {RATERS} · 대역 {payload['meta']['bands']}")


if __name__ == "__main__":
    main()
