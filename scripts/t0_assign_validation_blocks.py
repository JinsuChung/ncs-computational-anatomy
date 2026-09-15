"""판정 표본에 배정 블록을 부여한다 — 공통 앵커 + 고유 블록 설계.

왜 이 구조인가: 200쌍을 5명에게 겹침 없이 쪼개면 판정자 간 일치도를 측정할 수 없고,
일치도 없는 정밀도는 "다섯 사람이 제각각 다른 기준으로 찍은 것"이라는 비판에 답할 수 없다.
그래서 전원이 함께 보는 앵커 집합을 두고, 나머지를 균등 분배한다.

  앵커  N_ANCHOR 쌍  → 외부 판정자 전원이 판정 → Krippendorff α (명목 5범주, 다수 판정자)
  블록  나머지 ÷ 블록 수 → 각 1인 → 커버리지 확대
  연구자(AU) → 200쌍 전부 → 저자–외부 일치도 비교용(주 추정치가 아님)

배정은 층(코사인 대역 / 전이)별로 균형을 맞춘다. 한 판정자에게 특정 대역이 몰리면
그 사람의 관대·엄격 성향이 대역별 정밀도에 그대로 실리기 때문이다.

사용: .venv/bin/python scripts/t0_assign_validation_blocks.py --anchor 50 --blocks 5
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260909
SAMPLE = Path("docs/t0/t0_h3_validation_sample.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(SAMPLE))
    ap.add_argument("--anchor", type=int, default=50, help="전원 공통 앵커 문항 수")
    ap.add_argument("--blocks", type=int, default=5, help="고유 블록 수(= 외부 판정자 수)")
    args = ap.parse_args()

    p = Path(args.sample)
    d = json.loads(p.read_text(encoding="utf-8"))
    items = d["items"]
    rng = random.Random(SEED)

    # 층별로 모아서 앵커·블록을 비례 배정
    by_band = defaultdict(list)
    for it in items:
        by_band[it["_hidden"]["band"]].append(it)
    total = len(items)

    assigned = 0
    for band in sorted(by_band):
        group = by_band[band][:]
        rng.shuffle(group)
        n_anchor = round(args.anchor * len(group) / total)
        for it in group[:n_anchor]:
            it["block"] = "anchor"
        rest = group[n_anchor:]
        for i, it in enumerate(rest):
            it["block"] = (i % args.blocks) + 1   # 라운드로빈 → 블록 간 층 구성이 균형
        assigned += len(group)

    counts = Counter(it["block"] for it in items)
    per_block_bands = defaultdict(Counter)
    for it in items:
        per_block_bands[it["block"]][it["_hidden"]["band"]] += 1

    d["meta"]["assignment"] = {
        "design": "공통 앵커 + 고유 블록. 앵커는 외부 판정자 전원이 판정해 일치도(α)를 내고, "
                  "블록은 1인씩 맡아 커버리지를 채운다. 연구자 코드(AU-*)는 전체를 판정한다.",
        "seed": SEED,
        "n_anchor": counts["anchor"],
        "n_blocks": args.blocks,
        "per_rater_items": counts["anchor"] + max(counts[b] for b in range(1, args.blocks + 1)),
        "block_sizes": {str(k): v for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))},
        "band_balance": {str(k): dict(v) for k, v in sorted(per_block_bands.items(), key=lambda kv: str(kv[0]))},
    }
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[assign] 총 {assigned}쌍 · 앵커 {counts['anchor']} · 블록 {args.blocks}개")
    for b in ["anchor"] + list(range(1, args.blocks + 1)):
        print(f"  {str(b):>6}: {counts[b]:>3}쌍  {dict(per_block_bands[b])}")
    print(f"  → 외부 1인 부담 = 앵커 {counts['anchor']} + 블록 {counts[1]} = "
          f"{counts['anchor'] + counts[1]}쌍")


if __name__ == "__main__":
    main()
