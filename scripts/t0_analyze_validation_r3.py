# -*- coding: utf-8 -*-
"""T0 · H3 검증 3차(R3) 집계 — 전문가 천장을 쌍 1개에서 쌍 6개로.

1차의 "전문가 천장 α = .569"는 교육공학 박사 한 쌍(EV-1·EV-2)이 서로 맞은 정도였다. 3차에서 새 박사 2인
(EV-10·EV-11, 교외)이 **같은 앵커 50문항**을 같은 조건으로 판정했으므로, 이제 4인 α 와 쌍별 α 6개를 낼 수 있다.

내는 것
  1. 4인 α (5범주 · 병합 이진) + 문항 부트스트랩 CI — 논문이 쓸 새 '천장'
  2. 쌍별 α 6개 — 원래 쌍(EV-1·EV-2)이 전형적이었는가
  3. 새 쌍만의 α — 1차 쌍과 독립적으로 비교
  4. 엄격도 — 4인 각자의 「같음」 비율, 그리고 신·구 쌍의 판정 분포
  5. 박사 4인의 대역별 정밀도 (문항이 앵커 50뿐이라 참고치)

출력: docs/t0/t0_validation_r3_results.json · t0_validation_r3_report.md
사용: .venv/bin/python scripts/t0_analyze_validation_r3.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t0_analyze_validation import alpha_bootstrap, alpha_matrix, krippendorff_alpha, lenient, strict  # noqa: E402

OUT = Path("docs/t0")
VERD = ["same", "hierarchy", "related", "unrelated", "unsure"]
OLD_PHD = ["EV-1", "EV-2"]
NEW_PHD = ["EV-10", "EV-11"]
ALL_PHD = OLD_PHD + NEW_PHD


def main():
    rows = json.loads((OUT / "t0_judgments.json").read_text(encoding="utf-8"))["rows"]
    s1 = json.loads((OUT / "t0_h3_validation_sample.json").read_text(encoding="utf-8"))["items"]
    items = {it["id"]: it for it in s1}
    anchors = sorted(p for p, it in items.items() if it.get("block") == "anchor")
    assert len(anchors) == 50

    judg = [{"rater": r["rater"].upper(), "pair_id": r["pair_id"], "verdict": r["verdict"]}
            for r in rows if r["rater"].upper() in ALL_PHD and r["pair_id"] in set(anchors)]
    per = Counter(j["rater"] for j in judg)
    assert all(per[r] == 50 for r in ALL_PHD), per

    def alphas(raters, label):
        m5 = alpha_matrix(judg, raters, anchors, lambda v: VERD.index(v))
        mb = alpha_matrix(judg, raters, anchors, lambda v: int(strict(v)))
        return {"label": label, "n_raters": len(raters), "raters": raters,
                "five_way": krippendorff_alpha(m5), "five_way_ci": alpha_bootstrap(m5),
                "merge": krippendorff_alpha(mb), "merge_ci": alpha_bootstrap(mb)}

    R = {"n_anchor_items": len(anchors), "judgments_per_rater": dict(per)}
    R["alpha"] = {
        "four_phd": alphas(ALL_PHD, "교육공학 박사 4인 (새 천장)"),
        "old_pair": alphas(OLD_PHD, "1차 박사 2인 (기존 천장)"),
        "new_pair": alphas(NEW_PHD, "3차 박사 2인 (교외)"),
    }
    R["alpha_pairwise"] = {}
    for a, b in combinations(ALL_PHD, 2):
        R["alpha_pairwise"][f"{a}×{b}"] = alphas([a, b], f"{a}×{b}")

    # 엄격도 — 각자의 판정 분포
    by_r = defaultdict(list)
    for j in judg:
        by_r[j["rater"]].append(j["verdict"])
    R["per_rater"] = {r: {"n": len(v), "strict_rate": float(np.mean([strict(x) for x in v])),
                          "lenient_rate": float(np.mean([lenient(x) for x in v])),
                          "dist": {k: v.count(k) / len(v) for k in VERD if v.count(k)}}
                      for r, v in sorted(by_r.items())}
    for label, rs in (("old_pair", OLD_PHD), ("new_pair", NEW_PHD), ("four_phd", ALL_PHD)):
        vs = [x for r in rs for x in by_r[r]]
        R.setdefault("group_rates", {})[label] = {
            "n": len(vs), "strict": float(np.mean([strict(x) for x in vs])),
            "lenient": float(np.mean([lenient(x) for x in vs])),
            "dist": {k: vs.count(k) / len(vs) for k in VERD if vs.count(k)}}

    # 박사 4인의 대역별 정밀도 (앵커 50 한정)
    band = defaultdict(list)
    for j in judg:
        band[items[j["pair_id"]]["_hidden"]["band"]].append(j["verdict"])
    R["phd4_precision_by_band"] = {b: {"n_judgments": len(v), "n_items": len(v) // 4,
                                       "strict": float(np.mean([strict(x) for x in v])),
                                       "lenient": float(np.mean([lenient(x) for x in v]))}
                                   for b, v in sorted(band.items())}

    # 문항별 4인 합의 정도 — 전원 일치 / 3:1 / 2:2
    agree = Counter()
    for p in anchors:
        vs = [j["verdict"] for j in judg if j["pair_id"] == p]
        c = Counter(vs).most_common()
        agree["unanimous" if c[0][1] == 4 else "3-1" if c[0][1] == 3 else "2-2 또는 분산"] += 1
    R["item_agreement_four"] = dict(agree)

    (OUT / "t0_validation_r3_results.json").write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")

    f = lambda v: f"{v:.3f}"
    pct = lambda v: f"{v*100:.1f}%"
    L = ["# H3 검증 3차(R3) — 전문가 천장 4인", "",
         f"앵커 {len(anchors)}문항 · 판정자 {', '.join(ALL_PHD)} (각 50건)", "",
         "## Krippendorff α", "", "| 집단 | n | 5범주 [95% CI] | 병합 여부 [95% CI] |", "|---|---|---|---|"]
    for k in ("four_phd", "old_pair", "new_pair"):
        a = R["alpha"][k]
        L.append(f"| {a['label']} | {a['n_raters']} | {f(a['five_way'])} [{f(a['five_way_ci'][0])}, {f(a['five_way_ci'][1])}] | {f(a['merge'])} [{f(a['merge_ci'][0])}, {f(a['merge_ci'][1])}] |")
    L += ["", "## 쌍별 α (6쌍) — 원래 쌍이 전형적이었나", "", "| 쌍 | 5범주 | 병합 여부 |", "|---|---|---|"]
    for k, a in R["alpha_pairwise"].items():
        L.append(f"| {k} | {f(a['five_way'])} | {f(a['merge'])} |")
    fw = [a["five_way"] for a in R["alpha_pairwise"].values()]
    mg = [a["merge"] for a in R["alpha_pairwise"].values()]
    L += ["", f"쌍별 5범주 α 범위 {f(min(fw))}–{f(max(fw))} (중앙 {f(float(np.median(fw)))}) · 병합 {f(min(mg))}–{f(max(mg))} (중앙 {f(float(np.median(mg)))})",
          "", "## 엄격도", "", "| 판정자/집단 | 「같음」 | 「같음+상하위」 |", "|---|---|---|"]
    for r, v in R["per_rater"].items():
        L.append(f"| {r} | {pct(v['strict_rate'])} | {pct(v['lenient_rate'])} |")
    for k, v in R["group_rates"].items():
        L.append(f"| **{k}** | {pct(v['strict'])} | {pct(v['lenient'])} |")
    L += ["", "## 박사 4인 대역별 정밀도 (앵커 50 한정)", "", "| 대역 | 문항 | 엄격 | 관대 |", "|---|---|---|---|"]
    for b, v in R["phd4_precision_by_band"].items():
        L.append(f"| {b} | {v['n_items']} | {pct(v['strict'])} | {pct(v['lenient'])} |")
    L += ["", "문항별 4인 합의: " + " · ".join(f"{k} {v}" for k, v in R["item_agreement_four"].items())]
    (OUT / "t0_validation_r3_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
