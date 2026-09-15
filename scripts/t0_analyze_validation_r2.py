"""T0 · H3 병합 정밀도 검증 2차(R2) 집계 — 크기 21+ 군집 전이 정밀도 + 새 판정자 엄격도 보정.

입력: docs/t0/t0_judgments.json (서버 전량 회수본; 1차·2차 행이 섞여 있어도 판정자 코드로 가른다)
      docs/t0/t0_h3_validation_sample_r2.json · t0_h3_validation_sample.json · t0_validation_results.json(1차)
출력: docs/t0/t0_validation_r2_results.json · t0_validation_r2_report.md

집계 항목
  1. 층별(21–100 / 100+) 전이 정밀도 — 합산 판정률(엄격/관대), 문항 다수결, 문항 부트스트랩 95% CI
  2. 보정 — 앵커 10문항에서 새 4인 vs 1차 8인의 판정 분포·엄격률; 새 4인의 α(앵커 10 + 3인 교차 60, 결측 허용)
  3. 1차 합산값(21+ 19.0%)을 층별 2차 값으로 바꿨을 때 검증 회수율(t0_verified_recovery.json 재계산)
사용: .venv/bin/python scripts/t0_analyze_validation_r2.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t0_analyze_validation import alpha_bootstrap, alpha_matrix, krippendorff_alpha, lenient, strict  # noqa: E402

OUT = Path("docs/t0")
SEED = 20260911
R2_RATERS = {"EV-6", "EV-7", "EV-8", "EV-9"}
R1_RATERS = {"EV-1", "EV-2", "EV-3", "EV-4", "EV-5", "EV-3B", "EV-4B", "EV-5B"}  # 1차 8인 — 허용 목록(3차 유입 차단)
VERDICTS = ["same", "hierarchy", "related", "unrelated", "unsure"]


def load_rows():
    j = json.loads((OUT / "t0_judgments.json").read_text(encoding="utf-8"))
    rows = j["rows"] if isinstance(j, dict) else j
    return [{**r, "rater": r["rater"].upper()} for r in rows]


def prec_block(judgs, n_boot=2000):
    items = defaultdict(list)
    for j in judgs:
        items[j["pair_id"]].append(j["verdict"])
    vs = [v for j in judgs for v in [j["verdict"]]]
    rng = random.Random(SEED)
    ids = sorted(items)
    def pooled(sel, fn):
        x = [fn(v) for p in sel for v in items[p]]
        return float(np.mean(x)) if x else None
    boots = {"strict": [], "lenient": []}
    for _ in range(n_boot):
        sel = [ids[rng.randrange(len(ids))] for _ in ids]
        boots["strict"].append(pooled(sel, strict)); boots["lenient"].append(pooled(sel, lenient))
    maj = Counter()
    for p, vv in items.items():
        c = Counter(vv).most_common()
        top = c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else "tie"
        maj[top] += 1
    return {"n_items": len(ids), "n_judgments": len(vs),
            "strict": pooled(ids, strict), "lenient": pooled(ids, lenient),
            "strict_ci95": [float(np.percentile(boots["strict"], 2.5)), float(np.percentile(boots["strict"], 97.5))],
            "lenient_ci95": [float(np.percentile(boots["lenient"], 2.5)), float(np.percentile(boots["lenient"], 97.5))],
            "item_majority": {k: maj[k] / len(ids) for k in VERDICTS + ["tie"]},
            "verdict_dist": {k: vs.count(k) / len(vs) for k in VERDICTS}}


def main():
    rows = load_rows()
    s2 = json.loads((OUT / "t0_h3_validation_sample_r2.json").read_text(encoding="utf-8"))
    items2 = {it["id"]: it for it in s2["items"]}
    r2 = [r for r in rows if r["rater"] in R2_RATERS and r["pair_id"] in items2]
    if not r2:
        raise SystemExit("2차 판정 행이 없다 — 회수본을 docs/t0/t0_judgments.json 에 덮어쓴 뒤 다시 실행")
    R = {"n_rows_r2": len(r2), "per_rater": dict(Counter(r["rater"] for r in r2)),
         "expected_per_rater": s2["meta"]["per_rater_items"]}

    # 1. 층별 전이 정밀도
    R["transitive_by_stratum"] = {}
    for st in ("21-100", "100+"):
        js = [r for r in r2 if items2[r["pair_id"]]["_hidden"].get("stratum") == st]
        R["transitive_by_stratum"][st] = prec_block(js)
    R["transitive_21plus_pooled"] = prec_block([r for r in r2 if items2[r["pair_id"]]["_hidden"].get("stratum") in ("21-100", "100+")])

    # 2. 보정 — 앵커 10문항: 새 4인 vs 1차 8인(같은 문항)
    anchor_ids = {pid for pid, it in items2.items() if it["_hidden"].get("calibration")}
    new_a = [r for r in r2 if r["pair_id"] in anchor_ids]
    old_a = [r for r in rows if r["pair_id"] in anchor_ids and r["rater"] in R1_RATERS]
    def dist(js):
        vs = [j["verdict"] for j in js]
        return {"n": len(vs), "strict": float(np.mean([strict(v) for v in vs])) if vs else None,
                "lenient": float(np.mean([lenient(v) for v in vs])) if vs else None,
                "verdict_dist": {k: vs.count(k) / len(vs) for k in VERDICTS} if vs else None}
    # 문항 단위 다수결 일치: 새 4인 다수결 vs 1차 8인 다수결
    def majority(js):
        m = {}
        by = defaultdict(list)
        for j in js: by[j["pair_id"]].append(j["verdict"])
        for p, vv in by.items():
            c = Counter(vv).most_common(); m[p] = c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else "tie"
        return m
    mn, mo = majority(new_a), majority(old_a)
    common = [p for p in anchor_ids if p in mn and p in mo and mn[p] != "tie" and mo[p] != "tie"]
    R["calibration"] = {"n_anchor_items": len(anchor_ids), "new_raters": dist(new_a), "r1_raters_same_items": dist(old_a),
                        "majority_agreement_5way": float(np.mean([mn[p] == mo[p] for p in common])) if common else None,
                        "majority_agreement_merge": float(np.mean([strict(mn[p]) == strict(mo[p]) for p in common])) if common else None,
                        "n_common_decided": len(common)}
    # 새 4인 α — 앵커(전원) + 교차 60(3인, 결측 허용)
    raters = sorted(R2_RATERS)
    all_ids = sorted(items2)
    judg_for_alpha = [{"rater": r["rater"], "pair_id": r["pair_id"], "verdict": r["verdict"]} for r in r2]
    try:
        M5 = alpha_matrix(judg_for_alpha, raters, all_ids, lambda v: VERDICTS.index(v))
        Mb = alpha_matrix(judg_for_alpha, raters, all_ids, lambda v: int(strict(v)))
        R["alpha_new_raters"] = {"five_way": krippendorff_alpha(M5), "five_way_ci": alpha_bootstrap(M5),
                                 "merge": krippendorff_alpha(Mb), "merge_ci": alpha_bootstrap(Mb), "n_items": len(all_ids)}
    except Exception as e:  # 결측 구조가 함수 가정과 다르면 건너뛴다 — 정밀도 집계가 우선
        R["alpha_new_raters"] = {"error": str(e)}

    # 3. 검증 회수율 재계산 — 21–100 / 100+ 를 2차 층별 값으로 교체
    vr = json.loads((OUT / "t0_h3_verified_recovery.json").read_text(encoding="utf-8"))
    pb, n = vr["per_bucket"], vr["n_surface_forms"]
    p_direct = vr["p_direct_edge_cos_ge_090"]
    upd = {}
    for mode in ("strict", "lenient"):
        total = 0.0
        for b, v in pb.items():
            if b in ("21-100", "100+") and R["transitive_by_stratum"][b][mode] is not None:
                pt = R["transitive_by_stratum"][b][mode]
            elif b in vr["p_transitive_by_cluster_size"]:
                pt = vr["p_transitive_by_cluster_size"][b][mode]
            else:
                pt = p_direct[mode]
            share = v["direct_edge_share"]
            total += v["keys_absorbed"] * (share * p_direct[mode] + (1 - share) * pt)
        upd[mode] = {"embedding_verified_pp": total / n, "combined_verified": vr["stage1_rule_recovery"] + total / n,
                     "r1_combined_verified": vr["overall"][f"combined_verified_{mode}"]}
    R["verified_recovery_updated"] = upd

    (OUT / "t0_validation_r2_results.json").write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    pct = lambda v: "—" if v is None else f"{v*100:.1f}%"
    L = ["# H3 병합 정밀도 검증 2차(R2) 집계", "", f"판정 {R['n_rows_r2']}건 · 판정자별 {R['per_rater']} (기대 {R['expected_per_rater']})", "",
         "## 크기 21+ 군집 전이 정밀도", "", "| 층 | 문항 | 판정 | 엄격 [95% CI] | 관대 [95% CI] | 다수결 같음 |", "|---|---|---|---|---|---|"]
    for st, v in list(R["transitive_by_stratum"].items()) + [("21+ 합산", R["transitive_21plus_pooled"])]:
        L.append(f"| {st} | {v['n_items']} | {v['n_judgments']} | {pct(v['strict'])} [{pct(v['strict_ci95'][0])}, {pct(v['strict_ci95'][1])}] | "
                 f"{pct(v['lenient'])} [{pct(v['lenient_ci95'][0])}, {pct(v['lenient_ci95'][1])}] | {pct(v['item_majority']['same'])} |")
    c = R["calibration"]
    L += ["", "## 보정 — 앵커 10문항", "", f"새 4인 엄격 {pct(c['new_raters']['strict'])} / 관대 {pct(c['new_raters']['lenient'])} · 1차 8인 같은 문항 엄격 {pct(c['r1_raters_same_items']['strict'])} / 관대 {pct(c['r1_raters_same_items']['lenient'])} · "
          f"다수결 일치 5범주 {pct(c['majority_agreement_5way'])} · 병합 여부 {pct(c['majority_agreement_merge'])} (문항 {c['n_common_decided']})"]
    a = R["alpha_new_raters"]
    if "error" not in a:
        L.append(f"새 4인 α: 5범주 {a['five_way']:.3f} {a['five_way_ci']} · 병합 여부 {a['merge']:.3f} {a['merge_ci']}")
    L += ["", "## 검증 회수율 갱신", ""]
    for mode, v in upd.items():
        L.append(f"- {mode}: 1차 {pct(v['r1_combined_verified'])} → 2차 층별 대체 {pct(v['combined_verified'])}")
    (OUT / "t0_validation_r2_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
