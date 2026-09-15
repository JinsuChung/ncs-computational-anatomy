# -*- coding: utf-8 -*-
"""검증 회수율의 표집 불확실성 — 문항 부트스트랩을 정책별 검증 회수율까지 전파한다 (심사 지적 ③).

점추정은 t0_verified_recovery.py --r2 와 동일해야 한다 (직접 엣지 정밀도 = 1차 arm A cos≥.90 판정,
전이 정밀도 = 크기 구간별: 3–5·6–20 은 1차 arm B, 21–100·100+ 는 2차). 부트스트랩은 층마다 문항을 복원 추출하고
그 문항의 판정 전부를 가져와 층 정밀도를 다시 낸 뒤, 구간 가중식으로 정책별 검증 회수율을 재계산한다.
추가로 21+ 구간 정밀도를 0 / CI 하한 / CI 상한 / 1 로 고정한 경계값을 낸다 (심사 지적 ②).

출력: docs/t0/t0_h3_verified_recovery_ci.json · t0_h3_verified_recovery_ci.md
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("docs/t0")
SEED = 20260914
N_BOOT = 4000
BUCKETS = ["2", "3-5", "6-20", "21-100", "100+"]
R2_RATERS = {"EV-6", "EV-7", "EV-8", "EV-9"}
R1_RATERS = {"EV-1", "EV-2", "EV-3", "EV-4", "EV-5", "EV-3B", "EV-4B", "EV-5B"}  # 1차 판정자 8인 × 80문항.
# 제외가 아니라 허용 목록으로 둔다 — 라운드가 늘어날 때(EV-6~9 2차, EV-10~11 3차) 새 판정자가
# 조용히 1차 집계에 섞여 표본 설계(팔 A 30 + 앵커 50)를 깨뜨리는 것을 막기 위해서다.
POLICIES = [("size2_only", ["2"]), ("size<=5", ["2", "3-5"]), ("size<=20", ["2", "3-5", "6-20"]), ("all", BUCKETS)]


def bucket(k):
    return "2" if k == 2 else "3-5" if k <= 5 else "6-20" if k <= 20 else "21-100" if k <= 100 else "100+"


def main():
    V = json.loads((OUT / "t0_h3_verified_recovery_r2.json").read_text(encoding="utf-8"))
    C = json.loads((OUT / "t0_h3_cluster_sizes_090.json").read_text(encoding="utf-8"))["by_bucket"]
    rows = json.loads((OUT / "t0_judgments.json").read_text(encoding="utf-8"))["rows"]
    S1 = {it["id"]: it for it in json.loads((OUT / "t0_h3_validation_sample.json").read_text(encoding="utf-8"))["items"]}
    S2 = {it["id"]: it for it in json.loads((OUT / "t0_h3_validation_sample_r2.json").read_text(encoding="utf-8"))["items"]}
    n_surface, rule = V["n_surface_forms"], V["stage1_rule_recovery"]

    # 층별 문항 → 판정 목록
    strata = defaultdict(lambda: defaultdict(list))   # stratum → pair_id → verdicts
    for r in rows:
        rt = r["rater"].upper()
        if not rt.startswith("EV"):
            continue
        if rt not in R1_RATERS and rt not in R2_RATERS:
            continue   # 3차(EV-10·11)는 앵커만 판정했으므로 대역별 정밀도에 넣으면 가중이 깨진다
        if rt in R2_RATERS:
            it = S2.get(r["pair_id"])
            if it and it["block"] == "r2":
                strata[it["_hidden"]["stratum"]][r["pair_id"]].append(r["verdict"])
        else:
            it = S1.get(r["pair_id"])
            if not it:
                continue
            h = it["_hidden"]
            if h["arm"] == "A" and h["cosine"] >= 0.90:
                strata["direct"][r["pair_id"]].append(r["verdict"])
            elif h["arm"] == "B":
                b = bucket(h["cluster_size"])
                if b in ("3-5", "6-20"):
                    strata[b][r["pair_id"]].append(r["verdict"])
    names = ["direct", "3-5", "6-20", "21-100", "100+"]
    items = {s: sorted(strata[s]) for s in names}
    print({s: (len(items[s]), sum(len(strata[s][p]) for p in items[s])) for s in names})

    def prec(s, sel, fn):
        vs = [v for p in sel for v in strata[s][p]]
        return sum(fn(v) for v in vs) / len(vs)

    strict = lambda v: v == "same"
    lenient = lambda v: v in ("same", "hierarchy")

    def recovery(p_direct, p_trans, allowed):
        tot = 0.0
        for b in allowed:
            d = 1.0 if b == "2" else C[b]["direct_edge_share"]
            pt = p_trans.get(b, 0.0)
            tot += C[b]["keys_absorbed"] * (d * p_direct + (1 - d) * pt)
        return rule + tot / n_surface

    def all_policies(sel_fn, mode_fn):
        pd = prec("direct", sel_fn("direct"), mode_fn)
        pt = {b: prec(b, sel_fn(b), mode_fn) for b in ("3-5", "6-20", "21-100", "100+")}
        return {name: recovery(pd, pt, allowed) for name, allowed in POLICIES}, pd, pt

    point = {}
    for mode, fn in (("strict", strict), ("lenient", lenient)):
        pol, pd, pt = all_policies(lambda s: items[s], fn)
        point[mode] = {"policies": pol, "p_direct": pd, "p_transitive": pt}
    print("point strict:", {k: round(v * 100, 1) for k, v in point["strict"]["policies"].items()})

    rng = random.Random(SEED)
    boots = {m: {name: [] for name, _ in POLICIES} for m in ("strict", "lenient")}
    boot_prec = {m: {s: [] for s in names} for m in ("strict", "lenient")}
    for _ in range(N_BOOT):
        sel = {s: [items[s][rng.randrange(len(items[s]))] for _ in items[s]] for s in names}
        for mode, fn in (("strict", strict), ("lenient", lenient)):
            pol, pd, pt = all_policies(lambda s: sel[s], fn)
            for name, v in pol.items():
                boots[mode][name].append(v)
            boot_prec[mode]["direct"].append(pd)
            for b, v in pt.items():
                boot_prec[mode][b].append(v)
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    out = {"n_boot": N_BOOT, "seed": SEED, "n_items": {s: len(items[s]) for s in names},
           "n_judgments": {s: sum(len(strata[s][p]) for p in items[s]) for s in names},
           "policies": {m: {name: {"point": point[m]["policies"][name], "ci95": ci(boots[m][name])} for name, _ in POLICIES} for m in ("strict", "lenient")},
           "stratum_precision": {m: {s: {"point": (point[m]["p_direct"] if s == "direct" else point[m]["p_transitive"][s]), "ci95": ci(boot_prec[m][s])} for s in names} for m in ("strict", "lenient")}}
    # 경계: 21+ 전이 정밀도 고정 (엄격)
    pd_s, pt_s = point["strict"]["p_direct"], dict(point["strict"]["p_transitive"])
    lo21, hi21 = out["stratum_precision"]["strict"]["21-100"]["ci95"], out["stratum_precision"]["strict"]["100+"]["ci95"]
    bounds = {}
    for label, p in (("0", 0.0), ("ci_low", None), ("point", None), ("ci_high", None), ("1", 1.0)):
        pt = dict(pt_s)
        if label == "0" or label == "1":
            pt["21-100"] = pt["100+"] = p
        elif label == "ci_low":
            pt["21-100"], pt["100+"] = lo21[0], hi21[0]
        elif label == "ci_high":
            pt["21-100"], pt["100+"] = lo21[1], hi21[1]
        bounds[label] = recovery(pd_s, pt, BUCKETS)
    out["bounds_strict_all_by_21plus_precision"] = bounds
    (OUT / "t0_h3_verified_recovery_ci.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    pct = lambda v: f"{v*100:.1f}%"
    L = ["# 검증 회수율 부트스트랩 CI (문항 단위, %d회)" % N_BOOT, "",
         "| 정책 | 엄격 [95% CI] | 관대 [95% CI] |", "|---|---|---|"]
    for name, _ in POLICIES:
        s, l = out["policies"]["strict"][name], out["policies"]["lenient"][name]
        L.append(f"| {name} | {pct(s['point'])} [{pct(s['ci95'][0])}, {pct(s['ci95'][1])}] | {pct(l['point'])} [{pct(l['ci95'][0])}, {pct(l['ci95'][1])}] |")
    L += ["", "층별 정밀도(엄격): " + " · ".join(f"{s} {pct(v['point'])} [{pct(v['ci95'][0])}, {pct(v['ci95'][1])}] (문항 {out['n_items'][s]})" for s, v in out["stratum_precision"]["strict"].items()),
          "", "21+ 전이 정밀도 고정 시 전체 정책 검증 회수(엄격): " + " · ".join(f"{k}={pct(v)}" for k, v in bounds.items())]
    (OUT / "t0_h3_verified_recovery_ci.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
