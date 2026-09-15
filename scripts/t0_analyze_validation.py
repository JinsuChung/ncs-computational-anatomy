"""T0 · H3 병합 정밀도 수동 검증 — 회수 판정의 집계.

입력: docs/t0/t0_judgments.json (서버 회수) + docs/t0/t0_h3_validation_sample.json (표본 메타)
출력: docs/t0/t0_validation_results.json · docs/t0/t0_validation_report.md

계획(docs/t0-validation-plan.md §6)대로 보고한다:
  1. 대역별 엄격/관대 정밀도 — 엄격 = 「같은 개념」, 관대 = + 「상하위 관계」
  2. 팔별 정밀도 — A 직접 엣지 vs B 전이 동반
  3. Krippendorff α — 전체 외부 / 박사 2인(천장) / 직원 / 학생, 5범주와 병합 여부(이진)
  4. 저자–외부 일치도
  5. 「애매함」 비율
  6. 도메인 경험별 분포(강건성 확인용)

추정 원칙:
  · 주 정밀도는 외부 판정자(EV-*)만으로 낸다. 저자(AU-*)는 비교용.
  · 판정 단위로 합산하되, 신뢰구간은 **문항 단위 부트스트랩**(문항을 재표집하고 그 문항의 판정을 모두 가져온다).
    같은 문항의 판정은 독립이 아니므로 판정 단위 이항 CI 는 과신이다.
  · 문항 단위 다수결도 병기한다. 2인 판정 문항의 동률은 보수적으로(같은 개념 아님) 처리하고 동률 수를 보고한다.
"""

from __future__ import annotations

R1_RATERS = {"EV-1", "EV-2", "EV-3", "EV-4", "EV-5", "EV-3B", "EV-4B", "EV-5B"}  # 1차 판정자 8인 × 80문항.
# 제외가 아니라 허용 목록으로 둔다 — 라운드가 늘어날 때(EV-6~9 2차, EV-10~11 3차) 새 판정자가
# 조용히 1차 집계에 섞여 표본 설계(팔 A 30 + 앵커 50)를 깨뜨리는 것을 막기 위해서다.

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED = 20260909
VERDICTS = ["same", "hierarchy", "related", "unrelated", "unsure"]
VIDX = {v: i for i, v in enumerate(VERDICTS)}
BANDS = ["0.85-0.90", "0.90-0.95", "0.95-1.01", "transitive"]

GROUPS = {
    "external": lambda r: r.startswith("EV-"),
    "phd": lambda r: r in ("EV-1", "EV-2"),
    "staff": lambda r: r in ("EV-3", "EV-4", "EV-5"),
    "student": lambda r: r in ("EV-3B", "EV-4B", "EV-5B"),
    "author": lambda r: r.startswith("AU-"),
}


def load(judg_path, sample_path):
    j = json.loads(Path(judg_path).read_text(encoding="utf-8"))
    rows = j["rows"] if isinstance(j, dict) else j
    s = json.loads(Path(sample_path).read_text(encoding="utf-8"))
    items = {it["id"]: it for it in s["items"]}
    out = []
    missing = 0
    for r in rows:
        rt = r["rater"].upper()
        if rt.startswith("EV-") and rt not in R1_RATERS:
            continue   # 2·3차 판정자 — 1차 집계에서 제외(앵커 문항을 중복 판정했다)
        it = items.get(r["pair_id"])
        if not it:
            missing += 1
            continue
        h = it["_hidden"]
        out.append({
            "rater": r["rater"].upper(), "pair_id": r["pair_id"], "verdict": r["verdict"],
            "memo": r.get("memo") or "", "elapsed_ms": r.get("elapsed_ms"),
            "block": str(it.get("block")), "band": h["band"], "arm": h["arm"],
            "cosine": h.get("cosine"), "cluster_size": h.get("cluster_size"),
            "experience": (r.get("domain_experience") or None),
            "left": it["left"]["surfaces"][0], "right": it["right"]["surfaces"][0],
            "left_kta": it["left"]["kta"], "right_kta": it["right"]["kta"],
        })
    return out, items, missing


# ───────────────────────────────────────────── 정밀도

def strict(v): return v == "same"
def lenient(v): return v in ("same", "hierarchy")


def pooled_rate(judgs, fn):
    if not judgs:
        return None
    return sum(fn(j["verdict"]) for j in judgs) / len(judgs)


def item_bootstrap_ci(judgs, fn, n_boot=2000, seed=SEED):
    """문항 단위 부트스트랩 — 문항을 재표집하고 그 문항의 판정을 전부 가져온다."""
    by_item = defaultdict(list)
    for j in judgs:
        by_item[j["pair_id"]].append(fn(j["verdict"]))
    ids = sorted(by_item)   # 입력 행 순서와 무관하게 재현되도록 정렬 — 정렬하지 않으면 같은 시드로도 CI 가 달라진다
    if not ids:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        num = den = 0
        for _ in ids:
            picks = by_item[ids[rng.randrange(len(ids))]]
            num += sum(picks); den += len(picks)
        stats.append(num / den if den else 0)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def item_majority(judgs):
    """문항별 외부 다수결. 반환: {pair_id: (verdict|None, n_raters, tie)}"""
    by_item = defaultdict(list)
    for j in judgs:
        by_item[j["pair_id"]].append(j["verdict"])
    out = {}
    for pid, vs in by_item.items():
        c = Counter(vs).most_common()
        if len(c) > 1 and c[0][1] == c[1][1]:
            out[pid] = (None, len(vs), True)
        else:
            out[pid] = (c[0][0], len(vs), False)
    return out


def precision_block(judgs, label):
    maj = item_majority(judgs)
    ties = sum(1 for v in maj.values() if v[2])
    # 동률은 보수적으로: strict·lenient 모두 0
    item_strict = [strict(v[0]) if v[0] else False for v in maj.values()]
    item_lenient = [lenient(v[0]) if v[0] else False for v in maj.values()]
    return {
        "label": label,
        "n_judgments": len(judgs), "n_items": len(maj), "n_ties": ties,
        "pooled": {"strict": pooled_rate(judgs, strict), "lenient": pooled_rate(judgs, lenient),
                   "strict_ci95": item_bootstrap_ci(judgs, strict),
                   "lenient_ci95": item_bootstrap_ci(judgs, lenient)},
        "item_majority": {"strict": sum(item_strict) / len(item_strict) if item_strict else None,
                          "lenient": sum(item_lenient) / len(item_lenient) if item_lenient else None},
        "verdict_dist": {v: sum(1 for j in judgs if j["verdict"] == v) / len(judgs) for v in VERDICTS} if judgs else {},
        "unsure_rate": pooled_rate(judgs, lambda v: v == "unsure"),
    }


# ───────────────────────────────────────────── 일치도

def alpha_matrix(judgs, raters, items, mapper):
    """(raters × items) 행렬, 결측은 nan."""
    ridx = {r: i for i, r in enumerate(raters)}
    iidx = {p: i for i, p in enumerate(items)}
    M = np.full((len(raters), len(items)), np.nan)
    for j in judgs:
        if j["rater"] in ridx and j["pair_id"] in iidx:
            val = mapper(j["verdict"])
            if val is not None:
                M[ridx[j["rater"]], iidx[j["pair_id"]]] = val
    return M


def krippendorff_alpha(M):
    import krippendorff
    # 최소 2개 문항에서 2명 이상 겹쳐야 정의된다
    valid_cols = (~np.isnan(M)).sum(axis=0) >= 2
    if valid_cols.sum() < 2:
        return None
    try:
        return float(krippendorff.alpha(reliability_data=M[:, valid_cols], level_of_measurement="nominal"))
    except Exception:
        return None


def alpha_bootstrap(M, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    cols = np.arange(M.shape[1])
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(cols, size=len(cols), replace=True)
        a = krippendorff_alpha(M[:, pick])
        if a is not None and not math.isnan(a):
            vals.append(a)
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def pairwise_agreement(judgs, raters, items):
    """앵커에서 판정자 쌍별 단순 일치율."""
    by = defaultdict(dict)
    for j in judgs:
        if j["pair_id"] in items:
            by[j["rater"]][j["pair_id"]] = j["verdict"]
    out = {}
    for i, a in enumerate(raters):
        for b in raters[i + 1:]:
            common = set(by[a]) & set(by[b])
            if common:
                agree = sum(1 for p in common if by[a][p] == by[b][p]) / len(common)
                out[f"{a}|{b}"] = {"n": len(common), "agree": agree}
    return out


def cohen_kappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b))


# ───────────────────────────────────────────── main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", default="docs/t0/t0_judgments.json")
    ap.add_argument("--sample", default="docs/t0/t0_h3_validation_sample.json")
    ap.add_argument("--out", default="docs/t0")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    J, items, missing = load(args.judgments, args.sample)
    ext = [j for j in J if GROUPS["external"](j["rater"])]
    au = [j for j in J if GROUPS["author"](j["rater"])]
    raters = sorted({j["rater"] for j in J})
    ext_raters = [r for r in raters if r.startswith("EV-")]
    anchor_items = sorted({j["pair_id"] for j in J if j["block"] == "anchor"})
    all_items = sorted({j["pair_id"] for j in J})
    print(f"[val] 판정 {len(J)} (표본 미매칭 {missing}) · 판정자 {len(raters)} · 앵커 문항 {len(anchor_items)}", flush=True)

    R = {"meta": {
        "run_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "n_judgments": len(J), "n_external_judgments": len(ext), "n_author_judgments": len(au),
        "raters": raters, "n_items": len(all_items), "n_anchor_items": len(anchor_items),
        "seed": SEED, "n_boot": args.n_boot,
        "estimator": "주 정밀도 = 외부 판정 합산, CI = 문항 단위 부트스트랩; 문항 다수결 병기(동률은 보수적)",
    }}

    # 1·2. 대역별·팔별 정밀도 (외부)
    R["precision_external"] = {
        "overall": precision_block(ext, "외부 전체"),
        "by_band": {b: precision_block([j for j in ext if j["band"] == b], b) for b in BANDS},
        "by_arm": {a: precision_block([j for j in ext if j["arm"] == a], a) for a in ("A", "B")},
        "by_group": {g: precision_block([j for j in ext if GROUPS[g](j["rater"])], g) for g in ("phd", "staff", "student")},
    }
    # 저자 정밀도(비교용)
    R["precision_author"] = {
        "overall": precision_block(au, "저자"),
        "by_band": {b: precision_block([j for j in au if j["band"] == b], b) for b in BANDS},
    }
    # 임계값 누적 정밀도 — "cos≥t 를 채택하면 정밀도는?" (팔 A만, 외부)
    A_ext = [j for j in ext if j["arm"] == "A" and j["cosine"] is not None]
    cum = {}
    for t in (0.85, 0.90, 0.95):
        sub = [j for j in A_ext if j["cosine"] >= t]
        cum[str(t)] = {"n_judgments": len(sub), "strict": pooled_rate(sub, strict), "lenient": pooled_rate(sub, lenient),
                       "strict_ci95": item_bootstrap_ci(sub, strict, args.n_boot)}
    R["precision_cumulative_by_threshold"] = cum

    # 3. Krippendorff α
    def alpha_block(rlist, item_list, label):
        m5 = alpha_matrix(J, rlist, item_list, lambda v: VIDX[v])
        m_bin = alpha_matrix(J, rlist, item_list, lambda v: 1 if v == "same" else 0)
        m_len = alpha_matrix(J, rlist, item_list, lambda v: 1 if v in ("same", "hierarchy") else 0)
        m_3 = alpha_matrix(J, rlist, item_list, lambda v: {"same": 0, "hierarchy": 1}.get(v, 2) if v != "unsure" else None)
        return {"label": label, "n_raters": len(rlist), "n_items": len(item_list),
                "alpha_5cat": krippendorff_alpha(m5), "alpha_5cat_ci95": alpha_bootstrap(m5),
                "alpha_merge_binary": krippendorff_alpha(m_bin), "alpha_merge_binary_ci95": alpha_bootstrap(m_bin),
                "alpha_lenient_binary": krippendorff_alpha(m_len),
                "alpha_3cat_unsure_missing": krippendorff_alpha(m_3)}

    phd = [r for r in ext_raters if GROUPS["phd"](r)]
    staff = [r for r in ext_raters if GROUPS["staff"](r)]
    stud = [r for r in ext_raters if GROUPS["student"](r)]
    R["reliability"] = {
        "anchor_external_all": alpha_block(ext_raters, anchor_items, "앵커 · 외부 8인"),
        "anchor_phd": alpha_block(phd, anchor_items, "앵커 · 교육공학 박사 2인 (천장)"),
        "anchor_staff": alpha_block(staff, anchor_items, "앵커 · 직원 3인"),
        "anchor_student": alpha_block(stud, anchor_items, "앵커 · 학생 3인"),
        "anchor_all_incl_author": alpha_block(raters, anchor_items, "앵커 · 저자 포함 9인"),
        "all_items_external_missing": alpha_block(ext_raters, all_items, "전 문항 · 외부 8인 (결측 허용)"),
        "pairwise_anchor_agreement": pairwise_agreement(J, raters, set(anchor_items)),
    }

    # 4. 저자–외부 일치
    ext_maj = item_majority(ext)
    au_by = {j["pair_id"]: j["verdict"] for j in au}
    pairs = [(au_by[p], ext_maj[p][0]) for p in all_items if p in au_by and p in ext_maj and ext_maj[p][0]]
    if pairs:
        a, b = zip(*pairs)
        agree5 = sum(1 for x, y in pairs) and sum(1 for x, y in pairs if x == y) / len(pairs)
        agree_bin = sum(1 for x, y in pairs if strict(x) == strict(y)) / len(pairs)
        R["author_vs_external"] = {
            "n_items_compared": len(pairs), "n_ties_excluded": sum(1 for p in all_items if p in ext_maj and ext_maj[p][2]),
            "agreement_5cat": agree5, "kappa_5cat": cohen_kappa(a, b),
            "agreement_merge_binary": agree_bin,
            "kappa_merge_binary": cohen_kappa([int(strict(x)) for x in a], [int(strict(y)) for y in b]),
            "author_stricter_count": sum(1 for x, y in pairs if strict(y) and not strict(x)),
            "author_more_lenient_count": sum(1 for x, y in pairs if strict(x) and not strict(y)),
        }

    # 5. 애매함 · 6. 경험별 · 소요 시간
    R["unsure"] = {
        "by_rater": {r: pooled_rate([j for j in J if j["rater"] == r], lambda v: v == "unsure") for r in raters},
        "by_band_external": {b: pooled_rate([j for j in ext if j["band"] == b], lambda v: v == "unsure") for b in BANDS},
    }
    exp_levels = sorted({j["experience"] for j in ext if j["experience"]})
    R["by_experience_external"] = {
        e: {"n": len([j for j in ext if j["experience"] == e]),
            "raters": sorted({j["rater"] for j in ext if j["experience"] == e}),
            "verdict_dist": {v: pooled_rate([j for j in ext if j["experience"] == e], lambda x, v=v: x == v) for v in VERDICTS},
            "strict": pooled_rate([j for j in ext if j["experience"] == e], strict)}
        for e in exp_levels
    }
    R["timing"] = {r: {"median_s": float(np.median([j["elapsed_ms"] for j in J if j["rater"] == r and j["elapsed_ms"] is not None])) / 1000,
                       "share_under_2s": pooled_rate([j for j in J if j["rater"] == r and j["elapsed_ms"] is not None],
                                                     lambda v: False) if False else
                                         (sum(1 for j in J if j["rater"] == r and j["elapsed_ms"] is not None and j["elapsed_ms"] < 2000) /
                                          max(1, sum(1 for j in J if j["rater"] == r and j["elapsed_ms"] is not None)))}
                   for r in raters}

    # 정성 — 앵커에서 판정이 가장 갈린 문항, 박사 2인이 갈린 문항
    def item_view(pid):
        it = items[pid]
        vs = {j["rater"]: j["verdict"] for j in J if j["pair_id"] == pid}
        return {"pair_id": pid, "left": it["left"]["surfaces"][0], "right": it["right"]["surfaces"][0],
                "band": it["_hidden"]["band"], "cosine": it["_hidden"].get("cosine"),
                "verdicts": vs, "n_distinct": len(set(vs.values()))}
    anchor_views = [item_view(p) for p in anchor_items]
    R["qualitative"] = {
        "most_contested_anchor": sorted(anchor_views, key=lambda d: -d["n_distinct"])[:10],
        "phd_disagree": [d for d in anchor_views if d["verdicts"].get("EV-1") != d["verdicts"].get("EV-2")][:15],
        "unanimous_same_examples": [d for d in anchor_views if len(set(d["verdicts"].values())) == 1 and list(d["verdicts"].values())[0] == "same"][:6],
        "unanimous_not_same_examples": [d for d in anchor_views if len(set(d["verdicts"].values())) == 1 and list(d["verdicts"].values())[0] in ("related", "unrelated")][:6],
    }

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "t0_validation_results.json").write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out / "t0_validation_report.md", R)
    print(f"[val] 완료 → {out}/t0_validation_results.json · t0_validation_report.md")

    pe = R["precision_external"]
    print(f"\n외부 정밀도 전체: 엄격 {pe['overall']['pooled']['strict']:.1%} {pe['overall']['pooled']['strict_ci95']} · "
          f"관대 {pe['overall']['pooled']['lenient']:.1%}")
    for b in BANDS:
        x = pe["by_band"][b]["pooled"]
        print(f"  {b:>10}: 엄격 {x['strict']:.1%} [{x['strict_ci95'][0]:.2f},{x['strict_ci95'][1]:.2f}] · 관대 {x['lenient']:.1%}")
    rl = R["reliability"]
    for k in ("anchor_external_all", "anchor_phd", "anchor_staff", "anchor_student"):
        x = rl[k]
        print(f"  α {x['label']}: 5범주 {x['alpha_5cat']:.3f} · 병합이진 {x['alpha_merge_binary']:.3f}")
    if "author_vs_external" in R:
        x = R["author_vs_external"]
        print(f"  저자–외부: 5범주 일치 {x['agreement_5cat']:.1%} κ={x['kappa_5cat']:.3f} · 병합이진 일치 {x['agreement_merge_binary']:.1%} κ={x['kappa_merge_binary']:.3f}")


def fmt(v, nd=3):
    if v is None: return "—"
    if isinstance(v, float): return f"{v:.{nd}f}"
    return str(v)


def pct(v):
    return "—" if v is None else f"{v*100:.1f}%"


def write_report(path, R):
    L = []
    A = L.append
    m = R["meta"]
    A("# T0 · H3 병합 정밀도 수동 검증 결과\n")
    A(f"실행 {m['run_at']} · 판정 {m['n_judgments']}건(외부 {m['n_external_judgments']} · 저자 {m['n_author_judgments']}) · "
      f"판정자 {len(m['raters'])}명 · 문항 {m['n_items']}(앵커 {m['n_anchor_items']})\n")
    A(f"추정: {m['estimator']}\n")

    pe = R["precision_external"]
    A("\n## 1. 외부 판정 정밀도 — 대역별\n")
    A("| 대역 | 판정 n | 문항 | 엄격 정밀도 | 95% CI | 관대 정밀도 | 문항 다수결 엄격 | 동률 | 애매 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for key, blk in [("전체", pe["overall"])] + [(b, pe["by_band"][b]) for b in BANDS]:
        p = blk["pooled"]; ci = p["strict_ci95"]
        A(f"| {key} | {blk['n_judgments']} | {blk['n_items']} | **{pct(p['strict'])}** | [{pct(ci[0])}, {pct(ci[1])}] | "
          f"{pct(p['lenient'])} | {pct(blk['item_majority']['strict'])} | {blk['n_ties']} | {pct(blk['unsure_rate'])} |")
    A("\n### 임계값 누적 (팔 A, cos ≥ t 채택 시)\n")
    A("| 임계값 | n | 엄격 | 95% CI | 관대 |"); A("|---|---|---|---|---|")
    for t, x in R["precision_cumulative_by_threshold"].items():
        A(f"| ≥{t} | {x['n_judgments']} | {pct(x['strict'])} | [{pct(x['strict_ci95'][0])}, {pct(x['strict_ci95'][1])}] | {pct(x['lenient'])} |")

    A("\n## 2. 팔별 · 판정자군별\n")
    A("| 구분 | n | 엄격 | 95% CI | 관대 | 애매 |"); A("|---|---|---|---|---|---|")
    for key, blk in [("A 직접 엣지", pe["by_arm"]["A"]), ("B 전이 동반", pe["by_arm"]["B"]),
                     ("박사 2인", pe["by_group"]["phd"]), ("직원 3인", pe["by_group"]["staff"]), ("학생 3인", pe["by_group"]["student"]),
                     ("저자(비교용)", R["precision_author"]["overall"])]:
        p = blk["pooled"]; ci = p["strict_ci95"]
        A(f"| {key} | {blk['n_judgments']} | {pct(p['strict'])} | [{pct(ci[0])}, {pct(ci[1])}] | {pct(p['lenient'])} | {pct(blk['unsure_rate'])} |")

    A("\n## 3. 판정자 간 일치도 (Krippendorff α, 명목)\n")
    A("| 집단 | 판정자 | 문항 | α 5범주 | 95% CI | α 병합 여부(이진) | 95% CI | α 관대 이진 | α 3범주(애매=결측) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for k in ("anchor_phd", "anchor_staff", "anchor_student", "anchor_external_all", "anchor_all_incl_author", "all_items_external_missing"):
        x = R["reliability"][k]
        c5 = x["alpha_5cat_ci95"] or [None, None]; cb = x["alpha_merge_binary_ci95"] or [None, None]
        A(f"| {x['label']} | {x['n_raters']} | {x['n_items']} | **{fmt(x['alpha_5cat'])}** | [{fmt(c5[0])}, {fmt(c5[1])}] | "
          f"**{fmt(x['alpha_merge_binary'])}** | [{fmt(cb[0])}, {fmt(cb[1])}] | {fmt(x['alpha_lenient_binary'])} | {fmt(x['alpha_3cat_unsure_missing'])} |")

    if "author_vs_external" in R:
        x = R["author_vs_external"]
        A("\n## 4. 저자 vs 외부 다수결\n")
        A(f"- 비교 문항 {x['n_items_compared']}(동률 제외 {x['n_ties_excluded']}) · 5범주 일치 **{pct(x['agreement_5cat'])}** (κ={fmt(x['kappa_5cat'])}) · "
          f"병합 여부 일치 **{pct(x['agreement_merge_binary'])}** (κ={fmt(x['kappa_merge_binary'])})")
        A(f"- 외부가 「같은 개념」인데 저자가 아닌 경우 {x['author_stricter_count']} · 저자가 「같은 개념」인데 외부가 아닌 경우 {x['author_more_lenient_count']}")

    A("\n## 5. 애매함 · 경험 · 소요 시간\n")
    A("| 판정자 | 애매 비율 | 중앙 소요(초) | 2초 미만 비율 |"); A("|---|---|---|---|")
    for r in m["raters"]:
        A(f"| {r} | {pct(R['unsure']['by_rater'][r])} | {fmt(R['timing'][r]['median_s'],1)} | {pct(R['timing'][r]['share_under_2s'])} |")
    A("\n경험별(외부, 강건성 확인용):\n")
    A("| 경험 | 판정자 | n | 같은 개념 | 상하위 | 관련 | 무관 | 애매 |"); A("|---|---|---|---|---|---|---|---|")
    for e, x in R["by_experience_external"].items():
        d = x["verdict_dist"]
        A(f"| {e} | {', '.join(x['raters'])} | {x['n']} | {pct(d['same'])} | {pct(d['hierarchy'])} | {pct(d['related'])} | {pct(d['unrelated'])} | {pct(d['unsure'])} |")

    q = R["qualitative"]
    A("\n## 6. 정성 — 앵커에서 가장 갈린 문항\n")
    for d in q["most_contested_anchor"][:8]:
        vs = Counter(d["verdicts"].values())
        A(f"- **{d['left']}** ↔ **{d['right']}** ({d['band']}, cos {fmt(d['cosine'],3)}) → {dict(vs)}")
    A("\n박사 2인이 갈린 앵커 문항:\n")
    for d in q["phd_disagree"][:10]:
        A(f"- {d['left']} ↔ {d['right']} → EV-1 {d['verdicts'].get('EV-1')} / EV-2 {d['verdicts'].get('EV-2')}")
    Path(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
