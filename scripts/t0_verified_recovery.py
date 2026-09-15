"""정밀도 보정 회수율 — 임베딩이 '제안한' 병합 중 사람이 '옳다'고 본 몫만 센다.

입력
  docs/t0/t0_validation_results.json      수동 판정 집계 (대역·팔·군집크기별 정밀도)
  docs/t0/t0_judgments.json + 표본        군집 크기별 전이 정밀도 재계산용
  docs/t0/t0_h3_cluster_sizes_090.json    cos≥.90 Louvain 군집 크기별 흡수 키 수·직접 엣지 비율
  docs/t0/t0_h3_embedding.json            표면형 수(분모)

논리
  군집 안의 한 쌍은 '직접 엣지'(cos≥.90 으로 직접 연결)거나 '전이'(같은 군집이지만 직접 연결 없음)다.
  직접 엣지 쌍의 정밀도는 팔 A(cos≥.90 누적), 전이 쌍의 정밀도는 팔 B(군집 크기별)로 잰다.
  군집 크기 구간별로 두 정밀도를 직접 엣지 비율로 가중해 '그 구간에서 흡수된 키가 옳을 확률'을 만들고,
  흡수 키 수로 가중해 전체 검증 회수율을 낸다. 채택 정책(군집 크기 상한)별로도 낸다.

출력  docs/t0/t0_h3_verified_recovery.json
"""

from __future__ import annotations

R1_RATERS = {"EV-1", "EV-2", "EV-3", "EV-4", "EV-5", "EV-3B", "EV-4B", "EV-5B"}  # 1차 판정자 8인 × 80문항.
# 제외가 아니라 허용 목록으로 둔다 — 라운드가 늘어날 때(EV-6~9 2차, EV-10~11 3차) 새 판정자가
# 조용히 1차 집계에 섞여 표본 설계(팔 A 30 + 앵커 50)를 깨뜨리는 것을 막기 위해서다.

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20260909
BUCKETS = ["2", "3-5", "6-20", "21-100", "100+"]
STAGE1_RULE_RECOVERY = 0.08233842937703886


def bucket(k):
    return "2" if k == 2 else "3-5" if k <= 5 else "6-20" if k <= 20 else "21-100" if k <= 100 else "100+"


def rate(judgs, fn):
    return sum(fn(v) for v in judgs) / len(judgs) if judgs else None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2", action="store_true", help="2차(R2) 층별 전이 정밀도로 21-100/100+ 대체 → *_r2.json")
    args = ap.parse_args()
    V = json.loads(Path("docs/t0/t0_validation_results.json").read_text(encoding="utf-8"))
    C = json.loads(Path("docs/t0/t0_h3_cluster_sizes_090.json").read_text(encoding="utf-8"))
    E = json.loads(Path("docs/t0/t0_h3_embedding.json").read_text(encoding="utf-8"))
    J = json.loads(Path("docs/t0/t0_judgments.json").read_text(encoding="utf-8"))["rows"]
    S = {it["id"]: it for it in json.loads(Path("docs/t0/t0_h3_validation_sample.json").read_text(encoding="utf-8"))["items"]}
    n_surface = E["meta"]["n_surface_forms_total"]

    # 직접 엣지 정밀도: 팔 A, cos≥.90 누적 (외부)
    pa = V["precision_cumulative_by_threshold"]["0.9"]
    p_direct = {"strict": pa["strict"], "lenient": pa["lenient"], "n": pa["n_judgments"]}

    # 전이 정밀도: 팔 B, 군집 크기별 (외부). 표본이 얇은 구간(21+)은 합쳐서도 낸다.
    ext = [r for r in J if r["rater"].upper() in R1_RATERS and r["pair_id"] in S]
    byb = defaultdict(list)
    for r in ext:
        h = S[r["pair_id"]]["_hidden"]
        if h["arm"] == "B":
            byb[bucket(h["cluster_size"])].append(r["verdict"])
    p_trans = {}
    for b in BUCKETS:
        vs = byb.get(b, [])
        if not vs and b == "2":
            continue
        p_trans[b] = {"n": len(vs), "strict": rate(vs, lambda v: v == "same"),
                      "lenient": rate(vs, lambda v: v in ("same", "hierarchy"))}
    big = byb.get("21-100", []) + byb.get("100+", [])
    p_trans["21+"] = {"n": len(big), "strict": rate(big, lambda v: v == "same"),
                      "lenient": rate(big, lambda v: v in ("same", "hierarchy"))}
    # 21-100 / 100+ 은 각각 문항 4개라 얇다 — 합친 21+ 값을 두 구간에 공통 적용한다.
    for b in ("21-100", "100+"):
        p_trans[b] = dict(p_trans["21+"], note="21+ 합산값 적용(구간 표본 얇음)")
    if args.r2:
        R2 = json.loads(Path("docs/t0/t0_validation_r2_results.json").read_text(encoding="utf-8"))
        for b in ("21-100", "100+"):
            x = R2["transitive_by_stratum"][b]
            p_trans[b] = {"n": x["n_judgments"], "strict": x["strict"], "lenient": x["lenient"],
                          "strict_ci95": x["strict_ci95"], "lenient_ci95": x["lenient_ci95"], "note": "2차(R2) 층별 실측 — 문항 30, 판정자 3인"}

    # 구간별 가중 정밀도와 검증 키 수
    per = {}
    for b in BUCKETS:
        x = C["by_bucket"][b]
        d = 1.0 if b == "2" else (x["direct_edge_share"] or 0.0)
        pt = p_trans.get(b, {"strict": 0.0, "lenient": 0.0})
        strict = d * p_direct["strict"] + (1 - d) * (pt["strict"] or 0.0)
        lenient = d * p_direct["lenient"] + (1 - d) * (pt["lenient"] or 0.0)
        per[b] = {"keys_absorbed": x["keys_absorbed"], "clusters": x["clusters"],
                  "direct_edge_share": d, "precision_strict": strict, "precision_lenient": lenient,
                  "verified_keys_strict": x["keys_absorbed"] * strict,
                  "verified_keys_lenient": x["keys_absorbed"] * lenient}

    total_abs = sum(per[b]["keys_absorbed"] for b in BUCKETS)
    tot_s = sum(per[b]["verified_keys_strict"] for b in BUCKETS)
    tot_l = sum(per[b]["verified_keys_lenient"] for b in BUCKETS)

    # 채택 정책별: 군집 크기 상한을 두었을 때
    policies = {}
    for name, allowed in [("size2_only(직접 엣지만)", ["2"]), ("size<=5", ["2", "3-5"]),
                          ("size<=20", ["2", "3-5", "6-20"]), ("all(전부)", BUCKETS)]:
        ab = sum(per[b]["keys_absorbed"] for b in allowed)
        vs_ = sum(per[b]["verified_keys_strict"] for b in allowed)
        vl_ = sum(per[b]["verified_keys_lenient"] for b in allowed)
        policies[name] = {
            "keys_absorbed": ab, "raw_recovery_pp": ab / n_surface,
            "precision_strict": vs_ / ab if ab else None, "precision_lenient": vl_ / ab if ab else None,
            "verified_recovery_pp_strict": vs_ / n_surface, "verified_recovery_pp_lenient": vl_ / n_surface,
            "combined_verified_strict": STAGE1_RULE_RECOVERY + vs_ / n_surface,
            "combined_verified_lenient": STAGE1_RULE_RECOVERY + vl_ / n_surface,
            "combined_raw": STAGE1_RULE_RECOVERY + ab / n_surface,
        }

    out = {
        "n_surface_forms": n_surface, "keys_absorbed_at_090": total_abs,
        "stage1_rule_recovery": STAGE1_RULE_RECOVERY,
        "embedding_raw_contribution_pp": total_abs / n_surface,
        "p_direct_edge_cos_ge_090": p_direct, "p_transitive_by_cluster_size": p_trans,
        "per_bucket": per,
        "overall": {"precision_strict": tot_s / total_abs, "precision_lenient": tot_l / total_abs,
                    "verified_pp_strict": tot_s / n_surface, "verified_pp_lenient": tot_l / n_surface,
                    "combined_verified_strict": STAGE1_RULE_RECOVERY + tot_s / n_surface,
                    "combined_verified_lenient": STAGE1_RULE_RECOVERY + tot_l / n_surface},
        "policies": policies,
        "note": "규칙 단계(8.2%)의 병합 정밀도는 별도로 재지 않았다(접미사·공백 차이만 흡수하므로 오류 여지가 작다고 가정). "
                "전이 정밀도의 21-100/100+ 구간은 문항 4개씩이라 합산값(21+)을 적용했다.",
    }
    if args.r2:
        out["note"] = "21-100/100+ 전이 정밀도는 2차(R2, 2026-09-11) 층별 실측(각 문항 30, 3인 판정)으로 대체. 나머지는 1차와 동일."
    Path("docs/t0/t0_h3_verified_recovery_r2.json" if args.r2 else "docs/t0/t0_h3_verified_recovery.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"표면형 {n_surface:,} · cos≥.90 흡수 키 {total_abs:,} (= 원 기여 +{total_abs/n_surface:.1%}p)")
    print(f"직접 엣지 정밀도(cos≥.90): 엄격 {p_direct['strict']:.1%} 관대 {p_direct['lenient']:.1%}")
    print("구간별:")
    for b in BUCKETS:
        p = per[b]
        print(f"  크기 {b:>7}: 흡수 {p['keys_absorbed']:>6,} · 직접엣지 {p['direct_edge_share']:5.1%} · "
              f"정밀도 엄격 {p['precision_strict']:5.1%} 관대 {p['precision_lenient']:5.1%}")
    o = out["overall"]
    print(f"전체 병합 정밀도: 엄격 {o['precision_strict']:.1%} 관대 {o['precision_lenient']:.1%}")
    print(f"검증 회수율(결합): 엄격 {o['combined_verified_strict']:.1%} · 관대 {o['combined_verified_lenient']:.1%}  (원 수치 28.1%)")
    print("채택 정책별:")
    for k, p in policies.items():
        print(f"  {k:>22}: 원 {p['combined_raw']:.1%} → 검증 엄격 {p['combined_verified_strict']:.1%} "
              f"(정밀도 {p['precision_strict']:.0%}) · 관대 {p['combined_verified_lenient']:.1%}")


if __name__ == "__main__":
    main()
