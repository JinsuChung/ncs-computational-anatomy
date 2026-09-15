# -*- coding: utf-8 -*-
"""T0 · 임베딩 단계 후보 병합 파일 (상한 ≤ 5) — 공개 자산.

규칙 단계 별칭 사전(t0_alias_dictionary.csv)은 정규화 키 묶음만 담는다. 이 파일은 그 위에 얹는 임베딩 단계를
"후보(candidate)"로 공개한다: cos ≥ .90 유사도 그래프의 Louvain 커뮤니티 중 크기 2–5 인 것.
§4.6 이 잰 정밀도(직접 엣지 엄격 72.3% / 관대 85.2%, 크기 3–5 전이 32.9% / 56.9%)를 군집마다 직접 엣지 비율로
가중해 expected_strict_precision 으로 붙이고, 사람 판정이 있는 쌍은 판정을 그대로 싣는다.
검토되지 않은 병합을 푸는 위험은 validation_tier · 군집 크기 · 직접 엣지 비율 표기로 관리한다.

출력: docs/t0/t0_embedding_candidates_le5.csv (+ .meta.json) · docs/t0/t0_key_meta.json (키 캐시, 비추적)
사용: .venv/bin/python scripts/t0_build_embedding_candidates.py [--cap 5]
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t0_embed_recovery import SEED, cluster_louvain, load_keys  # noqa: E402

OUT = Path("docs/t0")
P_DIRECT = {"strict": 0.723, "lenient": 0.852}      # §4.6 1차 8인, 직접 엣지 cos ≥ .90 (120문항)
P_TRANS_35 = {"strict": 0.329, "lenient": 0.569}    # §4.6 전이, 군집 3–5
R_EXT = lambda r: r.upper().startswith("EV-")


def key_meta(dsn):
    cache = OUT / "t0_key_meta.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8")); print("[cand] 키 캐시 사용", flush=True)
        return d["keys"], d["meta"]
    print("[cand] 키 적재(DB) …", flush=True)
    key_surfs, key_jobs, key_type, _ = load_keys(dsn)
    keys = sorted(key_surfs)
    meta = {k: {"n_jobs": len(key_jobs[k]), "n_surfaces": len(key_surfs[k]), "type": key_type[k],
                "surfaces": sorted(key_surfs[k])[:5]} for k in keys}
    cache.write_text(json.dumps({"keys": keys, "meta": meta}, ensure_ascii=False), encoding="utf-8")
    return keys, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=os.environ.get("DATABASE_URL", "postgresql://ncs:ncs@localhost:55432/ncs"))
    ap.add_argument("--cap", type=int, default=5)
    a = ap.parse_args()
    keys, meta = key_meta(a.dsn); n = len(keys); kidx = {k: i for i, k in enumerate(keys)}
    kj_path = OUT / "t0_key_jobs.json"
    if kj_path.exists():
        key_jobs = {k: set(v) for k, v in json.loads(kj_path.read_text(encoding="utf-8")).items()}
    else:
        print("[cand] 키→직무 적재(DB) …", flush=True)
        _, key_jobs, _, _ = load_keys(a.dsn)
        kj_path.write_text(json.dumps({k: sorted(v) for k, v in key_jobs.items()}, ensure_ascii=False), encoding="utf-8")
    single = [len(key_jobs[k]) == 1 for k in keys]
    zp = np.load(OUT / "h3_vecs_326592_pairs_0.85.npz"); pi, pj, ps = zp["pi"], zp["pj"], zp["ps"]
    m = ps >= 0.90
    edge = {}
    for x, y, s in zip(pi[m].tolist(), pj[m].tolist(), ps[m].tolist()):
        edge[(min(x, y), max(x, y))] = s
    print("[cand] Louvain@.90 …", flush=True)
    groups, _ = cluster_louvain(n, pi, pj, ps, 0.90)
    # 사람 판정 — 외부 판정자 전원, 쌍 단위 다수결
    J = json.loads((OUT / "t0_judgments.json").read_text(encoding="utf-8"))["rows"]
    votes = defaultdict(list)
    for r in J:
        if R_EXT(r["rater"]) and r.get("left_key") in kidx and r.get("right_key") in kidx:
            x, y = kidx[r["left_key"]], kidx[r["right_key"]]
            votes[(min(x, y), max(x, y))].append(r["verdict"])
    rows, sizes, tiers, paths = [], Counter(), Counter(), Counter()
    for gid, mem in groups.items():
        if gid[0] != "c" or not (2 <= len(mem) <= a.cap):
            continue
        pairs = [(mem[i], mem[j]) for i in range(len(mem)) for j in range(i + 1, len(mem))]
        direct = [edge[p] for p in pairs if p in edge]
        w = len(direct) / len(pairs)
        exp_s = w * P_DIRECT["strict"] + (1 - w) * P_TRANS_35["strict"]
        exp_l = w * P_DIRECT["lenient"] + (1 - w) * P_TRANS_35["lenient"]
        s_ = [i for i in mem if single[i]]; sh = [i for i in mem if not single[i]]
        if sh: path = "c_into_shared"
        else:
            occs = {next(iter(key_jobs[keys[i]])) for i in s_}
            path = "a_cross_occupation" if len(occs) > 1 else "b_same_occupation"
        paths[path] += 1
        judged = [(p, votes[p]) for p in pairs if p in votes]
        if judged:
            vs = [v for _, vv in judged for v in vv]
            same = sum(v == "same" for v in vs); rel = sum(v in ("same", "hierarchy") for v in vs)
            tier = "human_validated"
            hv = f"same={same}/{len(vs)};same_or_hier={rel}/{len(vs)}"
        else:
            tier, hv = "candidate", ""
        sizes[len(mem)] += 1; tiers[tier] += 1
        rows.append({
            "candidate_id": f"C{gid[1]:06d}", "size": len(mem), "merge_path": path,
            "recommended": int(path != "b_same_occupation" and w == 1.0),
            "member_keys": "|".join(keys[i] for i in mem),
            "member_n_jobs": "|".join(str(meta[keys[i]]["n_jobs"]) for i in mem),
            "member_n_surfaces": "|".join(str(meta[keys[i]]["n_surfaces"]) for i in mem),
            "member_types": "|".join(meta[keys[i]]["type"] for i in mem),
            "example_surfaces": "|".join(meta[keys[i]]["surfaces"][0] for i in mem),
            "n_pairs": len(pairs), "n_direct_edges": len(direct), "direct_edge_share": round(w, 3),
            "min_cosine_direct": round(min(direct), 4) if direct else "", "mean_cosine_direct": round(float(np.mean(direct)), 4) if direct else "",
            "single_occupation_members": sum(meta[keys[i]]["n_jobs"] == 1 for i in mem),
            "expected_strict_precision": round(exp_s, 3), "expected_lenient_precision": round(exp_l, 3),
            "validation_tier": tier, "human_verdicts": hv,
        })
    rows.sort(key=lambda r: (-r["size"], r["candidate_id"]))
    path = OUT / f"t0_embedding_candidates_le{a.cap}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wri.writeheader(); wri.writerows(rows)
    n_keys = sum(r["size"] for r in rows)
    metaj = {
        "built_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "CANDIDATE — embedding-stage merges are proposals, not verified identities",
        "source": "bge-m3 embeddings of 326,592 normalization keys; pairs cos ≥ .85; Louvain (resolution 1.0, seed %d) on the cos ≥ .90 graph" % SEED,
        "policy": f"communities of size 2..{a.cap}",
        "n_candidate_groups": len(rows), "n_keys_covered": n_keys, "size_distribution": dict(sorted(sizes.items())),
        "validation_tiers": dict(tiers), "merge_paths": dict(paths),
        "merge_path_note": "c_into_shared: single-occupation keys join a concept another occupation already holds (judged direct-edge precision 72.5%); "
                           "a_cross_occupation: single-occupation keys of different occupations merge, creating a new shared concept (61.1%; carries the downstream gain); "
                           "b_same_occupation: keys of one occupation merge (43.9%; no downstream value) — not recommended. "
                           "recommended = 1 when the path is not b_same_occupation and every member pair is a direct edge (cos ≥ .90).",
        "precision_basis": {"direct_edge_cos>=0.90": P_DIRECT, "transitive_size_3_5": P_TRANS_35,
                            "note": "expected_*_precision = direct_edge_share × direct + (1 − share) × transitive; both from §4.6 (first-round 8 external raters). "
                                    "Policy-level verified figures for size ≤ 5 are in t0_h3_verified_recovery_ci.json."},
        "human_verdicts": "majority over all external raters' judgments on member pairs (rounds 1–3); 'same' = strict, 'same_or_hier' = lenient",
        "license": "Derived from NCS (KOGL Type 1, attribution). Candidate merges carry no claim of correctness; consumers should treat validation_tier='candidate' rows as unreviewed.",
    }
    (OUT / f"t0_embedding_candidates_le{a.cap}.meta.json").write_text(json.dumps(metaj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cand] 군집 {len(rows):,} · 키 {n_keys:,} · 크기분포 {dict(sorted(sizes.items()))} · 검증계층 {dict(tiers)} · 경로 {dict(paths)} · 권장 {sum(r['recommended'] for r in rows):,}")
    print(f"[cand] 직접 엣지 비율 평균 {np.mean([r['direct_edge_share'] for r in rows]):.3f} · 단일직무 멤버 비율 {sum(r['single_occupation_members'] for r in rows)/n_keys:.3f}")
    print(f"[cand] 기대 엄격 정밀도(키 가중) {sum(r['expected_strict_precision']*r['size'] for r in rows)/n_keys:.3f}")
    print("[cand] 완료", flush=True)


if __name__ == "__main__":
    main()
