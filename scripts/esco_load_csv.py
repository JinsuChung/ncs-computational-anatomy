"""ESCO 공식 CSV(classification, v1.2.x, 단일 언어) → esco_replication.py 가 읽는 JSON 으로 변환.

API 수집(esco_fetch.py)은 검색 상한·일괄 조회 500 오류로 완주하지 못했다. 공식 CSV 는 버전이 명시된
완전한 데이터라 1차 출처로 쓰고, API 부분 수집본은 대조용으로만 남긴다.

입력: 다운로드 zip 또는 압축 해제 디렉터리. 파일명은 ESCO 관례를 따른다고 가정하되
     (skills_*.csv, occupations_*.csv, occupationSkillRelations_*.csv, ISCOGroups_*.csv, skillsHierarchy_*.csv)
     열 이름은 실제 헤더에서 대소문자 무시로 찾고, 없으면 헤더를 출력하고 멈춘다.
출력: docs/esco/esco_skills.json · esco_occupations.json · esco_meta.json (원본 CSV 는 비추적)

사용: .venv/bin/python scripts/esco_load_csv.py --src "/path/to/ESCO dataset - v1.2.1 - classification - en - csv.zip"
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("docs/esco")
csv.field_size_limit(1 << 26)


def open_all(src: Path):
    """{파일명: 텍스트} — zip 이든 디렉터리든."""
    files = {}
    if src.is_dir():
        for p in src.rglob("*.csv"):
            files[p.name] = p.read_text(encoding="utf-8-sig")
    else:
        with zipfile.ZipFile(src) as z:
            for n in z.namelist():
                if n.lower().endswith(".csv"):
                    files[Path(n).name] = z.read(n).decode("utf-8-sig")
    return files


def pick(files, *needles):
    for name in files:
        low = name.lower()
        if all(nd.lower() in low for nd in needles):
            return name
    return None


def rows_of(text):
    return list(csv.DictReader(io.StringIO(text)))


def col(row_keys, *cands):
    keys = {k.lower().replace(" ", ""): k for k in row_keys}
    for c in cands:
        k = keys.get(c.lower().replace(" ", ""))
        if k:
            return k
    raise SystemExit(f"열을 찾지 못했다: {cands} — 실제 헤더: {list(row_keys)}")


def split_labels(cell):
    if not cell:
        return []
    return [x.strip() for x in re.split(r"[\n\r]+|\s*\|\s*", cell) if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    args = ap.parse_args()
    src = Path(args.src)
    files = open_all(src)
    print("[csv] 파일:", sorted(files))

    f_sk = pick(files, "skills_") or pick(files, "skills")
    f_oc = pick(files, "occupations_") or pick(files, "occupations")
    f_rel = pick(files, "occupationskillrelations")
    f_isco = pick(files, "iscogroups")
    if not (f_sk and f_oc and f_rel):
        raise SystemExit(f"필수 파일 누락: skills={f_sk} occupations={f_oc} relations={f_rel}")

    # ── skills
    sk_rows = rows_of(files[f_sk])
    k = sk_rows[0].keys()
    c_uri, c_pref, c_alt = col(k, "conceptUri", "uri"), col(k, "preferredLabel"), col(k, "altLabels", "alternativeLabels")
    c_type = col(k, "skillType"); c_reuse = col(k, "reuseLevel"); c_desc = col(k, "description"); c_status = col(k, "status")
    skills = []
    for r in sk_rows:
        if (r.get(c_status) or "released").lower() not in ("released", ""):
            continue  # obsolete/deprecated 제외
        skills.append({"uri": r[c_uri], "preferredLabel": r[c_pref], "altLabels": split_labels(r.get(c_alt)),
                       "description": r.get(c_desc), "status": r.get(c_status),
                       "skillType": [r.get(c_type)] if r.get(c_type) else [],
                       "reuseLevel": [r.get(c_reuse)] if r.get(c_reuse) else [], "broader": [], "essentialFor": []})
    # ── occupations
    oc_rows = rows_of(files[f_oc])
    k = oc_rows[0].keys()
    c_uri, c_pref, c_alt = col(k, "conceptUri", "uri"), col(k, "preferredLabel"), col(k, "altLabels", "alternativeLabels")
    c_isco = col(k, "iscoGroup"); c_code = col(k, "code"); c_desc = col(k, "description"); c_status = col(k, "status")
    occs = {}
    for r in oc_rows:
        if (r.get(c_status) or "released").lower() not in ("released", ""):
            continue
        occs[r[c_uri]] = {"uri": r[c_uri], "code": r.get(c_code) or r.get(c_isco), "iscoGroup": r.get(c_isco),
                          "preferredLabel": r[c_pref], "altLabels": split_labels(r.get(c_alt)),
                          "description": r.get(c_desc), "status": r.get(c_status),
                          "essentialSkills": [], "optionalSkills": [], "broaderOccupation": []}
    # ── relations
    rel_rows = rows_of(files[f_rel])
    k = rel_rows[0].keys()
    c_o, c_t, c_s = col(k, "occupationUri"), col(k, "relationType"), col(k, "skillUri")
    n_ess = n_opt = 0
    skill_uris = {s["uri"] for s in skills}
    ess_for = defaultdict(list)
    for r in rel_rows:
        o = occs.get(r[c_o])
        if not o or r[c_s] not in skill_uris:
            continue
        if (r[c_t] or "").lower().startswith("essential"):
            o["essentialSkills"].append(r[c_s]); ess_for[r[c_s]].append(r[c_o]); n_ess += 1
        else:
            o["optionalSkills"].append(r[c_s]); n_opt += 1
    for s in skills:
        s["essentialFor"] = ess_for.get(s["uri"], [])

    # ── ISCO 그룹(있으면 코드→라벨 사전만 보관)
    isco = {}
    if f_isco:
        ir = rows_of(files[f_isco]); k = ir[0].keys()
        c_code, c_pref = col(k, "code"), col(k, "preferredLabel")
        isco = {r[c_code]: r[c_pref] for r in ir}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "esco_skills.json").write_text(json.dumps(skills, ensure_ascii=False), encoding="utf-8")
    (OUT / "esco_occupations.json").write_text(json.dumps(list(occs.values()), ensure_ascii=False), encoding="utf-8")
    ver = re.search(r"v(\d+\.\d+(?:\.\d+)?)", src.name)
    meta = {"loaded_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "source": "ESCO 공식 CSV (esco.ec.europa.eu 다운로드, 사용자 수령)", "source_file": src.name,
            "version": ver.group(1) if ver else "파일명에서 미확인", "language": "en",
            "n_skills": len(skills), "n_occupations": len(occs), "n_isco_groups": len(isco),
            "n_en_altlabels": sum(len(s["altLabels"]) for s in skills),
            "n_essential_links": n_ess, "n_optional_links": n_opt,
            "license_note": "ESCO — European Commission, free reuse with attribution. 원본 CSV 비추적."}
    (OUT / "esco_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[csv] 완료: 버전 {meta['version']} · 스킬 {len(skills):,} · 직업 {len(occs):,} · ISCO 그룹 {len(isco):,} · "
          f"영문 대체 라벨 {meta['n_en_altlabels']:,} · 필수 {n_ess:,} · 선택 {n_opt:,}")


if __name__ == "__main__":
    main()
