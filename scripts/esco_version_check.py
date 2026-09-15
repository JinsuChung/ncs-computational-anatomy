"""로드된 ESCO 데이터(CSV 또는 API)의 버전 판정 — 사용자가 내려받은 v1.2.1 delta(변경 로그)와 대조한다.

delta.csv 는 v1.2.0 → v1.2.1 변경분이다. 그중 '영문 altLabel 추가' 행이 API 응답에 들어 있으면 API 는
v1.2.1(또는 그 이후)이고, 없으면 v1.2.0 이다. 응답 자체에는 버전 필드가 없으므로 이 대조가 유일한 증거다.

사용: .venv/bin/python scripts/esco_version_check.py --delta <delta.csv 경로>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", required=True)
    args = ap.parse_args()

    added, removed = {}, {}
    with open(args.delta, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["field"] != "altLabel" or row["language"] != "en":
                continue
            u = row["concept URI"]
            if row["action"] == "Added":
                added.setdefault(u, set()).add(row["new value"])
            elif row["action"] in ("Removed", "Deleted"):
                removed.setdefault(u, set()).add(row["old value"])

    data = {}
    for name in ("esco_skills.json", "esco_occupations.json"):
        for r in json.loads(Path("docs/esco", name).read_text(encoding="utf-8")):
            data[r["uri"]] = set(r.get("altLabels") or [])

    def score(chg):
        hit = tot = 0
        for u, labs in chg.items():
            if u not in data:
                continue
            for lab in labs:
                tot += 1
                hit += lab in data[u]
        return hit, tot

    ah, at = score(added)
    rh, rt = score(removed)
    print(f"v1.2.1 에서 추가된 영문 altLabel: 대조 가능 {at:,}건 중 데이터에 존재 {ah:,} ({ah/at:.1%})" if at else "추가 항목 대조 불가")
    print(f"v1.2.1 에서 제거된 영문 altLabel: 대조 가능 {rt:,}건 중 데이터에 잔존 {rh:,} ({rh/rt:.1%})" if rt else "제거 항목 대조 불가")
    verdict = "v1.2.1 이상" if at and ah / at > 0.9 and (not rt or rh / rt < 0.1) else ("v1.2.0" if at and ah / at < 0.1 else "판정 보류(혼재)")
    print(f"판정: 데이터 버전 = {verdict}")
    meta_p = Path("docs/esco/esco_meta.json")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta["version_check"] = {"added_en_altlabels_present": [ah, at], "removed_en_altlabels_still_present": [rh, rt], "verdict": verdict,
                             "delta_source": "ESCO dataset - v1.2.1 - delta (사용자 다운로드)"}
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
