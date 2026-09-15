"""ESCO 전수 수집 — 공개 REST API(인증 없음). 검색 API 는 쿼리당 200건 상한이라 열거에 못 쓴다.

열거 경로
  직업: ISCO 계층(C0~C9)을 resource/concept 의 narrowerConcept 로 내려가 단위군의 narrowerOccupation 을 모으고,
        직업 자체의 narrowerOccupation 도 폐포까지 따라간다.
  스킬: 모든 직업의 hasEssentialSkill/hasOptionalSkill 합집합 + 스킬 계층(S, L, ISCED-F 00~10) 순회.
조회: resource/{occupation,skill}?uris=… 일괄(25건/요청 — URL 길이 제한으로 100건은 연결이 끊긴다).

출력  docs/esco/esco_skills.json · esco_occupations.json · esco_meta.json  (원본 비추적, 집계만 커밋)
정중함: 요청 간 0.12s, 실패 시 지수 백오프.

사용: .venv/bin/python scripts/esco_fetch.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

API = "https://ec.europa.eu/esco/api"
OUT = Path("docs/esco")
LANG = "en"
BATCH = 25
SLEEP = 0.12
HEAD = {"Accept": "application/json", "User-Agent": "ncs-ontology-research/1.0 (academic; contact via repo)"}
n_req = 0


def get(url, tries=4):
    global n_req
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEAD), timeout=90) as r:
                n_req += 1
                time.sleep(SLEEP)
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            wait = 2 ** i
            print(f"  [retry {i+1}] {str(e)[:80]} — {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise SystemExit(f"실패: {url[:120]}")


def links(d, name):
    v = (d.get("_links") or {}).get(name) or []
    if isinstance(v, dict):
        v = [v]
    return [x.get("uri") or x.get("href") for x in v if x.get("uri") or x.get("href")]


def link_titles(d, name):
    v = (d.get("_links") or {}).get(name) or []
    if isinstance(v, dict):
        v = [v]
    return [x.get("title") for x in v if x.get("title")]


def concept(uri):
    return get(f"{API}/resource/concept?language={LANG}&uri={urllib.parse.quote(uri, safe='')}")


def batch(kind, uris):
    out = {}
    for i in range(0, len(uris), BATCH):
        chunk = uris[i:i + BATCH]
        q = "&".join("uris=" + urllib.parse.quote(u, safe="") for u in chunk)
        d = get(f"{API}/resource/{kind}?language={LANG}&{q}")
        emb = d.get("_embedded") or {}
        out.update(emb)
        if (i // BATCH) % 20 == 0:
            print(f"  {kind}: {len(out):,}/{len(uris):,}", flush=True)
    return out


def en(d, field):
    v = (d.get(field) or {}).get(LANG)
    if isinstance(v, dict):
        return v.get("literal")
    return v


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    # ── 1. ISCO 계층 순회 → 직업 URI
    print("[esco] ISCO 계층 순회 …", flush=True)
    occ_uris, seen = set(), set()
    q = deque(f"http://data.europa.eu/esco/isco/C{i}" for i in range(10))
    isco_nodes = 0
    while q:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        d = concept(u)
        isco_nodes += 1
        for c in links(d, "narrowerConcept"):
            q.append(c)
        for o in links(d, "narrowerOccupation"):
            occ_uris.add(o)
    print(f"[esco] ISCO 노드 {isco_nodes} · 단위군 직속 직업 {len(occ_uris):,}", flush=True)

    # ── 2. 직업 일괄 조회 + 하위 직업 폐포
    occs = {}
    pending = sorted(occ_uris)
    while pending:
        got = batch("occupation", pending)
        new = set()
        for u, d in got.items():
            occs[u] = d
            for c in links(d, "narrowerOccupation"):
                if c not in occs and c not in occ_uris:
                    new.add(c); occ_uris.add(c)
        pending = sorted(new)
        if pending:
            print(f"  하위 직업 추가 {len(pending)}", flush=True)
    print(f"[esco] 직업 {len(occs):,}", flush=True)

    # ── 3. 스킬 URI: 직업 링크 합집합 + 계층 순회
    skill_uris = set()
    for d in occs.values():
        skill_uris.update(links(d, "hasEssentialSkill")); skill_uris.update(links(d, "hasOptionalSkill"))
    print(f"[esco] 직업 연결 스킬 {len(skill_uris):,} · 스킬 계층 순회 …", flush=True)
    roots = ["http://data.europa.eu/esco/skill/S", "http://data.europa.eu/esco/skill/L", "http://data.europa.eu/esco/skill/T"] + \
            [f"http://data.europa.eu/esco/isced-f/{i:02d}" for i in range(0, 11)]
    q, seen, hier_nodes = deque(roots), set(), 0
    while q:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        try:
            d = get(f"{API}/resource/skill?language={LANG}&uri={urllib.parse.quote(u, safe='')}")
        except SystemExit:
            continue
        hier_nodes += 1
        for k, v in (d.get("_links") or {}).items():
            if k.startswith("narrower"):
                for x in (v if isinstance(v, list) else [v]):
                    cu = x.get("uri") or x.get("href")
                    if not cu:
                        continue
                    if "/skill/" in cu and not cu.rsplit("/", 1)[-1].startswith(("S", "L", "T")):
                        skill_uris.add(cu)   # 말단 스킬(UUID)
                    else:
                        q.append(cu)         # 그룹 노드
    print(f"[esco] 계층 노드 {hier_nodes} · 스킬 URI 합계 {len(skill_uris):,}", flush=True)

    # ── 4. 스킬 일괄 조회
    skills = batch("skill", sorted(skill_uris))
    print(f"[esco] 스킬 {len(skills):,}", flush=True)

    # ── 5. 정규화 저장 (esco_replication.py 가 읽는 형태)
    S = []
    for u, d in skills.items():
        S.append({"uri": u, "preferredLabel": en(d, "preferredLabel") or d.get("title"),
                  "altLabels": ((d.get("alternativeLabel") or {}).get(LANG)) or [],
                  "description": en(d, "description"), "status": d.get("status"),
                  "skillType": link_titles(d, "hasSkillType"), "reuseLevel": link_titles(d, "hasReuseLevel"),
                  "broader": links(d, "broaderHierarchyConcept") or links(d, "broaderConcept"),
                  "essentialFor": links(d, "isEssentialForOccupation")})
    O = []
    for u, d in occs.items():
        O.append({"uri": u, "code": d.get("code"), "preferredLabel": en(d, "preferredLabel") or d.get("title"),
                  "altLabels": ((d.get("alternativeLabel") or {}).get(LANG)) or [],
                  "description": en(d, "description"), "status": d.get("status"),
                  "essentialSkills": links(d, "hasEssentialSkill"), "optionalSkills": links(d, "hasOptionalSkill"),
                  "broaderOccupation": links(d, "broaderOccupation")})
    (OUT / "esco_skills.json").write_text(json.dumps(S, ensure_ascii=False), encoding="utf-8")
    (OUT / "esco_occupations.json").write_text(json.dumps(O, ensure_ascii=False), encoding="utf-8")
    meta = {"fetched_at": started, "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "source": API, "language": LANG, "requests": n_req,
            "enumeration": "ISCO C0-C9 narrowerConcept → narrowerOccupation (+직업 narrowerOccupation 폐포); "
                           "스킬 = 직업 링크 합집합 + S/L/T/ISCED-F 계층 순회",
            "search_totals_reported": {"skill": 13485, "occupation": 2942},
            "n_skills": len(S), "n_occupations": len(O),
            "n_en_altlabels": sum(len(s["altLabels"]) for s in S),
            "n_essential_links": sum(len(o["essentialSkills"]) for o in O),
            "license_note": "ESCO — European Commission, free reuse with attribution. 원본 JSON 비추적."}
    (OUT / "esco_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[esco] 완료: 스킬 {len(S):,}/13,485 · 직업 {len(O):,}/2,942 · 영문 대체 라벨 {meta['n_en_altlabels']:,} · "
          f"필수 링크 {meta['n_essential_links']:,} · 요청 {n_req:,}")


if __name__ == "__main__":
    main()
