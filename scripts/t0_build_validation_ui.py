"""검증 UI 빌더 — 표본 JSON과 판정자 명단을 판정 화면 HTML에 주입한다.

저장은 서버 API 한 곳뿐이다(`/api/judgments`). localStorage 는 저장소가 아니라 전송 실패 시
재시도 버퍼이자 이어하기용 진행 캐시이며, 기록의 원본은 언제나 서버다.

산출물은 Vercel 앱의 정적 파일 하나다: vercel-validation/public/index.html

사용:
  .venv/bin/python scripts/t0_build_validation_ui.py
  .venv/bin/python scripts/t0_build_validation_ui.py --api-base https://<앱>.vercel.app
      → 다른 출처에서 열 화면을 만들 때(절대 URL 사용). 이 경우 API 의 CORS 허용이 필요하다.
  .venv/bin/python scripts/t0_build_validation_ui.py --no-roster
      → 이름을 페이지에 넣지 않는다. 개인별 링크(?c=CODE)로만 진입.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "docs/t0/t0_h3_validation_sample.json"
TEMPLATE = ROOT / "scripts/templates/t0_validation_ui.html"
OUT = ROOT / "vercel-validation/public/index.html"

SAMPLE_MARKER = "/*__SAMPLE_JSON__*/null"
ROSTER_MARKER = "/*__ROSTER_JSON__*/[]"
SCRIPT_MARKER = "<script>"

# 판정자 실명은 저장소에 두지 않는다. 화면 명단이 필요하면 비추적 대조표(docs/t0/raters.local.json)를 읽고,
# 파일이 없으면 명단 없이 빌드한다 — 그 경우 판정자는 개인별 링크(?c=CODE)로만 들어온다.
# 서버에 저장되는 식별자는 어느 쪽이든 배정 코드뿐이다.
ROSTER_FILE = ROOT / "docs/t0/raters.local.json"
ROUND_CODES = {
    1: ["EV-1", "EV-2", "EV-3", "EV-4", "EV-5", "EV-3b", "EV-4b", "EV-5b", "AU-1"],
    2: ["EV-6", "EV-7", "EV-8", "EV-9"],
    3: ["EV-10", "EV-11"],
}


def load_roster(round_no: int, path: Path):
    """차수에 해당하는 [{name, code}] 목록. 대조표가 없으면 빈 목록(= 명단 없는 화면)."""
    if not path.exists():
        return []
    table = json.loads(path.read_text(encoding="utf-8")).get("raters", {})
    out = []
    for code in ROUND_CODES.get(round_no, []):
        rec = table.get(code)
        if rec and rec.get("name"):
            out.append({"name": rec["name"], "code": code})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="", help="절대 URL 로 API 를 가리킬 때의 오리진")
    ap.add_argument("--no-roster", action="store_true", help="이름을 페이지에 넣지 않는다(개인별 링크만 사용)")
    ap.add_argument("--roster-file", default=str(ROSTER_FILE), help="코드↔실명 대조표(비추적). 없으면 명단 없이 빌드")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--sample", default=str(SAMPLE), help="표본 JSON (2차: docs/t0/t0_h3_validation_sample_r2.json)")
    ap.add_argument("--round", type=int, default=1, help="명단 선택: 1 = 1차 8인+저자, 2 = 2차 4인, 3 = 3차 박사 2인")
    args = ap.parse_args()

    payload = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    html = TEMPLATE.read_text(encoding="utf-8")
    if SAMPLE_MARKER not in html or ROSTER_MARKER not in html:
        raise SystemExit("템플릿에 주입 지점이 없다 — 마커를 확인하라")

    roster = [] if args.no_roster else load_roster(args.round, Path(args.roster_file))
    api = (args.api_base.rstrip("/") + "/api/judgments") if args.api_base else "/api/judgments"

    html = html.replace(SAMPLE_MARKER, json.dumps(payload, ensure_ascii=False))
    html = html.replace(ROSTER_MARKER, json.dumps(roster, ensure_ascii=False))
    html = html.replace(
        SCRIPT_MARKER,
        f'<script>window.__T0_API__ = {json.dumps(api)};</script>\n{SCRIPT_MARKER}',
        1,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[ui] {out.relative_to(ROOT)} · 문항 {payload['meta']['n_items']}개 · "
          f"명단 {len(roster)}명 · API {api}")


if __name__ == "__main__":
    main()
