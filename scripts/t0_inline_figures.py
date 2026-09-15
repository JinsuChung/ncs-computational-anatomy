"""논문 초안(paper-draft-t0a.html)에 F1–F3 SVG 를 인라인으로 심는다.

아티팩트 CSP 는 외부 이미지를 막으므로 <img src> 대신 SVG 를 본문에 직접 넣는다.
그림 자리는 주석 마커 <!--FIG:F1--> … <!--/FIG:F1--> 사이이며, 재실행하면 그 구간만 교체된다
(마커가 없으면 최초 삽입 지점을 앵커 문자열로 찾는다).

사용: .venv/bin/python scripts/t0_inline_figures.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper-draft-t0a.html"
FIG = ROOT / "docs/t0/figures"

FIGS = {
    "F1": {
        "file": "F1_three_axis.svg",
        "caption": "Figure 1. Revision recency, competency level, and digital vocabulary by major category "
                   "(24 categories; bubble area ∝ share of KSA items matching lexicon A). Census data, 2026-02 release.",
        # 최초 삽입: 기존 초안 그림 블록(막대) 전체를 교체
        "anchor_start": '  <figure>\n    <div class="bar"><span class="lab">금융·보험</span>',
        "anchor_end": '</figcaption>\n  </figure>',
    },
    "F2": {
        "file": "F2_cluster_vs_major.svg",
        "caption": "Figure 2. Official major categories (rows) against Louvain communities of the shared-KSA occupation graph "
                   "(columns; column headers give community size). Cells are the share of a category's occupations "
                   "falling in each community; NMI = .329 against a permutation null of .041 ± .004.",
        "anchor_after": '<span class="ph">불일치 직무 전체 목록과 유형 분류<sup>F2</sup></span>\n  </p>',
    },
    "F3": {
        "file": "F3_structurability.svg",
        "caption": "Figure 3. Structurability map: major categories by template compliance, KSA uniqueness, "
                   "rule-stage fragmentation index, and K/S type inconsistency. Darker cells are relatively more "
                   "amenable to rule-based processing within each column; rows sorted by fragmentation index.",
        "anchor_after": 'skill items. Manual precision of the merges is assessed in §4.6.\n  </p>',
    },
}


def responsive(svg: str) -> str:
    # 캔버스 크기는 viewBox 가 갖고, 표시 크기는 문서 폭을 따르게 한다
    svg = re.sub(r'\swidth="\d+"\sheight="\d+"', ' width="100%"', svg, count=1)
    return svg


def block(fid: str, svg: str, caption: str) -> str:
    return (f'<!--FIG:{fid}-->\n  <figure style="padding:10px 12px;overflow-x:auto">\n{svg}\n'
            f'    <figcaption>{caption}</figcaption>\n  </figure>\n<!--/FIG:{fid}-->')


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--lang", default="ko", choices=["ko", "en"]); args = ap.parse_args()
    suf = "_en" if args.lang == "en" else ""
    html = PAPER.read_text(encoding="utf-8")
    for fid, cfg in FIGS.items():
        svg = responsive((FIG / cfg["file"].replace(".svg", suf + ".svg")).read_text(encoding="utf-8"))
        new = block(fid, svg, cfg["caption"])
        pat = re.compile(rf"<!--FIG:{fid}-->.*?<!--/FIG:{fid}-->", re.S)
        if pat.search(html):
            html = pat.sub(lambda m: new, html)
            print(f"[fig] {fid} 교체")
            continue
        if "anchor_start" in cfg:
            s = html.find(cfg["anchor_start"])
            e = html.find(cfg["anchor_end"], s) + len(cfg["anchor_end"])
            if s < 0 or e < len(cfg["anchor_end"]):
                raise SystemExit(f"{fid}: 앵커를 찾지 못했다")
            html = html[:s] + new + html[e:]
        else:
            a = cfg["anchor_after"]
            if a not in html:
                raise SystemExit(f"{fid}: 앵커를 찾지 못했다")
            html = html.replace(a, a + "\n" + new, 1)
        print(f"[fig] {fid} 삽입")
    PAPER.write_text(html, encoding="utf-8")
    print(f"[fig] {PAPER.name} 갱신")


if __name__ == "__main__":
    main()
