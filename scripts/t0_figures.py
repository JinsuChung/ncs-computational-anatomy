"""T0 그림 F1–F3 — 실측 데이터에서 SVG 로 직접 그린다 (외부 라이브러리·폰트 의존 없음).

  F1  3축 지도: 대분류 24 × (평균 개정 연도, L5+ 비율, 디지털 어휘) — 산점 + 버블
  F2  직무 군집 vs 공식 분류: 대분류 × Louvain 군집 점유율 히트맵 (H4)
  F3  구조화 가능성 지도: 대분류 × (템플릿 준수, 고유 비율, 파편화 지수, 유형 불일치) 히트맵

입력  docs/t0/t0_metrics.csv · docs/t0/t0_h4_clusters.csv · DB(파편화·유형 불일치는 대분류별 재계산)
출력  docs/t0/figures/F1_three_axis.svg · F2_cluster_vs_major.svg · F3_structurability.svg
      docs/t0/t0_major_aggregates.json (그림에 쓴 집계 — 재현·표 작성용)

사용: .venv/bin/python scripts/t0_figures.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ontology_keys import normalize_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/t0/figures"
TEAL, TEAL_DEEP, AMBER, INK, MUTED, LINE, PAPER = "#0E6E6D", "#0A5352", "#B07A18", "#1F2A2E", "#5C6B6E", "#DEE4DF", "#FFFFFF"
FONT = "'IBM Plex Sans KR','Apple SD Gothic Neo','Noto Sans KR',sans-serif"
MONO = "'IBM Plex Mono',ui-monospace,Menlo,monospace"

SHORT = {  # 라벨용 축약
    "법률·경찰·소방·교도·국방": "법률·경찰·소방", "이용·숙박·여행·오락·스포츠": "이용·숙박·여행",
    "문화·예술·디자인·방송": "문화·예술·방송", "인쇄·목재·가구·공예": "인쇄·목재·공예",
    "교육·자연·사회과학": "교육·자연·사회", "환경·에너지·안전": "환경·에너지",
}


# 영문 저널 제출용 라벨. NCS 대분류 24개의 영문명은 공식 표기 확인 필요(M11).
EN_MAJOR = {
    "사업관리": "Business Management", "경영·회계·사무": "Mgmt., Accounting & Office", "금융·보험": "Finance & Insurance",
    "교육·자연·사회과학": "Education, Nat. & Soc. Sci.", "법률·경찰·소방·교도·국방": "Law, Police, Fire & Defense",
    "보건·의료": "Health & Medical", "사회복지·종교": "Social Welfare & Religion", "문화·예술·디자인·방송": "Culture, Arts & Media",
    "운전·운송": "Driving & Transport", "영업판매": "Sales", "경비·청소": "Security & Cleaning",
    "이용·숙박·여행·오락·스포츠": "Beauty, Lodging, Travel & Sports", "음식서비스": "Food Service", "건설": "Construction",
    "기계": "Machinery", "재료": "Materials", "화학·바이오": "Chemistry & Bio", "섬유·의복": "Textiles & Apparel",
    "전기·전자": "Electricity & Electronics", "정보통신": "ICT", "식품가공": "Food Processing",
    "인쇄·목재·가구·공예": "Printing, Wood & Crafts", "환경·에너지·안전": "Environment, Energy & Safety", "농림어업": "Agri., Forestry & Fisheries",
}
LANG = "ko"
T = {  # 그림 안 문자열 — ko 원문 → en
    "그림 1. 대분류별 개정 연도·수준·디지털 어휘": "Figure 1. Revision year, level and digital vocabulary by major category",
    "평균 개발·개선 연도 (능력단위 코드 접미, 능력단위 수 가중)": "Mean development/revision year (unit-code suffix, weighted by units)",
    "L5 이상 능력단위 비율": "Share of competency units at L5 or above",
    "버블 크기 = KSA 중 디지털 어휘 비율": "Bubble size = share of KSA items with digital vocabulary",
    "그림 1. 대분류 24개의 개정 연도 × 수준 × 디지털 어휘 (원천 실측, 2026-02 릴리스)": "Figure 1. 24 major categories: revision year × level × digital vocabulary (census, 2026-02 release)",
    "그림 2. 공식 대분류 × 데이터 기반 군집": "Figure 2. Official major categories × data-driven communities",
    "그림 2. 공식 대분류(행) × 공유 KSA 기반 Louvain 군집(열) — 각 행은 그 대분류 직무의 군집 분포": "Figure 2. Official major categories (rows) × Louvain communities of the shared-KSA graph (columns) — each row is that category's distribution over communities",
    "군집 크기": "community size",
    "셀 = 해당 대분류 직무 중 그 군집에 속한 비율(%). 한 행이 한 열에 몰릴수록 공식 분류와 데이터 군집이 일치한다.": "Cell = share (%) of the category's occupations in that community. The more a row concentrates in one column, the better the official category and the data agree.",
    "그림 3. 구조화 가능성 지도": "Figure 3. Structurability map",
    "그림 3. 구조화 가능성 지도 — 대분류 × 기계 처리 특성 (파편화 지수 내림차순)": "Figure 3. Structurability map — major categories × machine-processing properties (sorted by fragmentation index)",
    "색은 열 안에서 상대적 위치(min–max). 템플릿 준수는 높을수록, 고유 비율·파편화·유형 불일치는 낮을수록 규칙 처리에 유리.": "Colour is the relative position within each column (min–max). Higher template compliance, and lower uniqueness, fragmentation and type mismatch, favour rule-based processing.",
    "수행준거\n템플릿 준수": "Template\ncompliance",
    "KSA\n고유 비율": "KSA\nuniqueness",
    "파편화 지수\n(표면형/키)": "Fragmentation\n(surfaces/keys)",
    "유형 불일치\n(K/S 혼재 키)": "Type mismatch\n(K/S mixed keys)",
    "진한 셀 = 그 열에서 규칙 기반 처리에 상대적으로 유리. 파편화 지수는 규칙 키 단계(임베딩 이전)의 값이다.": "Darker cell = relatively more amenable to rule-based processing within the column. Fragmentation index is the rule-key stage (before embedding).",
}


def tr_(s_):
    return T.get(s_, s_) if LANG == "en" else s_


def short(m):
    return EN_MAJOR.get(m, m) if LANG == "en" else SHORT.get(m, m)


def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ───────────────────────────────────────── 데이터

def load_metrics():
    rows = list(csv.DictReader(open(ROOT / "docs/t0/t0_metrics.csv", encoding="utf-8")))
    for r in rows:
        for k in ("n_units", "n_pc", "n_ksa", "n_ksa_unique"):
            r[k] = int(float(r[k])) if r[k] else 0
        for k in ("mean_level", "rate_l5plus", "mean_rev_year", "template_rate", "unique_rate", "digital_a"):
            r[k] = float(r[k]) if r[k] else None
    return rows


def major_aggregates(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["major"]].append(r)
    agg = {}
    for m, rs in by.items():
        units = sum(r["n_units"] for r in rs)
        agg[m] = {
            "n_jobs": len(rs), "n_units": units,
            "rate_l5plus": float(np.average([r["rate_l5plus"] for r in rs if r["rate_l5plus"] is not None],
                                            weights=[r["n_units"] for r in rs if r["rate_l5plus"] is not None])),
            "mean_rev_year": float(np.average([r["mean_rev_year"] for r in rs if r["mean_rev_year"] is not None],
                                              weights=[r["n_units"] for r in rs if r["mean_rev_year"] is not None])),
            "digital": float(np.average([r["digital_a"] for r in rs if r["digital_a"] is not None],
                                        weights=[r["n_ksa"] for r in rs if r["digital_a"] is not None])),
            "template_rate": float(np.average([r["template_rate"] for r in rs if r["template_rate"] is not None],
                                              weights=[r["n_pc"] for r in rs if r["template_rate"] is not None])),
            "unique_rate": sum(r["n_ksa_unique"] for r in rs) / sum(r["n_ksa"] for r in rs),
        }
    return agg


FRAG_SQL = """
SELECT DISTINCT cp.major_category_name AS major, ki.kta_type_code AS ktype, ki.description
FROM ncs.kta_items ki
JOIN ncs.performance_criteria p ON p.id = ki.performance_criterion_id
JOIN ncs.competency_unit_elements e ON e.code = p.competency_unit_element_code
JOIN ncs.competency_units cu ON cu.code = e.competency_unit_code
JOIN ncs.classification_paths cp ON cp.detail_category_full_code = cu.detail_category_full_code
"""


def fragmentation_by_major(dsn):
    """대분류별 표면형 수 / 규칙 키 수 (파편화 지수) 와 유형 불일치율."""
    surf = defaultdict(set); keys = defaultdict(set); ktypes = defaultdict(lambda: defaultdict(set))
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as c0:
            c0.execute("SET work_mem = '1GB'")
        with conn.cursor(name="frag") as cur:
            cur.itersize = 50000
            cur.execute(FRAG_SQL)
            for major, ktype, desc in cur:
                d = unicodedata.normalize("NFKC", (desc or "").strip())
                if not d:
                    continue
                k = normalize_key(d)
                if not k:
                    continue
                surf[major].add(d); keys[major].add(k); ktypes[major][k].add(ktype)
    out = {}
    for m in surf:
        mism = sum(1 for k, t in ktypes[m].items() if len(t) > 1)
        out[m] = {"n_surface": len(surf[m]), "n_keys": len(keys[m]),
                  "fragmentation_index": len(surf[m]) / len(keys[m]),
                  "type_mismatch_rate": mism / len(keys[m])}
    return out


# ───────────────────────────────────────── SVG 헬퍼

def svg_open(w, h, title):
    title = tr_(title)
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{esc(title)}" style="font-family:{FONT};background:{PAPER}">',
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="{PAPER}"/>']


def text(x, y, s, size=12, fill=INK, anchor="start", weight="normal", mono=False, rotate=None, dy=None):
    s = tr_(s)
    fam = f' font-family="{MONO}"' if mono else ""
    tr = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    d = f' dy="{dy}"' if dy is not None else ""
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{fam}{tr}{d}>{esc(s)}</text>'


def teal_scale(v):
    """0..1 → 연한 청록 ~ 진한 청록."""
    v = max(0.0, min(1.0, v))
    r0, g0, b0 = 0xEA, 0xF3, 0xF1
    r1, g1, b1 = 0x0A, 0x53, 0x52
    r = int(r0 + (r1 - r0) * v); g = int(g0 + (g1 - g0) * v); b = int(b0 + (b1 - b0) * v)
    return f"#{r:02X}{g:02X}{b:02X}"


def contrast_text(v): return PAPER if v > 0.55 else INK


# ───────────────────────────────────────── F1

def fig1(agg):
    W, H = 920, 560
    ml, mr, mt, mb = 70, 40, 40, 60
    x0, x1 = 2014.5, 2023.0
    y0, y1 = 0.0, 0.8
    def X(v): return ml + (v - x0) / (x1 - x0) * (W - ml - mr)
    def Y(v): return H - mb - (v - y0) / (y1 - y0) * (H - mt - mb)
    S = svg_open(W, H, "그림 1. 대분류별 개정 연도·수준·디지털 어휘")
    # 격자
    for yr in range(2015, 2023):
        S.append(f'<line x1="{X(yr)}" y1="{mt}" x2="{X(yr)}" y2="{H-mb}" stroke="{LINE}" stroke-width="1"/>')
        S.append(text(X(yr), H - mb + 18, str(yr), 11, MUTED, "middle", mono=True))
    for p in (0, 0.2, 0.4, 0.6, 0.8):
        S.append(f'<line x1="{ml}" y1="{Y(p)}" x2="{W-mr}" y2="{Y(p)}" stroke="{LINE}" stroke-width="1"/>')
        S.append(text(ml - 8, Y(p) + 4, f"{int(p*100)}%", 11, MUTED, "end", mono=True))
    S.append(text((ml + W - mr) / 2, H - 18, "평균 개발·개선 연도 (능력단위 코드 접미, 능력단위 수 가중)", 12, MUTED, "middle"))
    S.append(text(18, (mt + H - mb) / 2, "L5 이상 능력단위 비율", 12, MUTED, "middle", rotate=-90))
    # 버블
    dmax = max(a["digital"] for a in agg.values())
    items = sorted(agg.items(), key=lambda kv: -kv[1]["n_units"])
    for i, (m, a) in enumerate(items):
        r = 6 + 22 * (a["digital"] / dmax) ** 0.5
        cx, cy = X(a["mean_rev_year"]), Y(a["rate_l5plus"])
        S.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{TEAL}" fill-opacity="0.55" stroke="{TEAL_DEEP}" stroke-width="1"/>')
    # 라벨 — 오른쪽 절반의 점은 왼쪽에 라벨을 두어 캔버스 밖으로 나가지 않게 하고,
    # 같은 쪽 라벨끼리 겹치면 자리가 날 때까지 아래로 민다.
    placed = []
    xmid = (ml + W - mr) / 2
    for m, a in sorted(agg.items(), key=lambda kv: (kv[1]["rate_l5plus"], kv[1]["mean_rev_year"])):
        cx, cy = X(a["mean_rev_year"]), Y(a["rate_l5plus"])
        r = 6 + 22 * (a["digital"] / dmax) ** 0.5
        side = -1 if cx > xmid else 1
        lx, ly = cx + side * (r + 6), cy + 4
        moved = True
        while moved:
            moved = False
            for (px, py, ps) in placed:
                if ps == side and abs(px - lx) < 110 and abs(py - ly) < 13:
                    ly += 13; moved = True
        placed.append((lx, ly, side))
        S.append(text(lx, ly, short(m), 11, INK, "start" if side > 0 else "end"))
    # 범례
    lx, ly = W - mr - 200, mt + 6
    S.append(text(lx, ly + 4, "버블 크기 = KSA 중 디지털 어휘 비율", 11, MUTED))
    for k, (v, lab) in enumerate([(0.005, "0.5%"), (0.03, "3%"), (0.10, "10%")]):
        r = 6 + 22 * (v / dmax) ** 0.5
        S.append(f'<circle cx="{lx + 12 + k*58}" cy="{ly + 30}" r="{r:.1f}" fill="{TEAL}" fill-opacity="0.55" stroke="{TEAL_DEEP}"/>')
        S.append(text(lx + 12 + k * 58, ly + 58, lab, 10, MUTED, "middle", mono=True))
    S.append(text(ml, 22, "그림 1. 대분류 24개의 개정 연도 × 수준 × 디지털 어휘 (원천 실측, 2026-02 릴리스)", 13, INK, weight="bold"))
    S.append("</svg>")
    return "\n".join(S)


# ───────────────────────────────────────── F2

def fig2(clusters_csv, agg):
    rows = list(csv.DictReader(open(clusters_csv, encoding="utf-8")))
    cnt = defaultdict(Counter); tot = Counter()
    for r in rows:
        cnt[r["major"]][int(r["cluster"])] += 1; tot[r["major"]] += 1
    clus_size = Counter(int(r["cluster"]) for r in rows)
    cols = [c for c, _ in clus_size.most_common()]
    majors = sorted(cnt, key=lambda m: -tot[m])
    cw, rh = 46, 19
    ml, mt = 190, 96
    W = ml + cw * len(cols) + 40
    H = mt + rh * len(majors) + 60
    S = svg_open(W, H, "그림 2. 공식 대분류 × 데이터 기반 군집")
    S.append(text(16, 24, "그림 2. 공식 대분류(행) × 공유 KSA 기반 Louvain 군집(열) — 각 행은 그 대분류 직무의 군집 분포", 13, INK, weight="bold"))
    S.append(text(16, 42, (f"NMI (major) = .329 · permutation null .041 ± .004 · {len(cols)} communities · {len(rows):,} occupations" if LANG == "en"
                           else f"NMI(대분류) = .329 · 순열 영분포 .041 ± .004 · 군집 {len(cols)}개 · 직무 {len(rows):,}"), 11, MUTED))
    for j, c in enumerate(cols):
        S.append(text(ml + j * cw + cw / 2, mt - 8, f"C{c}", 10, MUTED, "middle", mono=True))
        S.append(text(ml + j * cw + cw / 2, mt - 22, f"{clus_size[c]}", 9, MUTED, "middle", mono=True))
    S.append(text(ml - 6, mt - 22, "군집 크기", 9, MUTED, "end"))
    for i, m in enumerate(majors):
        y = mt + i * rh
        S.append(text(ml - 8, y + 13, f"{short(m)} ({tot[m]})", 11, INK, "end"))
        for j, c in enumerate(cols):
            share = cnt[m][c] / tot[m]
            S.append(f'<rect x="{ml + j*cw}" y="{y}" width="{cw-1}" height="{rh-1}" fill="{teal_scale(share)}"/>')
            if share >= 0.15:
                S.append(text(ml + j * cw + cw / 2, y + 13, f"{int(round(share*100))}", 9.5, contrast_text(share), "middle", mono=True))
    S.append(text(16, H - 22, "셀 = 해당 대분류 직무 중 그 군집에 속한 비율(%). 한 행이 한 열에 몰릴수록 공식 분류와 데이터 군집이 일치한다.", 10.5, MUTED))
    S.append("</svg>")
    return "\n".join(S)


# ───────────────────────────────────────── F3

def fig3(agg, frag):
    # higher_good = 값이 클수록 규칙 기반 처리에 유리한가.
    # 고유 비율은 낮을수록(=중복이 많을수록) 빈도 기반 규칙이 잘 먹히므로 False.
    metrics = [("template_rate", "수행준거\n템플릿 준수", lambda v: f"{v*100:.1f}%", True),
               ("unique_rate", "KSA\n고유 비율", lambda v: f"{v*100:.1f}%", False),
               ("fragmentation_index", "파편화 지수\n(표면형/키)", lambda v: f"{v:.3f}", False),
               ("type_mismatch_rate", "유형 불일치\n(K/S 혼재 키)", lambda v: f"{v*100:.1f}%", False)]
    for m in agg:
        agg[m].update(frag.get(m, {}))
    majors = sorted(agg, key=lambda m: -agg[m]["fragmentation_index"])
    cw, rh = 118, 19
    ml, mt = 190, 78
    W = ml + cw * len(metrics) + 48
    H = mt + rh * len(majors) + 70
    S = svg_open(W, H, "그림 3. 구조화 가능성 지도")
    S.append(text(16, 24, "그림 3. 구조화 가능성 지도 — 대분류 × 기계 처리 특성 (파편화 지수 내림차순)", 13, INK, weight="bold"))
    S.append(text(16, 42, "색은 열 안에서 상대적 위치(min–max). 템플릿 준수는 높을수록, 고유 비율·파편화·유형 불일치는 낮을수록 규칙 처리에 유리.", 11, MUTED))
    vals = {k: [agg[m][k] for m in majors] for k, *_ in metrics}
    for j, (k, lab, fmt, higher_good) in enumerate(metrics):
        x = ml + j * cw
        for li, line in enumerate(tr_(lab).split("\n")):
            S.append(text(x + cw / 2, mt - 24 + li * 13, line, 10.5, MUTED, "middle"))
        lo, hi = min(vals[k]), max(vals[k])
        for i, m in enumerate(majors):
            v = agg[m][k]
            rel = (v - lo) / (hi - lo) if hi > lo else 0.5
            shade = rel if higher_good else 1 - rel   # 진할수록 '규칙에 유리'
            y = mt + i * rh
            S.append(f'<rect x="{x}" y="{y}" width="{cw-2}" height="{rh-1}" fill="{teal_scale(shade)}"/>')
            S.append(text(x + cw / 2, y + 13, fmt(v), 10, contrast_text(shade), "middle", mono=True))
    for i, m in enumerate(majors):
        S.append(text(ml - 8, mt + i * rh + 13, f"{short(m)} ({agg[m]['n_jobs']})", 11, INK, "end"))
    S.append(text(16, H - 26, "진한 셀 = 그 열에서 규칙 기반 처리에 상대적으로 유리. 파편화 지수는 규칙 키 단계(임베딩 이전)의 값이다.", 10.5, MUTED))
    S.append("</svg>")
    return "\n".join(S)


def main():
    import argparse
    global LANG
    ap = argparse.ArgumentParser(); ap.add_argument("--lang", default="ko", choices=["ko", "en"]); args = ap.parse_args()
    LANG = args.lang
    suf = "_en" if LANG == "en" else ""
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_metrics()
    agg = major_aggregates(rows)
    print(f"[fig] 대분류 {len(agg)} · 파편화 지수 계산(DB) …", flush=True)
    frag = fragmentation_by_major(os.environ.get("DATABASE_URL", "postgresql://ncs:ncs@localhost:55432/ncs"))
    (OUT / f"F1_three_axis{suf}.svg").write_text(fig1(agg), encoding="utf-8")
    (OUT / f"F2_cluster_vs_major{suf}.svg").write_text(fig2(ROOT / "docs/t0/t0_h4_clusters.csv", agg), encoding="utf-8")
    (OUT / f"F3_structurability{suf}.svg").write_text(fig3(agg, frag), encoding="utf-8")
    (ROOT / "docs/t0/t0_major_aggregates.json").write_text(
        json.dumps({m: {**agg[m], **frag.get(m, {})} for m in agg}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[fig] 완료 → docs/t0/figures/F1_three_axis.svg · F2_cluster_vs_major.svg · F3_structurability.svg")
    top = sorted(agg.items(), key=lambda kv: -kv[1]["fragmentation_index"])
    print("  파편화 지수 상위 3:", [(short(m), round(a["fragmentation_index"], 3)) for m, a in top[:3]],
          "· 하위 3:", [(short(m), round(a["fragmentation_index"], 3)) for m, a in top[-3:]])


if __name__ == "__main__":
    main()
