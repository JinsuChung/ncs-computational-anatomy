"""개념 라벨 정규화 키 — 추출(extraction) 결과와 규칙 후보(preprocess)를 같은 키로 맞추기 위한 공용 모듈.

preprocess_text.py 의 variant_key(띄어쓰기 제거)와 kta_core_concepts 의 역할 접미사 제거를 한 함수로 합친 것.
정렬(align_extractions.py)과 task 내보내기(export_extraction_tasks.py)가 같은 키를 쓰도록 여기 한 곳에만 둔다.
"""

from __future__ import annotations

import re
import unicodedata

# KTA 원문에서 붙는 역할 접미사. "…에 대한 지식", "…능력", "…기술" 을 떼어 개념 핵을 남긴다.
ROLE_SUFFIX_PATTERNS = [
    r"에\s*(?:대한|관한)\s*(?:지식|이해|기술|능력|태도|방법)$",
    r"(?:지식|이해|기술|능력|태도|방법|역량|스킬|기법|노하우)$",
]

_PUNCT = re.compile(r"[\s·•·\-–—_/\\(){}\[\]<>,.:;!?'\"“”‘’`~^*+=|]+")


def strip_role_suffix(text: str) -> str:
    out = text.strip()
    for pattern in ROLE_SUFFIX_PATTERNS:
        stripped = re.sub(pattern, "", out).strip()
        # 접미사만으로 된 라벨("능력")은 그대로 둔다 — 빈 키 방지
        if stripped and stripped != out:
            out = stripped
    return out


def normalize_key(text: str) -> str:
    """NFKC → 역할 접미사 제거 → 공백·구두점 제거 → 소문자. 빈 결과면 원문 정규화만 적용."""
    if text is None:
        return ""
    base = unicodedata.normalize("NFKC", str(text)).strip()
    core = strip_role_suffix(base)
    key = _PUNCT.sub("", core).lower()
    if not key:
        key = _PUNCT.sub("", base).lower()
    return key


def normalize_ws(text: str) -> str:
    """축자 인용 대조용: 공백 전부 제거 + NFKC."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))
