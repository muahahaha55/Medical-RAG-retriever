"""Làm sạch abstract: bỏ HTML, ký tự lạ, chuẩn hoá khoảng trắng, lọc rác."""
from __future__ import annotations

import html
import re
from typing import Iterable, Iterator

from medrag.config import Config, CONFIG
from medrag.utils.io import get_logger

logger = get_logger("medrag.preprocess")

_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_WS = re.compile(r"\s+")
# Giữ chữ-số, dấu câu y khoa thường gặp; bỏ ký tự điều khiển
_WEIRD = re.compile(r"[^\w\s.,;:%()/\-+°µ²³αβγ]")


def clean_text(text: str) -> str:
    """Chuẩn hoá một đoạn text."""
    if not text:
        return ""
    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    text = _WEIRD.sub(" ", text)
    text = _MULTI_WS.sub(" ", text)
    return text.strip()


def clean_records(
    records: Iterable[dict],
    config: Config = CONFIG,
) -> Iterator[dict]:
    """Làm sạch + lọc trùng + lọc abstract quá ngắn.

    records phải có khoá: pmid, title, abstract.
    """
    pp = config.raw.get("preprocessing", {})
    min_chars = int(pp.get("min_abstract_chars", 100))
    dedup = bool(pp.get("remove_duplicates", True))

    seen_pmids: set[str] = set()
    seen_hashes: set[int] = set()
    kept, dropped = 0, 0

    for rec in records:
        pmid = str(rec.get("pmid", "")).strip()
        title = clean_text(rec.get("title", ""))
        abstract = clean_text(rec.get("abstract", ""))

        if len(abstract) < min_chars:
            dropped += 1
            continue
        if dedup:
            if pmid and pmid in seen_pmids:
                dropped += 1
                continue
            h = hash(abstract)
            if h in seen_hashes:
                dropped += 1
                continue
            seen_pmids.add(pmid)
            seen_hashes.add(h)

        kept += 1
        yield {"pmid": pmid, "title": title, "abstract": abstract}

    logger.info("Cleaning xong: giữ %d, bỏ %d", kept, dropped)
