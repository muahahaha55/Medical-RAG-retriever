"""Thu thập abstract từ PubMed qua NCBI E-utilities (esearch + efetch).

Chỉ dùng `requests`, không phụ thuộc Biopython để giữ deps gọn nhẹ.
Tôn trọng rate limit: 3 req/s nếu không có API key, 10 req/s nếu có.

Kết quả ghi ra data/raw/pubmed_abstracts.jsonl với schema:
    {"pmid": "...", "title": "...", "abstract": "..."}
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import requests

from medrag.config import Config, CONFIG
from medrag.utils.io import get_logger, write_jsonl

logger = get_logger("medrag.collector")

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass
class PubMedRecord:
    pmid: str
    title: str
    abstract: str

    def to_dict(self) -> dict[str, str]:
        return {"pmid": self.pmid, "title": self.title, "abstract": self.abstract}


class PubMedCollector:
    """Thu thập PMID rồi fetch abstract theo lô."""

    def __init__(self, config: Config = CONFIG):
        self.cfg = config
        dc = config.raw.get("data_collection", {})
        self.api_key: str | None = dc.get("ncbi_api_key")
        self.email: str = dc.get("email", "anonymous@example.com")
        self.batch_size: int = int(dc.get("batch_size", 200))
        # NCBI: 10 req/s nếu có key, 3 req/s nếu không
        self._delay = 0.11 if self.api_key else 0.34

    # -- low level ---------------------------------------------------------
    def _params(self, extra: dict) -> dict:
        base = {"db": "pubmed", "email": self.email}
        if self.api_key:
            base["api_key"] = self.api_key
        base.update(extra)
        return base

    def _sleep(self) -> None:
        time.sleep(self._delay)

    def search_pmids(self, term: str, retmax: int) -> list[str]:
        """esearch: lấy danh sách PMID cho một truy vấn."""
        pmids: list[str] = []
        retstart = 0
        page = min(retmax, 9999)  # esearch giới hạn 9999/lần
        while len(pmids) < retmax:
            params = self._params(
                {
                    "term": term,
                    "retmax": min(page, retmax - len(pmids)),
                    "retstart": retstart,
                    "retmode": "json",
                }
            )
            resp = requests.get(ESEARCH_URL, params=params, timeout=30)
            resp.raise_for_status()
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                break
            pmids.extend(ids)
            retstart += len(ids)
            self._sleep()
            logger.info("term=%r thu được %d PMID (tổng %d)", term, len(ids), len(pmids))
        return pmids[:retmax]

    def fetch_abstracts(self, pmids: list[str]) -> Iterator[PubMedRecord]:
        """efetch: lấy title + abstract theo lô PMID."""
        for i in range(0, len(pmids), self.batch_size):
            batch = pmids[i : i + self.batch_size]
            params = self._params(
                {"id": ",".join(batch), "rettype": "abstract", "retmode": "xml"}
            )
            resp = requests.get(EFETCH_URL, params=params, timeout=60)
            resp.raise_for_status()
            yield from self._parse_efetch_xml(resp.text)
            self._sleep()
            logger.info("Đã fetch %d/%d PMID", min(i + self.batch_size, len(pmids)), len(pmids))

    @staticmethod
    def _parse_efetch_xml(xml_text: str) -> Iterator[PubMedRecord]:
        """Trích PMID/title/abstract từ XML của efetch."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:  # pragma: no cover
            logger.warning("Lỗi parse XML: %s", e)
            return
        for art in root.findall(".//PubmedArticle"):
            pmid_el = art.find(".//PMID")
            title_el = art.find(".//ArticleTitle")
            # Abstract có thể gồm nhiều đoạn (AbstractText với label)
            abstract_parts = [
                (el.text or "").strip() for el in art.findall(".//AbstractText")
            ]
            abstract = " ".join(p for p in abstract_parts if p)
            if pmid_el is None or not abstract:
                continue
            yield PubMedRecord(
                pmid=pmid_el.text or "",
                title=(title_el.text or "").strip() if title_el is not None else "",
                abstract=abstract,
            )

    # -- high level --------------------------------------------------------
    def collect(self, output_path: str | Path | None = None) -> Path:
        """Chạy toàn bộ pipeline thu thập theo config và ghi ra JSONL."""
        dc = self.cfg.raw.get("data_collection", {})
        terms: list[str] = dc.get("search_terms", [])
        max_total: int = int(dc.get("max_abstracts", 10000))
        per_term = max(1, max_total // max(1, len(terms)))

        out = Path(output_path) if output_path else self.cfg.path("paths.data_raw") / "pubmed_abstracts.jsonl"

        seen: set[str] = set()
        records: list[dict] = []
        for term in terms:
            pmids = self.search_pmids(term, per_term)
            for rec in self.fetch_abstracts(pmids):
                if rec.pmid in seen:
                    continue
                seen.add(rec.pmid)
                records.append(rec.to_dict())
                if len(records) >= max_total:
                    break
            if len(records) >= max_total:
                break

        n = write_jsonl(records, out)
        logger.info("Đã ghi %d abstract vào %s", n, out)
        return out
