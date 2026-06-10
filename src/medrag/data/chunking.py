"""Chia abstract thành các chunk theo số token (có overlap).

Dùng tokenizer của BioBERT để đếm token chính xác cho retriever.
Fallback sang đếm theo từ nếu transformers chưa cài (giúp test nhanh).

Mỗi chunk sinh ra một bản ghi:
    {"chunk_id", "pmid", "title", "chunk"}
"""
from __future__ import annotations

from typing import Iterable, Iterator

from medrag.config import Config, CONFIG
from medrag.utils.io import get_logger

logger = get_logger("medrag.chunking")


class Chunker:
    def __init__(self, config: Config = CONFIG):
        pp = config.raw.get("preprocessing", {})
        self.chunk_size = int(pp.get("chunk_size_tokens", 384))
        self.overlap = int(pp.get("chunk_overlap_tokens", 64))
        self.tokenizer_name = pp.get("tokenizer_for_chunking", "dmis-lab/biobert-base-cased-v1.2")
        self._tokenizer = None  # lazy load

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
                logger.info("Đã load tokenizer %s", self.tokenizer_name)
            except Exception as e:  # noqa: BLE001
                logger.warning("Không load được tokenizer (%s) -> dùng word split", e)
                self._tokenizer = False  # đánh dấu fallback
        return self._tokenizer

    def _split_tokens(self, text: str) -> list[str]:
        """Trả về danh sách 'token string' để ghép lại thành chunk."""
        tok = self.tokenizer
        if tok:  # transformers có sẵn
            ids = tok.encode(text, add_special_tokens=False)
            # cắt theo id rồi decode từng cửa sổ
            return ids  # type: ignore[return-value]
        return text.split()

    def _decode(self, units: list, tok) -> str:
        if tok:
            return tok.decode(units, skip_special_tokens=True)
        return " ".join(units)

    def chunk_text(self, text: str) -> list[str]:
        """Cắt 1 đoạn text thành nhiều chunk với overlap."""
        units = self._split_tokens(text)
        tok = self.tokenizer
        if len(units) <= self.chunk_size:
            return [self._decode(units, tok)]

        chunks: list[str] = []
        step = max(1, self.chunk_size - self.overlap)
        for start in range(0, len(units), step):
            window = units[start : start + self.chunk_size]
            if not window:
                break
            chunks.append(self._decode(window, tok))
            if start + self.chunk_size >= len(units):
                break
        return chunks

    def chunk_records(self, records: Iterable[dict]) -> Iterator[dict]:
        """Sinh chunk từ các bản ghi abstract đã làm sạch."""
        n_docs, n_chunks = 0, 0
        for rec in records:
            n_docs += 1
            pmid = rec["pmid"]
            title = rec.get("title", "")
            # Ghép title vào nội dung để tăng tín hiệu ngữ nghĩa
            body = f"{title}. {rec['abstract']}" if title else rec["abstract"]
            for j, chunk in enumerate(self.chunk_text(body)):
                if not chunk.strip():
                    continue
                n_chunks += 1
                yield {
                    "chunk_id": f"{pmid}_{j}",
                    "pmid": pmid,
                    "title": title,
                    "chunk": chunk,
                }
        logger.info("Chunking xong: %d docs -> %d chunks", n_docs, n_chunks)
