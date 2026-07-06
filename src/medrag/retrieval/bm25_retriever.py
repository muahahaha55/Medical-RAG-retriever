"""BM25 retriever (sparse) dùng thư viện bm25s.

Bổ sung cho dense retriever (FAISS): BM25 giỏi khớp từ khóa chính xác
(tên bệnh, thuốc, mã, viết tắt) — thứ embedding đôi khi bỏ lỡ. Kết hợp
hai loại bằng Reciprocal Rank Fusion (xem hybrid_retriever.py).

Index BM25 lưu song song với FAISS trong cùng thư mục index_dir:
    <index_dir>/bm25/           (bm25s.BM25.save)
    <index_dir>/bm25_meta.jsonl (metadata theo đúng thứ tự doc)
"""
from __future__ import annotations

import json
from pathlib import Path

from medrag.config import Config, CONFIG
from medrag.utils.io import get_logger

logger = get_logger("medrag.bm25")


def _tokenize(texts: list[str]):
    import bm25s
    return bm25s.tokenize(texts, stopwords="en", show_progress=False)


class BM25Retriever:
    def __init__(self, config: Config = CONFIG):
        self.cfg = config
        self.retriever = None
        self.metadata: list[dict] = []

    # -- build / persist ---------------------------------------------------
    def build(self, texts: list[str], metadata: list[dict]) -> None:
        import bm25s

        assert len(texts) == len(metadata), "Lệch số lượng text và metadata"
        logger.info("Build BM25 index trên %d document ...", len(texts))
        corpus_tokens = _tokenize(texts)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens, show_progress=False)
        self.metadata = metadata

    def save(self, index_dir: str | Path | None = None) -> Path:
        d = Path(index_dir) if index_dir else self.cfg.path("paths.index_dir")
        d.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(d / "bm25"))
        with open(d / "bm25_meta.jsonl", "w", encoding="utf-8") as fh:
            for m in self.metadata:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        logger.info("Đã lưu BM25 index vào %s", d)
        return d

    def load(self, index_dir: str | Path | None = None) -> "BM25Retriever":
        import bm25s

        d = Path(index_dir) if index_dir else self.cfg.path("paths.index_dir")
        self.retriever = bm25s.BM25.load(str(d / "bm25"), load_corpus=False)
        self.metadata = []
        with open(d / "bm25_meta.jsonl", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    self.metadata.append(json.loads(line))
        logger.info("Đã nạp BM25 index %d doc từ %s", len(self.metadata), d)
        return self

    # -- search ------------------------------------------------------------
    def search(self, query: str, top_k: int = 50) -> list[dict]:
        """Trả về top_k metadata kèm 'bm25_rank' (1-based) và 'bm25_score'."""
        import bm25s

        if self.retriever is None:
            raise RuntimeError("BM25 index chưa build/load")
        q_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        k = min(top_k, len(self.metadata))
        results, scores = self.retriever.retrieve(q_tokens, k=k, show_progress=False)
        out = []
        for rank, (idx, score) in enumerate(zip(results[0], scores[0]), start=1):
            item = dict(self.metadata[int(idx)])
            item["bm25_rank"] = rank
            item["bm25_score"] = float(score)
            out.append(item)
        return out