"""Retriever cấp cao: ghép EmbeddingModel + FaissIndex.

Cung cấp 2 thao tác chính:
  - index_corpus(): nhúng toàn bộ chunk và xây FAISS index.
  - retrieve(query): mã hoá query và trả về top-k chunk liên quan.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from medrag.config import Config, CONFIG
from medrag.retrieval.embeddings import EmbeddingModel
from medrag.retrieval.faiss_index import FaissIndex
from medrag.utils.io import get_logger, read_jsonl

logger = get_logger("medrag.retriever")


class Retriever:
    def __init__(
        self,
        config: Config = CONFIG,
        model_name_or_path: str | None = None,
    ):
        self.cfg = config
        self.embedder = EmbeddingModel(model_name_or_path, config=config)
        self.index = FaissIndex(config=config)
        self._loaded = False

    # -- indexing ----------------------------------------------------------
    def index_corpus(
        self,
        chunks_path: str | Path,
        index_dir: str | Path | None = None,
    ) -> Path:
        """Đọc file chunks .jsonl, nhúng và xây FAISS index."""
        records = list(read_jsonl(chunks_path))
        texts = [r["chunk"] for r in records]
        logger.info("Bắt đầu nhúng %d chunk ...", len(texts))
        embeddings = self.embedder.encode(texts, show_progress=True, is_query=False)
        self.index.build(embeddings, records)
        out = self.index.save(index_dir)
        self._loaded = True
        return out

    # -- loading -----------------------------------------------------------
    def load(self, index_dir: str | Path | None = None) -> "Retriever":
        self.index.load(index_dir)
        self._loaded = True
        return self

    # -- retrieval ---------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        if not self._loaded:
            self.load()
        if top_k is None:
            top_k = int(self.cfg.get("reranker.retrieve_top_k", 50))
        q_emb = self.embedder.encode([query], is_query=True)[0]
        return self.index.search(q_emb, top_k=top_k)

    def batch_retrieve(self, queries: Iterable[str], top_k: int | None = None) -> list[list[dict]]:
        return [self.retrieve(q, top_k=top_k) for q in queries]
