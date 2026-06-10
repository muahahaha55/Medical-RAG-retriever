"""Bộ mã hoá embedding — wrapper quanh sentence-transformers.

Hỗ trợ cả model baseline (BGE/BioBERT) lẫn model fine-tuned, vì cả hai
đều load được bằng SentenceTransformer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from medrag.config import Config, CONFIG
from medrag.utils.io import get_logger

logger = get_logger("medrag.embeddings")


class EmbeddingModel:
    """Mã hoá câu/đoạn văn thành vector dày (dense)."""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        config: Config = CONFIG,
    ):
        self.cfg = config
        r = config.raw.get("retriever", {})
        self.normalize = bool(r.get("normalize_embeddings", True))
        self.batch_size = int(r.get("embedding_batch_size", 64))
        self.max_seq_length = int(r.get("max_seq_length", 384))
        self.model_name = model_name_or_path or self._resolve_active_model()
        self._model = None

    def _resolve_active_model(self) -> str:
        """Chọn model theo retriever.active trong config."""
        r = self.cfg.raw.get("retriever", {})
        active = r.get("active", "baseline")
        if active == "finetuned":
            return r.get("finetuned_model_path", "data/models/MedicalRetriever-v1")
        return r.get("baseline_model", "BAAI/bge-small-en-v1.5")

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Sequence[str],
        show_progress: bool = False,
        is_query: bool = False,
    ) -> np.ndarray:
        """Trả về ma trận float32 shape (n, dim)."""
        prepared = list(texts)
        # BGE khuyến nghị thêm instruction cho query
        if is_query and "bge" in self.model_name.lower():
            prepared = [
                f"Represent this query for searching relevant medical passages: {t}"
                for t in prepared
            ]
        emb = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)
