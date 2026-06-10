"""Tầng rerank dùng Cross-Encoder MiniLM.

Retriever trả về top-N (vd 50) candidate; reranker chấm điểm lại từng cặp
(query, document) và giữ lại top-k (vd 5) liên quan nhất.
"""
from __future__ import annotations

from medrag.config import Config, CONFIG
from medrag.utils.io import get_logger

logger = get_logger("medrag.reranker")


class Reranker:
    def __init__(self, config: Config = CONFIG):
        self.cfg = config
        rr = config.raw.get("reranker", {})
        self.enabled = bool(rr.get("enabled", True))
        self.model_name = rr.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.top_k = int(rr.get("rerank_top_k", 5))
        self.batch_size = int(rr.get("batch_size", 32))
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Sắp xếp lại candidate theo điểm cross-encoder.

        candidates: list dict có khoá 'chunk'. Trả về list đã sort giảm dần,
        mỗi phần tử thêm khoá 'rerank_score'.
        """
        if not candidates:
            return []
        k = top_k or self.top_k

        if not self.enabled:
            # giữ nguyên thứ tự retriever, chỉ cắt top-k
            return candidates[:k]

        pairs = [[query, c["chunk"]] for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:k]
