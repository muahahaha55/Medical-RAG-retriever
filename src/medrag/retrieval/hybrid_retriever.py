"""Hybrid retriever: kết hợp dense (FAISS) + sparse (BM25) bằng Reciprocal Rank Fusion.

RRF (Cormack et al. 2009): với mỗi document d,
    score(d) = sum_over_lists  1 / (rrf_k + rank_d_trong_list)
rank bắt đầu từ 1. Document xuất hiện hạng cao ở NHIỀU danh sách được đẩy lên top.
RRF không cần chuẩn hoá điểm giữa hai hệ (cosine vs BM25) — chỉ dùng thứ hạng,
nên rất ổn định và là chuẩn phổ biến cho hybrid search.

Cần một khoá định danh doc chung để hợp nhất. Mặc định dùng 'chunk_id' nếu có,
không thì fallback sang 'pmid', không nữa thì dùng chính text chunk.
"""
from __future__ import annotations

from pathlib import Path

from medrag.config import Config, CONFIG
from medrag.retrieval.retriever import Retriever
from medrag.retrieval.bm25_retriever import BM25Retriever
from medrag.utils.io import get_logger

logger = get_logger("medrag.hybrid")


def _doc_key(item: dict) -> str:
    for k in ("chunk_id", "id", "pmid"):
        if item.get(k) not in (None, ""):
            return f"{k}:{item[k]}"
    return "chunk:" + str(item.get("chunk", ""))[:200]


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    rrf_k: int = 60,
    top_k: int = 50,
) -> list[dict]:
    """Hợp nhất nhiều danh sách đã xếp hạng thành 1 danh sách bằng RRF."""
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst, start=1):
            key = _doc_key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            # giữ lại payload đầy đủ nhất (ưu tiên lần gặp đầu)
            if key not in payload:
                payload[key] = item
    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
    out = []
    for key in ordered:
        item = dict(payload[key])
        item["rrf_score"] = scores[key]
        out.append(item)
    return out


class HybridRetriever:
    """Ghép Retriever (dense/FAISS) và BM25Retriever, hợp nhất bằng RRF."""

    def __init__(self, config: Config = CONFIG, model_name_or_path: str | None = None):
        self.cfg = config
        self.dense = Retriever(config, model_name_or_path=model_name_or_path)
        self.bm25 = BM25Retriever(config)
        self.rrf_k = int(config.get("retriever.rrf_k", 60))
        self._loaded = False

    def load(self, index_dir: str | Path | None = None) -> "HybridRetriever":
        self.dense.load(index_dir)
        self.bm25.load(index_dir)
        self._loaded = True
        return self

    def retrieve(self, query: str, top_k: int | None = None,
                 candidate_k: int | None = None) -> list[dict]:
        """Lấy candidate_k từ mỗi nhánh, fuse RRF, trả top_k."""
        if not self._loaded:
            self.load()
        if top_k is None:
            top_k = int(self.cfg.get("reranker.retrieve_top_k", 50))
        if candidate_k is None:
            candidate_k = max(top_k, int(self.cfg.get("retriever.hybrid_candidate_k", 50)))

        dense_hits = self.dense.retrieve(query, top_k=candidate_k)
        bm25_hits = self.bm25.search(query, top_k=candidate_k)
        return reciprocal_rank_fusion([dense_hits, bm25_hits], rrf_k=self.rrf_k, top_k=top_k)