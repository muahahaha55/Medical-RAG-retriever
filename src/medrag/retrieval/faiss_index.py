"""Bọc FAISS: xây dựng, lưu, nạp và tìm kiếm vector.

Hỗ trợ IndexFlatIP (corpus nhỏ) và IVF+PQ (corpus lớn).
Metadata (chunk_id, pmid, title, chunk) lưu song song ở file .jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from medrag.config import Config, CONFIG
from medrag.utils.io import get_logger

logger = get_logger("medrag.faiss")


class FaissIndex:
    def __init__(self, config: Config = CONFIG):
        self.cfg = config
        f = config.raw.get("faiss", {})
        self.index_type = f.get("index_type", "flat_ip")
        self.nlist = int(f.get("nlist", 1024))
        self.m_pq = int(f.get("m_pq", 16))
        self.nbits = int(f.get("nbits", 8))
        self.index = None
        self.metadata: list[dict[str, Any]] = []

    # -- build -------------------------------------------------------------
    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Xây index từ ma trận embedding và metadata tương ứng."""
        import faiss

        assert embeddings.shape[0] == len(metadata), "Lệch số lượng vector và metadata"
        dim = embeddings.shape[1]
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        if self.index_type == "ivf_pq":
            quantizer = faiss.IndexFlatIP(dim)
            nlist = min(self.nlist, max(1, embeddings.shape[0] // 39))
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, self.m_pq, self.nbits, faiss.METRIC_INNER_PRODUCT)
            logger.info("Training IVF-PQ (nlist=%d) trên %d vector", nlist, embeddings.shape[0])
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = min(16, nlist)
        else:  # flat_ip
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)

        self.index = index
        self.metadata = metadata
        logger.info("Đã build index %s với %d vector (dim=%d)", self.index_type, index.ntotal, dim)

    # -- persistence -------------------------------------------------------
    def save(self, index_dir: str | Path | None = None) -> Path:
        import faiss

        d = Path(index_dir) if index_dir else self.cfg.path("paths.index_dir")
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "index.faiss"))
        with open(d / "metadata.jsonl", "w", encoding="utf-8") as fh:
            for m in self.metadata:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        logger.info("Đã lưu index vào %s", d)
        return d

    def load(self, index_dir: str | Path | None = None) -> "FaissIndex":
        import faiss

        d = Path(index_dir) if index_dir else self.cfg.path("paths.index_dir")
        self.index = faiss.read_index(str(d / "index.faiss"))
        self.metadata = []
        with open(d / "metadata.jsonl", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    self.metadata.append(json.loads(line))
        logger.info("Đã nạp index %d vector từ %s", self.index.ntotal, d)
        return self

    # -- search ------------------------------------------------------------
    def search(self, query_emb: np.ndarray, top_k: int = 50) -> list[dict]:
        """Tìm top_k cho 1 query. Trả về metadata kèm điểm số."""
        if self.index is None:
            raise RuntimeError("Index chưa được build/load")
        q = np.ascontiguousarray(query_emb.reshape(1, -1).astype(np.float32))
        scores, idxs = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results
