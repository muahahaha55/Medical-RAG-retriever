"""Metric đánh giá retrieval, cài thuần Python (không phụ thuộc ngoài).

Quy ước đầu vào cho mỗi truy vấn:
  retrieved_ids: list[str]  — id tài liệu theo thứ tự xếp hạng (tốt -> kém)
  relevant_ids:  set[str]   — tập id tài liệu thực sự liên quan (ground truth)
"""
from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Tỉ lệ tài liệu liên quan xuất hiện trong top-k."""
    if not relevant_ids:
        return 0.0
    topk = set(retrieved_ids[:k])
    hit = len(topk & relevant_ids)
    return hit / len(relevant_ids)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """1/rank của tài liệu liên quan đầu tiên (0 nếu không có)."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # i bắt đầu từ 0 -> log2(rank+1)
    return dcg


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """nDCG nhị phân (relevance 0/1)."""
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    all_retrieved: list[Sequence[str]],
    all_relevant: list[set[str]],
    k_values: Sequence[int] = (5, 10, 20, 50),
) -> dict[str, float]:
    """Tính trung bình các metric trên toàn bộ tập truy vấn."""
    assert len(all_retrieved) == len(all_relevant), "Lệch số lượng truy vấn"
    n = len(all_retrieved)
    if n == 0:
        return {}

    results: dict[str, float] = {}
    for k in k_values:
        results[f"recall@{k}"] = sum(
            recall_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)
        ) / n
        results[f"ndcg@{k}"] = sum(
            ndcg_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)
        ) / n
    results["mrr"] = sum(
        reciprocal_rank(r, rel) for r, rel in zip(all_retrieved, all_relevant)
    ) / n
    return results
