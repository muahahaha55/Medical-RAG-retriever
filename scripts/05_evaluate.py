#!/usr/bin/env python
"""05 — Đánh giá retrieval trên tập PubMedQA.

Cách tiếp cận đơn giản cho khung ban đầu:
  - Lấy các câu hỏi PubMedQA.
  - Ground-truth relevant = chunk thuộc đúng PMID của câu hỏi đó.
  - Truy hồi top-k, tính recall@k / MRR / nDCG.

Lưu ý: cần index đã build từ corpus có chứa các PMID của PubMedQA.
Để demo nhanh, có thể build index trực tiếp từ context của PubMedQA.

Cách dùng:
    python scripts/05_evaluate.py --limit 200
"""
import argparse
import json

from medrag.config import CONFIG
from medrag.evaluation.retrieval_metrics import evaluate_retrieval
from medrag.retrieval.retriever import Retriever


def load_eval_queries(limit: int) -> list[dict]:
    """Trả về list {question, relevant_pmid}. Cần datasets."""
    from datasets import load_dataset

    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    out = []
    for ex in ds:
        pmid = str(ex.get("pubid", ""))
        q = (ex.get("question") or "").strip()
        if q and pmid:
            out.append({"question": q, "relevant_pmid": pmid})
        if len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description="Đánh giá retrieval")
    ap.add_argument("--limit", type=int, default=200, help="Số câu hỏi đánh giá")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    queries = load_eval_queries(args.limit)
    retriever = Retriever(CONFIG, model_name_or_path=args.model).load(args.index_dir)

    k_values = CONFIG.get("evaluation.k_values", [5, 10, 20, 50])
    max_k = max(k_values)

    all_retrieved, all_relevant = [], []
    for item in queries:
        hits = retriever.retrieve(item["question"], top_k=max_k)
        retrieved_pmids = [h.get("pmid", "") for h in hits]
        all_retrieved.append(retrieved_pmids)
        all_relevant.append({item["relevant_pmid"]})

    metrics = evaluate_retrieval(all_retrieved, all_relevant, k_values=k_values)
    print("=== Retrieval metrics ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
