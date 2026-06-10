"""Xây dựng cặp huấn luyện (positive / negative) cho retriever.

Nguồn: PubMedQA (HuggingFace: `pubmed_qa`, subset `pqa_labeled`).
  - Positive: (question, context liên quan)
  - Negative: (question, context của câu hỏi KHÁC) — in-batch negatives.

Với MultipleNegativesRankingLoss, ta chỉ cần các cặp positive; negatives
được lấy tự động trong batch. Hard negative (tuỳ chọn) lấy bằng cách
truy hồi top-k rồi loại passage đúng.

Output: data/processed/train_pairs.jsonl với schema:
    {"query": "...", "positive": "...", "negative": "..."(optional)}
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from medrag.config import Config, CONFIG
from medrag.utils.io import get_logger, write_jsonl

logger = get_logger("medrag.pairs")


def _load_pubmedqa(split: str = "train") -> list[dict]:
    """Tải PubMedQA (pqa_labeled). Trả về list dict {question, context}."""
    from datasets import load_dataset

    ds = load_dataset("pubmed_qa", "pqa_labeled", split=split)
    rows = []
    for ex in ds:
        ctxs = ex.get("context", {}).get("contexts", [])
        context = " ".join(ctxs) if ctxs else ""
        if ex.get("question") and context:
            rows.append({"question": ex["question"].strip(), "context": context.strip()})
    logger.info("Đã load %d mẫu PubMedQA", len(rows))
    return rows


def build_pairs(
    config: Config = CONFIG,
    output_path: Optional[str | Path] = None,
    add_random_negative: bool = True,
) -> Path:
    """Sinh file cặp huấn luyện từ PubMedQA."""
    random.seed(int(config.get("project.seed", 42)))
    rows = _load_pubmedqa("train")

    out = (
        Path(output_path)
        if output_path
        else config.path("paths.data_processed") / "train_pairs.jsonl"
    )

    contexts = [r["context"] for r in rows]
    pairs = []
    for i, r in enumerate(rows):
        pair = {"query": r["question"], "positive": r["context"]}
        if add_random_negative and len(contexts) > 1:
            j = random.randrange(len(contexts))
            while j == i:
                j = random.randrange(len(contexts))
            pair["negative"] = contexts[j]
        pairs.append(pair)

    n = write_jsonl(pairs, out)
    logger.info("Đã ghi %d cặp huấn luyện vào %s", n, out)
    return out
