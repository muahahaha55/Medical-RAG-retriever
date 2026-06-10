"""Wrapper RAGAS: Faithfulness, Context Precision/Recall, Answer Relevance.

RAGAS cần một LLM + embedding để chấm điểm, nên chỉ bật khi đã cấu hình
LLM thật (evaluation.ragas_enabled = true). Hàm trả về dict điểm số.
"""
from __future__ import annotations

from medrag.config import Config, CONFIG
from medrag.utils.io import get_logger

logger = get_logger("medrag.ragas")


def evaluate_with_ragas(
    samples: list[dict],
    config: Config = CONFIG,
) -> dict[str, float]:
    """samples: list dict {question, answer, contexts(list[str]), ground_truth}.

    Trả về dict điểm trung bình các metric của RAGAS.
    """
    if not config.get("evaluation.ragas_enabled", False):
        logger.warning("RAGAS đang tắt (evaluation.ragas_enabled=false). Bỏ qua.")
        return {}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:  # pragma: no cover
        logger.error("Chưa cài ragas/datasets: %s", e)
        return {}

    ds = Dataset.from_list(
        [
            {
                "question": s["question"],
                "answer": s["answer"],
                "contexts": s["contexts"],
                "ground_truth": s.get("ground_truth", ""),
            }
            for s in samples
        ]
    )
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    logger.info("RAGAS: %s", result)
    return dict(result)
