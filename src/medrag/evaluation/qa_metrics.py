"""Metric QA: Exact Match và token-level F1 (kiểu SQuAD)."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence


def normalize_answer(s: str) -> str:
    """Chuẩn hoá: lowercase, bỏ dấu câu, mạo từ, khoảng trắng thừa."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_qa(
    predictions: Sequence[str],
    references: Sequence[str],
) -> dict[str, float]:
    """Trung bình EM và F1 trên toàn bộ tập."""
    assert len(predictions) == len(references)
    n = len(predictions) or 1
    em = sum(exact_match(p, r) for p, r in zip(predictions, references)) / n
    f1 = sum(f1_score(p, r) for p, r in zip(predictions, references)) / n
    return {"exact_match": em, "f1": f1}
