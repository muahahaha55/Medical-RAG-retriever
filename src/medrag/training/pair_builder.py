"""Xây dựng cặp huấn luyện (positive / negative) cho retriever.

Nguồn: PubMedQA (HuggingFace: `pubmed_qa`, subset `pqa_artificial`).
  - Positive: (question, context liên quan)
  - Negative: 2 chiến lược:
      "random" — context của một câu hỏi KHÁC, chọn ngẫu nhiên (dễ, model học ít).
      "hard"   — context có điểm BM25 cao với câu hỏi nhưng KHÔNG phải context đúng
                 (khó phân biệt hơn, ép model học ranh giới ngữ nghĩa thật sự).

Với MultipleNegativesRankingLoss, in-batch negatives đã tự động có sẵn;
hard negative là lớp bổ sung, không bắt buộc phải phủ hết mọi dòng.

Vì BM25 mining có độ phức tạp gần bậc hai theo số mẫu, chạy hard-negative
trên CPU với toàn bộ ~211k mẫu (pqa_artificial) sẽ rất chậm. Mặc định chỉ
mine hard negative cho một subsample (hard_negative_sample_size), các dòng
còn lại vẫn giữ negative random — đây là đánh đổi hợp lý cho máy không GPU.

Output: data/processed/train_pairs.jsonl với schema:
    {"query": "...", "positive": "...", "negative": "...", "negative_type": "hard"|"random"}
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from medrag.config import Config, CONFIG
from medrag.utils.io import get_logger, write_jsonl

logger = get_logger("medrag.pairs")


def _load_pubmedqa(split: str = "train") -> list[dict]:
    """Tải PubMedQA (pqa_artificial, ~211k mẫu). Trả về list dict {question, context}."""
    from datasets import load_dataset

    ds = load_dataset("pubmed_qa", "pqa_artificial", split=split)
    rows = []
    for ex in ds:
        ctxs = ex.get("context", {}).get("contexts", [])
        context = " ".join(ctxs) if ctxs else ""
        if ex.get("question") and context:
            rows.append({"question": ex["question"].strip(), "context": context.strip()})
    logger.info("Đã load %d mẫu PubMedQA", len(rows))
    return rows


def _tokenize(text: str) -> list[str]:
    """Tokenize thô — đủ dùng cho BM25, không cần tokenizer nặng."""
    return text.lower().split()


def _mine_hard_negatives(
    rows: list[dict],
    sample_size: int,
    top_k: int = 10,
    skip_top: int = 1,
    seed: int = 42,
) -> dict[int, str]:
    """Mine hard negative bằng BM25 (thư viện `bm25s`) cho một subsample các dòng.

    Dùng bm25s thay vì rank_bm25: retrieve() trả top-k trực tiếp bằng batch
    vectorized (nhanh hơn rank_bm25 khoảng 50-100 lần trên corpus lớn, vì
    rank_bm25.get_scores() phải quét toàn bộ corpus cho từng query trong
    vòng lặp Python).

    Với mỗi câu hỏi trong subsample: lấy top_k context điểm BM25 cao nhất,
    bỏ qua `skip_top` vị trí đầu (thường là chính context đúng hoặc gần
    như trùng), rồi chọn ngẫu nhiên 1 trong số còn lại làm hard negative.
    """
    import bm25s

    rng = random.Random(seed)
    contexts = [r["context"] for r in rows]

    logger.info("Đang build BM25 index (bm25s) trên %d context ...", len(contexts))
    corpus_tokens = bm25s.tokenize(contexts, stopwords=None, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)

    sample_size = min(sample_size, len(rows))
    sample_idxs = rng.sample(range(len(rows)), sample_size)

    logger.info("Mining hard negative cho %d/%d mẫu (top_k=%d) ...", sample_size, len(rows), top_k)
    queries = [rows[i]["question"] for i in sample_idxs]
    q_tokens = bm25s.tokenize(queries, stopwords=None, show_progress=False)
    results, _scores = retriever.retrieve(q_tokens, k=top_k + skip_top, show_progress=False)

    hard_negatives: dict[int, str] = {}
    for row_i, orig_i in enumerate(sample_idxs):
        top_idxs = results[row_i]
        candidates = [j for j in top_idxs if j != orig_i][skip_top:]
        if not candidates:
            continue
        j = rng.choice(candidates)
        hard_negatives[orig_i] = contexts[j]

    logger.info("Hoàn tất mining: %d hard negative", len(hard_negatives))
    return hard_negatives


def build_pairs(
    config: Config = CONFIG,
    output_path: Optional[str | Path] = None,
    negative_strategy: str = "random",
    hard_negative_sample_size: int = 20000,
    hard_negative_top_k: int = 10,
) -> Path:
    """
    hard_negative_sample_size mặc định 20000 — với bm25s (batch retrieval),
    mining 20000 mẫu trên corpus 211k chỉ mất khoảng 1-3 phút trên CPU.
    Có thể tăng lên toàn bộ ~211000 nếu muốn (mất thêm vài chục giây).
    """
    """Sinh file cặp huấn luyện từ PubMedQA.

    negative_strategy:
      "random" — mọi dòng đều lấy negative ngẫu nhiên (nhanh).
      "hard"   — subsample (hard_negative_sample_size dòng) dùng hard negative
                 mine bằng BM25, phần còn lại vẫn dùng random negative.
    """
    seed = int(config.get("project.seed", 42))
    random.seed(seed)
    rows = _load_pubmedqa("train")

    out = (
        Path(output_path)
        if output_path
        else config.path("paths.data_processed") / "train_pairs.jsonl"
    )

    contexts = [r["context"] for r in rows]

    hard_negatives: dict[int, str] = {}
    if negative_strategy == "hard":
        hard_negatives = _mine_hard_negatives(
            rows,
            sample_size=hard_negative_sample_size,
            top_k=hard_negative_top_k,
            seed=seed,
        )

    pairs = []
    for i, r in enumerate(rows):
        pair = {"query": r["question"], "positive": r["context"]}
        if i in hard_negatives:
            pair["negative"] = hard_negatives[i]
            pair["negative_type"] = "hard"
        elif len(contexts) > 1:
            j = random.randrange(len(contexts))
            while j == i:
                j = random.randrange(len(contexts))
            pair["negative"] = contexts[j]
            pair["negative_type"] = "random"
        pairs.append(pair)

    n = write_jsonl(pairs, out)
    n_hard = sum(1 for p in pairs if p.get("negative_type") == "hard")
    logger.info("Đã ghi %d cặp huấn luyện vào %s (%d hard, %d random)", n, out, n_hard, n - n_hard)
    return out