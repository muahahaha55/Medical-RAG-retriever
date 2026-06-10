#!/usr/bin/env python
"""04 — Xây cặp huấn luyện và fine-tune retriever.

Bước 1: build cặp (query, positive[, negative]) từ PubMedQA.
Bước 2: fine-tune BioBERT bằng MultipleNegativesRankingLoss.

Output: data/models/MedicalRetriever-v1

Cách dùng:
    python scripts/04_train_retriever.py
    python scripts/04_train_retriever.py --pairs data/processed/train_pairs.jsonl
"""
import argparse

from medrag.config import CONFIG
from medrag.training.pair_builder import build_pairs
from medrag.training.train_retriever import train_retriever


def main():
    ap = argparse.ArgumentParser(description="Fine-tune retriever (contrastive)")
    ap.add_argument("--pairs", default=None, help="File cặp .jsonl có sẵn (bỏ qua bước build)")
    ap.add_argument("--out", default=None, help="Thư mục lưu model")
    ap.add_argument("--skip-build", action="store_true", help="Bỏ qua bước build pairs")
    args = ap.parse_args()

    pairs_path = args.pairs
    if not args.skip_build and not pairs_path:
        pairs_path = build_pairs(CONFIG)
    elif not pairs_path:
        pairs_path = str(CONFIG.path("paths.data_processed") / "train_pairs.jsonl")

    out = train_retriever(pairs_path, CONFIG, output_path=args.out)
    print(f"[OK] Đã fine-tune retriever -> {out}")
    print("Gợi ý: đặt retriever.active='finetuned' trong config rồi chạy lại script 03 để re-index.")


if __name__ == "__main__":
    main()
