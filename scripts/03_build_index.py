#!/usr/bin/env python
"""03 — Nhúng corpus và xây FAISS index.

Input:  data/processed/chunks.jsonl
Output: data/index/{index.faiss, metadata.jsonl}

Cách dùng:
    python scripts/03_build_index.py
    python scripts/03_build_index.py --model data/models/MedicalRetriever-v1
"""
import argparse

from medrag.config import CONFIG
from medrag.retrieval.retriever import Retriever


def main():
    ap = argparse.ArgumentParser(description="Xây FAISS index")
    ap.add_argument("--chunks", default=None, help="File chunks .jsonl")
    ap.add_argument("--model", default=None, help="Tên/đường dẫn embedding model")
    ap.add_argument("--index-dir", default=None, help="Thư mục lưu index")
    args = ap.parse_args()

    chunks = args.chunks or str(CONFIG.path("paths.data_processed") / "chunks.jsonl")

    retriever = Retriever(CONFIG, model_name_or_path=args.model)
    out = retriever.index_corpus(chunks, index_dir=args.index_dir)
    print(f"[OK] Đã xây index -> {out}")


if __name__ == "__main__":
    main()
