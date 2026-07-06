#!/usr/bin/env python
"""03 — Nhúng corpus và xây FAISS index (+ BM25 index cho hybrid search).

Input:  data/processed/chunks.jsonl
Output: data/index/{index.faiss, metadata.jsonl, bm25/, bm25_meta.jsonl}

Cách dùng:
    python scripts/03_build_index.py
    python scripts/03_build_index.py --model data/models/MedicalRetriever-v1
    python scripts/03_build_index.py --no-bm25      # chỉ build FAISS
"""
import argparse

from medrag.config import CONFIG
from medrag.retrieval.retriever import Retriever
from medrag.utils.io import read_jsonl


def main():
    ap = argparse.ArgumentParser(description="Xây FAISS (+ BM25) index")
    ap.add_argument("--chunks", default=None, help="File chunks .jsonl")
    ap.add_argument("--model", default=None, help="Tên/đường dẫn embedding model")
    ap.add_argument("--index-dir", default=None, help="Thư mục lưu index")
    ap.add_argument("--no-bm25", action="store_true", help="Bỏ qua build BM25")
    args = ap.parse_args()

    chunks = args.chunks or str(CONFIG.path("paths.data_processed") / "chunks.jsonl")

    retriever = Retriever(CONFIG, model_name_or_path=args.model)
    out = retriever.index_corpus(chunks, index_dir=args.index_dir)
    print(f"[OK] Đã xây FAISS index -> {out}")

    if not args.no_bm25:
        from medrag.retrieval.bm25_retriever import BM25Retriever

        records = list(read_jsonl(chunks))
        texts = [r["chunk"] for r in records]
        bm25 = BM25Retriever(CONFIG)
        bm25.build(texts, records)
        bm25.save(args.index_dir)
        print(f"[OK] Đã xây BM25 index -> {out}")


if __name__ == "__main__":
    main()