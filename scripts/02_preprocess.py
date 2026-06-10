#!/usr/bin/env python
"""02 — Làm sạch và chunk abstract thành corpus.

Input:  data/raw/pubmed_abstracts.jsonl
Output: data/processed/chunks.jsonl

Cách dùng:
    python scripts/02_preprocess.py
"""
import argparse

from medrag.config import CONFIG
from medrag.data.chunking import Chunker
from medrag.data.preprocess import clean_records
from medrag.utils.io import read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser(description="Tiền xử lý + chunking")
    ap.add_argument("--in", dest="inp", default=None, help="File abstract .jsonl")
    ap.add_argument("--out", default=None, help="File chunks .jsonl")
    args = ap.parse_args()

    inp = args.inp or str(CONFIG.path("paths.data_raw") / "pubmed_abstracts.jsonl")
    out = args.out or str(CONFIG.path("paths.data_processed") / "chunks.jsonl")

    cleaned = clean_records(read_jsonl(inp), CONFIG)
    chunker = Chunker(CONFIG)
    chunks = chunker.chunk_records(cleaned)
    n = write_jsonl(chunks, out)
    print(f"[OK] Đã tạo {n} chunk -> {out}")


if __name__ == "__main__":
    main()
