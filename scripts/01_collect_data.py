#!/usr/bin/env python
"""01 — Thu thập abstract từ PubMed.

Cách dùng:
    python scripts/01_collect_data.py
    python scripts/01_collect_data.py --max 5000 --out data/raw/sample.jsonl
"""
import argparse

from medrag.config import CONFIG
from medrag.data.pubmed_collector import PubMedCollector


def main():
    ap = argparse.ArgumentParser(description="Thu thập PubMed abstracts")
    ap.add_argument("--max", type=int, default=None, help="Số abstract tối đa")
    ap.add_argument("--out", type=str, default=None, help="Đường dẫn file output .jsonl")
    args = ap.parse_args()

    if args.max:
        CONFIG.raw.setdefault("data_collection", {})["max_abstracts"] = args.max

    collector = PubMedCollector(CONFIG)
    path = collector.collect(args.out)
    print(f"[OK] Đã thu thập dữ liệu -> {path}")


if __name__ == "__main__":
    main()
