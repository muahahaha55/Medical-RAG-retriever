"""Tiện ích chung: logging, seed, đọc/ghi JSONL."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "medrag", level: int = logging.INFO) -> logging.Logger:
    """Trả về logger đã cấu hình (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def set_seed(seed: int = 42) -> None:
    """Cố định seed cho random/numpy/torch (nếu có)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# JSONL helpers — định dạng chính cho corpus & dữ liệu trung gian
# ---------------------------------------------------------------------------
def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Ghi danh sách dict ra file .jsonl. Trả về số bản ghi đã ghi."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Đọc lần lượt từng bản ghi từ file .jsonl (generator, tiết kiệm RAM)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(path: str | Path) -> int:
    """Đếm số dòng (số bản ghi) trong file văn bản."""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)
