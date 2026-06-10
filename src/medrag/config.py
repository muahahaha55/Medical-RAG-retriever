"""Tải và xác thực cấu hình từ config/config.yaml.

Tự động đọc file .env ở thư mục gốc repo nếu python-dotenv đã cài.
Override bằng biến môi trường cho các trường nhạy cảm (API key, email).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Thư mục gốc của repo (…/medrag-retriever)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _load_dotenv() -> None:
    """Đọc file .env nếu python-dotenv có sẵn (không bắt buộc)."""
    try:
        from dotenv import load_dotenv
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)  # override=False: biến môi trường thật được ưu tiên
    except ImportError:
        pass  # python-dotenv chưa cài, bỏ qua


@dataclass
class Config:
    """Wrapper mỏng quanh dict cấu hình, cho phép truy cập bằng dot-path."""

    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Lấy giá trị theo path kiểu 'retriever.baseline_model'."""
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, dotted_key: str) -> Path:
        """Trả về Path tuyệt đối cho một entry trong section `paths`."""
        rel = self.get(dotted_key)
        if rel is None:
            raise KeyError(f"Không tìm thấy path config: {dotted_key}")
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Cho phép biến môi trường ghi đè vài trường quan trọng."""
    if os.getenv("NCBI_API_KEY"):
        data.setdefault("data_collection", {})["ncbi_api_key"] = os.environ["NCBI_API_KEY"]
    if os.getenv("NCBI_EMAIL"):
        data.setdefault("data_collection", {})["email"] = os.environ["NCBI_EMAIL"]
    if os.getenv("LLM_BASE_URL"):
        data.setdefault("llm", {})["base_url"] = os.environ["LLM_BASE_URL"]
    if os.getenv("LLM_API_KEY"):
        data.setdefault("llm", {})["api_key"] = os.environ["LLM_API_KEY"]
    return data


def load_config(path: str | Path | None = None) -> Config:
    """Đọc file .env rồi YAML, trả về đối tượng Config."""
    _load_dotenv()
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data = _apply_env_overrides(data)
    return Config(raw=data)


# Cho phép import nhanh: from medrag.config import CONFIG
CONFIG = load_config()
