#!/usr/bin/env python
"""Kiểm tra môi trường trước khi chạy pipeline.

Chạy script này đầu tiên sau khi cài dependencies:
    python setup_check.py

Script sẽ kiểm tra từng thứ và in ra hướng dẫn sửa nếu có vấn đề.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

# ── màu terminal (tắt nếu Windows không hỗ trợ) ──────────────────────────────
try:
    import colorama; colorama.init()
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
except ImportError:
    GREEN = YELLOW = RED = RESET = BOLD = ""

OK   = f"{GREEN}[OK]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"

ROOT = Path(__file__).resolve().parent
errors   : list[str] = []
warnings : list[str] = []


def check(label: str, ok: bool, fix: str = "", warn_only: bool = False) -> None:
    if ok:
        print(f"  {OK}   {label}")
    elif warn_only:
        print(f"  {WARN} {label}")
        if fix:
            print(f"         → {YELLOW}{fix}{RESET}")
        warnings.append(label)
    else:
        print(f"  {FAIL} {label}")
        if fix:
            print(f"         → {RED}{fix}{RESET}")
        errors.append(label)


def section(title: str) -> None:
    print(f"\n{BOLD}{'─'*50}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*50}{RESET}")


# ── 1. Python version ─────────────────────────────────────────────────────────
section("1. Python")
check(
    f"Python {sys.version.split()[0]}",
    sys.version_info >= (3, 10),
    fix="Cần Python >= 3.10  |  https://www.python.org/downloads/",
)

# ── 2. PYTHONPATH / src có trong path ────────────────────────────────────────
section("2. PYTHONPATH (src/)")
src_path = str(ROOT / "src")
in_path = src_path in sys.path or (ROOT / "src" in [Path(p) for p in sys.path])

# Thử tìm qua installed package (pip install -e .)
try:
    import medrag  # noqa: F401
    in_path = True
except ImportError:
    pass

check(
    "src/ nằm trong sys.path hoặc package đã được cài",
    in_path,
    fix="Chạy: pip install -e .   HOẶC trong PyCharm: chuột phải src/ → Mark as Sources Root",
)

# ── 3. Packages bắt buộc ─────────────────────────────────────────────────────
section("3. Dependencies bắt buộc")
required = [
    ("yaml",                 "pyyaml",                  "pip install pyyaml"),
    ("numpy",                "numpy",                   "pip install numpy"),
    ("requests",             "requests",                "pip install requests"),
    ("torch",                "torch",                   "pip install torch"),
    ("transformers",         "transformers",            "pip install transformers"),
    ("sentence_transformers","sentence-transformers",   "pip install sentence-transformers"),
    ("datasets",             "datasets",                "pip install datasets"),
    ("faiss",                "faiss-cpu",               "pip install faiss-cpu"),
    ("fastapi",              "fastapi",                 "pip install fastapi"),
    ("uvicorn",              "uvicorn",                 "pip install uvicorn[standard]"),
    ("streamlit",            "streamlit",               "pip install streamlit"),
    ("dotenv",               "python-dotenv",           "pip install python-dotenv"),
    ("pydantic",             "pydantic",                "pip install pydantic"),
]
for import_name, pkg_name, fix in required:
    found = importlib.util.find_spec(import_name) is not None
    check(f"{pkg_name}", found, fix=fix)

# ── 4. Packages tuỳ chọn ─────────────────────────────────────────────────────
section("4. Dependencies tuỳ chọn (chỉ cần khi dùng LLM thật)")
optional = [
    ("openai",        "openai",        "pip install openai          ← cần cho Ollama/vLLM"),
    ("bitsandbytes",  "bitsandbytes",  "pip install bitsandbytes    ← cần cho int4/int8 quantization"),
    ("accelerate",    "accelerate",    "pip install accelerate      ← cần kèm bitsandbytes"),
]
for import_name, pkg_name, fix in optional:
    found = importlib.util.find_spec(import_name) is not None
    check(pkg_name, found, fix=fix, warn_only=True)

# ── 5. GPU ────────────────────────────────────────────────────────────────────
section("5. GPU / CUDA")
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        check(f"CUDA available — {name} ({vram:.1f} GB VRAM)", True)
        check(
            f"VRAM đủ cho retriever + reranker (>= 4GB)",
            vram >= 4,
            fix="Máy ít VRAM, chỉ nên dùng mock LLM backend",
            warn_only=True,
        )
    else:
        check(
            "CUDA không khả dụng — sẽ chạy trên CPU",
            False,
            fix="Không sao, pipeline vẫn chạy được với mock LLM. Dùng Colab T4 để train.",
            warn_only=True,
        )
except Exception:
    check("Không kiểm tra được GPU (torch chưa cài?)", False, warn_only=True)

# ── 6. File cấu hình ─────────────────────────────────────────────────────────
section("6. File cấu hình")
check(
    "config/config.yaml tồn tại",
    (ROOT / "config" / "config.yaml").exists(),
    fix="File này phải có trong repo, không nên bị xoá",
)
env_exists = (ROOT / ".env").exists()
check(
    ".env tồn tại",
    env_exists,
    fix="Chạy:  copy .env.example .env   (Windows)   hoặc   cp .env.example .env   (Mac/Linux)",
)
if env_exists:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    email = os.getenv("NCBI_EMAIL", "")
    check(
        f"NCBI_EMAIL được set ({email or 'CHƯA SET'})",
        bool(email) and email != "your-email@example.com",
        fix="Mở file .env, điền NCBI_EMAIL=your-real-email@example.com",
    )

# ── 7. Thư mục data ───────────────────────────────────────────────────────────
section("7. Thư mục data")
for d in ["data/raw", "data/processed", "data/index", "data/models"]:
    p = ROOT / d
    p.mkdir(parents=True, exist_ok=True)
    check(f"{d}/ tồn tại", True)

data_ready   = any((ROOT / "data/raw").glob("*.jsonl"))
index_ready  = (ROOT / "data/index" / "index.faiss").exists()
check("data/raw/ có dữ liệu (.jsonl)",  data_ready,  fix="Chạy: python scripts/01_collect_data.py --max 500", warn_only=True)
check("data/index/ có FAISS index",     index_ready, fix="Chạy script 02 rồi 03 sau khi có data",            warn_only=True)

# ── Tổng kết ──────────────────────────────────────────────────────────────────
print(f"\n{'═'*50}")
if errors:
    print(f"{RED}{BOLD}  {len(errors)} lỗi cần sửa trước khi chạy:{RESET}")
    for e in errors:
        print(f"    • {e}")
    print(f"\n  Sau khi sửa xong, chạy lại:  python setup_check.py")
elif warnings:
    print(f"{YELLOW}{BOLD}  Môi trường OK — {len(warnings)} cảnh báo nhỏ (không bắt buộc sửa){RESET}")
    print(f"\n  Bước tiếp theo:")
    print(f"    python scripts/01_collect_data.py --max 500")
else:
    print(f"{GREEN}{BOLD}  Tất cả OK! Sẵn sàng chạy pipeline.{RESET}")
    if not data_ready:
        print(f"\n  Bước tiếp theo:")
        print(f"    python scripts/01_collect_data.py --max 500")
        print(f"    python scripts/02_preprocess.py")
        print(f"    python scripts/03_build_index.py")
    else:
        print(f"\n  Khởi động web app:")
        print(f"    uvicorn app.backend.main:app --port 8080")
        print(f"    streamlit run app/frontend/streamlit_app.py")
print(f"{'═'*50}\n")

sys.exit(1 if errors else 0)
