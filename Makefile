# ============================================================
# MedRAG-Retriever — Makefile
# Dùng: make <target>
# Windows: cài make qua  winget install GnuWin32.Make
#          hoặc dùng trực tiếp lệnh Python bên dưới
# ============================================================

.PHONY: install check collect preprocess index train eval backend frontend clean help

# ---------- Setup ----------
install:
	pip install -r requirements.txt
	pip install -e .

check:
	python setup_check.py

# ---------- Pipeline (chạy theo thứ tự) ----------
collect:
	python scripts/01_collect_data.py --max 500

collect-full:
	python scripts/01_collect_data.py

preprocess:
	python scripts/02_preprocess.py

index:
	python scripts/03_build_index.py

train:
	python scripts/04_train_retriever.py

eval:
	python scripts/05_evaluate.py --limit 200

# Chạy toàn bộ pipeline một lần (không tính train)
pipeline: collect preprocess index eval

# ---------- Web app ----------
backend:
	uvicorn app.backend.main:app --host 0.0.0.0 --port 8080 --reload

frontend:
	streamlit run app/frontend/streamlit_app.py

# ---------- Test ----------
test:
	pytest -q

# ---------- Dọn dẹp ----------
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clean-data:
	rm -f data/raw/*.jsonl data/processed/*.jsonl
	rm -f data/index/index.faiss data/index/metadata.jsonl

# ---------- Help ----------
help:
	@echo ""
	@echo "  make install      Cài tất cả dependencies"
	@echo "  make check        Kiểm tra môi trường"
	@echo "  make collect      Thu thập 500 abstract PubMed (test nhỏ)"
	@echo "  make collect-full Thu thập toàn bộ theo config"
	@echo "  make preprocess   Làm sạch + chunk"
	@echo "  make index        Xây FAISS index"
	@echo "  make train        Fine-tune retriever (cần GPU)"
	@echo "  make eval         Đánh giá retrieval"
	@echo "  make pipeline     Chạy collect→preprocess→index→eval"
	@echo "  make backend      Khởi động FastAPI backend"
	@echo "  make frontend     Khởi động Streamlit frontend"
	@echo "  make test         Chạy smoke tests"
	@echo "  make clean        Xoá __pycache__"
	@echo ""
