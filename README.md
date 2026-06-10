# MedRAG-Retriever

**Improving Medical Question Answering through Contrastively Fine-Tuned Biomedical Retrieval**

Hệ thống Medical RAG: truy hồi bằng chứng từ PubMed, rerank, rồi sinh câu trả lời có trích dẫn. Điểm cốt lõi là **tầng retrieval** — fine-tune retriever y sinh bằng contrastive learning và so sánh với các embedding model phổ thông.

---

## Mở trong PyCharm (Windows)

### Bước 1 — Cài đặt môi trường
Mở **Terminal** trong PyCharm (tab ở dưới cùng):
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Bước 2 — Điền email vào .env
Mở file `.env` ở thư mục gốc, sửa dòng:
```
NCBI_EMAIL=your-real-email@example.com
```
> Email dùng để gọi API NCBI PubMed — bắt buộc, không cần đăng ký gì.

### Bước 3 — Kiểm tra môi trường
```bash
python setup_check.py
```
Script này sẽ kiểm tra mọi thứ và in hướng dẫn sửa nếu có lỗi.

### Bước 4 — Chạy pipeline
```bash
python scripts/01_collect_data.py --max 500   # thu thập 500 abstract để test
python scripts/02_preprocess.py               # làm sạch + chunk
python scripts/03_build_index.py              # xây FAISS index
```

### Bước 5 — Khởi động web app
Mở 2 terminal:
```bash
# Terminal 1 — Backend
uvicorn app.backend.main:app --port 8080 --reload

# Terminal 2 — Frontend
streamlit run app/frontend/streamlit_app.py
```
Mở trình duyệt: **http://localhost:8501**

---

## Kiến trúc

```
User Question
   → Query Encoder
   → Fine-Tuned Retriever → Top-50 Docs (FAISS)
   → Cross-Encoder Reranker → Top-5 Evidence
   → LLM → Answer + Citations (PMID)
```

## Cấu trúc thư mục

```
medrag-retriever/
├── .env                        # ← ĐIỀN EMAIL VÀO ĐÂY
├── setup_check.py              # kiểm tra môi trường
├── config/config.yaml          # cấu hình tập trung
├── src/medrag/
│   ├── config.py               # loader (tự đọc .env)
│   ├── data/                   # collector PubMed, cleaning, chunking
│   ├── retrieval/              # embeddings, FAISS, retriever
│   ├── training/               # build pairs + fine-tune contrastive
│   ├── reranking/              # cross-encoder reranker
│   ├── llm/                    # prompts + generator (mock/hf/openai)
│   ├── rag/                    # pipeline end-to-end
│   ├── evaluation/             # recall@k, MRR, nDCG, EM/F1, RAGAS
│   └── utils/                  # io, logging, seed
├── scripts/                    # 01→05 chạy theo thứ tự
├── app/
│   ├── backend/main.py         # FastAPI
│   └── frontend/streamlit_app.py
├── docker/                     # Dockerfile + compose
└── tests/test_smoke.py
```

## Chạy trên Colab T4

```python
# Cell 1 — Clone / upload repo, rồi:
!pip install -r requirements.txt
!pip install -e .

# Cell 2 — Thu thập data
!python scripts/01_collect_data.py --max 100000

# Cell 3 — Preprocess + Index
!python scripts/02_preprocess.py
!python scripts/03_build_index.py

# Cell 4 — Fine-tune retriever (cần GPU)
!python scripts/04_train_retriever.py

# Cell 5 — Đánh giá
!python scripts/05_evaluate.py --limit 500
```

Để dùng LLM thật trên T4, sửa `config/config.yaml`:
```yaml
llm:
  backend: "hf"
  model: "Qwen/Qwen3-8B"
```
Và bỏ comment trong `requirements.txt`:
```
bitsandbytes>=0.43
accelerate>=0.30
```

## Đánh giá

| Nhóm | Metric |
|------|--------|
| Retrieval | Recall@5/10/20/50, MRR, nDCG |
| QA | Exact Match, F1 |
| RAG | Faithfulness, Context Precision/Recall, Answer Relevance (RAGAS) |

## Ablation (kế hoạch)

1. Baseline (BGE/BioBERT) vs MedicalRetriever-v1
2. Top-K: 5 / 10 / 20 / 50
3. Có vs không reranker
4. Corpus size: 50k / 100k / 500k

## Test

```bash
pytest -q    # 7 smoke tests, không cần model nặng
```

## Trạng thái

- [x] Pipeline end-to-end (mock LLM)
- [x] Collector PubMed + cleaning + chunking
- [x] Baseline retriever + FAISS + Cross-Encoder reranker
- [x] Contrastive fine-tuning (MNRL / InfoNCE)
- [x] Metric retrieval/QA + RAGAS wrapper
- [x] FastAPI + Streamlit + Docker
- [ ] Hard-negative mining nâng cao
- [ ] Hallucination detection
- [ ] Knowledge graph y khoa
