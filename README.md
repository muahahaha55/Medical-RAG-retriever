# MedRAG-Retriever

**Nâng cao khả năng truy hồi tài liệu y sinh bằng Contrastive Fine-Tuning và Hybrid Search**

Một hệ thống Medical RAG có khả năng truy xuất bằng chứng từ PubMed, xếp hạng lại các tài liệu liên quan và sinh câu trả lời kèm trích dẫn nguồn. Đóng góp cốt lõi của dự án nằm ở **tầng truy hồi** — một bộ truy hồi tài liệu y sinh được tinh chỉnh bằng học tương phản (contrastive learning), kết hợp với cơ chế tìm kiếm lai giữa dense retrieval và sparse retrieval, được đánh giá trên bộ dữ liệu chuẩn BioASQ.

---

## Kết quả

Mô hình được đánh giá trên **BioASQ** với quy trình tách tập kiểm thử an toàn nhằm tránh rò rỉ dữ liệu (300 câu hỏi, tập truy hồi gồm 40.000 đoạn văn). Nhãn đúng được lấy từ các đánh giá mức độ liên quan do chuyên gia gán, thay vì tự đối chiếu với ngữ cảnh đầu vào.

| Chỉ số | BGE gốc | Fine-tuned (v3) | v3 + Hybrid | Mức cải thiện |
|---|---|---|---|---|
| Recall@5 | 0.359 | 0.381 | **0.410** | **+14.3%** |
| Recall@10 | 0.454 | 0.472 | **0.517** | **+13.9%** |
| Recall@20 | 0.520 | 0.548 | **0.591** | **+13.7%** |
| Recall@50 | 0.589 | 0.610 | **0.638** | **+8.3%** |
| MRR | 0.774 | 0.769 | **0.815** | **+5.2%** |

Việc fine-tuning giúp Recall tăng khoảng 2–3 điểm phần trăm tuyệt đối, trong khi Hybrid Retrieval đóng góp thêm 3–4,5 điểm, đồng thời khắc phục sự suy giảm của MRR ở mô hình chỉ sử dụng dense retrieval.

BM25 đặc biệt hiệu quả trong việc truy xuất các tài liệu chứa từ khóa khớp chính xác mà biểu diễn embedding thường bỏ sót. Hai tín hiệu dense và sparse bổ trợ lẫn nhau thay vì thay thế nhau.

**Phát hiện quan trọng:** Fine-tuning trên dữ liệu sinh tự động (PubMedQA `pqa_artificial`) gây ra hiện tượng **negative transfer**, khiến hiệu năng còn thấp hơn mô hình gốc. Chỉ sau khi thay thế bằng tập dữ liệu được tổng hợp từ nhiều nguồn có chú thích bởi con người, mô hình mới vượt được baseline.

📄 Chi tiết toàn bộ thí nghiệm: [Báo cáo kỹ thuật](docs/TECHNICAL_REPORT.md)

---

## Kiến trúc hệ thống

```
Câu hỏi người dùng
      ↓
Query Encoder (BGE đã fine-tune)
      ↓
Hybrid Retrieval
      ├── Dense Search (FAISS)
      └── Sparse Search (BM25)
              ↓
Reciprocal Rank Fusion
              ↓
Top-50 tài liệu
              ↓
Cross-Encoder Reranker
              ↓
Top-5 bằng chứng
              ↓
LLM
              ↓
Câu trả lời + PMID của các tài liệu được trích dẫn
```

Hai chỉ mục dense và sparse đều được xây dựng trên cùng một tập tài liệu. Thuật toán Reciprocal Rank Fusion (RRF) kết hợp kết quả dựa trên **thứ hạng**, do đó không cần chuẩn hóa điểm số giữa cosine similarity và BM25.

---

## Các phiên bản Fine-Tuning

Ba lần huấn luyện được thực hiện trong điều kiện kiểm soát giống nhau và đều được đánh giá trên BioASQ.

| Phiên bản | Mô hình gốc | Dữ liệu huấn luyện | Hard Negative | Kết quả |
|---|---|---|---|---|
| v1 | BioBERT | `pqa_artificial` (dữ liệu tổng hợp) | BM25 | Thấp hơn baseline |
| v2 | BGE-small | `pqa_artificial` (dữ liệu tổng hợp) | Embedding | Thấp hơn baseline |
| **v3** | BGE-small | **5 bộ dữ liệu do con người xây dựng** | Embedding + lọc false negative | **Vượt baseline** |

Việc thay đổi mô hình nền (v1 → v2) gần như không mang lại cải thiện. Ngược lại, chỉ cần thay đổi chất lượng dữ liệu huấn luyện (v2 → v3), hiệu năng đã vượt qua baseline.

**Kết luận:** nút thắt không nằm ở kiến trúc mô hình, mà nằm ở chất lượng dữ liệu huấn luyện.

### Dữ liệu huấn luyện (v3)

Khoảng **164.000 cặp** `(query, positive)` được tổng hợp từ năm bộ dữ liệu khác nhau. Tất cả được chuẩn hóa về cùng một định dạng, giới hạn số lượng mẫu của từng nguồn nhằm tránh mất cân bằng, sau đó loại bỏ các bản ghi trùng lặp.

| Nguồn | Đặc điểm | Số lượng |
|---|---|---|
| MedMCQA | Câu hỏi y khoa do chuyên gia biên soạn | ~40.000 |
| MedQuAD | Hỏi đáp từ các nguồn NIH/NLM | ~16.000 |
| HealthcareMagic | Hội thoại thực tế giữa bác sĩ và bệnh nhân | ~40.000 |
| NFCorpus | Bộ dữ liệu IR y khoa với nhãn relevance | ~40.000 |
| BioASQ (train) | Câu hỏi chuyên gia và PMID liên quan | ~28.000 |

Hard negative được khai thác bằng chính encoder ban đầu thông qua FAISS: các tài liệu gần nhất nhưng không phải đáp án đúng sẽ được chọn làm negative. Để tránh nhiễu, các ứng viên có độ tương đồng quá cao với positive sẽ bị loại bỏ, vì chúng nhiều khả năng là false negative và có thể làm sai lệch tín hiệu huấn luyện.

Tập BioASQ được chia thành hai phần train và evaluation theo **Question ID**, đảm bảo không xảy ra hiện tượng rò rỉ dữ liệu.

---

## Tập truy hồi

Tách biệt hoàn toàn với dữ liệu huấn luyện. Hệ thống truy hồi trên các abstract PubMed thu thập qua NCBI E-utilities:

- **717 chunk** được lập chỉ mục (chunking theo token từ 437 abstract)
- Chỉ mục kép: FAISS (dense, 384 chiều) + BM25 (sparse)
- Có thể mở rộng — chạy lại collector với nhiều từ khóa tìm kiếm và `--max` lớn hơn

---

## Cài đặt

### 1. Môi trường
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 2. Cấu hình `.env`
```
NCBI_EMAIL=your-email@example.com
```
NCBI yêu cầu email để truy cập E-utilities (không cần đăng ký tài khoản). `NCBI_API_KEY` là tùy chọn — có key thì gọi được 10 req/s thay vì 3 req/s.

Để dùng LLM thật (tùy chọn), trỏ tới bất kỳ endpoint OpenAI-compatible nào:
```
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-v1-...
```

### 3. Kiểm tra môi trường
```powershell
python setup_check.py
```

### 4. Xây dựng pipeline
```powershell
python scripts/01_collect_data.py --max 500   # thu thập abstract từ PubMed
python scripts/02_preprocess.py               # làm sạch + chunking theo token
python scripts/03_build_index.py              # xây chỉ mục FAISS + BM25
```

> `03_build_index.py` xây dựng **cả hai** chỉ mục. Khi đổi bộ truy hồi, bắt buộc phải build lại — embedding của các mô hình khác nhau không tương thích.

### 5. Khởi động ứng dụng
```powershell
# Terminal 1 — Backend
uvicorn app.backend.main:app --port 8080 --reload

# Terminal 2 — Frontend
streamlit run app/frontend/streamlit_app.py
```
Mở trình duyệt tại **http://localhost:8501**

---

## Đánh giá

```powershell
# BioASQ benchmark (nhãn đúng do chuyên gia gán)
python scripts/05_evaluate.py --mode bioasq --limit 500 --model data/models/MedicalRetriever-v3
python scripts/05_evaluate.py --mode bioasq --limit 500 --hybrid --model data/models/MedicalRetriever-v3

# So sánh với baseline
python scripts/05_evaluate.py --mode bioasq --limit 500 --model BAAI/bge-small-en-v1.5
```

Các chế độ khác: `--mode selfcontained` (PubMedQA, dùng `--distractors N` để tăng độ khó), `--mode mainindex`.

> Việc encode tập truy hồi 40.000 đoạn văn trên CPU mất vài giờ. Nên dùng GPU hoặc giảm `--limit`.

| Tầng | Chỉ số |
|---|---|
| Retrieval | Recall@5/10/20/50, MRR, nDCG |
| QA | Exact Match, F1 |
| RAG | Faithfulness, Context Precision/Recall (RAGAS) |

---

## Cấu hình

Toàn bộ cấu hình tập trung tại `config/config.yaml`:

```yaml
retriever:
  active: "finetuned"                              # baseline | finetuned
  finetuned_model_path: "data/models/MedicalRetriever-v3"
  use_hybrid: true                                 # bật dense + BM25 với RRF
  rrf_k: 60

llm:
  backend: "openai_compatible"                     # mock | hf | openai_compatible
  model: "meta-llama/llama-3.3-70b-instruct:free"
```

`.env` **ghi đè** `base_url` và `api_key` trong `config.yaml`.

---

## Cấu trúc dự án

```
medrag-retriever/
├── config/config.yaml          # cấu hình tập trung
├── src/medrag/
│   ├── data/                   # collector PubMed, làm sạch, chunking
│   ├── retrieval/              # embeddings, FAISS, BM25, hybrid (RRF)
│   ├── training/               # (deprecated) quy trình huấn luyện v1
│   ├── reranking/              # cross-encoder reranker
│   ├── llm/                    # prompts + generator (mock/hf/openai)
│   ├── rag/                    # pipeline end-to-end
│   ├── evaluation/             # Recall@k, MRR, nDCG, EM/F1, RAGAS
│   └── utils/                  # io, logging, seeding
├── scripts/                    # 01→05, chạy theo thứ tự
├── notebooks/                  # quy trình fine-tuning (v1 tham chiếu, v3 bản chính)
├── app/
│   ├── backend/main.py         # FastAPI
│   └── frontend/streamlit_app.py
├── docker/                     # Dockerfile + compose
└── tests/test_smoke.py
```

---

## Tech Stack

**Retrieval:** BGE-small (fine-tuned) · FAISS · bm25s · sentence-transformers · cross-encoder
**Dữ liệu:** PubMed (NCBI E-utilities) · MedMCQA · MedQuAD · HealthcareMagic · NFCorpus · BioASQ
**Huấn luyện:** Kaggle/Colab T4 · MultipleNegativesRankingLoss · mixed precision (AMP)
**Triển khai:** FastAPI · Streamlit · Docker

---

## Kiểm thử

```powershell
pytest -q    # smoke tests, không cần model nặng
```