# Notebooks

Đặt notebook thử nghiệm (Google Colab T4) ở đây:

- `01_explore_pubmed.ipynb` — khám phá dữ liệu thu thập.
- `02_train_retriever.ipynb` — fine-tune retriever trên Colab GPU.
- `03_evaluation.ipynb` — chạy benchmark + vẽ biểu đồ ablation.

Gợi ý mở đầu mỗi notebook:
```python
import sys; sys.path.append("../src")
from medrag.config import CONFIG
```
