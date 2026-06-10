"""Fine-tune retriever bằng contrastive learning (Sentence Transformers).

Mặc định dùng MultipleNegativesRankingLoss (MNRL) — biến thể InfoNCE với
in-batch negatives, rất hiệu quả cho retrieval. Nếu file cặp có cột
'negative', nó sẽ được dùng làm hard negative.

Backbone: BioBERT. Kết quả lưu thành SentenceTransformer tại
data/models/MedicalRetriever-v1.
"""
from __future__ import annotations

from pathlib import Path

from medrag.config import Config, CONFIG
from medrag.utils.io import get_logger, read_jsonl, set_seed

logger = get_logger("medrag.train")


def _build_examples(pairs_path: str | Path):
    """Chuyển file cặp .jsonl thành list InputExample."""
    from sentence_transformers import InputExample

    examples = []
    for rec in read_jsonl(pairs_path):
        texts = [rec["query"], rec["positive"]]
        if rec.get("negative"):
            texts.append(rec["negative"])
        examples.append(InputExample(texts=texts))
    logger.info("Đã tạo %d InputExample", len(examples))
    return examples


def train_retriever(
    pairs_path: str | Path,
    config: Config = CONFIG,
    output_path: str | Path | None = None,
) -> Path:
    """Huấn luyện và lưu retriever fine-tuned."""
    from sentence_transformers import SentenceTransformer, losses, models
    from torch.utils.data import DataLoader

    t = config.raw.get("training", {})
    set_seed(int(config.get("project.seed", 42)))

    base_model = t.get("base_model", "dmis-lab/biobert-base-cased-v1.2")
    epochs = int(t.get("epochs", 3))
    batch_size = int(t.get("batch_size", 32))
    lr = float(t.get("learning_rate", 2e-5))
    warmup_ratio = float(t.get("warmup_ratio", 0.1))
    out = Path(output_path) if output_path else config.path("paths.models_dir") / "MedicalRetriever-v1"

    # BioBERT là model BERT thuần -> bọc thêm pooling để thành sentence encoder
    word_emb = models.Transformer(base_model, max_seq_length=int(config.get("retriever.max_seq_length", 384)))
    pooling = models.Pooling(word_emb.get_word_embedding_dimension(), pooling_mode="mean")
    model = SentenceTransformer(modules=[word_emb, pooling])

    examples = _build_examples(pairs_path)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)

    loss_name = t.get("loss", "mnrl")
    if loss_name == "mnrl":
        train_loss = losses.MultipleNegativesRankingLoss(model)
    else:  # InfoNCE-style cũng dùng MNRL trong sentence-transformers
        train_loss = losses.MultipleNegativesRankingLoss(model)
    logger.info("Loss: %s | epochs=%d | bs=%d | lr=%.1e", loss_name, epochs, batch_size, lr)

    warmup_steps = int(len(loader) * epochs * warmup_ratio)
    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=str(out),
        show_progress_bar=True,
    )
    logger.info("Đã lưu model fine-tuned tại %s", out)
    return out
