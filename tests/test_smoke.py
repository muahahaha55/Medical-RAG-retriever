"""Smoke tests — chạy nhanh, không cần tải model nặng.

Kiểm tra: config, cleaning, chunking (fallback word-split), và các metric.
    pytest -q
"""
from medrag.config import load_config
from medrag.data.preprocess import clean_text, clean_records
from medrag.data.chunking import Chunker
from medrag.evaluation.retrieval_metrics import (
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    evaluate_retrieval,
)
from medrag.evaluation.qa_metrics import exact_match, f1_score


def test_config_loads():
    cfg = load_config()
    assert cfg.get("project.name") == "MedRAG-Retriever"
    assert cfg.get("reranker.rerank_top_k") == 5


def test_clean_text():
    raw = "<p>Metformin   reduces&nbsp;HbA1c</p>"
    out = clean_text(raw)
    assert "<p>" not in out
    assert "Metformin" in out
    assert "  " not in out  # khoảng trắng đã gộp


def test_clean_records_filters_short():
    recs = [
        {"pmid": "1", "title": "A", "abstract": "short"},
        {"pmid": "2", "title": "B", "abstract": "x" * 200},
    ]
    out = list(clean_records(recs))
    assert len(out) == 1
    assert out[0]["pmid"] == "2"


def test_chunking_word_fallback(monkeypatch):
    chunker = Chunker()
    # ép fallback word-split để không tải tokenizer
    monkeypatch.setattr(Chunker, "tokenizer", property(lambda self: False))
    chunker._tokenizer = False
    chunker.chunk_size = 10
    chunker.overlap = 2
    text = " ".join(f"w{i}" for i in range(35))
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 3


def test_retrieval_metrics():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"c"}
    assert recall_at_k(retrieved, relevant, 5) == 1.0
    assert recall_at_k(retrieved, relevant, 2) == 0.0
    assert reciprocal_rank(retrieved, relevant) == 1 / 3
    assert 0 < ndcg_at_k(retrieved, relevant, 4) <= 1.0


def test_evaluate_retrieval_aggregate():
    res = evaluate_retrieval(
        [["a", "b"], ["x", "y"]],
        [{"a"}, {"y"}],
        k_values=[1, 2],
    )
    assert res["recall@2"] == 1.0
    assert res["mrr"] == (1.0 + 0.5) / 2


def test_qa_metrics():
    assert exact_match("The answer", "answer") == 1.0
    assert f1_score("metformin reduces glucose", "metformin lowers glucose") > 0.5
