#!/usr/bin/env python
"""05 — Đánh giá retrieval trên PubMedQA (self-contained).

VẤN ĐỀ CỦA BẢN CŨ:
  Bản cũ đo bằng cách khớp PMID câu hỏi PubMedQA với index chính (data/index),
  nhưng index đó build từ abstract thu theo search_terms (diabetes/hypertension/...)
  → gần như KHÔNG chứa PMID của PubMedQA → recall ~0, số liệu vô nghĩa.

CÁCH ĐÚNG (mặc định, --mode selfcontained):
  Build một index ĐÁNH GIÁ riêng NGAY TỪ context của PubMedQA:
  mỗi context là 1 "document" mang đúng PMID của nó. Ground-truth của một
  câu hỏi = context (PMID) tương ứng. Nhờ vậy corpus luôn chứa đáp án đúng,
  đo được Recall@k / MRR / nDCG có ý nghĩa cho việc so sánh baseline vs finetuned.

  Đây là đánh giá "retrieval thuần" (câu hỏi -> đúng passage của nó trong một
  hồ passage), chuẩn để so sánh chất lượng embedding của hai model.

CÁCH DÙNG:
  # đo model đang active trong config (đổi retriever.active để so sánh)
  python scripts/05_evaluate.py --limit 500

  # ép model cụ thể, bỏ qua config
  python scripts/05_evaluate.py --limit 500 --model data/models/MedicalRetriever-v1
  python scripts/05_evaluate.py --limit 500 --model BAAI/bge-small-en-v1.5

  # (cũ) đo trên index chính đã build sẵn — chỉ đúng nếu index chứa PMID PubMedQA
  python scripts/05_evaluate.py --mode mainindex --limit 200
"""
import argparse
import json

import numpy as np

from medrag.config import CONFIG
from medrag.evaluation.retrieval_metrics import evaluate_retrieval
from medrag.retrieval.embeddings import EmbeddingModel
from medrag.retrieval.faiss_index import FaissIndex
from medrag.retrieval.retriever import Retriever


def _load_pubmedqa(limit: int, subset: str = "pqa_labeled") -> list[dict]:
    """Trả về list {question, pmid, context}. Cần thư viện datasets."""
    from datasets import load_dataset

    ds = load_dataset("pubmed_qa", subset, split="train")
    out = []
    for ex in ds:
        pmid = str(ex.get("pubid", "")).strip()
        q = (ex.get("question") or "").strip()
        ctxs = ex.get("context", {}).get("contexts", []) if isinstance(ex.get("context"), dict) else []
        context = " ".join(ctxs).strip()
        if q and pmid and context:
            out.append({"question": q, "pmid": pmid, "context": context})
        if len(out) >= limit:
            break
    return out


def _load_distractors(n: int, exclude_pmids: set[str]) -> list[dict]:
    """Nạp thêm n passage phân tâm (distractor) từ pqa_artificial, loại pmid đã có trong gold.

    Distractor làm corpus lớn hơn -> bài đánh giá khó hơn -> phân biệt được model tốt/kém.
    """
    if n <= 0:
        return []
    from datasets import load_dataset

    ds = load_dataset("pubmed_qa", "pqa_artificial", split="train", streaming=True)
    seen = set(exclude_pmids)
    out = []
    for ex in ds:
        pmid = str(ex.get("pubid", "")).strip()
        ctxs = ex.get("context", {}).get("contexts", []) if isinstance(ex.get("context"), dict) else []
        context = " ".join(ctxs).strip()
        if pmid and context and pmid not in seen:
            out.append({"pmid": pmid, "context": context})
            seen.add(pmid)
        if len(out) >= n:
            break
    return out


def eval_selfcontained(rows: list[dict], model_name: str | None, k_values,
                       distractors: int = 0, use_hybrid: bool = False,
                       rrf_k: int = 60, batch_size: int = 64) -> dict:
    """Build index từ context PubMedQA (+ distractor tùy chọn) rồi truy hồi câu hỏi.

    use_hybrid=True: kết hợp dense (FAISS) + BM25 bằng RRF, so với dense-only.
    """
    embedder = EmbeddingModel(model_name_or_path=model_name, config=CONFIG)

    gold_pmids = {r["pmid"] for r in rows}
    extra = _load_distractors(distractors, gold_pmids)
    if extra:
        print(f"Đã thêm {len(extra)} distractor -> corpus = {len(rows) + len(extra)} passage")

    # gold trước, distractor sau; ground-truth vẫn là pmid của từng câu hỏi
    contexts = [r["context"] for r in rows] + [d["context"] for d in extra]
    pmids = [r["pmid"] for r in rows] + [d["pmid"] for d in extra]
    questions = [r["question"] for r in rows]

    # dense index trong RAM
    ctx_emb = embedder.encode(contexts, show_progress=True, is_query=False)
    index = FaissIndex(CONFIG)
    index.build(ctx_emb, [{"pmid": p} for p in pmids])
    q_emb = embedder.encode(questions, show_progress=True, is_query=True)
    max_k = max(k_values)

    # bm25 index (nếu hybrid)
    bm25 = None
    if use_hybrid:
        from medrag.retrieval.bm25_retriever import BM25Retriever
        from medrag.retrieval.hybrid_retriever import reciprocal_rank_fusion
        bm25 = BM25Retriever(CONFIG)
        bm25.build(contexts, [{"pmid": p} for p in pmids])

    all_retrieved, all_relevant = [], []
    for i in range(len(rows)):
        dense_hits = index.search(q_emb[i], top_k=max_k)
        if use_hybrid:
            bm25_hits = bm25.search(questions[i], top_k=max_k)
            fused = reciprocal_rank_fusion([dense_hits, bm25_hits], rrf_k=rrf_k, top_k=max_k)
            all_retrieved.append([h.get("pmid", "") for h in fused])
        else:
            all_retrieved.append([h.get("pmid", "") for h in dense_hits])
        all_relevant.append({pmids[i]})

    return evaluate_retrieval(all_retrieved, all_relevant, k_values=k_values)


def eval_mainindex(rows: list[dict], model_name: str | None, index_dir: str | None, k_values) -> dict:
    """Đo trên index chính đã build sẵn (chỉ đúng nếu index chứa PMID của PubMedQA)."""
    retriever = Retriever(CONFIG, model_name_or_path=model_name).load(index_dir)
    max_k = max(k_values)
    all_retrieved, all_relevant = [], []
    for r in rows:
        hits = retriever.retrieve(r["question"], top_k=max_k)
        all_retrieved.append([h.get("pmid", "") for h in hits])
        all_relevant.append({r["pmid"]})
    return evaluate_retrieval(all_retrieved, all_relevant, k_values=k_values)


def _parse_ids(raw) -> list[str]:
    """relevant_passage_ids có thể là string '[123, 456]' hoặc list."""
    import ast
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            return [str(x) for x in ast.literal_eval(raw.strip())]
        except Exception:
            return [t.strip() for t in raw.strip("[]").split(",") if t.strip()]
    return []


def _load_bioasq(limit: int, corpus_limit: int = 40000):
    """Nạp rag-mini-bioasq: (queries, corpus).

    queries: [{question, relevant_ids:set[str]}]
    corpus : ([texts], [ids]) — hồ passage để build index.
    Free trên HuggingFace, không cần đăng ký.
    """
    from datasets import load_dataset

    repo = "rag-datasets/rag-mini-bioasq"

    def _first_split(name: str):
        """Lấy split đầu tiên có sẵn của subset (tránh hard-code sai tên split)."""
        from datasets import get_dataset_split_names
        try:
            splits = get_dataset_split_names(repo, name)
        except Exception:
            splits = []
        return load_dataset(repo, name, split=splits[0] if splits else "train")

    qa = _first_split("question-answer-passages")
    corpus_ds = _first_split("text-corpus")

    # corpus: id -> passage text (bỏ nan)
    texts, ids = [], []
    for row in corpus_ds:
        pid = str(row.get("id", "")).strip()
        passage = row.get("passage")
        if pid and passage and isinstance(passage, str) and passage.strip():
            texts.append(passage.strip())
            ids.append(pid)
        if len(texts) >= corpus_limit:
            break
    corpus_id_set = set(ids)

    queries = []
    for row in qa:
        q = (row.get("question") or "").strip()
        rel = set(_parse_ids(row.get("relevant_passage_ids")))
        rel &= corpus_id_set  # chỉ giữ relevant nằm trong corpus đã nạp
        if q and rel:
            queries.append({"question": q, "relevant_ids": rel})
        if len(queries) >= limit:
            break
    return queries, (texts, ids)


def eval_bioasq(model_name: str | None, k_values, limit: int,
                use_hybrid: bool = False, rrf_k: int = 60) -> dict:
    """Đánh giá retrieval trên BioASQ (benchmark chuẩn, ground-truth chuyên gia)."""
    queries, (texts, ids) = _load_bioasq(limit)
    print(f"BioASQ: {len(queries)} câu hỏi | corpus {len(texts)} passage")

    embedder = EmbeddingModel(model_name_or_path=model_name, config=CONFIG)
    ctx_emb = embedder.encode(texts, show_progress=True, is_query=False)
    index = FaissIndex(CONFIG)
    index.build(ctx_emb, [{"pid": p} for p in ids])
    q_emb = embedder.encode([q["question"] for q in queries], show_progress=True, is_query=True)
    max_k = max(k_values)

    bm25 = None
    if use_hybrid:
        from medrag.retrieval.bm25_retriever import BM25Retriever
        from medrag.retrieval.hybrid_retriever import reciprocal_rank_fusion
        bm25 = BM25Retriever(CONFIG)
        bm25.build(texts, [{"pid": p} for p in ids])

    all_retrieved, all_relevant = [], []
    for i, q in enumerate(queries):
        dense_hits = index.search(q_emb[i], top_k=max_k)
        if use_hybrid:
            bm25_hits = bm25.search(q["question"], top_k=max_k)
            fused = reciprocal_rank_fusion(
                [[{**h, "pmid": h.get("pid")} for h in dense_hits],
                 [{**h, "pmid": h.get("pid")} for h in bm25_hits]],
                rrf_k=rrf_k, top_k=max_k)
            all_retrieved.append([h.get("pid") or h.get("pmid") for h in fused])
        else:
            all_retrieved.append([h.get("pid") for h in dense_hits])
        all_relevant.append(q["relevant_ids"])

    return evaluate_retrieval(all_retrieved, all_relevant, k_values=k_values)


def main():
    ap = argparse.ArgumentParser(description="Đánh giá retrieval trên PubMedQA")
    ap.add_argument("--limit", type=int, default=500, help="Số câu hỏi đánh giá")
    ap.add_argument("--mode", choices=["selfcontained", "mainindex", "bioasq"], default="selfcontained")
    ap.add_argument("--model", default=None, help="Ép model cụ thể (path hoặc HF id). Mặc định theo config.")
    ap.add_argument("--index-dir", default=None, help="(mode=mainindex) thư mục index")
    ap.add_argument("--subset", default="pqa_labeled", help="pqa_labeled | pqa_artificial")
    ap.add_argument("--distractors", type=int, default=0,
                    help="(mode=selfcontained) số passage phân tâm thêm vào corpus để tăng độ khó")
    ap.add_argument("--hybrid", action="store_true",
                    help="Kết hợp dense + BM25 bằng RRF (so với dense-only)")
    ap.add_argument("--rrf-k", type=int, default=60, help="Hằng số RRF (mặc định 60)")
    args = ap.parse_args()

    k_values = CONFIG.get("evaluation.k_values", [5, 10, 20, 50])

    # BioASQ có loader + corpus riêng, không dùng PubMedQA
    if args.mode == "bioasq":
        metrics = eval_bioasq(args.model, k_values, args.limit,
                              use_hybrid=args.hybrid, rrf_k=args.rrf_k)
        active = CONFIG.get("retriever.active", "baseline")
        print("\n=== Retrieval metrics (BioASQ) ===")
        print(f"model     : {args.model or f'(config active={active})'}")
        print(f"mode      : bioasq{' + HYBRID(RRF)' if args.hybrid else ' (dense-only)'}")
        print(json.dumps(metrics, indent=2))
        return

    rows = _load_pubmedqa(args.limit, subset=args.subset)
    if not rows:
        print("Không nạp được câu hỏi PubMedQA nào."); return
    print(f"Đã nạp {len(rows)} câu hỏi PubMedQA ({args.subset}).")

    if args.mode == "selfcontained":
        metrics = eval_selfcontained(rows, args.model, k_values,
                                     distractors=args.distractors,
                                     use_hybrid=args.hybrid, rrf_k=args.rrf_k)
    else:
        metrics = eval_mainindex(rows, args.model, args.index_dir, k_values)

    active = CONFIG.get("retriever.active", "baseline")
    model_shown = args.model or f"(config active={active})"
    print("\n=== Retrieval metrics ===")
    print(f"model     : {model_shown}")
    print(f"mode      : {args.mode}{' + HYBRID(RRF)' if args.hybrid else ' (dense-only)'}")
    print(f"n_queries : {len(rows)}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()