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
                       distractors: int = 0, batch_size: int = 64) -> dict:
    """Build index từ context PubMedQA (+ distractor tùy chọn) rồi truy hồi câu hỏi."""
    embedder = EmbeddingModel(model_name_or_path=model_name, config=CONFIG)

    gold_pmids = {r["pmid"] for r in rows}
    extra = _load_distractors(distractors, gold_pmids)
    if extra:
        print(f"Đã thêm {len(extra)} distractor -> corpus = {len(rows) + len(extra)} passage")

    # gold trước, distractor sau; ground-truth vẫn là pmid của từng câu hỏi
    contexts = [r["context"] for r in rows] + [d["context"] for d in extra]
    pmids = [r["pmid"] for r in rows] + [d["pmid"] for d in extra]
    questions = [r["question"] for r in rows]

    # 1) encode toàn bộ context làm "corpus" + build index tạm trong RAM
    ctx_emb = embedder.encode(contexts, show_progress=True, is_query=False)
    index = FaissIndex(CONFIG)
    index.build(ctx_emb, [{"pmid": p} for p in pmids])

    # 2) encode câu hỏi và truy hồi
    q_emb = embedder.encode(questions, show_progress=True, is_query=True)
    max_k = max(k_values)

    all_retrieved, all_relevant = [], []
    for i in range(len(rows)):
        hits = index.search(q_emb[i], top_k=max_k)
        all_retrieved.append([h.get("pmid", "") for h in hits])
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


def main():
    ap = argparse.ArgumentParser(description="Đánh giá retrieval trên PubMedQA")
    ap.add_argument("--limit", type=int, default=500, help="Số câu hỏi đánh giá")
    ap.add_argument("--mode", choices=["selfcontained", "mainindex"], default="selfcontained")
    ap.add_argument("--model", default=None, help="Ép model cụ thể (path hoặc HF id). Mặc định theo config.")
    ap.add_argument("--index-dir", default=None, help="(mode=mainindex) thư mục index")
    ap.add_argument("--subset", default="pqa_labeled", help="pqa_labeled | pqa_artificial")
    ap.add_argument("--distractors", type=int, default=0,
                    help="(mode=selfcontained) số passage phân tâm thêm vào corpus để tăng độ khó")
    args = ap.parse_args()

    rows = _load_pubmedqa(args.limit, subset=args.subset)
    if not rows:
        print("Không nạp được câu hỏi PubMedQA nào.");
        return
    print(f"Đã nạp {len(rows)} câu hỏi PubMedQA ({args.subset}).")

    k_values = CONFIG.get("evaluation.k_values", [5, 10, 20, 50])

    if args.mode == "selfcontained":
        metrics = eval_selfcontained(rows, args.model, k_values, distractors=args.distractors)
    else:
        metrics = eval_mainindex(rows, args.model, args.index_dir, k_values)

    active = CONFIG.get("retriever.active", "baseline")
    model_shown = args.model or f"(config active={active})"
    print("\n=== Retrieval metrics ===")
    print(f"model     : {model_shown}")
    print(f"mode      : {args.mode}")
    print(f"n_queries : {len(rows)}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()