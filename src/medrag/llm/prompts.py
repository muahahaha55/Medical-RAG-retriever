"""Mẫu prompt cho medical RAG. Tách riêng để dễ chỉnh sửa/version."""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful medical assistant. "
    "Answer ONLY using the provided evidence passages. "
    "If the evidence is insufficient to answer, explicitly say you do not know. "
    "Always cite the source documents by their PMID. "
    "Do not fabricate facts or citations."
)


def build_context_block(passages: list[dict]) -> str:
    """Định dạng các passage thành khối evidence đánh số kèm PMID."""
    lines = []
    for i, p in enumerate(passages, start=1):
        pmid = p.get("pmid", "N/A")
        chunk = p.get("chunk", "")
        lines.append(f"[{i}] (PMID: {pmid}) {chunk}")
    return "\n\n".join(lines)


def build_user_prompt(question: str, passages: list[dict]) -> str:
    """Ghép câu hỏi + evidence thành prompt cho người dùng."""
    context = build_context_block(passages)
    return (
        f"Evidence passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "1. Answer concisely based only on the evidence above.\n"
        "2. After the answer, add a 'Sources:' section listing the PMIDs you used.\n"
        "3. If the evidence does not contain the answer, say so clearly."
    )
