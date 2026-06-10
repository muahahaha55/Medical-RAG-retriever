"""Pipeline RAG đầu-cuối: retrieve -> rerank -> generate -> cite.

Đây là điểm vào chính cho cả backend API lẫn script đánh giá.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from medrag.config import Config, CONFIG
from medrag.llm.generator import LLMGenerator
from medrag.reranking.cross_encoder import Reranker
from medrag.retrieval.retriever import Retriever
from medrag.utils.io import get_logger

logger = get_logger("medrag.pipeline")


@dataclass
class RAGResult:
    question: str
    answer: str
    passages: list[dict] = field(default_factory=list)   # passage sau rerank
    retrieved: list[dict] = field(default_factory=list)  # passage trước rerank
    citations: list[str] = field(default_factory=list)   # danh sách PMID

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "passages": self.passages,
            "citations": self.citations,
        }


class RAGPipeline:
    def __init__(
        self,
        config: Config = CONFIG,
        index_dir: str | Path | None = None,
        retriever_model: str | None = None,
    ):
        self.cfg = config
        self.retriever = Retriever(config=config, model_name_or_path=retriever_model)
        self.retriever.load(index_dir)
        self.reranker = Reranker(config=config)
        self.llm = LLMGenerator(config=config)
        self.retrieve_k = int(config.get("reranker.retrieve_top_k", 50))
        self.rerank_k = int(config.get("reranker.rerank_top_k", 5))

    def answer(self, question: str) -> RAGResult:
        # 1) Retrieve
        retrieved = self.retriever.retrieve(question, top_k=self.retrieve_k)
        # 2) Rerank
        passages = self.reranker.rerank(question, retrieved, top_k=self.rerank_k)
        # 3) Generate
        answer = self.llm.generate(question, passages)
        # 4) Citations
        citations = sorted({p.get("pmid", "N/A") for p in passages})
        logger.info("Q: %s -> %d passage, %d citation", question[:60], len(passages), len(citations))
        return RAGResult(
            question=question,
            answer=answer,
            passages=passages,
            retrieved=retrieved,
            citations=citations,
        )
