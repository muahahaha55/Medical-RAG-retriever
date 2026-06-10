"""FastAPI backend cho MedRAG.

Endpoints:
  GET  /health        — kiểm tra service.
  POST /query         — nhận {question} -> trả answer + passages + citations.

Chạy:
    uvicorn app.backend.main:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from medrag.config import CONFIG
from medrag.rag.pipeline import RAGPipeline
from medrag.utils.io import get_logger

logger = get_logger("medrag.api")

# Giữ pipeline ở scope module để tái dùng giữa các request
_state: dict = {"pipeline": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo pipeline khi service start (lazy nếu index chưa sẵn sàng)."""
    try:
        _state["pipeline"] = RAGPipeline(CONFIG)
        logger.info("RAG pipeline đã sẵn sàng.")
    except Exception as e:  # noqa: BLE001
        logger.warning("Chưa khởi tạo được pipeline (index chưa build?): %s", e)
        _state["pipeline"] = None
    yield
    _state["pipeline"] = None


app = FastAPI(title="MedRAG-Retriever API", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Câu hỏi y khoa")


class PassageOut(BaseModel):
    pmid: str
    title: str = ""
    chunk: str
    score: float | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    passages: list[PassageOut]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "pipeline_ready": _state["pipeline"] is not None}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    pipeline = _state["pipeline"]
    if pipeline is None:
        # thử khởi tạo lại (có thể index vừa được build)
        try:
            pipeline = RAGPipeline(CONFIG)
            _state["pipeline"] = pipeline
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"Pipeline chưa sẵn sàng: {e}")

    result = pipeline.answer(req.question)
    passages = [
        PassageOut(
            pmid=p.get("pmid", ""),
            title=p.get("title", ""),
            chunk=p.get("chunk", ""),
            score=p.get("score"),
            rerank_score=p.get("rerank_score"),
        )
        for p in result.passages
    ]
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        citations=result.citations,
        passages=passages,
    )
