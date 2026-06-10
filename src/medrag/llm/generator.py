"""Bộ sinh câu trả lời LLM, hỗ trợ nhiều backend.

backend:
  - "mock"             : không cần model, dùng cho dev/test pipeline.
  - "openai_compatible": gọi endpoint OpenAI-compatible (vLLM, Ollama, ...).
  - "hf"               : load model HuggingFace cục bộ (cần GPU).
"""
from __future__ import annotations

from medrag.config import Config, CONFIG
from medrag.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from medrag.utils.io import get_logger

logger = get_logger("medrag.llm")


class LLMGenerator:
    def __init__(self, config: Config = CONFIG):
        self.cfg = config
        l = config.raw.get("llm", {})
        self.backend = l.get("backend", "mock")
        self.model = l.get("model", "Qwen/Qwen2.5-7B-Instruct")
        self.base_url = l.get("base_url", "http://localhost:8000/v1")
        self.api_key = l.get("api_key", "EMPTY")
        self.max_new_tokens = int(l.get("max_new_tokens", 512))
        self.temperature = float(l.get("temperature", 0.1))
        self._pipe = None

    # -- public ------------------------------------------------------------
    def generate(self, question: str, passages: list[dict]) -> str:
        user_prompt = build_user_prompt(question, passages)
        if self.backend == "mock":
            return self._mock(question, passages)
        if self.backend == "openai_compatible":
            return self._openai_compatible(user_prompt)
        if self.backend == "hf":
            return self._hf(user_prompt)
        raise ValueError(f"Backend không hỗ trợ: {self.backend}")

    # -- backends ----------------------------------------------------------
    def _mock(self, question: str, passages: list[dict]) -> str:
        """Trả lời giả lập để kiểm thử pipeline mà không cần model."""
        if not passages:
            return "I do not have enough evidence to answer this question."
        pmids = sorted({p.get("pmid", "N/A") for p in passages})
        snippet = passages[0].get("chunk", "")[:200]
        return (
            f"[MOCK ANSWER] Based on the retrieved evidence for: \"{question}\"\n\n"
            f"{snippet}...\n\n"
            f"Sources:\n" + "\n".join(f"PMID: {p}" for p in pmids)
        )

    def _openai_compatible(self, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""

    def _hf(self, user_prompt: str) -> str:
        if self._pipe is None:
            import torch
            from transformers import pipeline

            logger.info("Loading HF model: %s", self.model)
            self._pipe = pipeline(
                "text-generation",
                model=self.model,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        out = self._pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )
        return out[0]["generated_text"][-1]["content"]
