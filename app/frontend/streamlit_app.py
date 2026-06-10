"""Streamlit frontend cho MedRAG.

Hiển thị: cửa sổ chat, tài liệu truy hồi, panel trích dẫn, điểm tin cậy.

Chạy:
    streamlit run app/frontend/streamlit_app.py

Cấu hình URL backend qua biến môi trường BACKEND_URL (mặc định localhost:8080).
"""
from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

st.set_page_config(page_title="MedRAG-Retriever", page_icon="🩺", layout="wide")

st.title("🩺 MedRAG-Retriever")
st.caption("Hỏi đáp y khoa dựa trên bằng chứng từ PubMed (RAG + retriever fine-tuned)")

# Lịch sử hội thoại
if "history" not in st.session_state:
    st.session_state.history = []  # list[(question, response_dict)]

with st.sidebar:
    st.header("Cấu hình")
    backend = st.text_input("Backend URL", BACKEND_URL)
    try:
        h = requests.get(f"{backend}/health", timeout=5).json()
        ready = h.get("pipeline_ready")
        st.success("Backend OK" + (" — pipeline sẵn sàng" if ready else " — pipeline CHƯA sẵn sàng"))
    except Exception as e:  # noqa: BLE001
        st.error(f"Không kết nối được backend: {e}")

question = st.chat_input("Nhập câu hỏi y khoa...")

# Render lịch sử
for q, resp in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(resp.get("answer", ""))

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Đang truy hồi & sinh câu trả lời..."):
            try:
                r = requests.post(f"{backend}/query", json={"question": question}, timeout=120)
                r.raise_for_status()
                data = r.json()
            except Exception as e:  # noqa: BLE001
                st.error(f"Lỗi gọi API: {e}")
                data = None

        if data:
            st.write(data["answer"])

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📚 Tài liệu hỗ trợ")
                for i, p in enumerate(data.get("passages", []), 1):
                    score = p.get("rerank_score") or p.get("score")
                    with st.expander(f"[{i}] PMID {p['pmid']} — {p.get('title','')[:60]}"):
                        st.write(p["chunk"])
                        if score is not None:
                            st.caption(f"Điểm liên quan: {score:.4f}")
            with col2:
                st.subheader("🔖 Trích dẫn")
                for pmid in data.get("citations", []):
                    st.markdown(f"- PMID [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")

            st.session_state.history.append((question, data))
