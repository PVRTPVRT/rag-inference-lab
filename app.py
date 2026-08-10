"""Streamlit UI for local RAG with measured inference metrics."""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from backends.ollama_backend import generate as ollama_generate
from backends.vllm_backend import generate as vllm_generate
from rag.prompting import INSUFFICIENT_EVIDENCE, build_rag_prompt, retrieval_confidence, should_abstain
from rag.retriever import build_context, retrieve

st.set_page_config(page_title="RAG Inference Lab", layout="wide")
st.title("RAG Inference Lab")
st.caption("Local RAG with Ollama and vLLM; metrics are labeled by measurement source")

with st.sidebar:
    st.header("Engine configuration")
    engine = st.selectbox("Inference engine", ["Ollama", "vLLM"])
    if engine == "Ollama":
        model = st.selectbox(
            "Model",
            [
                "qwen2.5:7b-instruct-q4_K_M",
                "qwen2.5:7b-instruct-q8_0",
                "llama3.1:8b-instruct-q4_K_M",
            ],
        )
        base_url = None
        server_profile = "ollama"
    else:
        model = st.selectbox("Model", ["Qwen/Qwen2.5-7B-Instruct"])
        base_url = st.text_input("OpenAI-compatible URL", "http://localhost:8000/v1")
        server_profile = st.text_input(
            "Server profile label",
            "unspecified",
            help="Caching and speculation are server-startup settings; this is a label, not a toggle.",
        )
    top_k = st.slider("RAG top-k chunks", 1, 6, 3)
    retrieval_mode = st.selectbox("Retrieval mode", ["Dense", "Dense + BGE reranker"])
    use_reranker = retrieval_mode == "Dense + BGE reranker"
    candidate_k = st.slider(
        "Dense candidates before reranking",
        top_k,
        24,
        max(top_k, 12),
        disabled=not use_reranker,
    )
    use_abstention = st.checkbox(
        "Experimental retrieval-confidence gate",
        value=False,
        help="Diagnostic only: the default threshold was fitted on 20 small-corpus examples.",
    )
    abstention_threshold = st.slider(
        "Dense-score threshold", 0.0, 1.0, 0.5672, 0.0001, disabled=not use_abstention
    )
    if use_abstention:
        st.warning("This threshold is experimental and is not production-calibrated.")
    max_tokens = st.slider("Maximum output tokens", 32, 512, 128, step=32)

if "history" not in st.session_state:
    st.session_state.history = []

col_chat, col_metrics = st.columns([3, 2])

with col_chat:
    st.subheader("Ask a question")
    query = st.text_input("Query", placeholder="What is speculative decoding and when does it help?")
    if st.button("Generate", type="primary") and query:
        with st.spinner("Retrieving context..."):
            chunks = retrieve(
                query, top_k=top_k, rerank=use_reranker, candidate_k=candidate_k
            )
            context = build_context(chunks)
        st.markdown("**Retrieved chunks**")
        for index, chunk in enumerate(chunks):
            rerank_label = (
                f", rerank: {chunk['rerank_score']}" if "rerank_score" in chunk else ""
            )
            with st.expander(
                f"[S{index + 1}] {chunk['source']} "
                f"(dense: {chunk['score']}{rerank_label})"
            ):
                st.text(chunk["text"][:400] + "...")
        confidence = retrieval_confidence(chunks)
        if use_abstention and should_abstain(chunks, abstention_threshold):
            st.warning(
                f"{INSUFFICIENT_EVIDENCE} Best dense score: {confidence}. "
                "Generation was skipped by the experimental gate."
            )
            st.stop()
        prompt = build_rag_prompt(query, chunks)
        with st.spinner(f"Generating with {engine} / {model}..."):
            try:
                if engine == "Ollama":
                    answer, metrics = ollama_generate(
                        prompt, model=model, max_tokens=max_tokens
                    )
                else:
                    answer, metrics = vllm_generate(
                        prompt,
                        model=model,
                        base_url=base_url,
                        server_profile=server_profile,
                        max_tokens=max_tokens,
                    )
            except Exception as exc:
                st.error(f"Backend error: {exc}")
                st.stop()
        st.markdown(f"**Answer:**\n\n{answer}")
        st.session_state.history.append(metrics)

with col_metrics:
    st.subheader("Inference metrics")
    if st.session_state.history:
        latest = st.session_state.history[-1]
        ttft = latest.get("ttft_ms")
        tps = latest.get("decode_tps")
        vram = latest.get("vram_used_mb")
        metric_columns = st.columns(3)
        metric_columns[0].metric("Client TTFT", f"{ttft} ms" if ttft is not None else "N/A")
        metric_columns[1].metric("Decode TPS", f"{tps} tok/s" if tps is not None else "N/A")
        metric_columns[2].metric("Device VRAM used", f"{vram} MB" if vram and vram > 0 else "N/A")
        st.caption(
            f"Engine: {latest['engine']} | Model: {latest['model']} | "
            f"E2E: {latest['e2e_ms']} ms | Tokens: {latest.get('output_tokens')}"
        )
        st.caption(f"Token count source: {latest.get('token_count_source')}")
        if len(st.session_state.history) > 1:
            frame = pd.DataFrame(st.session_state.history)
            visible = [
                column
                for column in (
                    "engine",
                    "model",
                    "ttft_ms",
                    "decode_tps",
                    "vram_used_mb",
                    "e2e_ms",
                )
                if column in frame.columns
            ]
            st.dataframe(frame[visible], use_container_width=True)
    else:
        st.info("Run a query to see metrics.")

    st.divider()
    st.markdown(
        "- **TTFT** is client-observed time to the first non-empty text chunk.\n"
        "- **Decode TPS** uses Ollama `eval_count/eval_duration` or vLLM OpenAI usage tokens.\n"
        "- **VRAM used** is total device usage, not isolated model weight size."
    )
