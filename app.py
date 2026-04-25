"""
RAG Inference Lab — Streamlit UI
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from rag.retriever import retrieve, build_context
from backends.ollama_backend import generate as ollama_gen
from backends.vllm_backend import generate as vllm_gen

st.set_page_config(page_title="RAG Inference Lab", layout="wide")
st.title("RAG Inference Lab")
st.caption("Local RAG with live inference metrics — compare engines, quantization, and caching")

# ── Sidebar config ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Engine Config")
    engine = st.selectbox("Inference engine", ["Ollama", "vLLM"])

    if engine == "Ollama":
        model = st.selectbox(
            "Model (quantization)",
            ["qwen2.5:7b-instruct-q4_K_M", "qwen2.5:7b-instruct-q8_0", "llama3.1:8b-instruct-q4_K_M"],
        )
        prefix_caching = False
    else:
        model = st.selectbox("Model", ["Qwen/Qwen2.5-7B-Instruct"])
        prefix_caching = st.toggle("Prefix caching (must be enabled at server startup)", value=True)

    top_k = st.slider("RAG top-k chunks", 1, 6, 3)
    st.divider()
    st.markdown("**Start servers:**")
    st.code("ollama serve", language="bash")
    st.code(
        "vllm serve Qwen/Qwen2.5-7B-Instruct \\\n  --enable-prefix-caching --port 8000",
        language="bash",
    )

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of metrics dicts

# ── Main layout ───────────────────────────────────────────────────────────────
col_chat, col_metrics = st.columns([3, 2])

with col_chat:
    st.subheader("Ask a question")
    query = st.text_input("Query", placeholder="What is speculative decoding and when does it help?")

    if st.button("Generate", type="primary") and query:
        with st.spinner("Retrieving context..."):
            chunks = retrieve(query, top_k=top_k)
            context = build_context(chunks)

        prompt = f"""Use the following retrieved context to answer the question.
If the context doesn't contain the answer, say so clearly.

Context:
{context}

Question: {query}
Answer:"""

        st.markdown("**Retrieved chunks:**")
        for i, c in enumerate(chunks):
            with st.expander(f"[{i+1}] {c['source']} (score: {c['score']})"):
                st.text(c["text"][:400] + "...")

        answer_box = st.empty()
        st.markdown("---")

        with st.spinner(f"Generating with {engine} / {model}..."):
            try:
                if engine == "Ollama":
                    answer, metrics = ollama_gen(prompt, model=model)
                else:
                    answer, metrics = vllm_gen(prompt, model=model, prefix_caching=prefix_caching)
            except Exception as e:
                st.error(f"Backend error: {e}")
                st.stop()

        answer_box.markdown(f"**Answer:**\n\n{answer}")
        st.session_state.history.append(metrics)

with col_metrics:
    st.subheader("Inference Metrics")

    if st.session_state.history:
        latest = st.session_state.history[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("TTFT", f"{latest['ttft_ms']} ms")
        m2.metric("TPS", f"{latest['tps']} tok/s")
        m3.metric("VRAM", f"{latest['vram_mb']} MB" if latest["vram_mb"] > 0 else "N/A")

        st.caption(f"Engine: {latest['engine']} | Model: {latest['model']} | E2E: {latest['e2e_ms']} ms")

        if len(st.session_state.history) > 1:
            st.divider()
            st.markdown("**History (all runs)**")
            df = pd.DataFrame(st.session_state.history)[
                ["engine", "model", "ttft_ms", "tps", "vram_mb", "e2e_ms"]
            ]
            st.dataframe(df, use_container_width=True)

            st.line_chart(df[["ttft_ms", "tps"]].rename(columns={"ttft_ms": "TTFT (ms)", "tps": "TPS"}))
    else:
        st.info("Run a query to see metrics here.")

    st.divider()
    st.markdown("**What these numbers mean:**")
    st.markdown(
        "- **TTFT**: Time to first token — affected by prefill length and prefix cache hits\n"
        "- **TPS**: Decode throughput — affected by model size, quantization, batch size\n"
        "- **VRAM**: GPU memory used — Q4 ≈ 4GB, Q8 ≈ 8GB, FP16 ≈ 15GB for 7B"
    )
