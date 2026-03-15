"""
BRSR RAG Pipeline — Streamlit Frontend
Run with: streamlit run app.py
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BRSR Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f1e14 50%, #0d1117 100%);
    min-height: 100vh;
}

/* Hide default streamlit elements */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 4rem;}

/* Hero title */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    font-weight: 400;
    line-height: 1.15;
    color: #e8f5e2;
    letter-spacing: -0.02em;
    margin: 0;
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    color: #6b8f71;
    font-weight: 300;
    letter-spacing: 0.01em;
    margin-top: 0.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(74, 184, 96, 0.12);
    border: 1px solid rgba(74, 184, 96, 0.3);
    color: #4ab860;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 100px;
    margin-bottom: 1rem;
}

/* Cards */
.info-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.info-card:hover {
    border-color: rgba(74, 184, 96, 0.25);
}
.info-card-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4ab860;
    margin-bottom: 0.4rem;
}
.info-card-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #e8f5e2;
}

/* Answer block */
.answer-box {
    background: linear-gradient(135deg, rgba(74,184,96,0.06) 0%, rgba(30,80,40,0.1) 100%);
    border: 1px solid rgba(74, 184, 96, 0.2);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1rem;
}
.answer-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4ab860;
    margin-bottom: 0.8rem;
}
.answer-text {
    font-size: 1.05rem;
    line-height: 1.75;
    color: #d4ecda;
    white-space: pre-wrap;
}

/* Chunk cards */
.chunk-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
}
.chunk-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #4ab860, #1e8a3a);
}
.chunk-meta {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.6rem;
    flex-wrap: wrap;
}
.chunk-badge {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    background: rgba(74,184,96,0.12);
    color: #4ab860;
    border: 1px solid rgba(74,184,96,0.2);
}
.chunk-score {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    background: rgba(255,255,255,0.05);
    color: #aaa;
    border: 1px solid rgba(255,255,255,0.1);
}
.chunk-text {
    font-size: 0.88rem;
    line-height: 1.7;
    color: #8fac95;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #090e0b !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stSlider label {
    color: #6b8f71 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* Input styling */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e8f5e2 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(74,184,96,0.4) !important;
    box-shadow: 0 0 0 3px rgba(74,184,96,0.08) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #2d7a3a 0%, #1e5c28 100%) !important;
    color: #d4ecda !important;
    border: 1px solid rgba(74,184,96,0.3) !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #369943 0%, #26742f 100%) !important;
    box-shadow: 0 4px 20px rgba(74,184,96,0.2) !important;
    transform: translateY(-1px) !important;
}

/* Divider */
.green-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(74,184,96,0.3), transparent);
    margin: 2rem 0;
}

/* Selectbox */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8f5e2 !important;
}

/* Question pills */
.q-pill {
    display: inline-block;
    background: rgba(74,184,96,0.07);
    border: 1px solid rgba(74,184,96,0.15);
    color: #7ec98a;
    font-size: 0.8rem;
    padding: 0.35rem 0.9rem;
    border-radius: 100px;
    margin: 0.25rem;
    cursor: pointer;
    transition: all 0.15s;
}
.q-pill:hover {
    background: rgba(74,184,96,0.15);
    color: #a8e0b0;
}

/* Status messages */
.status-msg {
    font-size: 0.85rem;
    color: #6b8f71;
    font-style: italic;
    padding: 0.6rem 0;
}

/* Metrics row */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.metric-item {
    flex: 1;
    min-width: 120px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-item-val {
    font-size: 1.4rem;
    font-weight: 600;
    color: #4ab860;
}
.metric-item-lbl {
    font-size: 0.68rem;
    color: #6b8f71;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "DATA"
OUT_DIR  = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K               = 5
RELEVANCE_THRESHOLD = 0.45
DEFAULT_GEMINI_MODEL = "gemma-3-27b-it"

_ENCODING = tiktoken.get_encoding("cl100k_base")

SAMPLE_QUESTIONS = [
    "What are the Scope 1 greenhouse gas emissions?",
    "What is the total number of employees?",
    "Does the company have a sustainability or ESG policy?",
    "What is the total energy consumption?",
    "How much water was withdrawn or consumed?",
    "What percentage of employees are women?",
    "Is there a board-level committee for sustainability?",
    "What is the total waste generated?",
]

_COMPANY_ALIAS: dict[str, str] = {
    "heromotocorplimited":              "HEROMOTO",
    "larsentoubrolimited":              "L&T",
    "drreddyslaboratorieslimited":      "DRREDDY",
    "indianoilcorporationlimited":      "IOC",
    "infosyslimited":                   "INFOSYS",
    "oilandnaturalgascorporationlimited": "ONGC",
    "daburindialimited":                "DABURIN",
    "godrejconsumerproductslimited":    "GODREJCP",
    "nhpclimited":                      "NHPC",
    "tataconsultancyserviceslimitedtcs": "TCS",
    "tataconsultancyserviceslimited":   "TCS",
    "hdfcbanklimited":                  "HDFCBANK",
    "relianceindustrieslimited":        "RELIANCE",
    "hcltechnologieslimited":           "HCLTECH",
    "statebankofindia":                 "SBI",
    "nestleindialimited":               "NESTLEIND",
    "britanniaindustrieslimited":       "BRITANNIA",
    "tataconsumerproductslimited":      "TATACP",
    "asianpaintslimited":               "ASIANPAINT",
    "marutisuzukiindialimited":         "MARUTI",
    "suzlonenergylimited":              "SUZLON",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def token_len(text: str) -> int:
    return len(_ENCODING.encode(text or ""))


def parse_vector(x: Any) -> list[float]:
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        return ast.literal_eval(x)
    raise TypeError(f"Unsupported embedding type: {type(x)}")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower().replace("&", "and"))


# ── Cached resource loading ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def load_faiss_assets() -> tuple[faiss.IndexFlatIP, pd.DataFrame, np.ndarray] | None:
    index_path = OUT_DIR / "brsr_faiss_bge_small.index"
    meta_path  = OUT_DIR / "brsr_faiss_metadata.jsonl"
    emb_path   = OUT_DIR / "brsr_chunk_embeddings_bge_small.jsonl"

    if not (index_path.exists() and meta_path.exists()):
        return None

    faiss_index = faiss.read_index(str(index_path))
    meta_df     = pd.read_json(meta_path, lines=True)

    if emb_path.exists():
        embedding_df = pd.read_json(emb_path, lines=True)
        vectors = np.array([parse_vector(v) for v in embedding_df["embedding_vector"]], dtype=np.float32)
        faiss.normalize_L2(vectors)
    else:
        vectors = None

    return faiss_index, meta_df, vectors


def get_available_companies(meta_df: pd.DataFrame) -> list[str]:
    return sorted(meta_df["company"].astype(str).unique().tolist())


def retrieve_top_k(
    query: str,
    company: str,
    faiss_index: faiss.IndexFlatIP,
    meta_df: pd.DataFrame,
    all_vectors: np.ndarray,
    embedder: SentenceTransformer,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    company_mask = meta_df["company"].astype(str) == company
    company_ids  = np.where(company_mask.values)[0]

    if len(company_ids) == 0:
        return pd.DataFrame()

    sub_vecs  = all_vectors[company_ids].copy()
    sub_index = faiss.IndexFlatIP(sub_vecs.shape[1])
    sub_index.add(sub_vecs)

    qvec         = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, lids = sub_index.search(qvec, min(top_k, sub_index.ntotal))

    rows = []
    for score, lid in zip(scores[0], lids[0]):
        if lid < 0:
            continue
        row = meta_df.iloc[int(company_ids[lid])]
        rows.append({
            "score":      float(score),
            "company":    str(row["company"]),
            "page":       str(row.get("page", "")),
            "chunk_id":   str(row.get("chunk_id", "")),
            "chunk_text": str(row.get("chunk_text", "")),
        })
    return pd.DataFrame(rows)


def ask_gemini(company: str, question: str, retrieved_df: pd.DataFrame, model) -> str:
    blocks = [
        f"[Chunk {i+1} | chunk_id={r['chunk_id']} | page={r['page']} | score={r['score']:.4f}]\n{r['chunk_text']}"
        for i, r in retrieved_df.reset_index(drop=True).iterrows()
    ]
    prompt = f"""You are a professional ESG/BRSR analyst. Answer only from the retrieved context below.
If the answer is not present, write: Not found in retrieved context.

Company: {company}
Question: {question}

Retrieved context:
{chr(10).join(blocks)}

Provide:
1) A concise, direct answer
2) Key supporting evidence with chunk_id and page number
"""
    response = model.generate_content(prompt)
    return response.text if hasattr(response, "text") else str(response)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.5rem 0 1rem 0;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.4rem; color: #e8f5e2; line-height: 1.2;'>
            BRSR<br>Intelligence
        </div>
        <div style='font-size: 0.72rem; color: #4ab860; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.3rem;'>
            RAG · ESG · Analytics
        </div>
    </div>
    <div style='height: 1px; background: rgba(255,255,255,0.06); margin-bottom: 1.5rem;'></div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.72rem; color:#6b8f71; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem;'>Retrieval Settings</div>", unsafe_allow_html=True)

    top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=5, help="Number of document chunks to retrieve for context")

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem; color:#6b8f71; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem;'>LLM Settings</div>", unsafe_allow_html=True)

    use_llm = st.toggle("Enable Gemini LLM", value=False, help="Generate a synthesized answer using Gemini in addition to retrieval")

    if use_llm:
        api_key = st.text_input("Google API Key", type="password", placeholder="AIza...", help="Required for Gemini generation")
        gemini_model = st.selectbox("Gemini Model", ["gemma-3-27b-it", "gemini-1.5-pro", "gemini-1.5-flash"], index=0)
    else:
        api_key = ""
        gemini_model = DEFAULT_GEMINI_MODEL

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#3d5c43; line-height: 1.6; padding: 1rem; background: rgba(255,255,255,0.02); border-radius: 10px; border: 1px solid rgba(255,255,255,0.05);'>
        <strong style='color:#4ab860;'>How it works</strong><br><br>
        1. Your query is embedded using BGE-small<br>
        2. FAISS performs semantic search over company chunks<br>
        3. Top-K relevant passages are retrieved<br>
        4. (Optional) Gemini synthesizes a final answer
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
# Hero
st.markdown("""
<div style='margin-bottom: 0.5rem;'>
    <span class='hero-badge'>Business Responsibility & Sustainability Reporting</span>
</div>
<div class='hero-title'>Ask anything about<br>Indian ESG filings.</div>
<div class='hero-subtitle'>Semantic search over BRSR reports · Powered by FAISS + Sentence Transformers</div>
""", unsafe_allow_html=True)

st.markdown("<div class='green-divider'></div>", unsafe_allow_html=True)

# Load assets
assets = load_faiss_assets()

if assets is None:
    st.markdown("""
    <div class='info-card' style='border-color: rgba(255, 160, 80, 0.3);'>
        <div class='info-card-label' style='color: #f0a060;'>⚠ Index Not Found</div>
        <div style='color: #ccc; font-size: 0.95rem; line-height: 1.7;'>
            The FAISS index was not found in <code>outputs/</code>. 
            Please run the notebook end-to-end first to build the index, embeddings, and metadata files.<br><br>
            Expected files:<br>
            &nbsp;&nbsp;• <code>outputs/brsr_faiss_bge_small.index</code><br>
            &nbsp;&nbsp;• <code>outputs/brsr_faiss_metadata.jsonl</code><br>
            &nbsp;&nbsp;• <code>outputs/brsr_chunk_embeddings_bge_small.jsonl</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

faiss_index, meta_df, all_vectors = assets
available_companies = get_available_companies(meta_df)
embedder = load_embedding_model()

# Stats row
total_chunks = len(meta_df)
total_cos    = faiss_index.d
num_companies = len(available_companies)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f"""
    <div class='info-card'>
        <div class='info-card-label'>📄 Total Chunks</div>
        <div class='info-card-value'>{total_chunks:,}</div>
    </div>""", unsafe_allow_html=True)
with col_b:
    st.markdown(f"""
    <div class='info-card'>
        <div class='info-card-label'>🏢 Companies</div>
        <div class='info-card-value'>{num_companies}</div>
    </div>""", unsafe_allow_html=True)
with col_c:
    st.markdown(f"""
    <div class='info-card'>
        <div class='info-card-label'>🔢 Vector Dim</div>
        <div class='info-card-value'>{total_cos}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

# Query input section
col_left, col_right = st.columns([3, 1])

with col_left:
    st.markdown("<div style='font-size:0.8rem; color:#6b8f71; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.5rem;'>Your Question</div>", unsafe_allow_html=True)
    query = st.text_area(
        label="query_input",
        label_visibility="collapsed",
        placeholder="e.g. What are the Scope 1 greenhouse gas emissions?",
        height=100,
        key="query_input",
    )

with col_right:
    st.markdown("<div style='font-size:0.8rem; color:#6b8f71; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.5rem;'>Company</div>", unsafe_allow_html=True)
    selected_company = st.selectbox(
        label="company_select",
        label_visibility="collapsed",
        options=available_companies,
        index=0,
        key="company_select",
    )

# Sample questions
st.markdown("<div style='font-size:0.75rem; color:#3d5c43; margin-bottom:0.4rem; font-weight:500;'>Try a sample question:</div>", unsafe_allow_html=True)
pill_cols = st.columns(4)
for i, sample_q in enumerate(SAMPLE_QUESTIONS):
    with pill_cols[i % 4]:
        if st.button(sample_q[:45] + ("…" if len(sample_q) > 45 else ""), key=f"pill_{i}", use_container_width=True):
            st.session_state["query_input"] = sample_q
            st.rerun()

st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

run_btn = st.button("🔍  Search BRSR Reports", use_container_width=True)

# ── Run Query ─────────────────────────────────────────────────────────────────
if run_btn:
    if not query or not query.strip():
        st.warning("Please enter a question before searching.")
        st.stop()

    st.markdown("<div class='green-divider'></div>", unsafe_allow_html=True)

    # Retrieval
    with st.spinner(""):
        t0        = time.time()
        retrieved = retrieve_top_k(
            query=query.strip(),
            company=selected_company,
            faiss_index=faiss_index,
            meta_df=meta_df,
            all_vectors=all_vectors,
            embedder=embedder,
            top_k=top_k,
        )
        retrieval_time = time.time() - t0

    if retrieved.empty:
        st.markdown(f"""
        <div class='info-card' style='border-color: rgba(255,100,100,0.2);'>
            <div class='info-card-label' style='color:#e07070;'>No results found</div>
            <div style='color:#999; font-size:0.9rem;'>No chunks found for company <strong style='color:#ccc;'>{selected_company}</strong>. Ensure this company's PDF was indexed.</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # LLM answer
    llm_answer = None
    llm_time   = None

    if use_llm and api_key:
        with st.spinner("Generating LLM answer with Gemini…"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(gemini_model)
                t1          = time.time()
                llm_answer  = ask_gemini(selected_company, query, retrieved, model)
                llm_time    = time.time() - t1
            except Exception as e:
                llm_answer = f"⚠ Gemini error: {e}"

    # ── Results header
    avg_score = retrieved["score"].mean()
    top_score = retrieved["score"].max()
    n_chunks  = len(retrieved)

    st.markdown(f"""
    <div style='margin-bottom: 1rem;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.6rem; color: #e8f5e2; margin-bottom: 0.3rem;'>
            Results for <span style='color:#4ab860;'>{selected_company}</span>
        </div>
        <div style='font-size: 0.83rem; color: #4d7a55;'>
            {n_chunks} chunks retrieved &nbsp;·&nbsp; {retrieval_time*1000:.0f}ms &nbsp;·&nbsp; avg relevance {avg_score:.3f} &nbsp;·&nbsp; top score {top_score:.3f}
        </div>
    </div>""", unsafe_allow_html=True)

    # ── LLM Answer block
    if llm_answer:
        st.markdown(f"""
        <div class='answer-box'>
            <div class='answer-label'>✦ &nbsp; Gemini-synthesized Answer &nbsp; · &nbsp; {llm_time*1000:.0f}ms</div>
            <div class='answer-text'>{llm_answer}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

    elif use_llm and not api_key:
        st.markdown("""
        <div style='background: rgba(255,160,60,0.06); border: 1px solid rgba(255,160,60,0.15); border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 1rem;'>
            <span style='color:#e09050; font-size:0.85rem;'>⚠ Provide a Google API Key in the sidebar to enable LLM answer synthesis.</span>
        </div>""", unsafe_allow_html=True)

    # ── Retrieved chunks
    st.markdown(f"<div style='font-size:0.75rem; color:#4d7a55; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.8rem;'>Retrieved Passages ({n_chunks})</div>", unsafe_allow_html=True)

    for i, row in retrieved.iterrows():
        score_pct = int(row["score"] * 100)
        page_label = f"p.{row['page']}" if row["page"] else "—"
        chunk_preview = row["chunk_text"][:600] + ("…" if len(row["chunk_text"]) > 600 else "")

        st.markdown(f"""
        <div class='chunk-card'>
            <div class='chunk-meta'>
                <span class='chunk-badge'>Chunk {i+1}</span>
                <span class='chunk-badge'>{page_label}</span>
                <span class='chunk-score'>Relevance {score_pct}%</span>
                <span class='chunk-score' style='font-family:monospace; font-size:0.65rem; letter-spacing:0;'>{row['chunk_id']}</span>
            </div>
            <div class='chunk-text'>{chunk_preview}</div>
        </div>""", unsafe_allow_html=True)

    # ── Score distribution chart
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem; color:#4d7a55; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem;'>Relevance Score Distribution</div>", unsafe_allow_html=True)

    chart_df = pd.DataFrame({
        "Chunk": [f"Chunk {i+1}" for i in range(len(retrieved))],
        "Relevance Score": retrieved["score"].values,
    }).set_index("Chunk")

    st.bar_chart(chart_df, color="#4ab860", height=180)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top: 4rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.05); text-align: center;'>
    <div style='font-size: 0.75rem; color: #2d4a33; letter-spacing: 0.06em;'>
        BRSR Intelligence &nbsp;·&nbsp; BGE-small-en-v1.5 &nbsp;·&nbsp; FAISS IndexFlatIP &nbsp;·&nbsp; LangChain text splitters
    </div>
</div>
""", unsafe_allow_html=True)
