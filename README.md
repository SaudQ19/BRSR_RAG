# BRSR_RAG

# BRSR RAG Pipeline — End-to-End with Retrieval Benchmarking

A full **Retrieval-Augmented Generation (RAG)** pipeline built over **Business Responsibility and Sustainability Reports (BRSR)** for 20 Indian listed companies, complete with a rigorous retrieval evaluation benchmark comparing chunking strategies and embedding models.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Pipeline Steps](#pipeline-steps)
  - [Step 1 — PDF Extraction](#step-1--pdf-extraction)
  - [Step 2 & 3 — Chunking Strategies](#step-2--3--chunking-strategies)
  - [Step 4 — Embedding](#step-4--embedding)
  - [Step 5 — FAISS Indexing](#step-5--faiss-indexing)
  - [Step 6 — Retrieval (Company-Scoped)](#step-6--retrieval-company-scoped)
  - [Step 7 — LLM Generation (Gemini)](#step-7--llm-generation-gemini)
  - [Step 8 — Ground Truth Construction](#step-8--ground-truth-construction)
  - [Step 9 — Benchmarking Setups](#step-9--benchmarking-setups)
  - [Step 10 — Evaluation & Metrics](#step-10--evaluation--metrics)
- [Benchmark Results](#benchmark-results)
- [Failure Analysis](#failure-analysis)
- [Key Design Decisions](#key-design-decisions)
- [Extending the Pipeline](#extending-the-pipeline)

---

## Overview

BRSR (Business Responsibility and Sustainability Report) is a mandatory disclosure framework for India's top-listed companies, regulated by SEBI. These reports contain rich ESG (Environmental, Social, and Governance) data — from carbon emissions and water consumption to workforce diversity and governance policies.

This pipeline enables **natural-language question answering** over BRSR PDFs by:

1. Extracting and chunking PDF text from 20 Indian companies.
2. Building dense vector indexes using BGE embeddings.
3. Retrieving relevant passages at query time (scoped per company).
4. Optionally generating answers using **Google Gemini**.
5. Benchmarking retrieval quality with a curated ground-truth dataset.

---

## Pipeline Architecture

```
PDF Files (DATA/)
      │
      ▼
┌─────────────────┐
│  PDF Extraction  │  ── pdfplumber page-level text → brsr_pages.jsonl
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunking      │  ── Strategy A (300 tok) & B (800 tok) via LangChain
└────────┬────────┘       → brsr_chunks_strategy_{A,B}.jsonl
         │
         ▼
┌─────────────────┐
│   Embedding      │  ── BAAI/bge-small-en-v1.5 (384-dim, normalised)
└────────┬────────┘       → brsr_chunk_embeddings_bge_small.jsonl
         │
         ▼
┌─────────────────┐
│  FAISS Indexing  │  ── IndexFlatIP (cosine via inner product)
└────────┬────────┘       → brsr_faiss_bge_small.index + metadata.jsonl
         │
         ▼
┌─────────────────┐
│    Retrieval     │  ── Company-scoped top-k with cosine similarity
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Generation  │  ── Google Gemini (gemma-3-27b-it) — optional
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Benchmarking   │  ── Precision@k, Recall@k, MRR across 2 setups
└─────────────────┘
```

---

## Directory Structure

```
project_root/
├── DATA/                              ← Input PDF files (one per company, named COMPANY_*.pdf)
├── outputs/
│   ├── brsr_pages.jsonl               ← Page-level extracted text
│   ├── brsr_chunks_strategy_A.jsonl   ← 300-token chunks
│   ├── brsr_chunks_strategy_B.jsonl   ← 800-token chunks
│   ├── brsr_chunk_embeddings_bge_small.jsonl
│   ├── brsr_faiss_bge_small.index
│   ├── brsr_faiss_metadata.jsonl
│   ├── ground_truth_raw.json          ← Gemini-generated QA pairs per company
│   ├── evaluation_retrieval_metrics.jsonl
│   ├── evaluation_failure_analysis.jsonl
│   └── benchmarks/
│       └── <setup_name>/              ← e.g. 300_tokens_bge_small/
│           ├── index.faiss
│           ├── metadata.parquet
│           ├── vectors.npy
│           ├── evaluation_rows.jsonl
│           ├── failure_cases.jsonl
│           └── summary.json
└── brsr_rag_pipeline.ipynb
```

---

## Dependencies

Install all required packages:

```bash
pip install pdfplumber langchain langchain-text-splitters sentence-transformers \
            faiss-cpu tiktoken google-generativeai tqdm
```

| Package | Purpose |
|---|---|
| `pdfplumber` | PDF text extraction (page-level) |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` for chunking |
| `sentence-transformers` | BGE embedding models |
| `faiss-cpu` | Vector index (cosine similarity via IndexFlatIP) |
| `tiktoken` | Token counting with `cl100k_base` encoding |
| `google-generativeai` | Gemini API for answer generation & GT synthesis |
| `tqdm` | Progress bars |
| `pandas`, `numpy` | Data manipulation |

**Python version:** 3.12+

---

## Configuration

Key constants defined in the **Imports & Configuration** cell:

```python
# Chunking strategies
CHUNK_STRATEGIES = {
    "A": {"chunk_size": 300, "chunk_overlap": 50},
    "B": {"chunk_size": 800, "chunk_overlap": 100},
}

# Benchmark setups (strategy × embedding model)
SETUPS = {
    "300_tokens_bge_small": {"strategy": "A", "embedding_model": "BAAI/bge-small-en-v1.5"},
    "800_tokens_bge_small": {"strategy": "B", "embedding_model": "BAAI/bge-small-en-v1.5"},
}

TOP_K               = 5      # Number of chunks retrieved per query
RELEVANCE_THRESHOLD = 0.45   # Cosine-similarity threshold for relevance
DEFAULT_GEMINI_MODEL = "gemma-3-27b-it"
```

The tokeniser used for chunk-size measurement is OpenAI's `cl100k_base` (via `tiktoken`), ensuring consistent token counts independent of the embedding model's own tokeniser.

> **Caching:** All expensive steps (extraction, chunking, embedding, indexing) are cached to disk. Re-running the notebook skips already-computed artefacts automatically.

---

## Pipeline Steps

### Step 1 — PDF Extraction

**Cell:** `pdf-extraction`  
**Output:** `outputs/brsr_pages.jsonl`

Each PDF in `DATA/` is processed with `pdfplumber`. Text is extracted **page by page**, and the company name is inferred from the filename prefix (everything before the first `_`).

Each record in `brsr_pages.jsonl` contains:

| Field | Description |
|---|---|
| `company_name` | Inferred from filename (e.g. `ASIANPAINT`) |
| `page_number` | 1-indexed page number |
| `text` | Raw extracted page text |
| `file_name` | Source PDF filename |

Pages with empty or whitespace-only text are skipped. A summary table shows page counts and average token counts per company.

---

### Step 2 & 3 — Chunking Strategies

**Cell:** `chunking`  
**Output:** `outputs/brsr_chunks_strategy_A.jsonl`, `outputs/brsr_chunks_strategy_B.jsonl`

LangChain's `RecursiveCharacterTextSplitter` is used with a **token-based length function** (via `tiktoken`). Two strategies are compared:

| Strategy | Chunk Size | Overlap | Use Case |
|---|---|---|---|
| A | 300 tokens | 50 tokens | High precision, tight context windows |
| B | 800 tokens | 100 tokens | Better recall, more context per chunk |

Each chunk record contains:

| Field | Description |
|---|---|
| `chunk_id` | Stable ID: `{strategy}::{company}::{index}` |
| `company_name` | Company ticker/name |
| `chunk_index` | Sequential chunk number within the company |
| `chunk_text` | The actual text content |
| `page_start` / `page_end` | Source page range |
| `strategy` | `A` or `B` |
| `token_count` | Actual token count of the chunk |

---

### Step 4 — Embedding

**Cell:** `embedding`  
**Output:** `outputs/brsr_chunk_embeddings_bge_small.jsonl`

All chunks (both strategies combined) are embedded using **`BAAI/bge-small-en-v1.5`** — a lightweight, high-quality sentence embedding model producing **384-dimensional vectors**.

Key details:
- Embeddings are **L2-normalised** at encoding time (`normalize_embeddings=True`), making inner product equivalent to cosine similarity.
- Batch size: 64.
- Total chunks embedded: **6,894** across both strategies.
- Vector dimension: **384**.

The output JSONL stores `embedding_vector` as a list of floats alongside `chunk_id`, `company`, `page`, and `chunk_text`.

---

### Step 5 — FAISS Indexing

**Cell:** `faiss-index`  
**Output:** `outputs/brsr_faiss_bge_small.index` + `outputs/brsr_faiss_metadata.jsonl`

A **`faiss.IndexFlatIP`** (Inner Product index) is built over the pre-normalised vectors. Because vectors are unit-normalised, inner product equals cosine similarity — no approximation, exact nearest-neighbour search.

```python
index = faiss.IndexFlatIP(dim)
index.add(vectors)   # 6,894 vectors × 384 dims
```

The metadata DataFrame stored alongside the index maps each FAISS integer ID back to `chunk_id`, `company`, `page`, `chunk_text`, `strategy`, and `file_name`.

For the formal benchmark, **per-setup indexes** are built independently inside `get_setup_assets()`, keeping the two strategies cleanly separated.

---

### Step 6 — Retrieval (Company-Scoped)

**Function:** `retrieve(query, company, index, meta_df, top_k, threshold)`

Retrieval is intentionally **scoped to a single company** — the query is only matched against chunks belonging to the specified company. This reflects the real-world use case where a user interrogates one company's BRSR report at a time.

The retrieval flow:
1. Embed the query using the same BGE model.
2. Filter `meta_df` to rows where `company == target_company`.
3. Fetch the relevant FAISS vectors by integer ID.
4. Compute cosine similarity (inner product on normalised vectors).
5. Return top-k results above `RELEVANCE_THRESHOLD = 0.45`.

Each result includes `chunk_text`, `page`, `score`, `chunk_id`, and `company`.

---

### Step 7 — LLM Generation (Gemini)

**Function:** `generate_answer(query, retrieved_chunks, model_name)`

When retrieved chunks are available, they are formatted into a context block and passed to **Google Gemini** (`gemma-3-27b-it` by default) via the `google-generativeai` SDK.

The prompt instructs the model to:
- Answer using only the provided context.
- Cite page numbers where relevant.
- Acknowledge if the context is insufficient.

This step is **optional** — retrieval benchmarking runs independently without LLM calls.

> **API Key:** Set your Gemini API key via `genai.configure(api_key=YOUR_KEY)` before running generation cells.

---

### Step 8 — Ground Truth Construction

**Cell:** `ground-truth`  
**Output:** `outputs/ground_truth_raw.json`

A structured ground-truth dataset is built using Gemini to extract 21 factual fields from each company's BRSR report. The fields cover:

| Category | Fields |
|---|---|
| Identity | `cin`, `company_name`, `registered_office_address`, `financial_year`, `stock_exchanges` |
| Business | `main_business_activities`, `top_products`, `markets_served` |
| Workforce | `total_employees`, `percentage_women`, `differently_abled_employees` |
| Environment | `total_energy_consumption`, `scope_1_emissions`, `scope_2_emissions`, `water_consumption`, `total_waste_generated`, `waste_recycled_or_reused` |
| Governance | `sustainability_policy`, `climate_risk_strategy`, `board_sustainability_committee`, `external_assurance_brsr` |

For each field, a natural-language **question** is automatically generated alongside the extracted ground-truth answer. This produces a dataset of `20 companies × 21 questions = 420 QA pairs` that forms the evaluation benchmark.

---

### Step 9 — Benchmarking Setups

**Function:** `get_setup_assets(setup_name, setup_cfg, chunks_by_strategy)`

For each setup in `SETUPS`, the following is built and cached under `outputs/benchmarks/<setup_name>/`:

| Artefact | Description |
|---|---|
| `index.faiss` | Per-setup FAISS index |
| `metadata.parquet` | Chunk metadata (Parquet for efficiency) |
| `vectors.npy` | Raw numpy vectors |

This ensures that the 300-token and 800-token setups are evaluated with completely independent indexes, preventing any data leakage between configurations.

---

### Step 10 — Evaluation & Metrics

**Cell:** `evaluation`  
**Output:** `outputs/evaluation_retrieval_metrics.jsonl`, `outputs/evaluation_failure_analysis.jsonl`

For each setup and each QA pair in the ground truth, the pipeline:

1. Retrieves top-k chunks for the question (company-scoped).
2. Checks whether the ground-truth answer text appears in any retrieved chunk (lexical match after normalisation).
3. Records success or failure with a `failure_type` label.

**Metrics computed:**

| Metric | Description |
|---|---|
| **Precision@3** | Fraction of top-3 results that are relevant |
| **Precision@5** | Fraction of top-5 results that are relevant |
| **Recall@5** | Fraction of all relevant chunks found in top-5 |
| **MRR** | Mean Reciprocal Rank — position of the first relevant result |

**Failure types logged:**

- `correct_chunk_not_retrieved` — Relevant chunk exists in index but was not in top-k.
- `no_relevant_chunk_in_index` — The answer text could not be found in any indexed chunk.
- `below_threshold` — Retrieved chunks didn't meet the cosine similarity threshold.

---

## Benchmark Results

Results from 420 QA pairs (20 companies × 21 questions) across 2 setups:

| Setup | Precision@3 | Precision@5 | Recall@5 | MRR |
|---|---|---|---|---|
| `300_tokens_bge_small` | **0.797** | **0.796** | 0.0234 | **0.795** |
| `800_tokens_bge_small` | 0.797 | 0.790 | **0.0556** | 0.795 |

**Total failure cases logged: 40**

### Interpretation

**Precision and MRR are strong (~0.79–0.80):** The retriever reliably places at least one relevant chunk near the top of the ranking. For precision-sensitive applications (e.g. displaying a single best answer), either setup performs well.

**Recall is the bottleneck:** Recall@5 is low for both setups, meaning many relevant chunks are missed within the top-5 window. This is partly a consequence of the company-scoped design — some answers are spread across multiple document locations.

**Chunk-size trade-off:**
- **300-token** chunks yield marginally better Precision@5 (0.796 vs 0.790).
- **800-token** chunks more than double Recall@5 (0.0556 vs 0.0234) by capturing more context per chunk.

---

## Failure Analysis

The dominant failure mode is `correct_chunk_not_retrieved` — the answer text is present somewhere in the indexed corpus but doesn't surface in top-5. This points to **retrieval** as the primary improvement axis, not generation.

**Recommended next steps to improve recall:**

1. **Hybrid retrieval** — Combine dense (vector) retrieval with sparse BM25 to catch exact-match terms that semantic search misses.
2. **Query expansion / rewriting** — Use an LLM to rephrase the question before retrieval to broaden lexical overlap with source text.
3. **Re-ranking** — Add a cross-encoder re-ranker (e.g. `bge-reranker`) as a second-stage filter over a larger initial candidate set.
4. **Larger embedding model** — Swap `bge-small` for `bge-base-en-v1.5` (768-dim) or `bge-large` for higher embedding quality.
5. **Larger top-k** — Increase from `k=5` to `k=10` or `k=20` to trade latency for coverage in production.

---

## Key Design Decisions

**Company-scoped retrieval** rather than global search prevents cross-company contamination and mirrors the real-world analyst use case of interrogating one report at a time.

**Cosine similarity via IndexFlatIP** on L2-normalised vectors gives exact search (no HNSW approximation needed at this corpus size) with correct cosine semantics.

**Tiktoken for chunking length** ensures consistent token counting independent of the downstream embedding model — important when comparing strategies across model changes.

**Aggressive caching** at every step (pages → chunks → embeddings → index → benchmark assets) makes iterative experimentation fast without re-running expensive steps.

**Gemini for ground-truth synthesis** rather than manual annotation enables scalable QA pair generation across 20 companies × 21 structured fields, producing a reproducible evaluation benchmark.

---

## Extending the Pipeline

To add a new embedding model (e.g. `bge-base-en-v1.5`):

```python
SETUPS["300_tokens_bge_base"] = {
    "strategy": "A",
    "embedding_model": "BAAI/bge-base-en-v1.5"
}
```

Re-run from the benchmarking cell — caching means only the new setup is computed.

To add a new chunking strategy:

```python
CHUNK_STRATEGIES["C"] = {"chunk_size": 512, "chunk_overlap": 75}
```

Then add a corresponding entry to `SETUPS` and re-run from the chunking cell onwards.

To query interactively after setup:

```python
results = retrieve(
    query="What is the total Scope 1 emissions?",
    company="ASIANPAINT",
    index=faiss_index,
    meta_df=meta_df,
    top_k=5,
    threshold=0.45
)
answer = generate_answer("What is the total Scope 1 emissions?", results)
print(answer)
```
