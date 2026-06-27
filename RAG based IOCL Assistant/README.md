# 🤖 RAG based IOCL Knowledge Assistant

> **A production-grade, fully offline AI chatbot** that answers questions from a local CSV knowledge base — zero internet, zero API keys.

Production-grade offline RAG knowledge assistant for IOCL using Sentence Transformers, FAISS, Query Understanding, Cross-Encoder Reranking, Answer Validation, and optional Ollama-based local LLM generation — with an interactive Streamlit interface.

---

## 📋 Overview

The **IOCL Knowledge Assistant** is an intelligent Retrieval-Augmented Generation (RAG) system built for Indian Oil Corporation's knowledge management needs. It reads a structured CSV dataset, builds semantic embeddings, stores vectors in FAISS, retrieves and **reranks** contextually relevant answers, **validates them against a confidence threshold**, and generates a grounded response — all running entirely offline on local hardware.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| 🔍 **Semantic Search** | FAISS-powered vector similarity search using `all-MiniLM-L6-v2` |
| 🧠 **Hybrid Retrieval** | Combines semantic + keyword matching with configurable weights |
| 🧭 **Query Understanding** | Intent classification + entity extraction before retrieval |
| 🪜 **Cross-Encoder Reranking** | Re-scores top candidates for better precision |
| 🛡️ **Answer Validation** | Hallucination guard — refuses to answer when confidence is too low |
| 💬 **Multi-turn Chat** | Conversation memory with context-aware responses, welcome screen with suggested questions |
| 📊 **Analytics Dashboard** | Category/topic/keyword distribution with Plotly visualizations |
| 📄 **Export** | Chat history as TXT, PDF, and search results as CSV |
| ⚡ **Fully Offline** | No internet connection or API keys required |
| 🦙 **Local LLM** | Optional Ollama integration for natural language generation |
| 🎨 **Themed UI** | Custom dark-mode Streamlit interface with confidence color-coding |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (Dark Theme)                │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐              │
│  │   Chat   │   │Analytics │   │  Export Manager  │              │
│  └────┬─────┘   └────┬─────┘   └────────┬─────────┘              │
│       │              │                  │                        │
│  ┌────▼──────────────▼──────────────────▼──────────┐             │
│  │                 Application Core                │             │
│  └────┬───────┬───────────┬───────────┬─────────────┘            │
│       │       │           │           │                          │
│  ┌────▼───┐ ┌─▼────────┐ ┌▼─────────┐ ┌▼────────────────────┐    │
│  │Query   │ │Retriever │ │Reranker  │ │ Answer Validator    │    │
│  │Underst.│ │(FAISS    │ │(Cross-   │ │ (confidence/        │    │
│  │        │ │hybrid)   │ │ Encoder) │ │  hallucination gate)│    │
│  └────┬───┘ └─┬────────┘ └┬─────────┘ └┬────────────────────┘    │
│       │       │           │            │                         │
│  ┌────▼───────▼───────────▼────────────▼──────────┐  ┌─────────┐ │
│  │   Sentence Transformers + FAISS + Pandas       │  │ Ollama  │ │
│  │                                                │  │ LLM     │ │
│  └────────────────────────────────────────────────┘  │(Optional) │
│                                                      └─────────┘ │
│  ┌──────────────────────────────────────────────────┐            │
│  │                     data.csv                     │            │
│  └──────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow (RAG Pipeline)

```
User Query
  → expand_query()                  (synonym expansion)
  → QueryUnderstanding.analyze()    (intent + entities)
  → hybrid_search(threshold)        (semantic 0.7 + keyword 0.3, gated by sidebar threshold)
  → Reranker.rerank()               (cross-encoder re-scores top 20 → keep top 5)
  → AnswerValidator.validate()      (confidence/hallucination gate)
       ├─ invalid → "Information not found in the knowledge base."
       └─ valid   → build cited context (SOURCE:<Fact_ID> / CONTENT:<Answer>)
                      → OllamaLLM.generate()  (or raw context fallback if Ollama is down)
                      → Answer + source citations + similarity scores
```

---

## 🧬 RAG Pipeline

The assistant follows an advanced Retrieval-Augmented Generation architecture, not just a single-pass semantic search:

- ✅ Query Expansion
- ✅ Intent Detection
- ✅ Entity Extraction
- ✅ Hybrid Retrieval (semantic + keyword, confidence-threshold gated)
- ✅ Cross-Encoder Reranking
- ✅ Answer Validation (hallucination guard)
- ✅ Context Construction with source citations
- ✅ Optional Local LLM Generation (Ollama)
- ✅ Source Attribution in the final response

This pipeline improves retrieval accuracy, reduces hallucinations, and produces explainable, source-cited answers from the knowledge base — rather than a plain "top-1 nearest neighbor" lookup.

---

## 📁 Project Structure

```
project/
├── app.py                    # Main Streamlit application
├── data.csv                  # Knowledge base dataset
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── assets/                   # Static assets
├── cache/                    # FAISS index + metadata cache
│   ├── faiss.index
│   └── metadata.pkl
├── exports/                  # Generated export files
└── utils/
    ├── __init__.py           # Package exports
    ├── embeddings.py         # Sentence Transformers wrapper
    ├── retriever.py          # FAISS-based hybrid semantic/keyword retriever
    ├── reranker.py           # Cross-encoder reranking of retrieved candidates
    ├── query_understanding.py # Intent classification + entity extraction
    ├── validator.py          # Answer/hallucination confidence validation
    ├── ollama_helper.py      # Local LLM integration
    ├── exporter.py           # Chat/results export (TXT, PDF, CSV)
    └── analytics.py          # Dataset analytics & Plotly charts
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **4 GB RAM** (minimum for embedding + reranker models)
- **Optional:** [Ollama](https://ollama.ai) for local LLM generation

### Installation

```bash
# Clone the repository
cd project/

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
streamlit run app.py
```

The application opens at **http://localhost:8501**.

### Optional: Enable Local LLM

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the llama3 model
ollama pull llama3

# The app auto-detects Ollama — no configuration needed
```

---

## 🧩 Module Reference

### `utils/embeddings.py` — Embedding Engine

| Method | Description |
|--------|-------------|
| `embed_text(text)` | Returns a 384-dim vector for a single text |
| `embed_batch(texts)` | Returns an (N, 384) matrix for multiple texts |
| `get_embedding_engine()` | Returns the shared singleton instance |

### `utils/retriever.py` — Semantic Retriever

| Method | Description |
|--------|-------------|
| `build_index(df)` | Constructs FAISS index from a DataFrame |
| `semantic_search(query)` | Pure vector similarity search |
| `keyword_search(query)` | Keyword matching across searchable texts |
| `hybrid_search(query, top_k, threshold)` | Weighted combination of semantic + keyword, gated by a confidence `threshold` |
| `expand_query(query)` | Heuristic synonym expansion |
| `load_cache()` / `_save_cache()` | Persist/restore index to disk |
| `get_category_stats()` | Category frequency from indexed data |
| `get_topic_stats()` | Topic frequency from indexed data |
| `get_keyword_freq()` | Keyword frequency analysis |
| `query_log` | Running log of queries + scores, used by the Analytics tab |

### `utils/query_understanding.py` — Query Understanding *(new)*

| Method | Description |
|--------|-------------|
| `QueryUnderstanding.analyze(query)` | Returns a dict with `intent` (e.g. factual/definition/list) and `entities` extracted from the query, used to steer retrieval and downstream formatting |

### `utils/reranker.py` — Cross-Encoder Reranker *(new)*

| Method | Description |
|--------|-------------|
| `Reranker.rerank(query, results)` | Re-scores the hybrid-search candidates with a cross-encoder for higher precision before truncating to the top results shown to the user |

### `utils/validator.py` — Answer Validator *(new)*

| Method | Description |
|--------|-------------|
| `AnswerValidator.validate(results)` | Confidence/hallucination gate — returns `False` when no result is trustworthy enough, causing the app to respond with "Information not found in the knowledge base" instead of guessing |

### `utils/ollama_helper.py` — Local LLM

| Method | Description |
|--------|-------------|
| `is_available()` | Checks if Ollama is running locally |
| `generate(query, context)` | Generates answer from cited context using local LLM |
| Falls back to context | Returns raw retrieved context when Ollama is unavailable |

### `utils/exporter.py` — Export Manager

| Method | Description |
|--------|-------------|
| `chat_to_txt(messages)` | Converts chat history to plain text |
| `chat_to_pdf(messages)` | Generates a PDF report from chat history |
| `results_to_csv(results)` | Serializes search results as CSV |
| `save_session(messages)` | Persists full session to disk |

### `utils/analytics.py` — Analytics Engine

| Method | Description |
|--------|-------------|
| `fig_category_distribution()` | Pie chart of categories |
| `fig_topic_distribution()` | Horizontal bar chart of topics |
| `fig_keyword_cloud()` | Bar chart of top keywords |
| `fig_search_activity(log)` | Subplot of search scores and result counts (fed by `retriever.query_log`) |
| `fig_source_types()` | Donut chart of source types |

---

## 📊 Dataset Format

The application dynamically handles any CSV schema. The recommended format:

```csv
Fact_ID,Category,Subcategory,Topic,Question,Answer,Keywords,Source_Type
IOCL-001,Refinery,Operations,Panipat Refinery,Where is Panipat located?,Located in Haryana,Panipat Haryana location,Internal Document
```

### Auto-detection

- **ID column**: Automatically detected (`Fact_ID`, `id`, `record_id`, etc.)
- **Searchable text**: All columns concatenated into a single vector
- **Category/Topic**: Used for analytics and filtering

---

## ⚙️ Performance

| Metric | Value |
|--------|-------|
| Embedding dimension | 384 (all-MiniLM-L6-v2) |
| Index type | FAISS IndexFlatIP (Inner Product) |
| Batch encoding | 64 texts/batch |
| Retrieval depth | Top 20 hybrid candidates → reranked → top 5 used for answer |
| Cache | Persisted to disk (`cache/`) |
| Support capacity | 50,000+ rows |

### Caching

- **`@st.cache_resource`** — DataFrame, embedding engine, FAISS retriever, Ollama LLM, and reranker are each loaded once per session
- **FAISS cache** — Index persisted to disk; survives restarts
- Sidebar **Reload Data** / **Rebuild Index** buttons clear the relevant `st.cache_resource`/`st.cache_data` entries and force a rerun

---

## 🎨 UI Features

### Chat Interface
- 💬 Multi-turn conversation with memory
- 👋 **Welcome screen** with 5 clickable suggested questions shown when the chat is empty
- 📚 Source attribution with similarity scores, fact IDs, topic, and category
- 🔎 Expandable detailed source view (best similarity score, retrieved doc count, raw JSON per result)
- 🎨 Color-coded confidence indicators (green ≥0.50 / yellow ≥0.35 / red below)
- 🖤 Custom dark theme (CSS-styled chat bubbles, tabs, metric cards)

### Sidebar
- 📊 Live dataset statistics (records, categories, topics, vectors indexed)
- ⚙️ "Show source details" toggle
- ⚙️ **Functional** confidence threshold slider — now actually passed into `hybrid_search()`, so raising/lowering it changes what gets retrieved
- 🔍 Search history (last 10 queries)
- 🗑️ Clear Chat / 🔄 Reload Data / ♻️ Rebuild Index
- 📥 Collapsible export section (TXT + PDF) shown once a conversation exists

### Analytics Dashboard
- KPI row: total records, categories, topics, queries asked this session
- 📈 Category distribution (pie chart)
- 📊 Topic breakdown (bar chart)
- 🔤 Keyword frequency (bar chart)
- 📊 Source type distribution (donut chart)
- 🔍 Search activity trends (driven by `retriever.query_log`)
- 📋 Full dataset preview
- 📥 Download last search's results as CSV

---

## 🔒 Error Handling

The application gracefully handles:

| Error | Behavior |
|-------|----------|
| Missing `data.csv` | Shows clear error message with instructions, halts startup |
| Corrupted CSV | Falls back from UTF-8 to Latin-1 encoding |
| Empty dataset | Returns `None` from `load_dataframe`, app shows error |
| FAISS errors | Logs error, sidebar "Rebuild Index" offers recovery |
| Low-confidence / hallucination risk | `AnswerValidator` blocks the answer and returns a clear "not found" message instead of guessing |
| Ollama unavailable | Falls back to raw retrieved context, no LLM generation |
| Embedding failures | Shows error with recovery steps |

---

## 🧪 Technical Decisions

1. **FAISS IndexFlatIP** over IndexIVF — for the dataset sizes expected (<100K rows), flat index gives exact results with negligible speed difference
2. **Normalized embeddings + inner product** — equivalent to cosine similarity, but faster
3. **Hybrid search** — combines semantic (0.7 weight) + keyword (0.3 weight) for robustness
4. **Heuristic query expansion** — no external API needed; uses synonym map for common terms
5. **Query understanding before retrieval** — lightweight intent/entity extraction helps disambiguate short or ambiguous queries before they hit the retriever
6. **Two-stage retrieval (retrieve-then-rerank)** — hybrid search casts a wide net (top 20), then a cross-encoder reranker reorders for precision before truncating to the top 5 actually shown/used
7. **Explicit answer validation gate** — rather than always answering with whatever was retrieved, results are checked against a confidence threshold first; failing that check returns an explicit "not found" message instead of a low-confidence guess
8. **Sidebar threshold is wired into retrieval, not just display** — the confidence slider is passed directly into `hybrid_search()`, so it meaningfully changes retrieval behavior rather than only filtering what's shown
9. **Lazy model loading** — Sentence Transformers / reranker models loaded on first use, not on app start
10. **PDF via FPDF** — lightweight, no system dependencies; generates branded reports

---

## 📝 License

Internal use — Indian Oil Corporation Knowledge Management

---

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) — Embeddings
- [FAISS](https://faiss.ai/) — Vector search
- [Streamlit](https://streamlit.io/) — Web UI
- [Plotly](https://plotly.com/) — Visualizations
- [Ollama](https://ollama.ai/) — Local LLM inference