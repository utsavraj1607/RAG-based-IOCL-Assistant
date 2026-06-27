"""
IOCL Knowledge Assistant — Main Streamlit Application.

A production-grade, fully offline AI chatbot that answers questions from a
local CSV knowledge base using Sentence Transformers + FAISS + optional Ollama.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so utils can be imported
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.reranker import Reranker
from utils.query_understanding import QueryUnderstanding
from utils.analytics import AnalyticsEngine
from utils.embeddings import get_embedding_engine
from utils.exporter import ExportManager
from utils.ollama_helper import OllamaLLM
from utils.retriever import SemanticRetriever
from utils.validator import AnswerValidator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IOCL Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark theme CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .block-container {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
    }
    div[data-testid="stChatMessage"][aria-label="user"] {
        background-color: #1c2333;
        border-left: 4px solid #4da3ff;
    }
    div[data-testid="stChatMessage"][aria-label="assistant"] {
        background-color: #1a1f2e;
        border-left: 4px solid #00c853;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2740 0%, #161b22 100%);
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #30363d;
    }
    h1, h2, h3 { color: #fafafa !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1c2333 !important;
        border-bottom: 2px solid #4da3ff;
    }
    .confidence-high { color: #00c853; font-weight: bold; }
    .confidence-med  { color: #ffc107; font-weight: bold; }
    .confidence-low  { color: #ff5252; font-weight: bold; }
    .welcome-box {
        background: linear-gradient(135deg, #1c2333 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        margin-bottom: 24px;
    }
    .welcome-box h2 { margin-bottom: 8px; }
    .welcome-box p { color: #8b949e; font-size: 1rem; }
    .suggestion-chip {
        display: inline-block;
        background-color: #1c2333;
        border: 1px solid #4da3ff;
        border-radius: 20px;
        padding: 8px 16px;
        margin: 4px;
        color: #4da3ff;
        font-size: 0.85rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources — built once per session
# ---------------------------------------------------------------------------
@st.cache_resource
def load_dataframe(path: str) -> Optional[pd.DataFrame]:
    """Load and validate the CSV dataset."""
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding="latin-1")
        except Exception:
            return None
    except Exception:
        return None
    if df.empty:
        return None
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_resource
def build_retriever(_df_hash: str, df: pd.DataFrame) -> SemanticRetriever:
    """Build (or load from cache) the FAISS retriever."""
    engine = get_embedding_engine()
    retriever = SemanticRetriever(engine)
    if not retriever.load_cache():
        retriever.build_index(df)
    return retriever


@st.cache_resource
def get_llm() -> OllamaLLM:
    return OllamaLLM()


@st.cache_resource
def get_reranker() -> Reranker:
    return Reranker()


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def init_session() -> None:
    defaults = {
        "messages": [],
        "search_history": [],
        "show_sources": True,
        "confidence_threshold": 0.40,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(df: Optional[pd.DataFrame], retriever: Optional[SemanticRetriever]) -> None:
    with st.sidebar:
        st.markdown("## 🤖 IOCL Knowledge Assistant")
        st.markdown("---")

        st.markdown("### 📊 Dataset Statistics")
        if df is not None:
            st.metric("Total Records", len(df))
            if "Category" in df.columns:
                cats = df["Category"].nunique()
                st.metric("Categories", cats)
            if "Topic" in df.columns:
                topics = df["Topic"].nunique()
                st.metric("Topics", topics)
            if retriever and retriever.is_ready:
                st.metric("Vectors Indexed", retriever.total_vectors)
        else:
            st.warning("No dataset loaded.")

        st.markdown("---")
        st.markdown("### ⚙️ Settings")

        st.session_state.show_sources = st.toggle(
            "Show source details",
            value=st.session_state.show_sources,
        )
        st.session_state.confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.20,
            max_value=0.95,
            value=st.session_state.confidence_threshold,
            step=0.05,
        )

        st.markdown("---")
        st.markdown("### 🔍 Search History")
        history = st.session_state.search_history[-10:]
        if history:
            for h in reversed(history):
                st.caption(f"• {h[:60]}{'…' if len(h) > 60 else ''}")
        else:
            st.caption("No searches yet.")

        st.markdown("---")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.search_history = []
            st.rerun()

        if st.button("🔄 Reload Data", use_container_width=True):
            load_dataframe.clear()
            build_retriever.clear()
            st.cache_data.clear()
            st.rerun()

        if st.button("♻️ Rebuild Index", use_container_width=True):
            build_retriever.clear()
            st.cache_data.clear()
            st.rerun()

        # Export options — collapsed by default so they don't interrupt the chat
        if st.session_state.messages:
            st.markdown("---")
            with st.expander("📥 Export Conversation", expanded=False):
                st.caption("Save this conversation for later reference.")
                col1, col2 = st.columns(2)
                with col1:
                    txt = ExportManager.chat_to_txt(st.session_state.messages)
                    st.download_button(
                        label="📄 TXT",
                        data=txt,
                        file_name=f"chat_{datetime.now():%Y%m%d_%H%M%S}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="sidebar_export_txt",
                    )
                with col2:
                    pdf_bytes = ExportManager.chat_to_pdf(st.session_state.messages)
                    st.download_button(
                        label="📑 PDF",
                        data=pdf_bytes,
                        file_name=f"chat_{datetime.now():%Y%m%d_%H%M%S}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="sidebar_export_pdf",
                    )


# ---------------------------------------------------------------------------
# Chat processing — full RAG pipeline
# ---------------------------------------------------------------------------
def process_query(
    query: str,
    retriever: SemanticRetriever,
    llm: OllamaLLM,
    reranker: Reranker,
    threshold: float,
) -> Dict[str, Any]:
    """
    Run the full RAG pipeline:

        expand_query -> QueryUnderstanding -> hybrid_search -> rerank
        -> AnswerValidator -> build cited context -> LLM generate

    Always returns a dict with keys: answer, results, context_docs,
    intent, entities — so the caller never has to guard against None.
    """

    # ------------------------------------------------------------
    # STEP 1 — Query expansion + query understanding
    # ------------------------------------------------------------
    expanded = retriever.expand_query(query)

    analysis = QueryUnderstanding.analyze(query)
    intent = analysis["intent"]
    entities = analysis["entities"]

    # ------------------------------------------------------------
    # Retrieval — use the user-configurable confidence threshold
    # instead of a hardcoded 0.0 so the sidebar slider actually
    # has an effect on retrieval.
    # ------------------------------------------------------------
    results = retriever.hybrid_search(
        expanded,
        top_k=20,
        threshold=threshold,
    )

    # ------------------------------------------------------------
    # STEP 2 — Cross-encoder reranking
    # ------------------------------------------------------------
    if results:
        results = reranker.rerank(query, results)
    results = results[:5]

    # ------------------------------------------------------------
    # STEP 3 — Hallucination / confidence validation
    # ------------------------------------------------------------
    valid = AnswerValidator.validate(results)

    if not valid:
        return {
            "answer": "❌ Information not found in the knowledge base.",
            "results": [],
            "context_docs": [],
            "intent": intent,
            "entities": entities,
        }

    # ------------------------------------------------------------
    # STEP 5 — Build context with source citations
    # ------------------------------------------------------------
    context_docs: List[str] = []
    for r in results:
        answer_text = r.get("Answer", r.get("Topic", ""))
        source = r.get("Fact_ID", r.get("fact_id", "unknown"))
        context_docs.append(f"SOURCE:{source}\nCONTENT:\n{answer_text}")

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------
    if llm.is_available():
        answer = llm.generate(query, context_docs)
    else:
        # Fallback when Ollama isn't running — show raw retrieved text.
        answer = "\n\n".join(
            r.get("Answer", r.get("Topic", "")) for r in results
        )

    return {
        "answer": answer,
        "results": results,
        "context_docs": context_docs,
        "intent": intent,
        "entities": entities,
    }


def format_response(response: Dict[str, Any], show_sources: bool) -> str:
    """Format the pipeline output into markdown for the chat UI."""

    parts: List[str] = []

    answer = response.get("answer", "")
    if answer:
        parts.append(answer)
    else:
        parts.append("Information not found in the database.")

    if show_sources and response.get("results"):
        parts.append("\n\n---\n**📚 Source Documents:**\n")
        for i, res in enumerate(response["results"][:3], 1):
            score = res.get("_score", 0)
            source_type = res.get("_source", "N/A")
            confidence_cls = (
                "confidence-high" if score >= 0.50
                else "confidence-med" if score >= 0.35
                else "confidence-low"
            )
            topic = res.get("Topic", "N/A")
            category = res.get("Category", "N/A")
            fact_id = res.get("Fact_ID", res.get("fact_id", f"#{i}"))

            parts.append(
                f"**{i}.** `{fact_id}` — **{topic}** ({category})  \n"
                f"   Similarity: <span class=\"{confidence_cls}\">{score:.3f}</span> | "
                f"Source: `{source_type}`"
            )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------
WELCOME_QUESTIONS = [
    "What are the major refineries operated by IOCL?",
    "Explain the business segments of Indian Oil Corporation.",
    "Describe the Panipat refinery.",
    "What sustainability initiatives are undertaken by IOCL?",
    "What products are manufactured by IOCL?",
]


def render_welcome() -> None:
    """Show a welcome screen with suggested questions when chat is empty."""
    st.markdown(
        '<div class="welcome-box">'
        '<h2>👋 Welcome to IOCL Knowledge Assistant</h2>'
        '<p>Ask me anything about Indian Oil Corporation Limited — products, safety, '
        'operations, sustainability, and more. I answer from a local knowledge base, '
        'so no internet is needed!</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**💡 Try asking:**")
    cols = st.columns(min(len(WELCOME_QUESTIONS), 3))
    for idx, question in enumerate(WELCOME_QUESTIONS):
        col = cols[idx % len(cols)]
        with col:
            if st.button(question, key=f"suggestion_{idx}", use_container_width=True):
                # Inject the suggestion into the chat input via session state
                st.session_state["pending_question"] = question
                st.rerun()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main() -> None:
    # Header
    st.markdown(
        '<h1 style="text-align:center;">🤖 IOCL Knowledge Assistant</h1>'
        '<p style="text-align:center; color:#8b949e;">Offline AI-powered Knowledge Base — '
        'No Internet Required</p>',
        unsafe_allow_html=True,
    )

    # Load data
    csv_path = PROJECT_ROOT / "data.csv"
    df = load_dataframe(str(csv_path))

    if df is None:
        st.error(
            "❌ **data.csv not found or empty.**\n\n"
            "Place a valid `data.csv` in the project root and click **Reload Data**."
        )
        return

    # Build retriever / LLM / reranker / analytics
    with st.spinner("⚡ Initializing AI engine …"):
        df_hash = str(pd.util.hash_pandas_object(df).sum())
        retriever = build_retriever(df_hash, df)
        llm = get_llm()
        reranker = get_reranker()
        analytics = AnalyticsEngine(df)

    # Sidebar
    render_sidebar(df, retriever)

    # Tabs
    tab_chat, tab_analytics = st.tabs(["💬 Chat", "📊 Analytics Dashboard"])

    # -----------------------------------------------------------------------
    # Chat Tab
    # -----------------------------------------------------------------------
    with tab_chat:
        # Render existing messages
        if st.session_state.messages:
            for msg in st.session_state.messages:
                avatar = "👤" if msg["role"] == "user" else "🤖"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"], unsafe_allow_html=True)
        else:
            # Show welcome screen on first visit
            render_welcome()

        # Check if a suggestion button was clicked
        pending = st.session_state.pop("pending_question", None)

        # Chat input — always visible for continuous conversation
        user_query = st.chat_input("Ask a question about IOCL …")

        # Use the pending question if a suggestion was clicked
        if pending and not user_query:
            user_query = pending

        if user_query:
            # Display user message
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.search_history.append(user_query)

            with st.chat_message("user", avatar="👤"):
                st.markdown(user_query)

            # Generate response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🔍 Expanding query → Retrieving documents → Reranking → Generating response..."):
                    response = process_query(
                        user_query,
                        retriever,
                        llm,
                        reranker,
                        st.session_state.confidence_threshold,
                    )
                    formatted = format_response(response, st.session_state.show_sources)

                st.markdown(formatted, unsafe_allow_html=True)

                # Show raw sources in expander
                if response["results"]:
                    with st.expander("🔎 Detailed Source View"):
                        st.metric(
                            "Best Similarity Score",
                            f"{response['results'][0].get('_score', 0):.3f}",
                        )
                        st.metric(
                            "Retrieved Documents",
                            len(response["results"]),
                        )

                        for i, res in enumerate(response["results"], 1):
                            st.markdown(f"**Result {i}**")
                            st.json({k: v for k, v in res.items() if not k.startswith("_")})

            # Store assistant message
            st.session_state.messages.append({"role": "assistant", "content": formatted})

            # After answering, show a subtle hint to keep chatting
            st.markdown(
                '<p style="text-align:center; color:#8b949e; font-size:0.85rem; '
                'margin-top:8px;">💡 Ask another question below to continue the conversation</p>',
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------------------------
    # Analytics Tab
    # -----------------------------------------------------------------------
    with tab_analytics:
        st.markdown("## 📊 Knowledge Base Analytics")

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Total Records", analytics.total_records)
        with k2:
            st.metric("Categories", analytics.category_count)
        with k3:
            st.metric("Topics", analytics.topic_count)
        with k4:
            st.metric("Queries Asked", len(st.session_state.search_history))

        st.markdown("---")

        # Charts — two columns
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(analytics.fig_category_distribution(), use_container_width=True)
        with c2:
            st.plotly_chart(analytics.fig_topic_distribution(), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(analytics.fig_keyword_cloud(), use_container_width=True)
        with c4:
            st.plotly_chart(analytics.fig_source_types(), use_container_width=True)

        # Search activity
        if st.session_state.search_history:
            st.markdown("### 🔍 Search Activity")
            search_log = retriever.query_log if retriever else []
            st.plotly_chart(analytics.fig_search_activity(search_log), use_container_width=True)

        # Raw data view
        with st.expander("📂 Full Dataset Preview"):
            st.dataframe(df, use_container_width=True, height=400)

        # Download search results CSV
        if st.session_state.search_history:
            st.markdown("### 📥 Export Search Results")
            last_query = st.session_state.search_history[-1]
            last_results = retriever.semantic_search(last_query, top_k=20) if retriever else []
            if last_results:
                csv_data = ExportManager.results_to_csv(last_results)
                st.download_button(
                    "📄 Download Results as CSV",
                    data=csv_data,
                    file_name=f"search_results_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv",
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()