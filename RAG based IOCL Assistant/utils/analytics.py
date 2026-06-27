"""
Analytics Engine — dataset insights and Plotly visualizations.

Provides category distribution, topic breakdowns, keyword frequencies,
and search-activity analytics for the Streamlit dashboard.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Consistent colour palette across all charts
PALETTE = px.colors.qualitative.Set2


class AnalyticsEngine:
    """Compute statistics and produce Plotly figures from the dataset and query log."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()
        self._df.columns = [c.strip() for c in self._df.columns]

    # ------------------------------------------------------------------
    # Basic stats
    # ------------------------------------------------------------------
    @property
    def total_records(self) -> int:
        return len(self._df)

    @property
    def category_count(self) -> int:
        return self._df["Category"].nunique() if "Category" in self._df.columns else 0

    @property
    def topic_count(self) -> int:
        return self._df["Topic"].nunique() if "Topic" in self._df.columns else 0

    @property
    def columns(self) -> List[str]:
        return list(self._df.columns)

    def category_stats(self) -> Dict[str, int]:
        if "Category" not in self._df.columns:
            return {}
        return self._df["Category"].value_counts().to_dict()

    def topic_stats(self) -> Dict[str, int]:
        if "Topic" not in self._df.columns:
            return {}
        return self._df["Topic"].value_counts().head(15).to_dict()

    def keyword_freq(self, top_n: int = 20) -> Dict[str, int]:
        if "Keywords" not in self._df.columns:
            return {}
        counter: Counter = Counter()
        for val in self._df["Keywords"].dropna():
            for kw in str(val).split():
                kw = kw.strip().lower()
                if kw:
                    counter[kw] += 1
        return dict(counter.most_common(top_n))

    def source_type_stats(self) -> Dict[str, int]:
        if "Source_Type" not in self._df.columns:
            return {}
        return self._df["Source_Type"].value_counts().to_dict()

    # ------------------------------------------------------------------
    # Plotly figures
    # ------------------------------------------------------------------
    def fig_category_distribution(self) -> go.Figure:
        stats = self.category_stats()
        if not stats:
            return self._empty_fig("No category data available")
        fig = px.pie(
            names=list(stats.keys()),
            values=list(stats.values()),
            title="Category Distribution",
            color_discrete_sequence=PALETTE,
            hole=0.35,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(margin=dict(t=50, b=20, l=20, r=20), height=380)
        return fig

    def fig_topic_distribution(self) -> go.Figure:
        stats = self.topic_stats()
        if not stats:
            return self._empty_fig("No topic data available")
        fig = px.bar(
            x=list(stats.values()),
            y=list(stats.keys()),
            orientation="h",
            title="Top Topics",
            color=list(stats.values()),
            color_continuous_scale="Teal",
        )
        fig.update_layout(
            margin=dict(t=50, b=20, l=20, r=20),
            height=400,
            yaxis=dict(autorange="reversed"),
            showlegend=False,
        )
        return fig

    def fig_keyword_cloud(self) -> go.Figure:
        freq = self.keyword_freq(top_n=15)
        if not freq:
            return self._empty_fig("No keyword data available")
        fig = px.bar(
            x=list(freq.keys()),
            y=list(freq.values()),
            title="Top Keywords",
            color=list(freq.values()),
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            margin=dict(t=50, b=20, l=20, r=20),
            height=380,
            showlegend=False,
            xaxis_tickangle=-45,
        )
        return fig

    def fig_search_activity(self, query_log: List[Dict[str, Any]]) -> go.Figure:
        if not query_log:
            return self._empty_fig("No search queries recorded yet")

        queries = [q["query"][:40] for q in query_log[-20:]]
        scores = [q.get("top_score", 0) for q in query_log[-20:]]
        counts = [q.get("results_count", 0) for q in query_log[-20:]]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Top Relevance Scores", "Results per Query"),
        )
        fig.add_trace(
            go.Bar(x=queries, y=scores, name="Score", marker_color="teal"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=queries, y=counts, name="Count", marker_color="coral"),
            row=1, col=2,
        )
        fig.update_layout(
            height=380,
            margin=dict(t=60, b=20, l=20, r=20),
            showlegend=False,
        )
        fig.update_xaxes(tickangle=-45)
        return fig

    def fig_source_types(self) -> go.Figure:
        stats = self.source_type_stats()
        if not stats:
            return self._empty_fig("No source type data")
        fig = px.pie(
            names=list(stats.keys()),
            values=list(stats.values()),
            title="Source Type Breakdown",
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(margin=dict(t=50, b=20, l=20, r=20), height=350)
        return fig

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_fig(message: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text=message, showarrow=False, font=dict(size=16))],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=300,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        return fig
