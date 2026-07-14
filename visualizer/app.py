"""
PTS Visualizer - Interactive visualization for Pivotal Token Search

A Neuronpedia-inspired platform for exploring pivotal tokens, thought anchors,
and reasoning circuits in language models.
"""

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
import json
import html as html_lib
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import re
from collections import defaultdict

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_hf_dataset(dataset_id: str, split: str = "train") -> pd.DataFrame:
    """Load a dataset from HuggingFace Hub."""
    try:
        dataset = load_dataset(dataset_id, split=split)
        df = pd.DataFrame(dataset)
        return df, f"Loaded {len(df)} items from {dataset_id}"
    except Exception as e:
        return pd.DataFrame(), f"Error loading dataset: {str(e)}"


def load_jsonl_file(file_path: str) -> pd.DataFrame:
    """Load data from a local JSONL file."""
    try:
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return pd.DataFrame(data), f"Loaded {len(data)} items from file"
    except Exception as e:
        return pd.DataFrame(), f"Error loading file: {str(e)}"


def detect_dataset_type(df: pd.DataFrame) -> str:
    """Detect the type of PTS dataset.

    PTS v2 emits a single unified record type (``CausalReasoningEvent``) that
    carries ``event_type`` + ``granularity``. Those are checked first; the v1
    detection rules below them are unchanged so old datasets keep working.
    """
    columns = set(df.columns)

    # --- PTS v2 unified event schema -------------------------------------
    if 'event_type' in columns:
        event_types = set()
        if not df.empty:
            event_types = {
                str(v) for v in df['event_type'].dropna().unique().tolist()
            }
        # A file containing nothing but latent meta-tokens (e.g. `pts export
        # --format=metatokens`) gets its own type so the UI can lead with the
        # workspace views instead of the probability views.
        if event_types and event_types == {EVENT_LATENT}:
            return 'latent_events'
        if 'granularity' in columns:
            return 'causal_events'

    # --- PTS v1 ----------------------------------------------------------
    if 'sentence' in columns and 'sentence_id' in columns:
        return 'thought_anchors'
    elif 'steering_vector' in columns:
        return 'steering_vectors'
    elif 'chosen' in columns and 'rejected' in columns:
        return 'dpo_pairs'
    elif 'pivot_token' in columns:
        return 'pivotal_tokens'
    else:
        return 'unknown'


# ============================================================================
# PTS v2 event schema: constants and helpers
# ============================================================================

EVENT_LATENT = "latent_metatoken"
EVENT_TOKEN = "pivotal_token"
EVENT_SENTENCE = "thought_anchor"

GRANULARITY_FOR_EVENT_TYPE = {
    EVENT_LATENT: "latent",
    EVENT_TOKEN: "token",
    EVENT_SENTENCE: "sentence",
}

V2_TYPES = ('causal_events', 'latent_events')

# Green/red carry a claim about direction of effect. Latent events have no
# measured effect (prob_delta is null by construction), so they are never
# painted green or red -- purple means "observed, valence unknown".
COLOR_POSITIVE = '#22c55e'
COLOR_NEGATIVE = '#ef4444'
COLOR_LATENT = '#8b5cf6'
COLOR_UNKNOWN = '#6366f1'
COLOR_OUTCOME = '#f59e0b'

CATEGORY_COLORS = [
    '#6366f1', '#22c55e', '#ef4444', '#f59e0b', '#8b5cf6',
    '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16',
]


def is_v2_events(df: pd.DataFrame) -> bool:
    """True when the dataframe holds PTS v2 unified events."""
    return not df.empty and 'event_type' in df.columns


def _empty_fig(message: str, height: int = 400) -> go.Figure:
    """A dark-themed placeholder figure carrying an explanatory message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color="#a0a0a0")
    )
    fig.update_layout(
        template="plotly_dark",
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _val(row, key, default=None):
    """Read a field off a row, mapping NaN/None/missing onto ``default``."""
    try:
        value = row.get(key, default)
    except AttributeError:
        return default
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        # pd.isna over a list/array returns an array; those are real values.
        pass
    return value


def _as_list(value) -> list:
    """Normalize a link field (list / ndarray / None / scalar) into a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v is not None]
    if isinstance(value, np.ndarray):
        return [v for v in value.tolist() if v is not None]
    if isinstance(value, str):
        return [value] if value else []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [value]


def _num(value, default=0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(out):
        return default
    return out


def _granularity(row) -> str:
    gran = _val(row, 'granularity')
    if gran:
        return str(gran)
    return GRANULARITY_FOR_EVENT_TYPE.get(str(_val(row, 'event_type', '')), 'token')


def _event_color(row) -> str:
    """Positive=green, negative=red, latent/unscored=purple-blue.

    Never colors a latent event green or red: it has no measured valence.
    """
    if _granularity(row) == 'latent':
        return COLOR_LATENT
    is_positive = _val(row, 'is_positive')
    if is_positive is None:
        delta = _val(row, 'prob_delta')
        if delta is None:
            return COLOR_UNKNOWN
        is_positive = _num(delta) > 0
    return COLOR_POSITIVE if bool(is_positive) else COLOR_NEGATIVE


def _score_label(row) -> str:
    """Name the score honestly: latent scores are readout scores, not deltas."""
    if _granularity(row) == 'latent':
        return "Readout score"
    return "Score (|Δ probability|)"


def _filter_by_query(df: pd.DataFrame, selected_query: Optional[str]) -> pd.DataFrame:
    if (selected_query and isinstance(selected_query, str) and selected_query.strip()
            and 'query' in df.columns):
        return df[df['query'] == selected_query].copy()
    return df


def _event_hover(row) -> str:
    """Hover text shared by the v2 charts."""
    label = str(_val(row, 'label', _val(row, 'pivot_token', _val(row, 'sentence', ''))))
    if len(label) > 90:
        label = label[:87] + '...'
    parts = [
        f"<b>{html_lib.escape(label)}</b>",
        f"Event: {_val(row, 'event_type', 'unknown')}",
        f"Granularity: {_granularity(row)}",
        f"Category: {_val(row, 'category', 'n/a')}",
        f"{_score_label(row)}: {_num(_val(row, 'score')):.4f}",
    ]
    if _granularity(row) == 'latent':
        parts.append(f"Layer: {_val(row, 'layer', 'n/a')}")
        parts.append(f"Readout: {_val(row, 'readout_method', 'n/a')}")
        parts.append("<i>observational - no probability delta</i>")
    else:
        delta = _val(row, 'prob_delta')
        if delta is not None:
            parts.append(f"Δ probability: {_num(delta):+.4f}")
            parts.append(
                f"Before: {_num(_val(row, 'prob_before')):.3f} → "
                f"After: {_num(_val(row, 'prob_after')):.3f}"
            )
    return "<br>".join(parts)


def _compute_event_x(df: pd.DataFrame) -> List[float]:
    """Place every event on one shared x-axis (generation position / event order).

    Emitted events use their ``position`` (token index or sentence index),
    falling back to their order of appearance. Latent enrichment events carry a
    *negative* ``position``: an offset in tokens from the event they were read
    out before. Those are anchored to the linked event's x, so a latent event
    with position -3 lands three steps to the left of the token it precedes.
    """
    if df.empty:
        return []

    rows = list(df.iterrows())
    x_by_id: Dict[Any, float] = {}
    xs: List[Optional[float]] = [None] * len(rows)

    # Pass 1: emitted events anchor the axis.
    emitted_counter = 0
    for i, (_, row) in enumerate(rows):
        if _granularity(row) == 'latent':
            continue
        pos = _val(row, 'position')
        if pos is None or _num(pos, -1) < 0:
            x = float(emitted_counter)
        else:
            x = float(_num(pos))
        emitted_counter += 1
        xs[i] = x
        event_id = _val(row, 'event_id')
        if event_id is not None:
            x_by_id[event_id] = x

    known = [x for x in xs if x is not None]
    floor = min(known) if known else 0.0

    # Pass 2: latent events hang off the emitted event they precede.
    latent_counter = 0
    for i, (_, row) in enumerate(rows):
        if _granularity(row) != 'latent':
            continue
        pos = _val(row, 'position')
        links = (_as_list(_val(row, 'precedes_event_ids'))
                 + _as_list(_val(row, 'linked_event_ids')))
        anchor = next((x_by_id[l] for l in links if l in x_by_id), None)

        if anchor is not None:
            offset = _num(pos, 0.0) if pos is not None and _num(pos, 0.0) < 0 else 0.0
            xs[i] = anchor + offset
        elif pos is not None and _num(pos, -1) >= 0:
            xs[i] = float(_num(pos))
        else:
            # Unlinked, offset-only latent event: park it to the left of the
            # emitted timeline rather than pretending it has a position.
            xs[i] = floor - 1.0 - latent_counter * 0.1
            latent_counter += 1

    return [float(x) if x is not None else 0.0 for x in xs]


# ============================================================================
# Visualization Components
# ============================================================================

def create_token_highlight_html(context: str, token: str, prob_delta: float) -> str:
    """Create HTML with highlighted pivotal token showing full context."""
    # Escape HTML characters
    context_escaped = html_lib.escape(str(context))
    token_escaped = html_lib.escape(str(token))

    # Determine color based on probability delta
    if prob_delta > 0:
        # Positive impact - green gradient
        intensity = min(abs(prob_delta) * 2, 1.0)
        color = f"rgba(34, 197, 94, {intensity})"
        border_color = "#22c55e"
        impact_text = "Positive Impact"
    else:
        # Negative impact - red gradient
        intensity = min(abs(prob_delta) * 2, 1.0)
        color = f"rgba(239, 68, 68, {intensity})"
        border_color = "#ef4444"
        impact_text = "Negative Impact"

    # Create highlighted token span
    token_span = f'<span style="background-color: {color}; padding: 2px 6px; border-radius: 3px; border: 2px solid {border_color}; font-weight: bold; font-size: 1.1em;">{token_escaped}</span>'

    return f"""
    <div style="background-color: #1a1a2e; border-radius: 10px; padding: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="color: #a0a0a0; font-size: 0.9em;">Context Length: {len(context)} characters</span>
            <span style="background-color: {border_color}; color: white; padding: 4px 12px; border-radius: 5px; font-weight: bold;">
                {impact_text}: {'+' if prob_delta > 0 else ''}{prob_delta:.3f}
            </span>
        </div>
        <div style="font-family: monospace; padding: 15px; background-color: #0d1117; border-radius: 8px; color: #e0e0e0; line-height: 1.8; max-height: 500px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; border: 1px solid #30363d;">
            <span style="color: #8b949e;">{context_escaped}</span>{token_span}
        </div>
        <div style="margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap;">
            <span style="background-color: #238636; color: white; padding: 5px 10px; border-radius: 5px; font-size: 0.9em;">
                Token: <code style="background-color: rgba(0,0,0,0.3); padding: 2px 5px; border-radius: 3px;">{token_escaped}</code>
            </span>
        </div>
    </div>
    """


def create_probability_chart(prob_before: float, prob_after: float) -> go.Figure:
    """Create a bar chart showing probability change."""
    fig = go.Figure()

    # Ensure values are Python floats
    prob_before = float(prob_before) if prob_before is not None else 0.0
    prob_after = float(prob_after) if prob_after is not None else 0.0

    fig.add_trace(go.Bar(
        x=['Before Token', 'After Token'],
        y=[prob_before, prob_after],
        marker_color=['#6366f1', '#22c55e' if prob_after > prob_before else '#ef4444'],
        text=[f'{prob_before:.3f}', f'{prob_after:.3f}'],
        textposition='outside'
    ))

    fig.update_layout(
        title="Success Probability Change",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        template="plotly_dark",
        height=300
    )

    return fig


def create_pivotal_token_flow(df: pd.DataFrame, selected_query: str = None) -> go.Figure:
    """Create a visualization for pivotal tokens showing token impact flow."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    if 'prob_delta' not in df.columns:
        return _empty_fig(
            "This dataset has no <code>prob_delta</code> column, so token impact "
            "cannot be plotted."
        )

    # Filter by query if specified (handle None, empty string, or actual query)
    if selected_query and isinstance(selected_query, str) and selected_query.strip() and 'query' in df.columns:
        df = df[df['query'] == selected_query].copy()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data for selected query",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    # Create scatter plot of tokens by probability delta
    fig = go.Figure()

    # Separate positive and negative tokens
    positive_df = df[df.get('is_positive', df['prob_delta'] > 0) == True] if 'is_positive' in df.columns else df[df['prob_delta'] > 0]
    negative_df = df[df.get('is_positive', df['prob_delta'] > 0) == False] if 'is_positive' in df.columns else df[df['prob_delta'] <= 0]

    # Add positive tokens
    if not positive_df.empty:
        hover_text = [
            f"Token: {row.get('pivot_token', 'N/A')}<br>"
            f"Δ Prob: +{row.get('prob_delta', 0):.3f}<br>"
            f"Before: {row.get('prob_before', 0):.3f}<br>"
            f"After: {row.get('prob_after', 0):.3f}<br>"
            f"Query: {str(row.get('query', ''))[:50]}..."
            for _, row in positive_df.iterrows()
        ]
        y_vals = positive_df['prob_delta'].tolist()
        sizes = [10 + abs(v) * 30 for v in y_vals]
        fig.add_trace(go.Scatter(
            x=list(range(len(positive_df))),
            y=y_vals,
            mode='markers',
            name='Positive Impact',
            marker=dict(
                size=sizes,
                color='#22c55e',
                opacity=0.7
            ),
            hovertext=hover_text,
            hoverinfo='text'
        ))

    # Add negative tokens
    if not negative_df.empty:
        hover_text = [
            f"Token: {row.get('pivot_token', 'N/A')}<br>"
            f"Δ Prob: {row.get('prob_delta', 0):.3f}<br>"
            f"Before: {row.get('prob_before', 0):.3f}<br>"
            f"After: {row.get('prob_after', 0):.3f}<br>"
            f"Query: {str(row.get('query', ''))[:50]}..."
            for _, row in negative_df.iterrows()
        ]
        y_vals = negative_df['prob_delta'].tolist()
        sizes = [10 + abs(v) * 30 for v in y_vals]
        fig.add_trace(go.Scatter(
            x=list(range(len(negative_df))),
            y=y_vals,
            mode='markers',
            name='Negative Impact',
            marker=dict(
                size=sizes,
                color='#ef4444',
                opacity=0.7
            ),
            hovertext=hover_text,
            hoverinfo='text'
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Pivotal Token Impact Distribution",
        xaxis_title="Token Index",
        yaxis_title="Probability Delta",
        template="plotly_dark",
        height=500,
        showlegend=True
    )

    return fig


def create_thought_anchor_graph(df: pd.DataFrame, selected_query: str = None) -> go.Figure:
    """Create an interactive graph visualization of thought anchor dependencies."""
    dataset_type = detect_dataset_type(df)

    # PTS v2 events carry their own causal edges; use the unified graph.
    if dataset_type in V2_TYPES:
        return create_causal_event_graph(df, selected_query)

    # For pivotal tokens and steering vectors, create a token impact visualization
    if dataset_type in ('pivotal_tokens', 'steering_vectors'):
        return create_pivotal_token_flow(df, selected_query)

    if df.empty or 'sentence_id' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No thought anchor data available. Load a thought anchors dataset to see the reasoning graph.",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                          font=dict(size=14, color="#a0a0a0"))
        fig.update_layout(template="plotly_dark", height=400)
        return fig

    # Filter by query if specified (handle None, empty string, or actual query)
    if selected_query and isinstance(selected_query, str) and selected_query.strip():
        df = df[df['query'] == selected_query].copy()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data for selected query",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    # Create networkx graph
    G = nx.DiGraph()

    # Add nodes (sentences)
    for idx, row in df.iterrows():
        sentence_id = row.get('sentence_id', idx)
        importance = row.get('importance_score', abs(row.get('prob_delta', 0)))
        is_positive = row.get('is_positive', row.get('prob_delta', 0) > 0)
        sentence = row.get('sentence', '')[:50] + '...' if len(row.get('sentence', '')) > 50 else row.get('sentence', '')

        G.add_node(sentence_id,
                   importance=importance,
                   is_positive=is_positive,
                   sentence=sentence,
                   category=row.get('sentence_category', 'unknown'))

    # Add edges from causal dependencies
    for idx, row in df.iterrows():
        sentence_id = row.get('sentence_id', idx)
        dependencies = row.get('causal_dependencies', [])
        if isinstance(dependencies, list):
            for dep in dependencies:
                if dep in G.nodes():
                    G.add_edge(dep, sentence_id)

    # If no explicit dependencies, create sequential edges
    if G.number_of_edges() == 0:
        sorted_nodes = sorted(G.nodes())
        for i in range(len(sorted_nodes) - 1):
            G.add_edge(sorted_nodes[i], sorted_nodes[i+1])

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([float(x0), float(x1), None])
        edge_y.extend([float(y0), float(y1), None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_texts = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(float(x))
        node_y.append(float(y))

        node_data = G.nodes[node]
        is_positive = node_data.get('is_positive', True)
        importance = float(node_data.get('importance', 0.3))

        node_colors.append('#22c55e' if is_positive else '#ef4444')
        node_sizes.append(20 + importance * 50)

        hover_text = f"Sentence {node}<br>"
        hover_text += f"Category: {node_data.get('category', 'unknown')}<br>"
        hover_text += f"Importance: {importance:.3f}<br>"
        hover_text += f"Text: {node_data.get('sentence', 'N/A')}"
        node_texts.append(hover_text)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[str(n) for n in G.nodes()],
        textposition="top center",
        hovertext=node_texts,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=2, color='white')
        )
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title="Thought Anchor Reasoning Graph",
        showlegend=False,
        hovermode='closest',
        template="plotly_dark",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )

    return fig


# ============================================================================
# PTS v2: multiscale timeline, causal event graph, workspace heatmap
# ============================================================================

def create_multiscale_timeline(df: pd.DataFrame, selected_query: str = None) -> go.Figure:
    """Four scales of reasoning events on one shared generation axis.

    Row 1  latent meta-tokens        (diamonds, y = layer, size = readout score)
    Row 2  emitted pivotal tokens    (circles, y = Δ probability)
    Row 3  thought-anchor sentences  (wide bars, height = Δ probability)
    Row 4  success probability       (from prob_before/prob_after of emitted events)
    """
    if df is None or df.empty:
        return _empty_fig("No data loaded. Load a PTS dataset to see the multiscale timeline.", 640)

    if not is_v2_events(df):
        return _empty_fig(
            "The multiscale timeline needs PTS v2 causal events (event_type + granularity).<br>"
            "Load a causal_events or latent_events dataset, or run "
            "<code>pts migrate</code> on a v1 file.",
            640,
        )

    work = _filter_by_query(df, selected_query)
    if work.empty:
        return _empty_fig("No events for the selected query.", 640)

    work = work.reset_index(drop=True)
    xs = _compute_event_x(work)
    work = work.assign(_x=xs)

    gran = work.apply(_granularity, axis=1)
    latent_df = work[gran == 'latent']
    token_df = work[gran == 'token']
    sentence_df = work[gran == 'sentence']

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.24, 0.22, 0.22, 0.32],
        subplot_titles=(
            "Latent meta-tokens (readout score - NOT a probability delta)",
            "Emitted pivotal tokens (Δ success probability)",
            "Thought-anchor sentences (Δ success probability)",
            "Success probability across emitted events",
        ),
    )

    # --- Row 1: latent meta-tokens -------------------------------------
    if not latent_df.empty:
        scores = [_num(_val(r, 'score')) for _, r in latent_df.iterrows()]
        max_score = max(scores) if scores else 1.0
        max_score = max_score if max_score > 0 else 1.0
        fig.add_trace(
            go.Scatter(
                x=latent_df['_x'].tolist(),
                y=[_num(_val(r, 'layer')) for _, r in latent_df.iterrows()],
                mode='markers',
                name='Latent meta-token',
                marker=dict(
                    symbol='diamond',
                    size=[6 + 10 * (s / max_score) for s in scores],
                    color=COLOR_LATENT,
                    opacity=0.85,
                    line=dict(width=1, color='#d8b4fe'),
                ),
                hovertext=[_event_hover(r) for _, r in latent_df.iterrows()],
                hoverinfo='text',
            ),
            row=1, col=1,
        )
    else:
        fig.add_annotation(
            text="no latent meta-token events in this dataset",
            xref="x domain", yref="y domain", x=0.5, y=0.5,
            showarrow=False, font=dict(size=11, color="#6b7280"),
            row=1, col=1,
        )

    # --- Row 2: emitted pivotal tokens ---------------------------------
    if not token_df.empty:
        fig.add_trace(
            go.Scatter(
                x=token_df['_x'].tolist(),
                y=[_num(_val(r, 'prob_delta')) for _, r in token_df.iterrows()],
                mode='markers',
                name='Pivotal token',
                marker=dict(
                    symbol='circle',
                    size=[10 + 25 * abs(_num(_val(r, 'score'))) for _, r in token_df.iterrows()],
                    color=[_event_color(r) for _, r in token_df.iterrows()],
                    opacity=0.8,
                    line=dict(width=1, color='#111827'),
                ),
                hovertext=[_event_hover(r) for _, r in token_df.iterrows()],
                hoverinfo='text',
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    else:
        fig.add_annotation(
            text="no emitted pivotal-token events",
            xref="x domain", yref="y domain", x=0.5, y=0.5,
            showarrow=False, font=dict(size=11, color="#6b7280"),
            row=2, col=1,
        )

    # --- Row 3: thought-anchor sentences -------------------------------
    if not sentence_df.empty:
        deltas = [_num(_val(r, 'prob_delta')) for _, r in sentence_df.iterrows()]
        # A sentence with no measured delta still deserves a bar: fall back to
        # its score, drawn in neutral blue rather than green/red.
        heights = [
            d if d != 0 else _num(_val(r, 'score'))
            for d, (_, r) in zip(deltas, sentence_df.iterrows())
        ]
        fig.add_trace(
            go.Bar(
                x=sentence_df['_x'].tolist(),
                y=heights,
                name='Thought anchor',
                width=0.8,
                marker=dict(
                    color=[_event_color(r) for _, r in sentence_df.iterrows()],
                    opacity=0.65,
                    line=dict(width=1, color='#111827'),
                ),
                hovertext=[_event_hover(r) for _, r in sentence_df.iterrows()],
                hoverinfo='text',
            ),
            row=3, col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    else:
        fig.add_annotation(
            text="no thought-anchor sentence events",
            xref="x domain", yref="y domain", x=0.5, y=0.5,
            showarrow=False, font=dict(size=11, color="#6b7280"),
            row=3, col=1,
        )

    # --- Row 4: success probability curve ------------------------------
    emitted = work[gran != 'latent']
    curve_x: List[float] = []
    curve_y: List[float] = []
    curve_hover: List[str] = []
    if not emitted.empty:
        ordered = emitted.sort_values('_x')
        for _, row in ordered.iterrows():
            before = _val(row, 'prob_before')
            after = _val(row, 'prob_after')
            if before is None or after is None:
                continue
            x = float(row['_x'])
            curve_x.extend([x, x])
            curve_y.extend([_num(before), _num(after)])
            hover = _event_hover(row)
            curve_hover.extend([f"before<br>{hover}", f"after<br>{hover}"])

    if curve_x:
        fig.add_trace(
            go.Scatter(
                x=curve_x,
                y=curve_y,
                mode='lines+markers',
                name='Success probability',
                line=dict(color=COLOR_UNKNOWN, width=2),
                marker=dict(size=7, color=COLOR_UNKNOWN),
                hovertext=curve_hover,
                hoverinfo='text',
            ),
            row=4, col=1,
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=4, col=1)
    else:
        fig.add_annotation(
            text="no prob_before/prob_after on any emitted event",
            xref="x domain", yref="y domain", x=0.5, y=0.5,
            showarrow=False, font=dict(size=11, color="#6b7280"),
            row=4, col=1,
        )

    fig.update_yaxes(title_text="Layer", row=1, col=1)
    fig.update_yaxes(title_text="Δ prob", row=2, col=1)
    fig.update_yaxes(title_text="Δ prob", row=3, col=1)
    fig.update_yaxes(title_text="P(success)", range=[0, 1], row=4, col=1)
    fig.update_xaxes(title_text="Generation position / event order", row=4, col=1)

    fig.update_layout(
        title="Multiscale Reasoning Timeline",
        template="plotly_dark",
        height=760,
        showlegend=True,
        hovermode='closest',
        bargap=0.2,
    )

    return fig


def create_causal_event_graph(df: pd.DataFrame, selected_query: str = None) -> go.Figure:
    """Causal graph over v2 events: latent -> token -> sentence -> outcome.

    Edges come from ``precedes_event_ids`` / ``linked_event_ids`` /
    ``parent_event_id``. Node shape encodes granularity, node color encodes
    valence (latent nodes stay purple - they have no valence).
    """
    if df is None or df.empty:
        return _empty_fig("No data loaded. Load a PTS dataset to see the causal event graph.", 550)

    # v1 datasets keep the old graph exactly as it was.
    if not is_v2_events(df):
        return create_thought_anchor_graph(df, selected_query)

    work = _filter_by_query(df, selected_query)
    if work.empty:
        return _empty_fig("No events for the selected query.", 550)

    work = work.reset_index(drop=True)
    xs = _compute_event_x(work)
    work = work.assign(_x=xs)

    G = nx.DiGraph()
    id_by_index: Dict[int, str] = {}

    for i, (_, row) in enumerate(work.iterrows()):
        event_id = _val(row, 'event_id') or f"event_{i}"
        id_by_index[i] = event_id
        label = str(_val(row, 'label', ''))
        G.add_node(
            event_id,
            label=label[:60] + ('...' if len(label) > 60 else ''),
            granularity=_granularity(row),
            event_type=str(_val(row, 'event_type', 'unknown')),
            category=_val(row, 'category', 'unknown'),
            score=_num(_val(row, 'score')),
            color=_event_color(row),
            hover=_event_hover(row),
            x_hint=float(row['_x']),
        )

    for i, (_, row) in enumerate(work.iterrows()):
        source = id_by_index[i]
        targets = set(_as_list(_val(row, 'precedes_event_ids')))
        targets |= set(_as_list(_val(row, 'linked_event_ids')))
        for target in targets:
            if target in G.nodes() and target != source:
                G.add_edge(source, target)

        parent = _val(row, 'parent_event_id')
        if parent and parent in G.nodes() and parent != source:
            G.add_edge(parent, source)

    # No links attached (e.g. `pts link` was never run): fall back to
    # generation order so the graph still shows the reasoning sequence.
    if G.number_of_edges() == 0 and G.number_of_nodes() > 1:
        ordered = sorted(G.nodes(), key=lambda n: G.nodes[n]['x_hint'])
        for a, b in zip(ordered, ordered[1:]):
            G.add_edge(a, b)

    # Terminal outcome node: everything that leads nowhere leads to the outcome.
    OUTCOME = "__outcome__"
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
    if leaves:
        G.add_node(
            OUTCOME,
            label="outcome",
            granularity='outcome',
            event_type='outcome',
            category='n/a',
            score=1.0,
            color=COLOR_OUTCOME,
            hover="Task outcome<br>(terminal node: success / failure of the completion)",
            x_hint=max((G.nodes[n]['x_hint'] for n in G.nodes() if n != OUTCOME), default=0.0) + 1.0,
        )
        for leaf in leaves:
            G.add_edge(leaf, OUTCOME)

    pos = nx.spring_layout(G, k=1.6, iterations=60, seed=42)

    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x.extend([float(x0), float(x1), None])
        edge_y.extend([float(y0), float(y1), None])

    traces = [go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#4b5563'),
        hoverinfo='none',
        mode='lines',
        showlegend=False,
    )]

    symbols = {
        'latent': 'diamond',
        'token': 'circle',
        'sentence': 'square',
        'outcome': 'star',
    }
    names = {
        'latent': 'Latent meta-token',
        'token': 'Pivotal token',
        'sentence': 'Thought anchor',
        'outcome': 'Outcome',
    }

    for gran, symbol in symbols.items():
        nodes = [n for n in G.nodes() if G.nodes[n]['granularity'] == gran]
        if not nodes:
            continue
        traces.append(go.Scatter(
            x=[float(pos[n][0]) for n in nodes],
            y=[float(pos[n][1]) for n in nodes],
            mode='markers',
            name=names[gran],
            marker=dict(
                symbol=symbol,
                size=[14 + 30 * min(G.nodes[n]['score'], 1.0) for n in nodes],
                color=[G.nodes[n]['color'] for n in nodes],
                line=dict(width=1.5, color='white'),
            ),
            hovertext=[G.nodes[n]['hover'] for n in nodes],
            hoverinfo='text',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Causal Event Graph (latent → token → sentence → outcome)",
        showlegend=True,
        hovermode='closest',
        template="plotly_dark",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
    )
    return fig


def create_workspace_heatmap(df: pd.DataFrame, selected_query: str = None) -> go.Figure:
    """Latent workspace: meta-token (or category) x position, colored by readout score.

    The color scale is a *readout* score (how strongly the lens surfaces that
    meta-token), never a probability delta.
    """
    if df is None or df.empty:
        return _empty_fig("No data loaded. Load a PTS dataset with latent events.", 500)

    if not is_v2_events(df):
        return _empty_fig(
            "The workspace heatmap needs latent meta-token events (PTS v2).<br>"
            "Generate them with <code>pts run --scale=latent</code> or "
            "<code>pts enrich</code>.",
            500,
        )

    work = _filter_by_query(df, selected_query)
    latent = work[work.apply(_granularity, axis=1) == 'latent'] if not work.empty else work

    if latent.empty:
        return _empty_fig(
            "No latent meta-token events in this selection.<br>"
            "The workspace heatmap only applies to <code>latent_metatoken</code> events.",
            500,
        )

    latent = latent.reset_index(drop=True)
    latent = latent.assign(_x=_compute_event_x(latent))

    labels = [str(_val(r, 'label', '')) for _, r in latent.iterrows()]
    unique_labels = sorted(set(labels))

    # Too many distinct meta-tokens to read as rows: fall back to categories.
    row_key = 'meta-token'
    if len(unique_labels) > 30 and 'category' in latent.columns:
        keys = [str(_val(r, 'category', 'unknown')) for _, r in latent.iterrows()]
        row_key = 'category'
    else:
        keys = labels

    xs = [float(x) for x in latent['_x'].tolist()]
    scores = [_num(_val(r, 'score')) for _, r in latent.iterrows()]
    layers = [_val(r, 'layer', 'n/a') for _, r in latent.iterrows()]
    readouts = [_val(r, 'readout_method', 'n/a') for _, r in latent.iterrows()]

    row_values = sorted(set(keys))
    col_values = sorted(set(xs))
    row_index = {v: i for i, v in enumerate(row_values)}
    col_index = {v: i for i, v in enumerate(col_values)}

    z = np.full((len(row_values), len(col_values)), np.nan)
    hover = [["" for _ in col_values] for _ in row_values]

    for key, x, score, layer, readout in zip(keys, xs, scores, layers, readouts):
        r, c = row_index[key], col_index[x]
        # Several layers can surface the same meta-token at one position; keep
        # the strongest readout.
        if np.isnan(z[r, c]) or score > z[r, c]:
            z[r, c] = score
            hover[r][c] = (
                f"<b>{html_lib.escape(str(key)[:60])}</b><br>"
                f"Position: {x:g}<br>"
                f"Readout score: {score:.4f}<br>"
                f"Layer: {layer}<br>"
                f"Readout method: {readout}<br>"
                f"<i>readout score, not a probability delta</i>"
            )

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"{c:g}" for c in col_values],
        y=[str(v)[:45] for v in row_values],
        colorscale='Viridis',
        hoverongaps=False,
        text=hover,
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(title="Readout<br>score"),
    ))

    fig.update_layout(
        title=f"Latent Workspace Heatmap ({row_key} x position, color = readout score)",
        xaxis_title="Generation position / event order",
        yaxis_title=row_key.capitalize(),
        template="plotly_dark",
        height=max(400, 60 + 22 * len(row_values)),
    )
    return fig


def create_event_trace(df: pd.DataFrame, selected_query: str) -> Tuple[str, go.Figure]:
    """Step-by-step HTML cards + probability progression for v2 events."""
    if df is None or df.empty:
        return "No events found for this query", _empty_fig("No events", 300)

    work = df.reset_index(drop=True)
    work = work.assign(_x=_compute_event_x(work)).sort_values('_x')

    query_text = str(selected_query or "")
    html_parts = [f"""
    <div style="font-family: sans-serif; padding: 20px; background-color: #1a1a2e; border-radius: 10px;">
        <h3 style="color: #e0e0e0; border-bottom: 2px solid #6366f1; padding-bottom: 10px;">
            Query: {html_lib.escape(query_text[:100])}{'...' if len(query_text) > 100 else ''}
        </h3>
        <p style="color: #a0a0a0; margin: 10px 0;">{len(work)} causal reasoning events for this query</p>
        <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 20px;">
    """]

    curve_x, curve_y = [], []

    for _, row in work.iterrows():
        gran = _granularity(row)
        color = _event_color(row)
        label = html_lib.escape(str(_val(row, 'label', '')))
        category = html_lib.escape(str(_val(row, 'category', 'unknown')))
        score = _num(_val(row, 'score'))
        x = float(row['_x'])

        if gran == 'latent':
            metric_html = (
                f'<span style="color: {color}; font-weight: bold;">'
                f'readout score {score:.3f}</span>'
            )
            extra = (
                f'<span style="background-color: #333; padding: 3px 8px; border-radius: 3px; '
                f'font-size: 0.8em; color: #a0a0a0;">Layer {_val(row, "layer", "n/a")}</span>'
                f'<span style="background-color: #333; padding: 3px 8px; border-radius: 3px; '
                f'font-size: 0.8em; color: #a0a0a0;">{_val(row, "readout_method", "n/a")}</span>'
                f'<span style="background-color: #3b2f5e; padding: 3px 8px; border-radius: 3px; '
                f'font-size: 0.8em; color: #d8b4fe;">observational - no probability delta</span>'
            )
        else:
            delta = _num(_val(row, 'prob_delta'))
            metric_html = (
                f'<span style="color: {color}; font-weight: bold;">'
                f'{"+" if delta > 0 else ""}{delta:.3f} Δ probability</span>'
            )
            extra = (
                f'<span style="background-color: #333; padding: 3px 8px; border-radius: 3px; '
                f'font-size: 0.8em; color: #a0a0a0;">Before: {_num(_val(row, "prob_before")):.3f}</span>'
                f'<span style="background-color: #333; padding: 3px 8px; border-radius: 3px; '
                f'font-size: 0.8em; color: #a0a0a0;">After: {_num(_val(row, "prob_after")):.3f}</span>'
            )
            before, after = _val(row, 'prob_before'), _val(row, 'prob_after')
            if before is not None and after is not None:
                curve_x.extend([x, x])
                curve_y.extend([_num(before), _num(after)])

        html_parts.append(f"""
        <div style="background-color: rgba(255,255,255,0.03); border-left: 4px solid {color};
                    padding: 15px; border-radius: 5px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #a0a0a0; font-size: 0.9em;">
                    {html_lib.escape(str(_val(row, 'event_type', 'event')))} | {gran} | pos {x:g} | {category}
                </span>
                {metric_html}
            </div>
            <p style="color: #e0e0e0; margin: 10px 0; font-family: monospace; white-space: pre-wrap; word-break: break-word;">{label}</p>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">{extra}</div>
        </div>
        """)

    html_parts.append("</div></div>")

    if curve_x:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve_x, y=curve_y,
            mode='lines+markers',
            name='Success probability',
            line=dict(color=COLOR_UNKNOWN, width=2),
            marker=dict(size=8),
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Success Probability Across Emitted Events",
            xaxis_title="Generation position / event order",
            yaxis_title="Success Probability",
            yaxis_range=[0, 1],
            template="plotly_dark",
            height=300,
        )
    else:
        fig = _empty_fig(
            "No emitted events with prob_before/prob_after for this query.<br>"
            "Latent events are observational and carry no probability delta.",
            300,
        )

    return "\n".join(html_parts), fig


def create_probability_space_visualization(df: pd.DataFrame, color_by: str = 'is_positive') -> go.Figure:
    """Create a probability space visualization for pivotal tokens (prob_before vs prob_after)."""
    fig = go.Figure()

    # Color palette for categorical values
    CATEGORY_COLORS = [
        '#6366f1', '#22c55e', '#ef4444', '#f59e0b', '#8b5cf6',
        '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16'
    ]

    # Determine color column
    use_colorscale = False
    if color_by in df.columns:
        color_col = df[color_by]
        if color_by == 'is_positive':
            colors = ['#22c55e' if v else '#ef4444' for v in color_col]
        else:
            # Convert to list
            values = color_col.tolist() if hasattr(color_col, 'tolist') else list(color_col)

            if len(values) > 0:
                # Check if numeric
                if isinstance(values[0], (int, float)) and not isinstance(values[0], bool):
                    colors = values
                    use_colorscale = True
                else:
                    # Categorical - map to colors
                    unique_vals = list(set(values))
                    color_map = {val: CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i, val in enumerate(unique_vals)}
                    colors = [color_map[v] for v in values]
            else:
                colors = ['#6366f1'] * len(df)
    else:
        colors = ['#6366f1'] * len(df)

    # Create hover text
    hover_texts = []
    for _, row in df.iterrows():
        # v1 calls it pivot_token, v2 calls it label.
        token_text = _val(row, 'pivot_token', _val(row, 'label', 'N/A'))
        text = f"Token: {token_text}<br>"
        text += f"Before: {_num(_val(row, 'prob_before')):.3f}<br>"
        text += f"After: {_num(_val(row, 'prob_after')):.3f}<br>"
        text += f"Delta: {_num(_val(row, 'prob_delta')):+.3f}<br>"
        text += f"Query: {str(_val(row, 'query', ''))[:40]}..."
        hover_texts.append(text)

    fig.add_trace(go.Scatter(
        x=df['prob_before'].tolist(),
        y=df['prob_after'].tolist(),
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            opacity=0.6,
            colorscale='Viridis' if use_colorscale else None,
            showscale=use_colorscale
        ),
        hovertext=hover_texts,
        hoverinfo='text',
        name='Pivotal Tokens'
    ))

    # Add diagonal line (no change)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray', width=1),
        name='No Change Line',
        showlegend=True
    ))

    fig.update_layout(
        title="Probability Space: Before vs After Pivotal Token",
        xaxis_title="Probability Before Token",
        yaxis_title="Probability After Token",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_dark",
        height=500
    )

    # Add annotations
    fig.add_annotation(
        x=0.2, y=0.8,
        text="Positive Impact ↑",
        showarrow=False,
        font=dict(color="#22c55e", size=12)
    )
    fig.add_annotation(
        x=0.8, y=0.2,
        text="Negative Impact ↓",
        showarrow=False,
        font=dict(color="#ef4444", size=12)
    )

    return fig


def create_embedding_visualization(df: pd.DataFrame, color_by: str = 'is_positive') -> go.Figure:
    """Create UMAP/t-SNE visualization of embeddings or alternative visualization for pivotal tokens."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    dataset_type = detect_dataset_type(df)

    # Check for embeddings
    embedding_col = None
    for col in ['sentence_embedding', 'steering_vector']:
        if col in df.columns:
            embedding_col = col
            break

    # For pivotal tokens without embeddings, create a probability space visualization
    if embedding_col is None:
        if dataset_type == 'pivotal_tokens' and 'prob_before' in df.columns and 'prob_after' in df.columns:
            return create_probability_space_visualization(df, color_by)

        # v2 events: only emitted events live in probability space. Latent
        # events have no prob_before/prob_after and must not be plotted there.
        if dataset_type in V2_TYPES and 'prob_before' in df.columns and 'prob_after' in df.columns:
            emitted = df[df.apply(_granularity, axis=1) != 'latent']
            emitted = emitted.dropna(subset=['prob_before', 'prob_after'])
            if emitted.empty:
                return _empty_fig(
                    "No emitted events with probabilities to plot.<br>"
                    "Latent meta-token events are observational: they carry a readout "
                    "score, not a probability delta, so they have no place in "
                    "probability space.",
                    450,
                )
            if color_by not in emitted.columns:
                color_by = 'is_positive' if 'is_positive' in emitted.columns else 'event_type'
            return create_probability_space_visualization(emitted.reset_index(drop=True), color_by)

        fig = go.Figure()
        fig.add_annotation(
            text="No embedding data found. Embeddings are available in thought_anchors and steering_vectors datasets.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="#a0a0a0")
        )
        fig.update_layout(template="plotly_dark", height=400)
        return fig

    # Extract embeddings
    embeddings = []
    valid_indices = []

    for idx, row in df.iterrows():
        emb = row.get(embedding_col, [])
        # Handle both list and numpy array formats
        if emb is not None:
            if isinstance(emb, np.ndarray) and len(emb) > 0:
                embeddings.append(emb.tolist())
                valid_indices.append(idx)
            elif isinstance(emb, list) and len(emb) > 0:
                embeddings.append(emb)
                valid_indices.append(idx)

    if len(embeddings) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Not enough embeddings for visualization (need at least 3)",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    embeddings = np.array(embeddings)

    # Reduce dimensionality
    n_samples = len(embeddings)
    # t-SNE requires perplexity < n_samples; the clamp keeps small datasets
    # (a handful of embeddings) from blowing up.
    perplexity = min(30, max(5, n_samples // 3))
    perplexity = max(2, min(perplexity, n_samples - 1))

    if embeddings.shape[1] > 50:
        # First reduce with PCA
        pca = PCA(n_components=min(50, n_samples - 1))
        embeddings = pca.fit_transform(embeddings)

    # Then t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings)

    # Create dataframe for plotting
    plot_df = df.iloc[valid_indices].copy()
    plot_df['x'] = coords[:, 0].tolist()
    plot_df['y'] = coords[:, 1].tolist()

    # Handle color column
    if color_by not in plot_df.columns:
        color_by = 'is_positive' if 'is_positive' in plot_df.columns else None

    fig = go.Figure()

    # Determine text field for hover (v1: sentence/pivot_token, v2: label)
    if 'sentence' in plot_df.columns:
        text_field = 'sentence'
    elif 'pivot_token' in plot_df.columns:
        text_field = 'pivot_token'
    else:
        text_field = 'label'

    if color_by and color_by in plot_df.columns:
        # Group by color column for separate traces
        if color_by == 'is_positive':
            # Special handling for boolean is_positive
            for is_pos in [True, False]:
                mask = plot_df[color_by] == is_pos
                subset = plot_df[mask]
                if len(subset) > 0:
                    hover_texts = [str(row.get(text_field, ''))[:100] for _, row in subset.iterrows()]
                    fig.add_trace(go.Scatter(
                        x=subset['x'].tolist(),
                        y=subset['y'].tolist(),
                        mode='markers',
                        name='Positive' if is_pos else 'Negative',
                        marker=dict(
                            size=8,
                            color='#22c55e' if is_pos else '#ef4444',
                            opacity=0.7
                        ),
                        hovertext=hover_texts,
                        hoverinfo='text'
                    ))
        else:
            # Categorical coloring
            unique_vals = plot_df[color_by].unique()
            colors = ['#6366f1', '#22c55e', '#ef4444', '#f59e0b', '#8b5cf6',
                      '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16']
            for i, val in enumerate(unique_vals):
                mask = plot_df[color_by] == val
                subset = plot_df[mask]
                if len(subset) > 0:
                    hover_texts = [str(row.get(text_field, ''))[:100] for _, row in subset.iterrows()]
                    fig.add_trace(go.Scatter(
                        x=subset['x'].tolist(),
                        y=subset['y'].tolist(),
                        mode='markers',
                        name=str(val),
                        marker=dict(
                            size=8,
                            color=colors[i % len(colors)],
                            opacity=0.7
                        ),
                        hovertext=hover_texts,
                        hoverinfo='text'
                    ))
    else:
        # No color grouping
        hover_texts = [str(row.get(text_field, ''))[:100] for _, row in plot_df.iterrows()]
        fig.add_trace(go.Scatter(
            x=plot_df['x'].tolist(),
            y=plot_df['y'].tolist(),
            mode='markers',
            name='Embeddings',
            marker=dict(
                size=8,
                color='#6366f1',
                opacity=0.7
            ),
            hovertext=hover_texts,
            hoverinfo='text'
        ))

    fig.update_layout(
        title="Embedding Space Visualization (t-SNE)",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        template="plotly_dark",
        height=500,
        showlegend=True
    )

    return fig


def create_pivotal_token_trace(df: pd.DataFrame, selected_query: str) -> Tuple[str, go.Figure]:
    """Create a trace visualization for pivotal tokens in a query."""
    if df.empty:
        return "No tokens found for this query", go.Figure()

    # Build HTML for token cards
    html_parts = [f"""
    <div style="font-family: sans-serif; padding: 20px; background-color: #1a1a2e; border-radius: 10px;">
        <h3 style="color: #e0e0e0; border-bottom: 2px solid #6366f1; padding-bottom: 10px;">
            Query: {selected_query[:100]}{'...' if len(selected_query) > 100 else ''}
        </h3>
        <p style="color: #a0a0a0; margin: 10px 0;">Found {len(df)} pivotal tokens for this query</p>
        <div style="display: flex; flex-direction: column; gap: 15px; margin-top: 20px;">
    """]

    prob_deltas = []
    token_indices = []

    for idx, (_, row) in enumerate(df.iterrows()):
        token = row.get('pivot_token', 'N/A')
        context = row.get('pivot_context', '')
        is_positive = row.get('is_positive', row.get('prob_delta', 0) > 0)
        prob_delta = row.get('prob_delta', 0)
        prob_before = row.get('prob_before', 0)
        prob_after = row.get('prob_after', 0)
        task_type = row.get('task_type', 'unknown')

        # Color based on impact
        bg_color = "rgba(34, 197, 94, 0.2)" if is_positive else "rgba(239, 68, 68, 0.2)"
        border_color = "#22c55e" if is_positive else "#ef4444"

        # Show full context in a scrollable container - no truncation
        # Escape HTML characters in context and token
        context_escaped = html_lib.escape(str(context))
        token_escaped = html_lib.escape(str(token))

        # Build token card with full context (scrollable)
        card_html = f"""
        <div style="background-color: {bg_color}; border-left: 4px solid {border_color};
                    padding: 15px; border-radius: 5px; margin-bottom: 5px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="color: #a0a0a0; font-size: 0.9em;">Token #{idx + 1} | {task_type}</span>
                <span style="color: {border_color}; font-weight: bold; font-size: 1.1em;">
                    {'+'if prob_delta > 0 else ''}{prob_delta:.3f}
                </span>
            </div>
            <div style="background-color: #1a1a2e; padding: 10px; border-radius: 5px; max-height: 200px; overflow-y: auto; margin: 10px 0;">
                <span style="color: #888; font-family: monospace; font-size: 0.85em; white-space: pre-wrap; word-break: break-word;">{context_escaped}</span><span style="background-color: {border_color}; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-family: monospace;">{token_escaped}</span>
            </div>
            <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                <span style="background-color: #333; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; color: #a0a0a0;">
                    Before: {prob_before:.3f}
                </span>
                <span style="background-color: #333; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; color: #a0a0a0;">
                    After: {prob_after:.3f}
                </span>
                <span style="background-color: #333; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; color: #6366f1;">
                    Context: {len(context)} chars
                </span>
            </div>
        </div>
        """
        html_parts.append(card_html)
        prob_deltas.append(prob_delta)
        token_indices.append(idx)

    html_parts.append("</div></div>")

    # Create probability delta chart
    fig = go.Figure()

    # Ensure all values are Python native types
    prob_deltas = [float(d) for d in prob_deltas]
    colors = ['#22c55e' if d > 0 else '#ef4444' for d in prob_deltas]

    fig.add_trace(go.Bar(
        x=token_indices,
        y=prob_deltas,
        marker_color=colors,
        name='Probability Delta',
        hovertemplate='Token #%{x}<br>Δ Prob: %{y:.3f}<extra></extra>'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Probability Impact per Token",
        xaxis_title="Token Index",
        yaxis_title="Probability Delta",
        template="plotly_dark",
        height=300
    )

    return "\n".join(html_parts), fig


def create_circuit_visualization(df: pd.DataFrame, query_idx: int = 0) -> Tuple[str, go.Figure]:
    """Create step-by-step circuit visualization for reasoning trace."""
    if df.empty:
        return "No data available", go.Figure()

    dataset_type = detect_dataset_type(df)

    # Get unique queries
    queries = df['query'].unique() if 'query' in df.columns else []
    if len(queries) == 0:
        return "No queries found", go.Figure()

    query_idx = min(query_idx, len(queries) - 1)
    selected_query = queries[query_idx]

    # Filter to this query
    query_df = df[df['query'] == selected_query].copy()

    # PTS v2 unified events
    if dataset_type in V2_TYPES:
        return create_event_trace(query_df, selected_query)

    # For pivotal tokens and steering vectors, use the token trace visualization
    if dataset_type in ('pivotal_tokens', 'steering_vectors'):
        return create_pivotal_token_trace(query_df, selected_query)

    # Sort by sentence_id if available, otherwise keep original order
    if 'sentence_id' in query_df.columns:
        query_df = query_df.sort_values('sentence_id')
    else:
        query_df = query_df.reset_index(drop=True)

    # Build HTML for step-by-step view
    html_parts = [f"""
    <div style="font-family: sans-serif; padding: 20px; background-color: #1a1a2e; border-radius: 10px;">
        <h3 style="color: #e0e0e0; border-bottom: 2px solid #6366f1; padding-bottom: 10px;">
            Query: {selected_query[:100]}{'...' if len(selected_query) > 100 else ''}
        </h3>
        <div style="display: flex; flex-direction: column; gap: 15px; margin-top: 20px;">
    """]

    prob_values = []
    sentence_ids = []

    for idx, row in query_df.iterrows():
        sentence = row.get('sentence', 'N/A')
        sentence_id = row.get('sentence_id', idx)
        is_positive = row.get('is_positive', row.get('prob_delta', 0) > 0)
        prob_delta = row.get('prob_delta', 0)
        category = row.get('sentence_category', 'unknown')
        importance = row.get('importance_score', abs(prob_delta))

        # Verification info
        verification_score = row.get('verification_score', None)
        arithmetic_errors = row.get('arithmetic_errors', [])

        # Color based on impact
        bg_color = "rgba(34, 197, 94, 0.2)" if is_positive else "rgba(239, 68, 68, 0.2)"
        border_color = "#22c55e" if is_positive else "#ef4444"

        # Build step card
        step_html = f"""
        <div style="background-color: {bg_color}; border-left: 4px solid {border_color};
                    padding: 15px; border-radius: 5px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #a0a0a0; font-size: 0.9em;">Step {sentence_id} | {category}</span>
                <span style="color: {border_color}; font-weight: bold;">
                    {'+'if prob_delta > 0 else ''}{prob_delta:.3f}
                </span>
            </div>
            <p style="color: #e0e0e0; margin: 10px 0;">{sentence}</p>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span style="background-color: #333; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; color: #a0a0a0;">
                    Importance: {importance:.3f}
                </span>
        """

        if verification_score is not None:
            v_color = "#22c55e" if verification_score > 0.5 else "#ef4444"
            step_html += f"""
                <span style="background-color: #333; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; color: {v_color};">
                    Verification: {verification_score:.2f}
                </span>
            """

        if arithmetic_errors:
            step_html += """
                <span style="background-color: #7f1d1d; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; color: #fca5a5;">
                    Has Errors
                </span>
            """

        step_html += """
            </div>
        </div>
        """

        html_parts.append(step_html)
        prob_values.append(row.get('prob_with_sentence', 0.5))
        sentence_ids.append(sentence_id)

    html_parts.append("</div></div>")

    # Create probability progression chart
    fig = go.Figure()

    colors = ['#22c55e' if p > 0.5 else '#ef4444' for p in prob_values]

    fig.add_trace(go.Scatter(
        x=[int(s) if isinstance(s, (int, np.integer)) else s for s in sentence_ids],
        y=[float(p) for p in prob_values],
        mode='lines+markers',
        name='Success Probability',
        line=dict(color='#6366f1', width=2),
        marker=dict(size=10, color=colors)
    ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="50% threshold")

    fig.update_layout(
        title="Probability Progression Through Reasoning",
        xaxis_title="Sentence ID",
        yaxis_title="Success Probability",
        yaxis_range=[0, 1],
        template="plotly_dark",
        height=300
    )

    return "\n".join(html_parts), fig


def create_statistics_dashboard(df: pd.DataFrame) -> Tuple[str, go.Figure]:
    """Create statistics dashboard for the dataset."""
    if df.empty:
        return "No data available", go.Figure()

    dataset_type = detect_dataset_type(df)
    is_v2 = dataset_type in V2_TYPES

    # Build statistics
    stats = {
        "Total Items": len(df),
        "Dataset Type": dataset_type,
    }

    if is_v2:
        gran = df.apply(_granularity, axis=1)
        stats["Latent Events"] = int((gran == 'latent').sum())
        stats["Token Events"] = int((gran == 'token').sum())
        stats["Sentence Events"] = int((gran == 'sentence').sum())

        link_count = 0
        for col in ('linked_event_ids', 'precedes_event_ids'):
            if col in df.columns:
                link_count += int(sum(len(_as_list(v)) for v in df[col]))
        stats["Event Links"] = link_count

        if 'query' in df.columns:
            stats["Queries"] = int(df['query'].nunique())
        if 'category' in df.columns:
            stats["Categories"] = int(df['category'].dropna().nunique())
        if 'layer' in df.columns and df['layer'].notna().any():
            stats["Layer Range"] = (
                f"{int(df['layer'].min())}-{int(df['layer'].max())}"
            )
        if 'readout_method' in df.columns and df['readout_method'].notna().any():
            methods = sorted({str(m) for m in df['readout_method'].dropna().unique()})
            stats["Readout"] = ", ".join(methods)

    if 'is_positive' in df.columns:
        # Latent events are null here by design; count them separately rather
        # than folding them into "negative".
        positive_count = int(
            df['is_positive'].map(lambda v: bool(v) if v is not None and v is not pd.NA and v == v else False).sum()
        )
        unscored = int(df['is_positive'].isna().sum())
        stats["Positive Items"] = positive_count
        stats["Negative Items"] = int(len(df) - positive_count - unscored)
        if unscored:
            stats["Unscored (latent)"] = unscored

    if 'prob_delta' in df.columns and df['prob_delta'].notna().any():
        stats["Avg Prob Delta"] = f"{df['prob_delta'].mean():.3f}"
        stats["Max Prob Delta"] = f"{df['prob_delta'].max():.3f}"

    if 'importance_score' in df.columns:
        stats["Avg Importance"] = f"{df['importance_score'].mean():.3f}"

    if 'sentence_category' in df.columns:
        category_counts = df['sentence_category'].value_counts()
        stats["Categories"] = len(category_counts)

    if 'model_id' in df.columns:
        stats["Models"] = df['model_id'].nunique()

    # Build HTML
    html_parts = ['<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">']

    for key, value in stats.items():
        html_parts.append(f"""
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
                    padding: 20px; border-radius: 10px; text-align: center;">
            <div style="color: #6366f1; font-size: 1.5em; font-weight: bold;">{value}</div>
            <div style="color: #a0a0a0; font-size: 0.9em; margin-top: 5px;">{key}</div>
        </div>
        """)

    html_parts.append('</div>')

    # v2 gets its own panel set: emitted deltas and latent readout scores are
    # different quantities and are never binned into the same histogram.
    if is_v2:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Δ probability (emitted events)",
                "Events by type",
                "Readout score (latent events)",
            ),
        )

        gran = df.apply(_granularity, axis=1)
        emitted = df[gran != 'latent']
        latent = df[gran == 'latent']

        if 'prob_delta' in emitted.columns and emitted['prob_delta'].notna().any():
            data = emitted['prob_delta'].dropna().values
            counts, bin_edges = np.histogram(data, bins=30)
            centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
            fig.add_trace(
                go.Bar(x=centers, y=counts.tolist(), name="Δ probability",
                       marker_color=COLOR_UNKNOWN,
                       width=(bin_edges[1] - bin_edges[0]) * 0.9),
                row=1, col=1,
            )

        if 'event_type' in df.columns:
            type_counts = df['event_type'].value_counts()
            colors = [
                COLOR_LATENT if t == EVENT_LATENT else COLOR_POSITIVE
                for t in type_counts.index.tolist()
            ]
            fig.add_trace(
                go.Bar(x=type_counts.index.tolist(), y=type_counts.values.tolist(),
                       name="Events", marker_color=colors),
                row=1, col=2,
            )

        if not latent.empty and 'score' in latent.columns and latent['score'].notna().any():
            data = latent['score'].dropna().values
            counts, bin_edges = np.histogram(data, bins=20)
            centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
            fig.add_trace(
                go.Bar(x=centers, y=counts.tolist(), name="Readout score",
                       marker_color=COLOR_LATENT,
                       width=(bin_edges[1] - bin_edges[0]) * 0.9),
                row=1, col=3,
            )
        else:
            fig.add_annotation(
                text="no latent events",
                xref="x3 domain", yref="y3 domain", x=0.5, y=0.5,
                showarrow=False, font=dict(size=11, color="#6b7280"),
                row=1, col=3,
            )

        fig.update_xaxes(title_text="Δ probability", row=1, col=1)
        fig.update_xaxes(title_text="Readout score (not a Δ probability)", row=1, col=3)
        fig.update_layout(template="plotly_dark", height=380, showlegend=False)
        return "\n".join(html_parts), fig

    # Determine what to show in second chart
    second_chart_title = "Category Distribution"
    if 'sentence_category' in df.columns:
        second_chart_title = "Sentence Category"
    elif 'reasoning_pattern' in df.columns:
        second_chart_title = "Reasoning Pattern"
    elif 'task_type' in df.columns:
        second_chart_title = "Task Type"
    elif 'is_positive' in df.columns:
        second_chart_title = "Positive vs Negative"

    # Create distribution charts
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Probability Delta Distribution", second_chart_title))

    # First chart: Probability Delta histogram (using numpy for binning)
    if 'prob_delta' in df.columns and len(df['prob_delta'].dropna()) > 0:
        prob_data = df['prob_delta'].dropna().values
        # Create histogram manually using numpy
        counts, bin_edges = np.histogram(prob_data, bins=30)
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
        fig.add_trace(
            go.Bar(x=bin_centers, y=counts.tolist(), name="Prob Delta",
                   marker_color='#6366f1', width=(bin_edges[1]-bin_edges[0])*0.9),
            row=1, col=1
        )
    elif 'prob_after' in df.columns and len(df['prob_after'].dropna()) > 0:
        # Fallback: show prob_after distribution
        prob_data = df['prob_after'].dropna().values
        counts, bin_edges = np.histogram(prob_data, bins=30)
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
        fig.add_trace(
            go.Bar(x=bin_centers, y=counts.tolist(), name="Prob After",
                   marker_color='#6366f1', width=(bin_edges[1]-bin_edges[0])*0.9),
            row=1, col=1
        )

    # Second chart: Categories, patterns, or task types
    if 'sentence_category' in df.columns:
        category_counts = df['sentence_category'].value_counts()
        fig.add_trace(
            go.Bar(x=category_counts.index.tolist(), y=category_counts.values.tolist(), name="Categories",
                  marker_color='#22c55e'),
            row=1, col=2
        )
    elif 'reasoning_pattern' in df.columns:
        pattern_counts = df['reasoning_pattern'].value_counts()
        fig.add_trace(
            go.Bar(x=pattern_counts.index.tolist(), y=pattern_counts.values.tolist(), name="Patterns",
                  marker_color='#22c55e'),
            row=1, col=2
        )
    elif 'task_type' in df.columns:
        task_counts = df['task_type'].value_counts()
        fig.add_trace(
            go.Bar(x=task_counts.index.tolist(), y=task_counts.values.tolist(), name="Task Types",
                  marker_color='#22c55e'),
            row=1, col=2
        )
    elif 'is_positive' in df.columns:
        pos_neg_counts = df['is_positive'].value_counts()
        labels = ['Positive' if v else 'Negative' for v in pos_neg_counts.index.tolist()]
        fig.add_trace(
            go.Bar(x=labels, y=pos_neg_counts.values.tolist(), name="Impact",
                  marker_color=['#22c55e' if l == 'Positive' else '#ef4444' for l in labels]),
            row=1, col=2
        )

    fig.update_layout(
        template="plotly_dark",
        height=350,
        showlegend=False
    )

    return "\n".join(html_parts), fig


# ============================================================================
# Gradio Interface
# ============================================================================

# Global state for loaded data. "filtered" holds the Event Explorer's current
# selection so the detail slider indexes into what the user is actually looking at.
current_data = {"df": pd.DataFrame(), "type": "unknown", "filtered": pd.DataFrame()}

DPO_NOTICE_HTML = """
<div style="padding: 40px; text-align: center; background-color: #1a1a2e; border-radius: 10px;">
    <h3 style="color: #f59e0b;">DPO Pairs Dataset</h3>
    <p style="color: #a0a0a0;">This visualization is not available for DPO pairs datasets.</p>
    <p style="color: #a0a0a0;">DPO pairs contain prompt/chosen/rejected structure without token-level context.</p>
    <p style="color: #6366f1; margin-top: 20px;">
        Try loading a <strong>causal_events</strong>, <strong>pivotal_tokens</strong> or
        <strong>thought_anchors</strong> dataset instead.
    </p>
</div>
"""


def _query_choices(df: pd.DataFrame) -> List[str]:
    """Truncated dropdown labels for each unique query."""
    if df.empty or 'query' not in df.columns:
        return []
    choices = []
    for i, q in enumerate(df['query'].unique().tolist()):
        q_str = str(q) if q is not None else ""
        if len(q_str) > 80:
            choices.append(f"[{i+1}] {q_str[:77]}...")
        else:
            choices.append(f"[{i+1}] {q_str}")
    return choices


def load_dataset_action(source_type: str, dataset_id: str, file_upload):
    """Handle dataset loading and return all visualization updates."""
    global current_data

    def blank(message: str):
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark")
        return (message, "", "No data", empty_fig, empty_fig, empty_fig,
                "No data", empty_fig, empty_fig, empty_fig,
                gr.update(maximum=0, value=0),
                gr.update(choices=[], value=None),
                gr.update(choices=["All"], value="All"),
                gr.update(choices=["All"], value="All"),
                "No data", empty_fig, "No data", empty_fig)

    if source_type == "HuggingFace Hub":
        if not dataset_id:
            return blank("Please enter a dataset ID")
        df, msg = load_hf_dataset(dataset_id)
    else:  # Local File
        if file_upload is None:
            return blank("Please upload a file")
        df, msg = load_jsonl_file(file_upload.name)

    if df.empty:
        return blank(msg)

    current_data["df"] = df
    current_data["type"] = detect_dataset_type(df)
    current_data["filtered"] = df

    columns_info = f"Columns: {', '.join(df.columns[:10])}"
    if len(df.columns) > 10:
        columns_info += f" ... and {len(df.columns) - 10} more"

    # Generate all visualizations
    stats_html, stats_fig = create_statistics_dashboard(df)
    graph_fig = create_causal_event_graph(df)
    embed_fig = create_embedding_visualization(df)
    circuit_html, circuit_fig = create_circuit_visualization(df)

    first_query = df['query'].iloc[0] if 'query' in df.columns and len(df) else None
    timeline_fig = create_multiscale_timeline(df, first_query)
    heatmap_fig = create_workspace_heatmap(df, first_query)

    # Event Explorer filter choices come from the data itself.
    type_choices = ["All"]
    if 'event_type' in df.columns:
        type_choices += sorted({str(v) for v in df['event_type'].dropna().unique()})

    category_choices = ["All"]
    for col in ('category', 'sentence_category'):
        if col in df.columns:
            category_choices += sorted({str(v) for v in df[col].dropna().unique()})
            break

    ev_summary, ev_fig = describe_event_selection(df)
    detail_html, detail_fig = get_event_details(0)

    return (msg, f"Dataset type: {current_data['type']}\n{columns_info}",
            stats_html, stats_fig, graph_fig, embed_fig, circuit_html, circuit_fig,
            timeline_fig, heatmap_fig,
            gr.update(maximum=max(0, len(df) - 1), value=0),
            gr.update(choices=_query_choices(df), value=None),
            gr.update(choices=type_choices, value="All"),
            gr.update(choices=category_choices, value="All"),
            ev_summary, ev_fig, detail_html, detail_fig)


def create_readout_score_chart(row) -> go.Figure:
    """Detail chart for a latent event.

    Deliberately not a before/after probability chart: a latent event has no
    prob_before or prob_after. Showing one would assert a causal effect that was
    never measured.
    """
    score = _num(_val(row, 'score'))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Readout score'],
        y=[score],
        marker_color=COLOR_LATENT,
        text=[f"{score:.4f}"],
        textposition='outside',
    ))
    fig.update_layout(
        title=(
            f"Latent readout score (layer {_val(row, 'layer', 'n/a')}, "
            f"{_val(row, 'readout_method', 'n/a')}) - not a probability delta"
        ),
        yaxis_title="Readout score",
        yaxis_range=[0, max(1.0, score * 1.2)],
        template="plotly_dark",
        height=300,
    )
    return fig


def create_latent_detail_html(row) -> str:
    """HTML card for a latent meta-token event."""
    context = html_lib.escape(str(_val(row, 'context', '')))
    label = html_lib.escape(str(_val(row, 'label', '')))
    score = _num(_val(row, 'score'))

    return f"""
    <div style="background-color: #1a1a2e; border-radius: 10px; padding: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="color: #a0a0a0; font-size: 0.9em;">
                Latent meta-token | layer {_val(row, 'layer', 'n/a')} |
                readout {_val(row, 'readout_method', 'n/a')} |
                offset {_val(row, 'position', 'n/a')} |
                category {html_lib.escape(str(_val(row, 'category', 'unknown')))}
            </span>
            <span style="background-color: {COLOR_LATENT}; color: white; padding: 4px 12px; border-radius: 5px; font-weight: bold;">
                Readout score: {score:.4f}
            </span>
        </div>
        <div style="font-family: monospace; padding: 15px; background-color: #0d1117; border-radius: 8px; color: #e0e0e0; line-height: 1.8; max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; border: 1px solid #30363d;">
            <span style="color: #8b949e;">{context}</span><span style="background-color: {COLOR_LATENT}; padding: 2px 6px; border-radius: 3px; border: 2px solid #d8b4fe; font-weight: bold;">{label}</span>
        </div>
        <p style="color: #d8b4fe; margin-top: 15px; font-size: 0.9em;">
            This event is <strong>observational</strong>: it was read out of the model's residual
            stream, not intervened on. Its score is a <strong>readout score</strong>, not a
            probability delta, and is not comparable to the scores on emitted token and
            sentence events.
        </p>
    </div>
    """


def get_event_details(idx: int) -> Tuple[str, go.Figure]:
    """Detail view for one event from the current Event Explorer selection."""
    df = current_data.get("filtered")
    if df is None or df.empty:
        df = current_data["df"]
    dataset_type = current_data.get("type", "unknown")

    if df is None or df.empty:
        return "No data available. Please load a dataset first.", go.Figure()

    if dataset_type == 'dpo_pairs':
        return DPO_NOTICE_HTML, go.Figure()

    try:
        idx = int(idx)
    except (TypeError, ValueError):
        idx = 0

    if idx >= len(df) or idx < 0:
        return "Index out of range", go.Figure()

    row = df.iloc[idx]

    # v2 latent events get their own card: no before/after probability exists.
    if is_v2_events(df) and _granularity(row) == 'latent':
        return create_latent_detail_html(row), create_readout_score_chart(row)

    # v1 pivot_context/prefix_context, v2 context. Same for the label.
    context = _val(row, 'pivot_context', _val(row, 'prefix_context', _val(row, 'context', '')))
    token = _val(row, 'pivot_token', _val(row, 'sentence', _val(row, 'label', '')))
    prob_delta = _num(_val(row, 'prob_delta'))
    prob_before = _num(_val(row, 'prob_before', _val(row, 'prob_without_sentence', 0.5)), 0.5)
    prob_after = _num(_val(row, 'prob_after', _val(row, 'prob_with_sentence', 0.5)), 0.5)

    if not context and not token:
        html = """
        <div style="padding: 40px; text-align: center; background-color: #1a1a2e; border-radius: 10px;">
            <h3 style="color: #ef4444;">Missing Data</h3>
            <p style="color: #a0a0a0;">This dataset doesn't have the expected fields for token visualization.</p>
        </div>
        """
        return html, go.Figure()

    html = create_token_highlight_html(context, token, prob_delta)
    chart = create_probability_chart(prob_before, prob_after)

    return html, chart


# Kept for backwards compatibility with anything importing the old name.
get_token_details = get_event_details


def _row_score(row) -> float:
    """The score used by the Event Explorer's min-score filter.

    v2 events carry ``score`` directly. v1 records get |prob_delta| (or the
    thought-anchor importance score), which is the same quantity v2 stores.
    """
    score = _val(row, 'score')
    if score is not None:
        return _num(score)
    importance = _val(row, 'importance_score')
    if importance is not None:
        return _num(importance)
    return abs(_num(_val(row, 'prob_delta')))


def describe_event_selection(df: pd.DataFrame) -> Tuple[str, go.Figure]:
    """Summary cards + score distribution for the Event Explorer's selection."""
    if df is None or df.empty:
        return (
            '<div style="padding: 20px; color: #a0a0a0; background-color: #1a1a2e; '
            'border-radius: 10px;">No events match these filters.</div>',
            _empty_fig("No events match these filters", 320),
        )

    v2 = is_v2_events(df)
    if v2:
        gran = df.apply(_granularity, axis=1)
        counts = {
            'latent': int((gran == 'latent').sum()),
            'token': int((gran == 'token').sum()),
            'sentence': int((gran == 'sentence').sum()),
        }
    else:
        gran = pd.Series(['token'] * len(df), index=df.index)
        counts = {'latent': 0, 'token': len(df), 'sentence': 0}

    cards = [
        ("Events", len(df)),
        ("Latent", counts['latent']),
        ("Token", counts['token']),
        ("Sentence", counts['sentence']),
    ]
    html_parts = ['<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px;">']
    for name, value in cards:
        html_parts.append(f"""
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
                    padding: 15px; border-radius: 10px; text-align: center;">
            <div style="color: #6366f1; font-size: 1.4em; font-weight: bold;">{value}</div>
            <div style="color: #a0a0a0; font-size: 0.85em; margin-top: 4px;">{name}</div>
        </div>
        """)
    html_parts.append('</div>')

    fig = go.Figure()
    emitted_scores = [
        _row_score(r) for _, r in df[gran != 'latent'].iterrows()
    ] if (gran != 'latent').any() else []
    latent_scores = [
        _num(_val(r, 'score')) for _, r in df[gran == 'latent'].iterrows()
    ] if (gran == 'latent').any() else []

    # Two separate traces, never one merged histogram: |Δ probability| and
    # readout score are different quantities on different scales.
    if emitted_scores:
        fig.add_trace(go.Histogram(
            x=emitted_scores, name="Emitted |Δ probability|",
            marker_color=COLOR_POSITIVE, opacity=0.7, nbinsx=30,
        ))
    if latent_scores:
        fig.add_trace(go.Histogram(
            x=latent_scores, name="Latent readout score",
            marker_color=COLOR_LATENT, opacity=0.7, nbinsx=30,
        ))

    fig.update_layout(
        title="Score distribution of the current selection",
        xaxis_title="Score (|Δ probability| for emitted events, readout score for latent events)",
        yaxis_title="Events",
        barmode='overlay',
        template="plotly_dark",
        height=320,
    )
    return "\n".join(html_parts), fig


def apply_event_filters(event_type: str, granularity: str, polarity: str,
                        category: str, min_score: float,
                        layer_min: float, layer_max: float):
    """Filter the loaded events and refresh the Event Explorer."""
    df = current_data["df"]
    if df is None or df.empty:
        current_data["filtered"] = pd.DataFrame()
        summary, fig = describe_event_selection(pd.DataFrame())
        return (summary, fig, gr.update(maximum=0, value=0),
                "No data available. Please load a dataset first.", go.Figure())

    if current_data.get("type") == 'dpo_pairs':
        current_data["filtered"] = df
        summary, fig = describe_event_selection(pd.DataFrame())
        return (summary, fig, gr.update(maximum=0, value=0), DPO_NOTICE_HTML, go.Figure())

    work = df.copy()
    v2 = is_v2_events(work)

    if v2 and event_type and event_type != "All" and 'event_type' in work.columns:
        work = work[work['event_type'].astype(str) == event_type]

    if v2 and granularity and granularity != "All" and not work.empty:
        work = work[work.apply(_granularity, axis=1) == granularity]

    if polarity and polarity != "All" and not work.empty:
        def valence(row):
            """True / False / None -- latent events are always None."""
            if _granularity(row) == 'latent':
                return None
            is_positive = _val(row, 'is_positive')
            if is_positive is not None:
                return bool(is_positive)
            delta = _val(row, 'prob_delta')
            if delta is None:
                return None
            return _num(delta) > 0

        positives = work.apply(valence, axis=1)
        if polarity == "Positive":
            work = work[positives.apply(lambda v: v is True)]
        elif polarity == "Negative":
            work = work[positives.apply(lambda v: v is False)]
        elif polarity.startswith("Unscored"):
            work = work[positives.apply(lambda v: v is None)]

    if category and category != "All" and not work.empty:
        cat_col = 'category' if 'category' in work.columns else (
            'sentence_category' if 'sentence_category' in work.columns else None
        )
        if cat_col:
            work = work[work[cat_col].astype(str) == category]

    if not work.empty and min_score:
        scores = work.apply(_row_score, axis=1)
        work = work[scores >= float(min_score)]

    # Layer range only constrains latent events; emitted events have no layer.
    if not work.empty and 'layer' in work.columns:
        lo, hi = float(layer_min), float(layer_max)
        if lo > hi:
            lo, hi = hi, lo

        def layer_ok(row) -> bool:
            if _granularity(row) != 'latent':
                return True
            layer = _val(row, 'layer')
            if layer is None:
                return True
            return lo <= _num(layer) <= hi

        work = work[work.apply(layer_ok, axis=1)]

    work = work.reset_index(drop=True)
    current_data["filtered"] = work

    summary, fig = describe_event_selection(work)
    detail_html, detail_fig = get_event_details(0)

    return (summary, fig,
            gr.update(maximum=max(0, len(work) - 1), value=0),
            detail_html, detail_fig)


def get_original_query_from_label(label: str) -> str:
    """Extract original query from truncated dropdown label like '[1] query...'"""
    if not label or not isinstance(label, str):
        return None

    df = current_data["df"]
    if df.empty or 'query' not in df.columns:
        return None

    # Extract index from "[N] query..." format
    match = re.match(r'\[(\d+)\]', label)
    if match:
        idx = int(match.group(1)) - 1  # Convert to 0-based index
        queries = df['query'].unique().tolist()
        if 0 <= idx < len(queries):
            return queries[idx]

    return None


def update_graph_visualization(query_dropdown: str = None):
    """Update the causal event graph (falls back to the v1 reasoning graph)."""
    dataset_type = current_data.get("type", "unknown")
    if dataset_type == 'dpo_pairs':
        return _empty_fig(
            "Causal Event Graph is not available for DPO pairs datasets.<br>"
            "Load a causal_events, pivotal_tokens or thought_anchors dataset.",
        )

    # Convert truncated label back to original query
    original_query = get_original_query_from_label(query_dropdown)
    return create_causal_event_graph(current_data["df"], original_query)


def update_embedding_visualization(color_by: str):
    """Update the embedding visualization."""
    dataset_type = current_data.get("type", "unknown")
    if dataset_type == 'dpo_pairs':
        return _empty_fig(
            "Embedding Space is not available for DPO pairs datasets.<br>"
            "Load a pivotal_tokens, thought_anchors, or steering_vectors dataset.",
        )
    return create_embedding_visualization(current_data["df"], color_by)


def _query_at_index(df: pd.DataFrame, query_idx: int) -> Optional[str]:
    if df.empty or 'query' not in df.columns:
        return None
    queries = df['query'].unique().tolist()
    if not queries:
        return None
    idx = max(0, min(int(query_idx), len(queries) - 1))
    return queries[idx]


def update_circuit_view(query_idx: int):
    """Update the Multiscale Reasoning Timeline tab (timeline + heatmap + trace)."""
    dataset_type = current_data.get("type", "unknown")
    df = current_data["df"]

    if dataset_type == 'dpo_pairs':
        html = """
        <div style="padding: 40px; text-align: center; background-color: #1a1a2e; border-radius: 10px;">
            <h3 style="color: #f59e0b;">DPO Pairs Dataset</h3>
            <p style="color: #a0a0a0;">The reasoning timeline is not available for DPO pairs datasets.</p>
            <p style="color: #6366f1; margin-top: 20px;">
                Load a <strong>causal_events</strong>, <strong>pivotal_tokens</strong> or
                <strong>thought_anchors</strong> dataset to explore reasoning circuits.
            </p>
        </div>
        """
        empty = _empty_fig("Not available for DPO pairs datasets")
        return html, go.Figure(), empty, empty

    try:
        query_idx = int(query_idx)
    except (TypeError, ValueError):
        query_idx = 0

    selected_query = _query_at_index(df, query_idx)
    circuit_html, circuit_fig = create_circuit_visualization(df, query_idx)
    timeline_fig = create_multiscale_timeline(df, selected_query)
    heatmap_fig = create_workspace_heatmap(df, selected_query)

    return circuit_html, circuit_fig, timeline_fig, heatmap_fig


def update_statistics():
    """Update the statistics dashboard."""
    return create_statistics_dashboard(current_data["df"])


def get_query_list():
    """Get list of unique queries with truncated display labels."""
    df = current_data["df"]
    if df.empty or 'query' not in df.columns:
        return gr.update(choices=[], value=None)
    return gr.update(choices=_query_choices(df), value=None)


def refresh_all():
    """Refresh all visualizations."""
    df = current_data["df"]
    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark")
        return (
            "No data loaded",
            empty_fig,
            empty_fig,
            empty_fig,
            "No data loaded",
            empty_fig,
            empty_fig,
            empty_fig,
        )

    stats_html, stats_fig = create_statistics_dashboard(df)
    graph_fig = create_causal_event_graph(df)
    embed_fig = create_embedding_visualization(df)
    circuit_html, circuit_fig = create_circuit_visualization(df)
    first_query = _query_at_index(df, 0)
    timeline_fig = create_multiscale_timeline(df, first_query)
    heatmap_fig = create_workspace_heatmap(df, first_query)

    return (stats_html, stats_fig, graph_fig, embed_fig, circuit_html, circuit_fig,
            timeline_fig, heatmap_fig)


# ============================================================================
# Build Gradio App
# ============================================================================

# Pre-defined HuggingFace datasets.
# The v1 datasets below still load and render exactly as they did before; the
# v2 causal-event datasets use the unified CausalReasoningEvent schema.
# Only datasets that actually exist on the Hub. The visualizer reads v2
# causal-event files too -- upload one and add it here, or load it from disk with
# the file upload. Listing v2 datasets that have not been published yet would put
# entries in the dropdown that can only fail.
HF_DATASETS = [
    "codelion/Qwen3-0.6B-pts",
    "codelion/Qwen3-0.6B-pts-thought-anchors",
    "codelion/Qwen3-0.6B-pts-steering-vectors",
    "codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts",
    "codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts-thought-anchors",
    "codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts-steering-vectors",
]

# Default to a v1 dataset that is known to exist on the Hub.
DEFAULT_DATASET = "codelion/Qwen3-0.6B-pts"

# CSS configuration
CSS = """
.gradio-container { max-width: 1400px !important; }
.main-header { text-align: center; margin-bottom: 20px; }
"""

with gr.Blocks(title="PTS Visualizer", css=CSS) as demo:

    # Header
    gr.Markdown("""
    # PTS Visualizer
    ### Interactive Exploration of Latent Meta-Tokens, Pivotal Tokens & Thought Anchors

    A [Neuronpedia](https://neuronpedia.org/)-inspired platform for understanding how language models reason.
    Load datasets from HuggingFace Hub or upload your own JSONL files.

    Supports the PTS v2 unified event schema (`CausalReasoningEvent`: latent / token / sentence)
    as well as all v1 pivotal-token, thought-anchor and steering-vector datasets.

    🔗 [Browse more PTS datasets on HuggingFace](https://huggingface.co/datasets?other=pts)
    """)

    # Data Loading Section
    with gr.Accordion("Load Dataset", open=True):
        with gr.Row():
            source_type = gr.Radio(
                choices=["HuggingFace Hub", "Local File"],
                value="HuggingFace Hub",
                label="Data Source"
            )

        with gr.Row():
            with gr.Column(scale=3):
                dataset_dropdown = gr.Dropdown(
                    choices=HF_DATASETS,
                    value=DEFAULT_DATASET,
                    label="Select Dataset",
                    info="Choose a pre-defined dataset or enter your own HuggingFace dataset ID"
                )
            with gr.Column(scale=1):
                file_upload = gr.File(
                    label="Or Upload JSONL",
                    file_types=[".jsonl", ".json"]
                )

        with gr.Row():
            load_btn = gr.Button("Load Dataset", variant="primary")
            refresh_btn = gr.Button("Refresh Visualizations", variant="secondary")

        with gr.Row():
            load_status = gr.Textbox(label="Status", interactive=False)
            dataset_info = gr.Textbox(label="Dataset Info", interactive=False)

    # Main Visualization Tabs
    with gr.Tabs():

        # Overview Tab (unified event overview)
        with gr.TabItem("Overview"):
            gr.Markdown("### Dataset Statistics")
            gr.Markdown(
                "*For PTS v2 datasets this counts events at all three scales "
                "(latent / token / sentence) and the causal links between them. "
                "Emitted Δ-probabilities and latent readout scores are charted "
                "separately: they are different quantities and are not comparable.*"
            )
            stats_html = gr.HTML()
            stats_chart = gr.Plot()

        # Event Explorer Tab (generalization of the old Token Explorer)
        with gr.TabItem("Event Explorer"):
            gr.Markdown("### Explore Causal Reasoning Events")
            gr.Markdown(
                "*Filter events across all three scales. Latent meta-tokens are "
                "observational: they carry a **readout score**, not a probability delta, "
                "and are shown in purple rather than green/red.*"
            )
            with gr.Row():
                ev_type = gr.Dropdown(
                    choices=["All"], value="All", label="Event Type"
                )
                ev_granularity = gr.Dropdown(
                    choices=["All", "latent", "token", "sentence"],
                    value="All", label="Granularity"
                )
                ev_polarity = gr.Radio(
                    choices=["All", "Positive", "Negative", "Unscored (latent)"],
                    value="All", label="Impact"
                )
            with gr.Row():
                ev_category = gr.Dropdown(
                    choices=["All"], value="All", label="Category"
                )
                ev_min_score = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01, value=0.0,
                    label="Min Score (|Δ prob| for emitted, readout score for latent)"
                )
            with gr.Row():
                ev_layer_min = gr.Slider(
                    minimum=0, maximum=128, step=1, value=0,
                    label="Min Layer (latent events only)"
                )
                ev_layer_max = gr.Slider(
                    minimum=0, maximum=128, step=1, value=128,
                    label="Max Layer (latent events only)"
                )

            ev_summary = gr.HTML()
            ev_score_plot = gr.Plot(label="Score Distribution")

            gr.Markdown("#### Event Detail")
            with gr.Row():
                with gr.Column(scale=1):
                    token_slider = gr.Slider(
                        minimum=0, maximum=100, step=1, value=0,
                        label="Event Index (within current filter)"
                    )
                with gr.Column(scale=3):
                    token_html = gr.HTML(label="Event in Context")
            prob_chart = gr.Plot(label="Probability Change / Readout Score")

        # Causal Event Graph Tab
        with gr.TabItem("Causal Event Graph"):
            gr.Markdown("### Causal Event Graph")
            gr.Markdown("""
            *Latent meta-tokens (diamonds) → pivotal tokens (circles) → thought anchors
            (squares) → outcome (star). Edges come from the event links
            (`precedes_event_ids` / `linked_event_ids` / `parent_event_id`).
            Green = positive impact, red = negative impact, purple = latent (no measured valence).
            Node size reflects score. v1 datasets fall back to the original reasoning graph.*
            """)
            with gr.Row():
                query_filter = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Filter by Query"
                )
            graph_plot = gr.Plot()

        # Embedding Visualization Tab
        with gr.TabItem("Embedding Space"):
            gr.Markdown("### Embedding Space Visualization")
            gr.Markdown("*t-SNE projection of sentence/token embeddings. Explore clusters and patterns.*")
            with gr.Row():
                color_dropdown = gr.Dropdown(
                    choices=["is_positive", "event_type", "granularity", "category",
                             "sentence_category", "reasoning_pattern", "task_type"],
                    value="is_positive",
                    label="Color By"
                )
            embed_plot = gr.Plot()

        # Multiscale Reasoning Timeline Tab (was: Circuit Tracer)
        with gr.TabItem("Multiscale Reasoning Timeline"):
            gr.Markdown("### Multiscale Reasoning Timeline")
            gr.Markdown(
                "*One shared generation axis across four scales: latent meta-tokens, "
                "emitted pivotal tokens, thought-anchor sentences, and the resulting "
                "success probability. Latent events are placed by their offset from the "
                "emitted event they precede.*"
            )
            with gr.Row():
                circuit_query_idx = gr.Slider(
                    minimum=0, maximum=100, step=1, value=0,
                    label="Query Index"
                )
            timeline_plot = gr.Plot(label="Multiscale Timeline")

            gr.Markdown("#### Latent Workspace Heatmap")
            gr.Markdown(
                "*Meta-token (or category) x generation position, colored by **readout score** "
                "— how strongly the lens surfaces that concept. This is not a probability delta.*"
            )
            heatmap_plot = gr.Plot(label="Workspace Heatmap")

            gr.Markdown("#### Step-by-Step Reasoning Circuit")
            circuit_html = gr.HTML()
            circuit_chart = gr.Plot()

    # Event handlers - using api_name=False to prevent schema generation issues
    load_btn.click(
        fn=load_dataset_action,
        inputs=[source_type, dataset_dropdown, file_upload],
        outputs=[load_status, dataset_info, stats_html, stats_chart, graph_plot,
                 embed_plot, circuit_html, circuit_chart, timeline_plot, heatmap_plot,
                 token_slider, query_filter, ev_type, ev_category,
                 ev_summary, ev_score_plot, token_html, prob_chart],
        api_name=False
    )

    refresh_btn.click(
        fn=refresh_all,
        outputs=[stats_html, stats_chart, graph_plot, embed_plot, circuit_html,
                 circuit_chart, timeline_plot, heatmap_plot],
        api_name=False
    )

    token_slider.change(
        fn=get_event_details,
        inputs=[token_slider],
        outputs=[token_html, prob_chart],
        api_name=False
    )

    event_filter_inputs = [ev_type, ev_granularity, ev_polarity, ev_category,
                           ev_min_score, ev_layer_min, ev_layer_max]
    event_filter_outputs = [ev_summary, ev_score_plot, token_slider, token_html, prob_chart]

    for filter_component in event_filter_inputs:
        filter_component.change(
            fn=apply_event_filters,
            inputs=event_filter_inputs,
            outputs=event_filter_outputs,
            api_name=False
        )

    query_filter.change(
        fn=update_graph_visualization,
        inputs=[query_filter],
        outputs=[graph_plot],
        api_name=False
    )

    color_dropdown.change(
        fn=update_embedding_visualization,
        inputs=[color_dropdown],
        outputs=[embed_plot],
        api_name=False
    )

    circuit_query_idx.change(
        fn=update_circuit_view,
        inputs=[circuit_query_idx],
        outputs=[circuit_html, circuit_chart, timeline_plot, heatmap_plot],
        api_name=False
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    demo.launch()
