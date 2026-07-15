"""
PTS Visualizer - Interactive visualization for Pivotal Token Search

A Neuronpedia-inspired platform for exploring pivotal tokens, thought anchors,
and reasoning circuits in language models.
"""

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
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
    """Load a dataset from HuggingFace Hub.

    Reads the raw JSONL directly rather than going through
    ``datasets.load_dataset``. The events schema is deeply nested (link arrays
    with hundreds of entries, metadata dicts), and Arrow schema inference over
    that is slow -- ~15s for a 30 MB file vs ~2.5s for a direct download + parse.
    Falls back to the datasets library for anything not stored as a flat JSONL.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    try:
        files = list_repo_files(dataset_id, repo_type="dataset")
        jsonl = [f for f in files if f.endswith(".jsonl")]
        if jsonl:
            # Prefer the canonical event file when present.
            preferred = next((f for f in jsonl if "causal_events" in f), jsonl[0])
            path = hf_hub_download(dataset_id, preferred, repo_type="dataset")
            rows = [json.loads(l) for l in open(path) if l.strip()]
            return pd.DataFrame(rows), f"Loaded {len(rows)} items from {dataset_id}"

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

    PTS emits a single unified record type (``CausalReasoningEvent``) that
    carries ``event_type`` + ``granularity``. Those are checked first; the v1
    detection rules below them are unchanged so old datasets keep working.
    """
    columns = set(df.columns)

    # --- PTS unified event schema -------------------------------------
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
# PTS event schema: constants and helpers
# ============================================================================

EVENT_LATENT = "latent_metatoken"
EVENT_TOKEN = "pivotal_token"
EVENT_SENTENCE = "thought_anchor"

GRANULARITY_FOR_EVENT_TYPE = {
    EVENT_LATENT: "latent",
    EVENT_TOKEN: "token",
    EVENT_SENTENCE: "sentence",
}

PTS_TYPES = ('causal_events', 'latent_events')

# The design system lives at the bottom of this file (see "Design system"), but
# the palette is needed up here by the chart helpers. Two axes, kept strictly
# apart:
#
#   CHANNEL  which scale an event belongs to   (latent / token / sentence)
#   VALENCE  which way it moved success        (positive / negative)
#
# Latent events have a channel but NO valence -- prob_delta is null by
# construction -- so they must never be painted green or red. Violet means
# "observed in the workspace, direction unknown", and that is the whole point.
C_CANVAS      = '#0A0D13'
C_PANEL       = '#111722'
C_PANEL_HI    = '#161E2C'
C_LINE        = '#212C3D'
C_LINE_SOFT   = '#1A2333'
C_TEXT        = '#CBD5E4'
C_TEXT_MUTED  = '#A6B2C7'
C_TEXT_FAINT  = '#8592A8'

C_LATENT      = '#A78BFA'   # violet   -- hidden, deep
C_TOKEN       = '#38BDF8'   # sky      -- emitted, a single decision point
C_SENTENCE    = '#FBBF24'   # amber    -- emitted, an extended step
C_POSITIVE    = '#34D399'   # emerald
C_NEGATIVE    = '#FB7185'   # rose
C_NEUTRAL     = '#64748B'

COLOR_POSITIVE = C_POSITIVE
COLOR_NEGATIVE = C_NEGATIVE
COLOR_LATENT = C_LATENT
COLOR_UNKNOWN = C_NEUTRAL
COLOR_OUTCOME = C_SENTENCE

CHANNEL = {'latent': C_LATENT, 'token': C_TOKEN, 'sentence': C_SENTENCE}

CATEGORY_COLORS = [
    C_LATENT, C_TOKEN, C_SENTENCE, C_POSITIVE, C_NEGATIVE,
    '#F472B6', '#2DD4BF', '#FB923C', '#818CF8', '#A3E635',
]


def is_pts_events(df: pd.DataFrame) -> bool:
    """True when the dataframe holds PTS unified events."""
    return not df.empty and 'event_type' in df.columns


def as_pts_events(df: pd.DataFrame) -> pd.DataFrame:
    """Present a v1 dataframe as PTS events, in memory.

    This is the whole PTS thesis applied to the UI: a legacy pivotal-token file *is*
    a token-scale event stream, and a legacy thought-anchor file *is* a
    sentence-scale one. They differ only in what the columns are called.
    Upgrading them here means the reasoning views work on every published
    dataset instead of showing an empty box until someone re-exports.

    Returns the frame unchanged if it is already a PTS event frame, or neither shape
    (steering vectors, DPO pairs).
    """
    if df.empty or is_pts_events(df):
        return df

    out = df.copy()
    cols = set(out.columns)

    if 'pivot_token' in cols:
        out['event_type'] = 'pivotal_token'
        out['granularity'] = 'token'
        out['visibility'] = 'emitted'
        out['label'] = out['pivot_token']
        out['context'] = out.get('pivot_context', '')
        if 'pivot_token_id' in cols:
            out['token_id'] = out['pivot_token_id']
    elif {'sentence', 'sentence_id'} <= cols:
        out['event_type'] = 'thought_anchor'
        out['granularity'] = 'sentence'
        out['visibility'] = 'emitted'
        out['label'] = out['sentence']
        out['context'] = out.get('prefix_context', '')
        out['position'] = out['sentence_id']
        # legacy named these prob_with/prob_without; PTS calls them after/before.
        if 'prob_with_sentence' in cols:
            out['prob_after'] = out['prob_with_sentence']
        if 'prob_without_sentence' in cols:
            out['prob_before'] = out['prob_without_sentence']
        if 'sentence_category' in cols:
            out['category'] = out['sentence_category']
    else:
        return df

    if 'prob_delta' in out.columns:
        out['score'] = out['prob_delta'].abs()
        if 'is_positive' not in out.columns:
            out['is_positive'] = out['prob_delta'] > 0

    for c in ('layer', 'readout_method', 'linked_event_ids', 'precedes_event_ids',
              'follows_event_ids', 'parent_event_id', 'event_id'):
        if c not in out.columns:
            out[c] = None
    if out['event_id'].isna().all():
        out['event_id'] = [f'v1_evt_{i}' for i in range(len(out))]

    return out


def _empty_fig(message: str, height: int = 400) -> go.Figure:
    """A placeholder that reads as a deliberate state, not a broken chart."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(family="'IBM Plex Mono', monospace", size=12, color=C_TEXT_MUTED),
        align="center", bgcolor=C_PANEL, bordercolor=C_LINE, borderwidth=1,
        borderpad=14,
    )
    fig.update_layout(
        template="pts",
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


def _granularity_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized granularity for a whole frame.

    ``_granularity_series(df)`` is a row-wise Python loop -- seconds over
    14k events, and it ran on every dataset load. The data already has a
    ``granularity`` column, so this is a column read with an event_type fallback.
    """
    if 'granularity' in df.columns:
        g = df['granularity'].astype('object').where(df['granularity'].notna(), None)
    else:
        g = pd.Series([None] * len(df), index=df.index, dtype='object')
    if 'event_type' in df.columns:
        fallback = df['event_type'].map(lambda t: GRANULARITY_FOR_EVENT_TYPE.get(str(t), 'token'))
        g = g.where(g.notna(), fallback)
    return g.fillna('token').astype(str)


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


def _default_query(df: pd.DataFrame, selected_query: Optional[str]) -> Optional[str]:
    """Resolve a per-query view to one query when none is chosen.

    The causal graph and timeline are per-query by nature -- laying out every
    event from every query at once is both meaningless and, on an enriched
    dataset (14k events, 69k links), pathologically slow (a spring layout over
    the whole graph takes minutes). So a per-query view with no selection shows
    the first query, not the entire dataset.
    """
    if selected_query and isinstance(selected_query, str) and selected_query.strip():
        return selected_query
    if 'query' in df.columns and len(df):
        return str(df['query'].iloc[0])
    return selected_query


def _event_hover(row) -> str:
    """Hover text shared by the charts."""
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
        border_color = "#34D399"
        impact_text = "Positive Impact"
    else:
        # Negative impact - red gradient
        intensity = min(abs(prob_delta) * 2, 1.0)
        color = f"rgba(239, 68, 68, {intensity})"
        border_color = "#FB7185"
        impact_text = "Negative Impact"

    # Create highlighted token span
    token_span = f'<span style="background-color: {color}; padding: 2px 6px; border-radius: 3px; border: 2px solid {border_color}; font-weight: bold; font-size: 1.1em;">{token_escaped}</span>'

    return f"""
    <div style="background-color: #111722; border: 1px solid #212C3D; border-radius: 4px; padding: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="color: #71809A; font-size: 0.9em;">Context Length: {len(context)} characters</span>
            <span style="background-color: {border_color}; color: white; padding: 4px 12px; border-radius: 5px; font-weight: bold;">
                {impact_text}: {'+' if prob_delta > 0 else ''}{prob_delta:.3f}
            </span>
        </div>
        <div style="font-family: monospace; padding: 15px; background-color: #0d1117; border-radius: 8px; color: #CBD5E4; line-height: 1.8; max-height: 500px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; border: 1px solid #30363d;">
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
        marker_color=[C_NEUTRAL, C_POSITIVE if prob_after > prob_before else C_NEGATIVE],
        text=[f'{prob_before:.3f}', f'{prob_after:.3f}'],
        textposition='outside'
    ))

    fig.update_layout(
        title="Success Probability Change",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        template="pts",
        height=300
    )

    return fig


def create_pivotal_token_flow(df: pd.DataFrame, selected_query: str = None) -> go.Figure:
    """Create a visualization for pivotal tokens showing token impact flow."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="pts")
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
        fig.update_layout(template="pts")
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
                color=C_POSITIVE,
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
                color=C_NEGATIVE,
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
        template="pts",
        height=500,
        showlegend=True
    )

    return fig


def create_thought_anchor_graph(df: pd.DataFrame, selected_query: str = None) -> go.Figure:
    """Create an interactive graph visualization of thought anchor dependencies."""
    dataset_type = detect_dataset_type(df)

    # PTS events carry their own causal edges; use the unified graph.
    if dataset_type in PTS_TYPES:
        return create_causal_event_graph(df, selected_query)

    # For pivotal tokens and steering vectors, create a token impact visualization
    if dataset_type in ('pivotal_tokens', 'steering_vectors'):
        return create_pivotal_token_flow(df, selected_query)

    if df.empty or 'sentence_id' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No thought anchor data available. Load a thought anchors dataset to see the reasoning graph.",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                          font=dict(size=14, color="#a0a0a0"))
        fig.update_layout(template="pts", height=400)
        return fig

    # Filter by query if specified (handle None, empty string, or actual query)
    if selected_query and isinstance(selected_query, str) and selected_query.strip():
        df = df[df['query'] == selected_query].copy()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data for selected query",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="pts")
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
        line=dict(width=1, color=C_LINE),
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

        node_colors.append(C_POSITIVE if is_positive else C_NEGATIVE)
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
        template="pts",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )

    return fig


# ============================================================================
# PTS: reasoning timeline, causal event graph, workspace heatmap
# ============================================================================

def create_reasoning_timeline(df: pd.DataFrame, selected_query: str = None) -> go.Figure:
    """Four scales of reasoning events on one shared generation axis.

    Row 1  latent meta-tokens        (diamonds, y = layer, size = readout score)
    Row 2  emitted pivotal tokens    (circles, y = Δ probability)
    Row 3  thought-anchor sentences  (wide bars, height = Δ probability)
    Row 4  success probability       (from prob_before/prob_after of emitted events)
    """
    if df is None or df.empty:
        return _empty_fig("No data loaded. Load a PTS dataset to see the reasoning timeline.", 640)

    # A legacy pivotal-token file IS a token-scale event stream, and a v1
    # thought-anchor file IS a sentence-scale one -- that is the whole PTS
    # thesis. So upgrade them here rather than refusing, and every published
    # dataset renders instead of showing an empty box. Latent rows will simply
    # be absent until the file is enriched.
    df = as_pts_events(df)
    if not is_pts_events(df):
        return _empty_fig(
            "This view needs reasoning events.<br>"
            "Steering-vector and DPO files contain none.",
            640,
        )

    work = _filter_by_query(df, selected_query)
    if work.empty:
        return _empty_fig("No events for the selected query.", 640)

    work = work.reset_index(drop=True)
    xs = _compute_event_x(work)
    work = work.assign(_x=xs)

    gran = _granularity_series(work)
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
                    line=dict(width=1, color=C_LATENT),
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
                    line=dict(width=1, color=C_CANVAS),
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
                    line=dict(width=1, color=C_CANVAS),
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
        title="Reasoning Timeline",
        template="pts",
        height=760,
        showlegend=True,
        hovermode='closest',
        bargap=0.2,
    )

    return fig


def create_causal_event_graph(df: pd.DataFrame, selected_query: str = None) -> go.Figure:
    """Causal graph over unified events: latent -> token -> sentence -> outcome.

    Edges come from ``precedes_event_ids`` / ``linked_event_ids`` /
    ``parent_event_id``. Node shape encodes granularity, node color encodes
    valence (latent nodes stay purple - they have no valence).
    """
    if df is None or df.empty:
        return _empty_fig("No data loaded. Load a PTS dataset to see the causal event graph.", 550)

    # legacy datasets keep the old graph exactly as it was.
    if not is_pts_events(df):
        return create_thought_anchor_graph(df, selected_query)

    # Per-query view: never lay out the whole dataset (minutes on 14k nodes).
    selected_query = _default_query(df, selected_query)
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
        # Semi-transparent so a dense single-query graph reads as a web rather
        # than a solid mass, but light enough to actually be visible against the
        # near-black canvas -- C_LINE (#212C3D) is the panel-border colour and
        # was indistinguishable from the background.
        line=dict(width=1, color='rgba(148,163,190,0.45)'),
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
        template="pts",
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

    df = as_pts_events(df)
    if not is_pts_events(df):
        return _empty_fig(
            "The workspace heatmap needs latent meta-token events (PTS).<br>"
            "Generate them with <code>pts run --scale=latent</code> or "
            "<code>pts enrich</code>.",
            500,
        )

    work = _filter_by_query(df, selected_query)
    latent = work[_granularity_series(work) == 'latent'] if not work.empty else work

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
        template="pts",
        height=max(400, 60 + 22 * len(row_values)),
    )
    return fig


def create_event_trace(df: pd.DataFrame, selected_query: str) -> Tuple[str, go.Figure]:
    """Step-by-step HTML cards + probability progression for PTS events."""
    if df is None or df.empty:
        return "No events found for this query", _empty_fig("No events", 300)

    work = df.reset_index(drop=True)
    work = work.assign(_x=_compute_event_x(work)).sort_values('_x')

    query_text = str(selected_query or "")
    html_parts = [f"""
    <div style="font-family: sans-serif; padding: 20px; background-color: #111722; border: 1px solid #212C3D; border-radius: 4px;">
        <h3 style="color: #CBD5E4; border-bottom: 2px solid #A78BFA; padding-bottom: 10px;">
            Query: {html_lib.escape(query_text[:100])}{'...' if len(query_text) > 100 else ''}
        </h3>
        <p style="color: #71809A; margin: 10px 0;">{len(work)} causal reasoning events for this query</p>
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
                f'<span style="background-color: #161E2C; padding: 3px 8px; border-radius: 3px; '
                f'font-size: 0.8em; color: #71809A;">Layer {_val(row, "layer", "n/a")}</span>'
                f'<span style="background-color: #161E2C; padding: 3px 8px; border-radius: 3px; '
                f'font-size: 0.8em; color: #71809A;">{_val(row, "readout_method", "n/a")}</span>'
                f'<span style="background-color: #3b2f5e; padding: 3px 8px; border-radius: 3px; '
                f'font-size: 0.8em; color: #A78BFA;">observational - no probability delta</span>'
            )
        else:
            delta = _num(_val(row, 'prob_delta'))
            metric_html = (
                f'<span style="color: {color}; font-weight: bold;">'
                f'{"+" if delta > 0 else ""}{delta:.3f} Δ probability</span>'
            )
            extra = (
                f'<span style="background-color: #161E2C; padding: 3px 8px; border-radius: 3px; '
                f'font-size: 0.8em; color: #71809A;">Before: {_num(_val(row, "prob_before")):.3f}</span>'
                f'<span style="background-color: #161E2C; padding: 3px 8px; border-radius: 3px; '
                f'font-size: 0.8em; color: #71809A;">After: {_num(_val(row, "prob_after")):.3f}</span>'
            )
            before, after = _val(row, 'prob_before'), _val(row, 'prob_after')
            if before is not None and after is not None:
                curve_x.extend([x, x])
                curve_y.extend([_num(before), _num(after)])

        html_parts.append(f"""
        <div style="background-color: rgba(255,255,255,0.03); border-left: 4px solid {color};
                    padding: 15px; border-radius: 5px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #71809A; font-size: 0.9em;">
                    {html_lib.escape(str(_val(row, 'event_type', 'event')))} | {gran} | pos {x:g} | {category}
                </span>
                {metric_html}
            </div>
            <p style="color: #CBD5E4; margin: 10px 0; font-family: monospace; white-space: pre-wrap; word-break: break-word;">{label}</p>
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
            template="pts",
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
        '#A78BFA', '#34D399', '#FB7185', '#FBBF24', '#A78BFA',
        '#F472B6', '#2DD4BF', '#FB923C', '#38BDF8', '#A3E635'
    ]

    # Determine color column
    use_colorscale = False
    if color_by in df.columns:
        color_col = df[color_by]
        if color_by == 'is_positive':
            colors = ['#34D399' if v else '#FB7185' for v in color_col]
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
                colors = ['#A78BFA'] * len(df)
    else:
        colors = ['#A78BFA'] * len(df)

    # Create hover text
    hover_texts = []
    for _, row in df.iterrows():
        # legacy calls it pivot_token, PTS calls it label.
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
        template="pts",
        height=500
    )

    # Add annotations
    fig.add_annotation(
        x=0.2, y=0.8,
        text="Positive Impact ↑",
        showarrow=False,
        font=dict(color="#34D399", size=12)
    )
    fig.add_annotation(
        x=0.8, y=0.2,
        text="Negative Impact ↓",
        showarrow=False,
        font=dict(color="#FB7185", size=12)
    )

    return fig


def create_embedding_visualization(df: pd.DataFrame, color_by: str = 'is_positive') -> go.Figure:
    """Create UMAP/t-SNE visualization of embeddings or alternative visualization for pivotal tokens."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="pts")
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

        # unified events: only emitted events live in probability space. Latent
        # events have no prob_before/prob_after and must not be plotted there.
        if dataset_type in PTS_TYPES and 'prob_before' in df.columns and 'prob_after' in df.columns:
            emitted = df[_granularity_series(df) != 'latent']
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
        fig.update_layout(template="pts", height=400)
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
        fig.update_layout(template="pts")
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

    # Determine text field for hover (legacy: sentence/pivot_token; PTS: label)
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
                            color='#34D399' if is_pos else '#FB7185',
                            opacity=0.7
                        ),
                        hovertext=hover_texts,
                        hoverinfo='text'
                    ))
        else:
            # Categorical coloring
            unique_vals = plot_df[color_by].unique()
            colors = ['#A78BFA', '#34D399', '#FB7185', '#FBBF24', '#A78BFA',
                      '#F472B6', '#2DD4BF', '#FB923C', '#38BDF8', '#A3E635']
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
                color='#A78BFA',
                opacity=0.7
            ),
            hovertext=hover_texts,
            hoverinfo='text'
        ))

    fig.update_layout(
        title="Embedding Space Visualization (t-SNE)",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        template="pts",
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
    <div style="font-family: sans-serif; padding: 20px; background-color: #111722; border: 1px solid #212C3D; border-radius: 4px;">
        <h3 style="color: #CBD5E4; border-bottom: 2px solid #A78BFA; padding-bottom: 10px;">
            Query: {selected_query[:100]}{'...' if len(selected_query) > 100 else ''}
        </h3>
        <p style="color: #71809A; margin: 10px 0;">Found {len(df)} pivotal tokens for this query</p>
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
        border_color = "#34D399" if is_positive else "#FB7185"

        # Show full context in a scrollable container - no truncation
        # Escape HTML characters in context and token
        context_escaped = html_lib.escape(str(context))
        token_escaped = html_lib.escape(str(token))

        # Build token card with full context (scrollable)
        card_html = f"""
        <div style="background-color: {bg_color}; border-left: 4px solid {border_color};
                    padding: 15px; border-radius: 5px; margin-bottom: 5px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="color: #71809A; font-size: 0.9em;">Token #{idx + 1} | {task_type}</span>
                <span style="color: {border_color}; font-weight: bold; font-size: 1.1em;">
                    {'+'if prob_delta > 0 else ''}{prob_delta:.3f}
                </span>
            </div>
            <div style="background-color: #111722; padding: 10px; border-radius: 5px; max-height: 200px; overflow-y: auto; margin: 10px 0;">
                <span style="color: #71809A; font-family: monospace; font-size: 0.85em; white-space: pre-wrap; word-break: break-word;">{context_escaped}</span><span style="background-color: {border_color}; color: white; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-family: monospace;">{token_escaped}</span>
            </div>
            <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                <span style="background-color: #161E2C; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; color: #71809A;">
                    Before: {prob_before:.3f}
                </span>
                <span style="background-color: #161E2C; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; color: #71809A;">
                    After: {prob_after:.3f}
                </span>
                <span style="background-color: #161E2C; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; color: #A78BFA;">
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
    colors = ['#34D399' if d > 0 else '#FB7185' for d in prob_deltas]

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
        template="pts",
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

    # PTS unified events
    if dataset_type in PTS_TYPES:
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
    <div style="font-family: sans-serif; padding: 20px; background-color: #111722; border: 1px solid #212C3D; border-radius: 4px;">
        <h3 style="color: #CBD5E4; border-bottom: 2px solid #A78BFA; padding-bottom: 10px;">
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
        border_color = "#34D399" if is_positive else "#FB7185"

        # Build step card
        step_html = f"""
        <div style="background-color: {bg_color}; border-left: 4px solid {border_color};
                    padding: 15px; border-radius: 5px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #71809A; font-size: 0.9em;">Step {sentence_id} | {category}</span>
                <span style="color: {border_color}; font-weight: bold;">
                    {'+'if prob_delta > 0 else ''}{prob_delta:.3f}
                </span>
            </div>
            <p style="color: #CBD5E4; margin: 10px 0;">{sentence}</p>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span style="background-color: #161E2C; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; color: #71809A;">
                    Importance: {importance:.3f}
                </span>
        """

        if verification_score is not None:
            v_color = "#34D399" if verification_score > 0.5 else "#FB7185"
            step_html += f"""
                <span style="background-color: #161E2C; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; color: {v_color};">
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

    colors = ['#34D399' if p > 0.5 else '#FB7185' for p in prob_values]

    fig.add_trace(go.Scatter(
        x=[int(s) if isinstance(s, (int, np.integer)) else s for s in sentence_ids],
        y=[float(p) for p in prob_values],
        mode='lines+markers',
        name='Success Probability',
        line=dict(color='#A78BFA', width=2),
        marker=dict(size=10, color=colors)
    ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="50% threshold")

    fig.update_layout(
        title="Probability Progression Through Reasoning",
        xaxis_title="Sentence ID",
        yaxis_title="Success Probability",
        yaxis_range=[0, 1],
        template="pts",
        height=300
    )

    return "\n".join(html_parts), fig


def create_statistics_dashboard(df: pd.DataFrame) -> Tuple[str, go.Figure]:
    """Create statistics dashboard for the dataset."""
    if df.empty:
        return "No data available", go.Figure()

    dataset_type = detect_dataset_type(df)
    is_pts = dataset_type in PTS_TYPES

    # Build statistics
    stats = {
        "Total Items": len(df),
        "Dataset Type": dataset_type,
    }

    if is_pts:
        gran = _granularity_series(df)
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

    # Build HTML. Tiles are tinted by the channel they report on, so the three
    # scales stay visually distinct even in a plain count.
    TILE_CHANNEL = {
        'Latent Events': C_LATENT,
        'Token Events': C_TOKEN,
        'Sentence Events': C_SENTENCE,
        'Positive Items': C_POSITIVE,
        'Negative Items': C_NEGATIVE,
        'Causal Links': C_LATENT,
    }

    html_parts = ['<div class="pts-stats">']
    for key, value in stats.items():
        c = TILE_CHANNEL.get(key)
        accent = ' accent' if c else ''
        style = f' style="--c:{c}"' if c else ''
        html_parts.append(
            f'<div class="pts-stat"{style}>'
            f'<div class="l">{html_lib.escape(str(key))}</div>'
            f'<div class="n{accent}">{html_lib.escape(str(value))}</div>'
            f'</div>'
        )
    html_parts.append('</div>')

    # causal-event datasets get their own panel set: emitted deltas and latent readout scores are
    # different quantities and are never binned into the same histogram.
    if is_pts:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Δ probability (emitted events)",
                "Events by type",
                "Readout score (latent events)",
            ),
        )

        gran = _granularity_series(df)
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
        fig.update_layout(template="pts", height=380, showlegend=False)
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
                   marker_color='#A78BFA', width=(bin_edges[1]-bin_edges[0])*0.9),
            row=1, col=1
        )
    elif 'prob_after' in df.columns and len(df['prob_after'].dropna()) > 0:
        # Fallback: show prob_after distribution
        prob_data = df['prob_after'].dropna().values
        counts, bin_edges = np.histogram(prob_data, bins=30)
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
        fig.add_trace(
            go.Bar(x=bin_centers, y=counts.tolist(), name="Prob After",
                   marker_color='#A78BFA', width=(bin_edges[1]-bin_edges[0])*0.9),
            row=1, col=1
        )

    # Second chart: Categories, patterns, or task types
    if 'sentence_category' in df.columns:
        category_counts = df['sentence_category'].value_counts()
        fig.add_trace(
            go.Bar(x=category_counts.index.tolist(), y=category_counts.values.tolist(), name="Categories",
                  marker_color='#34D399'),
            row=1, col=2
        )
    elif 'reasoning_pattern' in df.columns:
        pattern_counts = df['reasoning_pattern'].value_counts()
        fig.add_trace(
            go.Bar(x=pattern_counts.index.tolist(), y=pattern_counts.values.tolist(), name="Patterns",
                  marker_color='#34D399'),
            row=1, col=2
        )
    elif 'task_type' in df.columns:
        task_counts = df['task_type'].value_counts()
        fig.add_trace(
            go.Bar(x=task_counts.index.tolist(), y=task_counts.values.tolist(), name="Task Types",
                  marker_color='#34D399'),
            row=1, col=2
        )
    elif 'is_positive' in df.columns:
        pos_neg_counts = df['is_positive'].value_counts()
        labels = ['Positive' if v else 'Negative' for v in pos_neg_counts.index.tolist()]
        fig.add_trace(
            go.Bar(x=labels, y=pos_neg_counts.values.tolist(), name="Impact",
                  marker_color=['#34D399' if l == 'Positive' else '#FB7185' for l in labels]),
            row=1, col=2
        )

    fig.update_layout(
        template="pts",
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
<div style="padding: 40px; text-align: center; background-color: #111722; border: 1px solid #212C3D; border-radius: 4px;">
    <h3 style="color: #FBBF24;">DPO Pairs Dataset</h3>
    <p style="color: #71809A;">This visualization is not available for DPO pairs datasets.</p>
    <p style="color: #71809A;">DPO pairs contain prompt/chosen/rejected structure without token-level context.</p>
    <p style="color: #A78BFA; margin-top: 20px;">
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
        empty_fig.update_layout(template="pts")
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

    # Render only what is visible on load -- the Overview (and the cheap Event
    # Explorer below). The graph, embedding, and timeline tabs each render six
    # Plotly figures whose serialization + browser paint dominate load on a weak
    # Space CPU (~90s for all six). They are computed on demand when their tab is
    # opened (see the tab.select handlers), which keeps load to a couple of
    # seconds. The Python for each is <0.5s; the cost is figure transfer/paint.
    stats_html, stats_fig = create_statistics_dashboard(df)

    lazy = _empty_fig("Open this tab to render.", 480)
    graph_fig = lazy
    embed_fig = lazy
    circuit_html = "<div style='padding:24px;color:#8592A8;font-family:monospace'>Open this tab to render.</div>"
    circuit_fig = lazy
    timeline_fig = lazy
    heatmap_fig = lazy

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
        template="pts",
        height=300,
    )
    return fig


def create_latent_detail_html(row) -> str:
    """HTML card for a latent meta-token event."""
    context = html_lib.escape(str(_val(row, 'context', '')))
    label = html_lib.escape(str(_val(row, 'label', '')))
    score = _num(_val(row, 'score'))

    return f"""
    <div style="background-color: #111722; border: 1px solid #212C3D; border-radius: 4px; padding: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="color: #71809A; font-size: 0.9em;">
                Latent meta-token | layer {_val(row, 'layer', 'n/a')} |
                readout {_val(row, 'readout_method', 'n/a')} |
                offset {_val(row, 'position', 'n/a')} |
                category {html_lib.escape(str(_val(row, 'category', 'unknown')))}
            </span>
            <span style="background-color: {COLOR_LATENT}; color: white; padding: 4px 12px; border-radius: 5px; font-weight: bold;">
                Readout score: {score:.4f}
            </span>
        </div>
        <div style="font-family: monospace; padding: 15px; background-color: #0d1117; border-radius: 8px; color: #CBD5E4; line-height: 1.8; max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; border: 1px solid #30363d;">
            <span style="color: #8b949e;">{context}</span><span style="background-color: {COLOR_LATENT}; padding: 2px 6px; border-radius: 3px; border: 2px solid #A78BFA; font-weight: bold;">{label}</span>
        </div>
        <p style="color: #A78BFA; margin-top: 15px; font-size: 0.9em;">
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

    # latent events get their own card: no before/after probability exists.
    if is_pts_events(df) and _granularity(row) == 'latent':
        return create_latent_detail_html(row), create_readout_score_chart(row)

    # legacy pivot_context/prefix_context, PTS context. Same for the label.
    context = _val(row, 'pivot_context', _val(row, 'prefix_context', _val(row, 'context', '')))
    token = _val(row, 'pivot_token', _val(row, 'sentence', _val(row, 'label', '')))
    prob_delta = _num(_val(row, 'prob_delta'))
    prob_before = _num(_val(row, 'prob_before', _val(row, 'prob_without_sentence', 0.5)), 0.5)
    prob_after = _num(_val(row, 'prob_after', _val(row, 'prob_with_sentence', 0.5)), 0.5)

    if not context and not token:
        html = """
        <div style="padding: 40px; text-align: center; background-color: #111722; border: 1px solid #212C3D; border-radius: 4px;">
            <h3 style="color: #FB7185;">Missing Data</h3>
            <p style="color: #71809A;">This dataset doesn't have the expected fields for token visualization.</p>
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

    PTS events carry ``score`` directly. legacy records get |prob_delta| (or the
    thought-anchor importance score), which is the same quantity PTS stores.
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
            '<div style="padding: 20px; color: #71809A; background-color: #111722; '
            'border-radius: 4px;">No events match these filters.</div>',
            _empty_fig("No events match these filters", 320),
        )

    is_pts = is_pts_events(df)
    if is_pts:
        gran = _granularity_series(df)
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
        <div style="background: #111722; border: 1px solid #212C3D;
                    padding: 15px; border-radius: 4px; text-align: center;">
            <div style="color: #A78BFA; font-size: 1.4em; font-weight: bold;">{value}</div>
            <div style="color: #71809A; font-size: 0.85em; margin-top: 4px;">{name}</div>
        </div>
        """)
    html_parts.append('</div>')

    fig = go.Figure()
    # Vectorized: iterrows over 14k events took seconds. Emitted score is
    # |prob_delta| (fall back to |score|); latent score is the readout score.
    # Every series is aligned to df's index so masks never mismatch on datasets
    # that lack a `score`/`prob_delta` column (e.g. steering vectors).
    nan = pd.Series([float('nan')] * len(df), index=df.index)
    score_col = pd.to_numeric(df['score'], errors='coerce') if 'score' in df.columns else nan
    delta = pd.to_numeric(df['prob_delta'], errors='coerce').abs() if 'prob_delta' in df.columns else nan
    emitted_all = delta.where(delta.notna(), score_col.abs())
    is_latent = pd.Series(gran.values == 'latent', index=df.index)
    emitted_scores = emitted_all[~is_latent].dropna().tolist()
    latent_scores = score_col[is_latent].dropna().tolist()

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
        template="pts",
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
    is_pts = is_pts_events(work)

    if is_pts and event_type and event_type != "All" and 'event_type' in work.columns:
        work = work[work['event_type'].astype(str) == event_type]

    if is_pts and granularity and granularity != "All" and not work.empty:
        work = work[_granularity_series(work) == granularity]

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
        # Apply the threshold only to emitted events, whose score is a
        # probability delta. A latent event's score is a readout probability on
        # a different scale entirely -- one slider across both would drop a
        # pivotal token worth +0.45 while keeping a banal " the" readout at 0.92.
        def score_ok(row) -> bool:
            if _granularity(row) == 'latent':
                return True
            return _row_score(row) >= float(min_score)

        work = work[work.apply(score_ok, axis=1)]

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
    """Update the Reasoning Timeline tab (timeline + heatmap + trace)."""
    dataset_type = current_data.get("type", "unknown")
    df = current_data["df"]

    if dataset_type == 'dpo_pairs':
        html = """
        <div style="padding: 40px; text-align: center; background-color: #111722; border: 1px solid #212C3D; border-radius: 4px;">
            <h3 style="color: #FBBF24;">DPO Pairs Dataset</h3>
            <p style="color: #71809A;">The reasoning timeline is not available for DPO pairs datasets.</p>
            <p style="color: #A78BFA; margin-top: 20px;">
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
    timeline_fig = create_reasoning_timeline(df, selected_query)
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
        empty_fig.update_layout(template="pts")
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
    timeline_fig = create_reasoning_timeline(df, first_query)
    heatmap_fig = create_workspace_heatmap(df, first_query)

    return (stats_html, stats_fig, graph_fig, embed_fig, circuit_html, circuit_fig,
            timeline_fig, heatmap_fig)


# ============================================================================
# Build Gradio App
# ============================================================================

# Pre-defined HuggingFace datasets.
# The legacy datasets below still load and render exactly as they did before; the
# causal-event datasets use the unified CausalReasoningEvent schema.
# Only datasets that actually exist on the Hub. The visualizer reads PTS
# causal-event files too -- upload one and add it here, or load it from disk with
# the file upload. Listing PTS datasets that have not been published yet would put
# entries in the dropdown that can only fail.
# The unified `-pts` datasets are what this visualizer is for: pivotal reasoning
# events at all three scales, with the multiscale timeline, causal graph, and
# workspace heatmap. The other PTS repos don't earn a slot here:
#   - `-thought-anchors`  duplicate the sentence events already in `-pts`
#   - `-dpo-pairs`        are training pairs, not events (no charts render)
#   - `-steering-vectors` are activation vectors -- only the embedding tab
#                         applies; the multiscale views are token-only
# All of them remain linked in this Space's README `datasets:` list for
# discovery. Any of them can still be loaded by pasting its id or uploading it.
HF_DATASETS = [
    "codelion/Qwen3-0.6B-pts",
    "codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts",
]

DEFAULT_DATASET = "codelion/Qwen3-0.6B-pts"


# ============================================================================
# Design system
#
# PTS reads signals out of a model's internals at three depths. The interface is
# built as an *instrument* for that -- a spectrometer readout, not an admin
# dashboard. Three scales, three channels, each with its own signal colour; that
# colour system is the identity and it is used with discipline everywhere:
# charts, panel accents, badges, legends.
#
#   LATENT   violet   hidden, deep in the residual stream
#   TOKEN    sky      emitted, sharp, a single decision point
#   SENTENCE amber    emitted, extended over a reasoning step
#
# Valence (green/rose) is deliberately a SEPARATE axis from channel, because
# latent events have no valence -- they are never green or red, and the palette
# must make that impossible to render by accident.
# ============================================================================

# (palette defined at the top of this file)

# One Plotly template for every figure in the app. Registered as "pts" and
# swapped in for plotly_dark throughout, so no chart can drift off-system.
_pts_template = go.layout.Template()
_pts_template.layout = go.Layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'IBM Plex Sans', system-ui, sans-serif", size=12, color=C_TEXT),
    title=dict(
        font=dict(family="'IBM Plex Mono', monospace", size=14, color=C_TEXT),
        x=0.01, xanchor="left", pad=dict(l=4, t=8, b=8),
    ),
    colorway=[C_LATENT, C_TOKEN, C_SENTENCE, C_POSITIVE, C_NEGATIVE, "#F472B6", "#2DD4BF"],
    xaxis=dict(
        gridcolor=C_LINE_SOFT, zerolinecolor=C_LINE, linecolor=C_LINE,
        tickfont=dict(family="'IBM Plex Mono', monospace", size=10, color=C_TEXT_MUTED),
        title=dict(font=dict(size=11, color=C_TEXT_MUTED)),
        showline=True, ticks="outside", tickcolor=C_LINE, ticklen=4,
    ),
    yaxis=dict(
        gridcolor=C_LINE_SOFT, zerolinecolor=C_LINE, linecolor=C_LINE,
        tickfont=dict(family="'IBM Plex Mono', monospace", size=10, color=C_TEXT_MUTED),
        title=dict(font=dict(size=11, color=C_TEXT_MUTED)),
        showline=True, ticks="outside", tickcolor=C_LINE, ticklen=4,
    ),
    legend=dict(
        bgcolor="rgba(17,23,34,0.85)", bordercolor=C_LINE, borderwidth=1,
        font=dict(family="'IBM Plex Mono', monospace", size=10, color=C_TEXT_MUTED),
    ),
    hoverlabel=dict(
        bgcolor=C_PANEL_HI, bordercolor=C_LINE,
        font=dict(family="'IBM Plex Mono', monospace", size=11, color=C_TEXT),
        align="left",
    ),
    margin=dict(l=56, r=24, t=48, b=48),
    colorscale=dict(sequential=[[0, "#161E2C"], [0.5, "#5B4BC4"], [1, C_LATENT]]),
)
pio.templates["pts"] = _pts_template


PTS_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
    text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill=C_CANVAS,
    body_text_color=C_TEXT,
    background_fill_primary=C_PANEL,
    background_fill_secondary=C_PANEL_HI,
    border_color_primary=C_LINE,
    block_background_fill=C_PANEL,
    block_border_color=C_LINE,
    block_label_background_fill="transparent",
    block_label_text_color=C_TEXT_MUTED,
    block_title_text_color=C_TEXT,
    input_background_fill=C_CANVAS,
    input_border_color=C_LINE,
    button_primary_background_fill=C_LATENT,
    button_primary_background_fill_hover="#B9A3FB",
    button_primary_text_color="#0A0D13",
    button_secondary_background_fill=C_PANEL_HI,
    button_secondary_border_color=C_LINE,
    button_secondary_text_color=C_TEXT,
    panel_background_fill=C_PANEL,
)


CSS = f"""
/* ---- canvas: a faint graticule, like an instrument screen ---- */
.gradio-container {{
  max-width: 1580px !important;
  background:
    linear-gradient(to right, {C_LINE_SOFT}22 1px, transparent 1px) 0 0 / 32px 32px,
    linear-gradient(to bottom, {C_LINE_SOFT}22 1px, transparent 1px) 0 0 / 32px 32px,
    radial-gradient(ellipse 90% 55% at 50% -12%, #241C4E55 0%, transparent 70%),
    {C_CANVAS} !important;
}}
footer {{ display: none !important; }}

/* ---- masthead ---- */
.pts-masthead {{
  border: 1px solid {C_LINE};
  border-radius: 4px;
  background: linear-gradient(160deg, {C_PANEL_HI} 0%, {C_PANEL} 60%);
  padding: 26px 30px 22px;
  margin-bottom: 18px;
  position: relative;
  overflow: hidden;
}}
/* the three channels, drawn as a signal bar across the top edge */
.pts-masthead::before {{
  content: "";
  position: absolute; inset: 0 0 auto 0; height: 2px;
  background: linear-gradient(90deg,
    {C_LATENT} 0%, {C_LATENT} 33%,
    {C_TOKEN} 33%, {C_TOKEN} 66%,
    {C_SENTENCE} 66%, {C_SENTENCE} 100%);
}}
.pts-wordmark {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 30px; font-weight: 600; letter-spacing: -0.02em;
  color: {C_TEXT}; margin: 0 0 2px;
}}
.pts-wordmark .tag {{
  font-size: 11px; font-weight: 500; letter-spacing: 0.16em;
  color: {C_LATENT}; border: 1px solid {C_LATENT}44;
  border-radius: 3px; padding: 3px 7px; margin-left: 12px;
  vertical-align: middle; background: {C_LATENT}12;
}}
.pts-tagline {{
  font-size: 13.5px; color: {C_TEXT_MUTED}; margin: 0 0 20px; max-width: 78ch;
  line-height: 1.6;
}}
.pts-tagline b {{ color: {C_TEXT}; font-weight: 500; }}

/* ---- the three-channel key: the conceptual spine of the whole tool ---- */
.pts-channels {{ display: flex; gap: 10px; flex-wrap: wrap; }}
.pts-chan {{
  flex: 1 1 190px;
  border: 1px solid {C_LINE};
  border-left: 2px solid var(--c);
  border-radius: 3px;
  background: {C_CANVAS}99;
  padding: 10px 13px;
}}
.pts-chan .k {{
  font-family: 'IBM Plex Mono', monospace; font-size: 10px;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--c); margin-bottom: 3px;
}}
.pts-chan .v {{ font-size: 12.5px; color: {C_TEXT}; line-height: 1.4; }}
.pts-chan .m {{ font-size: 11px; color: {C_TEXT_FAINT}; margin-top: 4px;
  font-family: 'IBM Plex Mono', monospace; }}

/* ---- stat tiles: tabular figures, instrument readout ---- */
.pts-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
  gap: 9px; margin: 4px 0 16px; }}
.pts-stat {{
  border: 1px solid {C_LINE}; border-radius: 3px;
  background: {C_PANEL}; padding: 11px 13px 10px;
  border-top: 2px solid var(--c, {C_LINE});
}}
.pts-stat .l {{
  font-family: 'IBM Plex Mono', monospace; font-size: 9.5px;
  letter-spacing: 0.13em; text-transform: uppercase;
  color: {C_TEXT_FAINT}; margin-bottom: 5px; white-space: nowrap;
}}
.pts-stat .n {{
  font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 600;
  color: {C_TEXT}; font-variant-numeric: tabular-nums; line-height: 1.1;
}}
.pts-stat .n.accent {{ color: var(--c); }}
.pts-stat .s {{ font-size: 10.5px; color: {C_TEXT_FAINT}; margin-top: 2px; }}

/* ---- section rule ---- */
.pts-rule {{
  font-family: 'IBM Plex Mono', monospace; font-size: 10.5px;
  letter-spacing: 0.16em; text-transform: uppercase; color: {C_TEXT_FAINT};
  display: flex; align-items: center; gap: 12px; margin: 20px 0 10px;
}}
.pts-rule::after {{ content: ""; flex: 1; height: 1px; background: {C_LINE}; }}

/* ---- event cards ---- */
.pts-card {{
  border: 1px solid {C_LINE}; border-left: 3px solid var(--c, {C_NEUTRAL});
  border-radius: 3px; background: {C_PANEL}; padding: 14px 16px; margin-bottom: 10px;
}}
.pts-card .hdr {{ display: flex; align-items: center; gap: 8px; margin-bottom: 9px; flex-wrap: wrap; }}
.pts-badge {{
  font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; font-weight: 500;
  letter-spacing: 0.1em; text-transform: uppercase;
  padding: 3px 7px; border-radius: 2px;
  border: 1px solid var(--c, {C_NEUTRAL})55;
  color: var(--c, {C_NEUTRAL}); background: var(--c, {C_NEUTRAL})12;
}}
.pts-metric {{
  font-family: 'IBM Plex Mono', monospace; font-size: 12px;
  font-variant-numeric: tabular-nums; color: {C_TEXT_MUTED};
}}
.pts-metric b {{ color: {C_TEXT}; font-weight: 600; }}

/* the token, shown inline in its context */
.pts-context {{
  font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; line-height: 1.75;
  color: {C_TEXT_MUTED}; background: {C_CANVAS}; border: 1px solid {C_LINE_SOFT};
  border-radius: 3px; padding: 12px 14px; white-space: pre-wrap;
  max-height: 260px; overflow-y: auto;
}}
.pts-hit {{
  padding: 1px 4px; border-radius: 2px; font-weight: 600;
  color: #0A0D13; background: var(--c);
  box-shadow: 0 0 14px -2px var(--c);
}}

/* a latent readout is NOT a probability delta -- it never gets a valence colour */
.pts-obs {{
  font-family: 'IBM Plex Mono', monospace; font-size: 10px;
  color: {C_LATENT}; border: 1px dashed {C_LATENT}44;
  border-radius: 2px; padding: 2px 6px; letter-spacing: 0.06em;
}}

/* ---- gradio surface tuning ---- */
.tabs button {{
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 11.5px !important; letter-spacing: 0.09em !important;
  text-transform: uppercase !important;
}}
.tabs button.selected {{
  color: {C_LATENT} !important;
  border-bottom: 2px solid {C_LATENT} !important;
}}
.block, .form {{ border-radius: 4px !important; }}
label span {{
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 10px !important; letter-spacing: 0.11em !important;
  text-transform: uppercase !important; color: {C_TEXT_FAINT} !important;
}}
.plot-container, .js-plotly-plot {{ border-radius: 3px; }}
::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: {C_CANVAS}; }}
::-webkit-scrollbar-thumb {{ background: {C_LINE}; border-radius: 5px; }}
::-webkit-scrollbar-thumb:hover {{ background: #2E3A50; }}

/* ---- source bar: one compact row, so the charts stay above the fold ---- */
.pts-loadbar {{
  border: 1px solid {C_LINE}; border-radius: 4px; background: {C_PANEL};
  padding: 10px 12px; margin-bottom: 8px; align-items: center !important;
}}
.pts-loadbar .wrap, .pts-loadbar .file-preview {{ min-height: 0 !important; }}
/* the drop zone is a secondary affordance -- shrink its chrome so it does not
   shout louder than the dataset picker next to it */
.pts-upload .wrap {{ font-size: 11px !important; gap: 2px !important; }}
.pts-upload svg {{ width: 16px !important; height: 16px !important; }}
.pts-upload .or {{ display: none !important; }}
.pts-status textarea {{
  background: {C_CANVAS} !important; border: 1px solid {C_LINE_SOFT} !important;
  font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important;
  color: {C_TEXT_MUTED} !important; padding: 7px 10px !important;
}}

/* ---- sliders: instrument, not stock ---- */
input[type=range] {{ accent-color: {C_LATENT}; }}
.gradio-container input[type=range]::-webkit-slider-runnable-track {{
  background: {C_LINE} !important; height: 3px !important;
}}
.gradio-container input[type=range]::-webkit-slider-thumb {{
  background: {C_LATENT} !important; border: none !important;
  width: 13px !important; height: 13px !important; border-radius: 50% !important;
  margin-top: -5px !important; box-shadow: 0 0 10px -1px {C_LATENT};
}}

/* numbers anywhere gradio renders them */
.gradio-container input[type=number] {{
  font-family: 'IBM Plex Mono', monospace !important;
  font-variant-numeric: tabular-nums;
}}
"""


MASTHEAD = f"""
<div class="pts-masthead">
  <div class="pts-wordmark">PTS<span class="tag">PIVOTAL TOKEN SEARCH</span></div>
  <p class="pts-tagline">
    A causal-event search framework for model reasoning. PTS finds the
    <b>pivotal reasoning events</b> that shift a model's probability of solving a task,
    at three representational scales &mdash; latent, token, and sentence &mdash; as a
    single kind of object.
  </p>
  <div class="pts-channels">
    <div class="pts-chan" style="--c:{C_LATENT}">
      <div class="k">Latent PTS</div>
      <div class="v">Meta-tokens read out of the hidden workspace</div>
      <div class="m">J-lens · observational</div>
    </div>
    <div class="pts-chan" style="--c:{C_TOKEN}">
      <div class="k">Token PTS</div>
      <div class="v">Emitted tokens that flip success probability</div>
      <div class="m">&Delta;P(success) · measured</div>
    </div>
    <div class="pts-chan" style="--c:{C_SENTENCE}">
      <div class="k">Sentence PTS</div>
      <div class="v">Reasoning steps that flip success probability</div>
      <div class="m">&Delta;P(success) · measured</div>
    </div>
  </div>
</div>
"""


with gr.Blocks(title="PTS · Pivotal Token Search", theme=PTS_THEME, css=CSS) as demo:

    gr.HTML(MASTHEAD)

    # Source bar. Kept to a single row: the old stacked layout ate the whole
    # viewport and pushed every chart below the fold, which is a worse problem
    # than any colour choice.
    with gr.Row(elem_classes="pts-loadbar"):
        with gr.Column(scale=2, min_width=190):
            source_type = gr.Radio(
                choices=["HuggingFace Hub", "Local File"],
                value="HuggingFace Hub",
                label="Source",
                container=False,
            )
        with gr.Column(scale=5, min_width=280):
            dataset_dropdown = gr.Dropdown(
                choices=HF_DATASETS,
                value=DEFAULT_DATASET,
                label="Dataset",
                allow_custom_value=True,
                container=False,
            )
        with gr.Column(scale=3, min_width=180):
            file_upload = gr.File(
                label="Or upload JSONL",
                file_types=[".jsonl", ".json"],
                height=88,
                elem_classes="pts-upload",
            )
        with gr.Column(scale=2, min_width=140):
            load_btn = gr.Button("Load", variant="primary", size="sm")
            refresh_btn = gr.Button("Refresh", variant="secondary", size="sm")

    with gr.Row():
        load_status = gr.Textbox(interactive=False, container=False, show_label=False,
                                 lines=1, max_lines=1, elem_classes="pts-status",
                                 placeholder="no dataset loaded")
        dataset_info = gr.Textbox(interactive=False, container=False, show_label=False,
                                  lines=1, max_lines=1, elem_classes="pts-status",
                                  placeholder="—")

    # Main Visualization Tabs
    with gr.Tabs():

        # Overview Tab (unified event overview)
        with gr.TabItem("Overview"):
            gr.Markdown("### Dataset Statistics")
            gr.Markdown(
                "*For PTS datasets this counts events at all three scales "
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
        with gr.TabItem("Causal Event Graph") as tab_graph:
            gr.Markdown("### Causal Event Graph")
            gr.Markdown("""
            *Latent meta-tokens (diamonds) → pivotal tokens (circles) → thought anchors
            (squares) → outcome (star). Edges come from the event links
            (`precedes_event_ids` / `linked_event_ids` / `parent_event_id`).
            Green = positive impact, red = negative impact, purple = latent (no measured valence).
            Node size reflects score. legacy datasets fall back to the original reasoning graph.*
            """)
            with gr.Row():
                query_filter = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Filter by Query"
                )
            graph_plot = gr.Plot()

        # Embedding Visualization Tab
        with gr.TabItem("Embedding Space") as tab_embed:
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

        # Reasoning Timeline Tab (was: Circuit Tracer)
        with gr.TabItem("Reasoning Timeline") as tab_timeline:
            gr.Markdown("### Reasoning Timeline")
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
            timeline_plot = gr.Plot(label="Reasoning Timeline")

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

    # Lazy tab rendering: each heavy tab computes its figures only when opened,
    # so load never pays for all six at once. Recompute-on-open is cheap in
    # Python and keeps the wiring simple (no stale-cache bugs).
    def _current_df():
        df = current_data.get("df")
        return df if df is not None and hasattr(df, "empty") and not df.empty else None

    def render_graph_tab():
        df = _current_df()
        return create_causal_event_graph(df) if df is not None else _empty_fig("No data loaded.", 480)

    def render_embed_tab():
        df = _current_df()
        return create_embedding_visualization(df) if df is not None else _empty_fig("No data loaded.", 480)

    def render_timeline_tab():
        df = _current_df()
        if df is None:
            e = _empty_fig("No data loaded.", 480)
            return "", e, e, e
        first_q = df['query'].iloc[0] if 'query' in df.columns and len(df) else None
        c_html, c_fig = create_circuit_visualization(df)
        return c_html, c_fig, create_reasoning_timeline(df, first_q), create_workspace_heatmap(df, first_q)

    tab_graph.select(fn=render_graph_tab, outputs=[graph_plot], api_name=False)
    tab_embed.select(fn=render_embed_tab, outputs=[embed_plot], api_name=False)
    tab_timeline.select(
        fn=render_timeline_tab,
        outputs=[circuit_html, circuit_chart, timeline_plot, heatmap_plot],
        api_name=False,
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
