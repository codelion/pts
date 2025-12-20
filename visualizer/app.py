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
    """Detect the type of PTS dataset."""
    columns = set(df.columns)

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
        fig.add_trace(go.Scatter(
            x=list(range(len(positive_df))),
            y=positive_df['prob_delta'].values,
            mode='markers',
            name='Positive Impact',
            marker=dict(
                size=10 + positive_df['prob_delta'].abs().values * 30,
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
        fig.add_trace(go.Scatter(
            x=list(range(len(negative_df))),
            y=negative_df['prob_delta'].values,
            mode='markers',
            name='Negative Impact',
            marker=dict(
                size=10 + negative_df['prob_delta'].abs().values * 30,
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
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

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
        node_x.append(x)
        node_y.append(y)

        node_data = G.nodes[node]
        is_positive = node_data.get('is_positive', True)
        importance = node_data.get('importance', 0.3)

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
        text = f"Token: {row.get('pivot_token', 'N/A')}<br>"
        text += f"Before: {row.get('prob_before', 0):.3f}<br>"
        text += f"After: {row.get('prob_after', 0):.3f}<br>"
        text += f"Delta: {row.get('prob_delta', 0):+.3f}<br>"
        text += f"Query: {str(row.get('query', ''))[:40]}..."
        hover_texts.append(text)

    fig.add_trace(go.Scatter(
        x=df['prob_before'],
        y=df['prob_after'],
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
        if isinstance(emb, list) and len(emb) > 0:
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
    perplexity = min(30, max(5, n_samples // 3))

    if embeddings.shape[1] > 50:
        # First reduce with PCA
        pca = PCA(n_components=min(50, n_samples - 1))
        embeddings = pca.fit_transform(embeddings)

    # Then t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings)

    # Create dataframe for plotting
    plot_df = df.iloc[valid_indices].copy()
    plot_df['x'] = coords[:, 0]
    plot_df['y'] = coords[:, 1]

    # Handle color column
    if color_by not in plot_df.columns:
        color_by = 'is_positive' if 'is_positive' in plot_df.columns else None

    if color_by and color_by in plot_df.columns:
        fig = px.scatter(
            plot_df, x='x', y='y',
            color=color_by,
            hover_data=['sentence' if 'sentence' in plot_df.columns else 'pivot_token'],
            title="Embedding Space Visualization (t-SNE)",
            template="plotly_dark"
        )
    else:
        fig = px.scatter(
            plot_df, x='x', y='y',
            hover_data=['sentence' if 'sentence' in plot_df.columns else 'pivot_token'],
            title="Embedding Space Visualization (t-SNE)",
            template="plotly_dark"
        )

    fig.update_layout(height=500)

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
        x=sentence_ids,
        y=prob_values,
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

    # Build statistics
    stats = {
        "Total Items": len(df),
        "Dataset Type": dataset_type,
    }

    if 'is_positive' in df.columns:
        positive_count = df['is_positive'].sum()
        stats["Positive Items"] = int(positive_count)
        stats["Negative Items"] = int(len(df) - positive_count)

    if 'prob_delta' in df.columns:
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

    # Create distribution charts
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Probability Delta Distribution", "Category Distribution"))

    if 'prob_delta' in df.columns:
        fig.add_trace(
            go.Histogram(x=df['prob_delta'], nbinsx=30, name="Prob Delta",
                        marker_color='#6366f1'),
            row=1, col=1
        )

    if 'sentence_category' in df.columns:
        category_counts = df['sentence_category'].value_counts()
        fig.add_trace(
            go.Bar(x=category_counts.index, y=category_counts.values, name="Categories",
                  marker_color='#22c55e'),
            row=1, col=2
        )
    elif 'reasoning_pattern' in df.columns:
        pattern_counts = df['reasoning_pattern'].value_counts()
        fig.add_trace(
            go.Bar(x=pattern_counts.index, y=pattern_counts.values, name="Patterns",
                  marker_color='#22c55e'),
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

# Global state for loaded data
current_data = {"df": pd.DataFrame(), "type": "unknown"}


def load_dataset_action(source_type: str, dataset_id: str, file_upload) -> Tuple[str, str]:
    """Handle dataset loading."""
    global current_data

    if source_type == "HuggingFace Hub":
        if not dataset_id:
            return "Please enter a dataset ID", ""
        df, msg = load_hf_dataset(dataset_id)
    else:  # Local File
        if file_upload is None:
            return "Please upload a file", ""
        df, msg = load_jsonl_file(file_upload.name)

    if df.empty:
        return msg, ""

    current_data["df"] = df
    current_data["type"] = detect_dataset_type(df)

    columns_info = f"Columns: {', '.join(df.columns[:10])}"
    if len(df.columns) > 10:
        columns_info += f" ... and {len(df.columns) - 10} more"

    return msg, f"Dataset type: {current_data['type']}\n{columns_info}"


def get_token_details(idx: int) -> Tuple[str, go.Figure]:
    """Get details for a specific pivotal token."""
    df = current_data["df"]
    dataset_type = current_data.get("type", "unknown")

    if df.empty:
        return "No data available. Please load a dataset first.", go.Figure()

    # Handle unsupported dataset types
    if dataset_type == 'dpo_pairs':
        html = """
        <div style="padding: 40px; text-align: center; background-color: #1a1a2e; border-radius: 10px;">
            <h3 style="color: #f59e0b;">DPO Pairs Dataset</h3>
            <p style="color: #a0a0a0;">This visualization is not available for DPO pairs datasets.</p>
            <p style="color: #a0a0a0;">DPO pairs contain prompt/chosen/rejected structure without token-level context.</p>
            <p style="color: #6366f1; margin-top: 20px;">
                Try loading a <strong>pivotal_tokens</strong> or <strong>thought_anchors</strong> dataset instead.
            </p>
        </div>
        """
        return html, go.Figure()

    if idx >= len(df):
        return "Index out of range", go.Figure()

    row = df.iloc[idx]

    context = row.get('pivot_context', row.get('prefix_context', ''))
    token = row.get('pivot_token', row.get('sentence', ''))
    prob_delta = row.get('prob_delta', 0)
    prob_before = row.get('prob_before', row.get('prob_with_sentence', 0.5))
    prob_after = row.get('prob_after', row.get('prob_without_sentence', 0.5))

    # Handle missing data
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
    """Update the thought anchor graph."""
    dataset_type = current_data.get("type", "unknown")
    if dataset_type == 'dpo_pairs':
        fig = go.Figure()
        fig.add_annotation(
            text="Reasoning Graph is not available for DPO pairs datasets.<br>Load a pivotal_tokens or thought_anchors dataset.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#a0a0a0")
        )
        fig.update_layout(template="plotly_dark", height=400)
        return fig

    # Convert truncated label back to original query
    original_query = get_original_query_from_label(query_dropdown)
    return create_thought_anchor_graph(current_data["df"], original_query)


def update_embedding_visualization(color_by: str):
    """Update the embedding visualization."""
    dataset_type = current_data.get("type", "unknown")
    if dataset_type == 'dpo_pairs':
        fig = go.Figure()
        fig.add_annotation(
            text="Embedding Space is not available for DPO pairs datasets.<br>Load a pivotal_tokens, thought_anchors, or steering_vectors dataset.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#a0a0a0")
        )
        fig.update_layout(template="plotly_dark", height=400)
        return fig
    return create_embedding_visualization(current_data["df"], color_by)


def update_circuit_view(query_idx: int):
    """Update the circuit view."""
    dataset_type = current_data.get("type", "unknown")
    if dataset_type == 'dpo_pairs':
        html = """
        <div style="padding: 40px; text-align: center; background-color: #1a1a2e; border-radius: 10px;">
            <h3 style="color: #f59e0b;">DPO Pairs Dataset</h3>
            <p style="color: #a0a0a0;">Circuit Tracer is not available for DPO pairs datasets.</p>
            <p style="color: #6366f1; margin-top: 20px;">
                Load a <strong>pivotal_tokens</strong> or <strong>thought_anchors</strong> dataset to explore reasoning circuits.
            </p>
        </div>
        """
        return html, go.Figure()
    return create_circuit_visualization(current_data["df"], int(query_idx))


def update_statistics():
    """Update the statistics dashboard."""
    return create_statistics_dashboard(current_data["df"])


def get_query_list():
    """Get list of unique queries with truncated display labels."""
    df = current_data["df"]
    if df.empty or 'query' not in df.columns:
        return gr.update(choices=[], value=None)

    queries = df['query'].unique().tolist()
    # Return simple truncated strings for dropdown choices
    truncated_queries = []
    for i, q in enumerate(queries):
        q_str = str(q) if q is not None else ""
        if len(q_str) > 80:
            truncated_queries.append(f"[{i+1}] {q_str[:77]}...")
        else:
            truncated_queries.append(f"[{i+1}] {q_str}")

    return gr.update(choices=truncated_queries, value=None)


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
            empty_fig
        )

    stats_html, stats_fig = create_statistics_dashboard(df)
    graph_fig = create_thought_anchor_graph(df)
    embed_fig = create_embedding_visualization(df)
    circuit_html, circuit_fig = create_circuit_visualization(df)

    return stats_html, stats_fig, graph_fig, embed_fig, circuit_html, circuit_fig


# ============================================================================
# Build Gradio App
# ============================================================================

# Pre-defined HuggingFace datasets
HF_DATASETS = [
    "codelion/Qwen3-0.6B-pts",
    "codelion/Qwen3-0.6B-pts-thought-anchors",
    "codelion/Qwen3-0.6B-pts-steering-vectors",
    "codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts",
    "codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts-thought-anchors",
    "codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts-steering-vectors",
]

# CSS configuration
CSS = """
.gradio-container { max-width: 1400px !important; }
.main-header { text-align: center; margin-bottom: 20px; }
"""

with gr.Blocks(title="PTS Visualizer", css=CSS) as demo:

    # Header
    gr.Markdown("""
    # PTS Visualizer
    ### Interactive Exploration of Pivotal Tokens, Thought Anchors & Reasoning Circuits

    A [Neuronpedia](https://neuronpedia.org/)-inspired platform for understanding how language models reason.
    Load datasets from HuggingFace Hub or upload your own JSONL files.

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
                    value=HF_DATASETS[0],
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

        # Overview Tab
        with gr.TabItem("Overview"):
            gr.Markdown("### Dataset Statistics")
            stats_html = gr.HTML()
            stats_chart = gr.Plot()

        # Token Explorer Tab
        with gr.TabItem("Token Explorer"):
            gr.Markdown("### Explore Pivotal Tokens")
            with gr.Row():
                with gr.Column(scale=1):
                    token_slider = gr.Slider(
                        minimum=0, maximum=100, step=1, value=0,
                        label="Token Index"
                    )
                with gr.Column(scale=3):
                    token_html = gr.HTML(label="Token in Context")
            prob_chart = gr.Plot(label="Probability Change")

        # Thought Anchor Graph Tab
        with gr.TabItem("Reasoning Graph"):
            gr.Markdown("### Thought Anchor Dependency Graph")
            gr.Markdown("""
            *Visualizes causal dependencies between reasoning steps.
            Green nodes indicate positive impact, red nodes indicate negative impact.
            Node size reflects importance score.*
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
                    choices=["is_positive", "sentence_category", "reasoning_pattern", "task_type"],
                    value="is_positive",
                    label="Color By"
                )
            embed_plot = gr.Plot()

        # Circuit Tracer Tab
        with gr.TabItem("Circuit Tracer"):
            gr.Markdown("### Step-by-Step Reasoning Circuit")
            gr.Markdown("*Walk through the reasoning process step by step. See how each step affects the probability of success.*")
            with gr.Row():
                circuit_query_idx = gr.Slider(
                    minimum=0, maximum=100, step=1, value=0,
                    label="Query Index"
                )
            circuit_html = gr.HTML()
            circuit_chart = gr.Plot()

    # Event handlers - using api_name=False to prevent schema generation issues
    load_btn.click(
        fn=load_dataset_action,
        inputs=[source_type, dataset_dropdown, file_upload],
        outputs=[load_status, dataset_info],
        api_name=False
    ).then(
        fn=refresh_all,
        outputs=[stats_html, stats_chart, graph_plot, embed_plot, circuit_html, circuit_chart],
        api_name=False
    ).then(
        fn=lambda: gr.update(maximum=max(0, len(current_data["df"]) - 1)),
        outputs=[token_slider],
        api_name=False
    ).then(
        fn=get_query_list,
        outputs=[query_filter],
        api_name=False
    )

    refresh_btn.click(
        fn=refresh_all,
        outputs=[stats_html, stats_chart, graph_plot, embed_plot, circuit_html, circuit_chart],
        api_name=False
    )

    token_slider.change(
        fn=get_token_details,
        inputs=[token_slider],
        outputs=[token_html, prob_chart],
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
        outputs=[circuit_html, circuit_chart],
        api_name=False
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    demo.launch()
