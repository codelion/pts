#!/usr/bin/env python3
"""
Deep Analysis of Thought Anchors

Incorporates ideas from "Thought Branches: Interpreting LLM Reasoning Requires Resampling"
(arXiv:2510.27484) - focuses on contrastive analysis and branch-based interpretation.

Key analyses:
1. Prob delta distributions and impact analysis
2. Sentence pattern analysis (what types become anchors)
3. Entropy-impact correlation
4. Per-query "branch" analysis (reasoning path variations)
5. Contrastive analysis: correct vs incorrect anchor patterns
6. Causal dependency structure

Usage:
    python research/reasoning_circuits/scripts/deep_analysis.py
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import re

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
FIGURES_DIR = SCRIPT_DIR.parent / "figures"

QWEN_FILE = DATA_DIR / "qwen3_verified_anchors.jsonl"
DEEPSEEK_FILE = DATA_DIR / "deepseek_verified_anchors.jsonl"


def load_anchors(filepath):
    """Load thought anchors from JSONL file."""
    anchors = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                anchors.append(json.loads(line))
    return anchors


# =============================================================================
# 1. PROB DELTA ANALYSIS
# =============================================================================

def analyze_prob_deltas(anchors, model_name):
    """Analyze probability delta distributions - the 'impact' of each anchor."""
    print(f"\n{'='*60}")
    print(f"PROB DELTA ANALYSIS: {model_name}")
    print('='*60)

    deltas = [a.get('prob_delta', 0) for a in anchors]
    positive_deltas = [d for d in deltas if d > 0]
    negative_deltas = [d for d in deltas if d < 0]

    print(f"\nTotal anchors: {len(deltas)}")
    print(f"  Positive impact (helps answer): {len(positive_deltas)} ({100*len(positive_deltas)/len(deltas):.1f}%)")
    print(f"  Negative impact (hurts answer): {len(negative_deltas)} ({100*len(negative_deltas)/len(deltas):.1f}%)")

    if deltas:
        print(f"\nProb delta statistics:")
        print(f"  Mean: {np.mean(deltas):+.4f}")
        print(f"  Std: {np.std(deltas):.4f}")
        print(f"  Min: {np.min(deltas):+.4f}")
        print(f"  Max: {np.max(deltas):+.4f}")
        print(f"  Median: {np.median(deltas):+.4f}")

    # Find most impactful anchors (both positive and negative)
    sorted_by_impact = sorted(anchors, key=lambda x: abs(x.get('prob_delta', 0)), reverse=True)

    print(f"\nTop 5 MOST IMPACTFUL anchors (highest |delta|):")
    for i, anchor in enumerate(sorted_by_impact[:5]):
        delta = anchor.get('prob_delta', 0)
        sentence = anchor.get('sentence', '')[:70]
        is_pos = anchor.get('is_positive', True)
        marker = "✓" if is_pos else "✗"
        print(f"  {i+1}. [{marker}] delta={delta:+.3f}: \"{sentence}...\"")

    return deltas


# =============================================================================
# 2. SENTENCE PATTERN ANALYSIS
# =============================================================================

def analyze_sentence_patterns(anchors, model_name):
    """Analyze what linguistic patterns appear in thought anchors."""
    print(f"\n{'='*60}")
    print(f"SENTENCE PATTERN ANALYSIS: {model_name}")
    print('='*60)

    # Define patterns to look for
    patterns = {
        'has_calculation': r'\d+\s*[+\-*/]\s*\d+',
        'has_equals': r'\d+\s*=\s*\d+',
        'has_number': r'\d+',
        'starts_so/therefore': r'^(So|Therefore|Thus|Hence)\b',
        'has_question': r'\?',
        'has_conclusion': r'\b(answer|result|total|equals|gives?)\b',
        'has_step_word': r'\b(step|first|next|then|now)\b',
        'has_verification': r'\b(check|verify|wait|let me|hmm)\b',
        'has_because': r'\b(because|since|as)\b',
        'is_short (<50 chars)': lambda s: len(s) < 50,
        'is_long (>200 chars)': lambda s: len(s) > 200,
    }

    # Separate correct vs incorrect for contrastive analysis
    correct_anchors = [a for a in anchors if a.get('is_positive', True)]
    incorrect_anchors = [a for a in anchors if not a.get('is_positive', True)]

    print(f"\nPattern frequency comparison (Correct vs Incorrect):")
    print(f"{'Pattern':<25} {'Correct':>10} {'Incorrect':>10} {'Diff':>10}")
    print("-" * 55)

    for name, pattern in patterns.items():
        correct_count = 0
        incorrect_count = 0

        for anchor in correct_anchors:
            sentence = anchor.get('sentence', '')
            if callable(pattern):
                if pattern(sentence):
                    correct_count += 1
            elif re.search(pattern, sentence, re.IGNORECASE):
                correct_count += 1

        for anchor in incorrect_anchors:
            sentence = anchor.get('sentence', '')
            if callable(pattern):
                if pattern(sentence):
                    incorrect_count += 1
            elif re.search(pattern, sentence, re.IGNORECASE):
                incorrect_count += 1

        correct_pct = 100 * correct_count / len(correct_anchors) if correct_anchors else 0
        incorrect_pct = 100 * incorrect_count / len(incorrect_anchors) if incorrect_anchors else 0
        diff = correct_pct - incorrect_pct

        print(f"{name:<25} {correct_pct:>9.1f}% {incorrect_pct:>9.1f}% {diff:>+9.1f}%")


# =============================================================================
# 3. ENTROPY-IMPACT CORRELATION (Paper-inspired)
# =============================================================================

def analyze_entropy_correlation(anchors, model_name):
    """
    Analyze correlation between attention entropy and reasoning impact.

    Inspired by Thought Branches paper: internal state markers correlate with quality.
    """
    print(f"\n{'='*60}")
    print(f"ENTROPY-IMPACT CORRELATION: {model_name}")
    print('='*60)

    data = []
    for anchor in anchors:
        entropy = anchor.get('attention_entropy')
        focus = anchor.get('attention_focus_score')
        delta = anchor.get('prob_delta', 0)

        if entropy is not None and focus is not None:
            try:
                entropy = float(entropy)
                focus = float(focus)
                if not np.isnan(entropy) and not np.isnan(focus):
                    data.append({
                        'entropy': entropy,
                        'focus': focus,
                        'delta': delta,
                        'abs_delta': abs(delta),
                        'is_correct': anchor.get('is_positive', True)
                    })
            except (TypeError, ValueError):
                continue

    if len(data) < 5:
        print("Insufficient data for correlation analysis")
        return

    entropies = np.array([d['entropy'] for d in data])
    focuses = np.array([d['focus'] for d in data])
    deltas = np.array([d['delta'] for d in data])
    abs_deltas = np.array([d['abs_delta'] for d in data])

    print(f"\nCorrelations (n={len(data)}):")

    # Key correlations
    r, p = stats.pearsonr(entropies, abs_deltas)
    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  Entropy vs |prob_delta|: r={r:+.3f}, p={p:.4f} {sig}")

    r, p = stats.pearsonr(focuses, abs_deltas)
    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  Focus vs |prob_delta|:   r={r:+.3f}, p={p:.4f} {sig}")

    r, p = stats.pearsonr(entropies, deltas)
    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  Entropy vs prob_delta:   r={r:+.3f}, p={p:.4f} {sig}")

    # High impact vs low impact comparison
    median_impact = np.median(abs_deltas)
    high_impact_entropy = [d['entropy'] for d in data if d['abs_delta'] > median_impact]
    low_impact_entropy = [d['entropy'] for d in data if d['abs_delta'] <= median_impact]

    if len(high_impact_entropy) >= 2 and len(low_impact_entropy) >= 2:
        t, p = stats.ttest_ind(high_impact_entropy, low_impact_entropy)
        print(f"\nHigh vs Low impact anchors:")
        print(f"  High impact entropy: {np.mean(high_impact_entropy):.4f} (n={len(high_impact_entropy)})")
        print(f"  Low impact entropy:  {np.mean(low_impact_entropy):.4f} (n={len(low_impact_entropy)})")
        print(f"  t={t:.3f}, p={p:.4f}")


# =============================================================================
# 4. BRANCH ANALYSIS (Paper-inspired: per-query reasoning paths)
# =============================================================================

def analyze_branches(anchors, model_name):
    """
    Analyze thought anchors grouped by query (reasoning branches).

    Inspired by Thought Branches paper: resampling reveals structure.
    Each query may have multiple anchors = multiple pivotal points in reasoning.
    """
    print(f"\n{'='*60}")
    print(f"BRANCH ANALYSIS (Per-Query): {model_name}")
    print('='*60)

    # Group anchors by query
    by_query = defaultdict(list)
    for anchor in anchors:
        query = anchor.get('query', '')[:100]  # Truncate for grouping
        by_query[query].append(anchor)

    num_queries = len(by_query)
    anchors_per_query = [len(v) for v in by_query.values()]

    print(f"\nQueries analyzed: {num_queries}")
    print(f"Anchors per query: mean={np.mean(anchors_per_query):.1f}, max={np.max(anchors_per_query)}")

    # Analyze queries with multiple anchors (multiple branch points)
    multi_anchor_queries = [(q, a) for q, a in by_query.items() if len(a) > 1]

    print(f"\nQueries with multiple branch points: {len(multi_anchor_queries)}")

    # For queries with multiple anchors, analyze the trajectory
    if multi_anchor_queries:
        print(f"\nBranch point analysis:")

        all_first_deltas = []
        all_last_deltas = []

        for query, query_anchors in multi_anchor_queries[:5]:  # Show first 5
            # Sort by sentence_id to get temporal order
            sorted_anchors = sorted(query_anchors, key=lambda x: x.get('sentence_id', 0))

            deltas = [a.get('prob_delta', 0) for a in sorted_anchors]
            correctness = ["✓" if a.get('is_positive', True) else "✗" for a in sorted_anchors]

            if deltas:
                all_first_deltas.append(deltas[0])
                all_last_deltas.append(deltas[-1])

            delta_str = " → ".join([f"{d:+.2f}" for d in deltas])
            corr_str = " → ".join(correctness)

            print(f"\n  Query: \"{query[:50]}...\"")
            print(f"    Deltas: {delta_str}")
            print(f"    Correct: {corr_str}")

        # Statistical comparison of first vs last anchor in reasoning
        if len(all_first_deltas) >= 3:
            print(f"\n  First vs Last anchor in multi-branch queries:")
            print(f"    First anchor mean delta: {np.mean(all_first_deltas):+.3f}")
            print(f"    Last anchor mean delta:  {np.mean(all_last_deltas):+.3f}")


# =============================================================================
# 5. CONTRASTIVE ANALYSIS (Paper-inspired)
# =============================================================================

def contrastive_analysis(anchors, model_name):
    """
    Contrastive analysis between correct and incorrect reasoning.

    Key insight from paper: comparing success/failure branches reveals
    systematic differences in model computation.
    """
    print(f"\n{'='*60}")
    print(f"CONTRASTIVE ANALYSIS: {model_name}")
    print('='*60)

    correct = [a for a in anchors if a.get('is_positive', True)]
    incorrect = [a for a in anchors if not a.get('is_positive', True)]

    print(f"\nCorrect: {len(correct)}, Incorrect: {len(incorrect)}")

    if not correct or not incorrect:
        print("Need both correct and incorrect anchors for contrastive analysis")
        return

    # Compare various metrics
    metrics_to_compare = [
        ('prob_delta', 'Prob Delta'),
        ('attention_entropy', 'Attention Entropy'),
        ('attention_focus_score', 'Focus Score'),
    ]

    print(f"\n{'Metric':<20} {'Correct':>12} {'Incorrect':>12} {'p-value':>10}")
    print("-" * 54)

    for field, name in metrics_to_compare:
        correct_vals = []
        incorrect_vals = []

        for a in correct:
            val = a.get(field)
            if val is not None:
                try:
                    val = float(val)
                    if not np.isnan(val):
                        correct_vals.append(val)
                except (TypeError, ValueError):
                    pass

        for a in incorrect:
            val = a.get(field)
            if val is not None:
                try:
                    val = float(val)
                    if not np.isnan(val):
                        incorrect_vals.append(val)
                except (TypeError, ValueError):
                    pass

        if len(correct_vals) >= 2 and len(incorrect_vals) >= 2:
            t, p = stats.ttest_ind(correct_vals, incorrect_vals)
            sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{name:<20} {np.mean(correct_vals):>+12.4f} {np.mean(incorrect_vals):>+12.4f} {p:>9.4f} {sig}")
        else:
            print(f"{name:<20} {'N/A':>12} {'N/A':>12} {'N/A':>10}")

    # Sentence length comparison
    correct_lens = [len(a.get('sentence', '')) for a in correct]
    incorrect_lens = [len(a.get('sentence', '')) for a in incorrect]

    t, p = stats.ttest_ind(correct_lens, incorrect_lens)
    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{'Sentence Length':<20} {np.mean(correct_lens):>12.1f} {np.mean(incorrect_lens):>12.1f} {p:>9.4f} {sig}")

    # Position in reasoning (sentence_id)
    correct_pos = [a.get('sentence_id', 0) for a in correct]
    incorrect_pos = [a.get('sentence_id', 0) for a in incorrect]

    if correct_pos and incorrect_pos:
        t, p = stats.ttest_ind(correct_pos, incorrect_pos)
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{'Position (sent_id)':<20} {np.mean(correct_pos):>12.1f} {np.mean(incorrect_pos):>12.1f} {p:>9.4f} {sig}")


# =============================================================================
# 6. FAILURE MODE DEEP DIVE
# =============================================================================

def analyze_failure_modes(anchors, model_name):
    """Detailed analysis of failure modes and their characteristics."""
    print(f"\n{'='*60}")
    print(f"FAILURE MODE ANALYSIS: {model_name}")
    print('='*60)

    failures = [a for a in anchors if not a.get('is_positive', True)]

    if not failures:
        print("No failures to analyze")
        return

    print(f"\nTotal failures: {len(failures)}")

    # Group by failure mode
    by_mode = defaultdict(list)
    for anchor in failures:
        mode = anchor.get('failure_mode', 'unknown')
        by_mode[mode].append(anchor)

    print(f"\nFailure modes breakdown:")
    for mode, items in sorted(by_mode.items(), key=lambda x: -len(x[1])):
        avg_delta = np.mean([a.get('prob_delta', 0) for a in items])
        avg_entropy = np.mean([a.get('attention_entropy', 0) or 0 for a in items])

        print(f"\n  {mode}: {len(items)} failures")
        print(f"    Avg prob_delta: {avg_delta:+.3f}")
        print(f"    Avg entropy: {avg_entropy:.4f}")

        # Show example sentence
        example = items[0]
        sentence = example.get('sentence', '')[:80]
        print(f"    Example: \"{sentence}...\"")

        # Show correction if available
        correction = example.get('correction_suggestion')
        if correction:
            print(f"    Correction: {correction}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_branch_analysis(qwen_anchors, deepseek_anchors):
    """Visualize branch analysis results."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (name, anchors) in enumerate([("Qwen3-0.6B", qwen_anchors), ("DeepSeek-R1-1.5B", deepseek_anchors)]):
        if not anchors:
            continue

        # Prob delta distribution
        ax = axes[0, idx]
        deltas = [a.get('prob_delta', 0) for a in anchors]
        correct_deltas = [a.get('prob_delta', 0) for a in anchors if a.get('is_positive', True)]
        incorrect_deltas = [a.get('prob_delta', 0) for a in anchors if not a.get('is_positive', True)]

        ax.hist(correct_deltas, bins=15, alpha=0.7, label='Correct', color='green')
        ax.hist(incorrect_deltas, bins=15, alpha=0.7, label='Incorrect', color='red')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Prob Delta')
        ax.set_ylabel('Count')
        ax.set_title(f'{name}: Prob Delta Distribution')
        ax.legend()

        # Entropy vs Impact scatter
        ax = axes[1, idx]
        data_points = []
        for a in anchors:
            entropy = a.get('attention_entropy')
            delta = a.get('prob_delta', 0)
            is_correct = a.get('is_positive', True)
            if entropy is not None:
                try:
                    entropy = float(entropy)
                    if not np.isnan(entropy):
                        data_points.append((entropy, abs(delta), is_correct))
                except:
                    pass

        if data_points:
            correct_pts = [(e, d) for e, d, c in data_points if c]
            incorrect_pts = [(e, d) for e, d, c in data_points if not c]

            if correct_pts:
                ax.scatter([p[0] for p in correct_pts], [p[1] for p in correct_pts],
                          alpha=0.7, label='Correct', color='green', s=50)
            if incorrect_pts:
                ax.scatter([p[0] for p in incorrect_pts], [p[1] for p in incorrect_pts],
                          alpha=0.7, label='Incorrect', color='red', s=50)

            ax.set_xlabel('Attention Entropy')
            ax.set_ylabel('|Prob Delta| (Impact)')
            ax.set_title(f'{name}: Entropy vs Impact')
            ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'deep_analysis.png', dpi=150)
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'deep_analysis.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("DEEP ANALYSIS OF THOUGHT ANCHORS")
    print("Inspired by 'Thought Branches' (arXiv:2510.27484)")
    print("="*60)

    qwen_exists = QWEN_FILE.exists()
    deepseek_exists = DEEPSEEK_FILE.exists()

    if not qwen_exists and not deepseek_exists:
        print("\nNo data files found!")
        print(f"Expected: {QWEN_FILE}")
        print(f"Expected: {DEEPSEEK_FILE}")
        return

    qwen_anchors = load_anchors(QWEN_FILE) if qwen_exists else []
    deepseek_anchors = load_anchors(DEEPSEEK_FILE) if deepseek_exists else []

    print(f"\nLoaded: Qwen3={len(qwen_anchors)}, DeepSeek={len(deepseek_anchors)}")

    # Run analyses for each model
    for name, anchors in [("Qwen3-0.6B", qwen_anchors), ("DeepSeek-R1-1.5B", deepseek_anchors)]:
        if anchors:
            analyze_prob_deltas(anchors, name)
            analyze_sentence_patterns(anchors, name)
            analyze_entropy_correlation(anchors, name)
            analyze_branches(anchors, name)
            contrastive_analysis(anchors, name)
            analyze_failure_modes(anchors, name)

    # Generate visualizations
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print('='*60)
    plot_branch_analysis(qwen_anchors, deepseek_anchors)

    print("\n" + "="*60)
    print("DEEP ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
