#!/usr/bin/env python3
"""
Cross-Model Reasoning Circuit Analysis

Compares attention patterns and error types between Qwen3-0.6B and DeepSeek-R1-Distill-Qwen-1.5B
on verified thought anchors from GSM8K.

Incorporates ideas from "Thought Branches" paper (arXiv:2510.27484):
- Cross-model comparison of reasoning patterns
- Effect size calculations for robust comparison
- Reasoning efficiency metrics

Usage:
    python research/reasoning_circuits/scripts/compare_models.py
"""

import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Paths
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


def extract_metrics(anchors):
    """Extract key metrics from anchors."""
    metrics = {
        'entropy_correct': [],
        'entropy_incorrect': [],
        'focus_correct': [],
        'focus_incorrect': [],
        'verification_scores': [],
        'is_positive': [],
        'error_types': defaultdict(int),
        'prob_deltas': [],
    }

    for anchor in anchors:
        is_correct = anchor.get('is_positive', True)
        entropy = anchor.get('attention_entropy')
        focus = anchor.get('attention_focus_score')
        verification = anchor.get('verification_score', 0)

        # Skip None or NaN values
        if entropy is None or focus is None:
            continue
        try:
            entropy = float(entropy)
            focus = float(focus)
            if np.isnan(entropy) or np.isnan(focus):
                continue
        except (TypeError, ValueError):
            continue

        metrics['verification_scores'].append(verification)
        metrics['is_positive'].append(is_correct)
        metrics['prob_deltas'].append(anchor.get('prob_delta', 0))

        if is_correct:
            metrics['entropy_correct'].append(entropy)
            metrics['focus_correct'].append(focus)
        else:
            metrics['entropy_incorrect'].append(entropy)
            metrics['focus_incorrect'].append(focus)

            # Count error types
            failure_mode = anchor.get('failure_mode', 'unknown')
            if failure_mode:
                metrics['error_types'][failure_mode] += 1

            # Count arithmetic error operations
            for err in anchor.get('arithmetic_errors', []):
                op = err.get('operator', 'unknown')
                metrics['error_types'][f'arithmetic_{op}'] += 1

    return metrics


def plot_entropy_comparison(qwen_metrics, deepseek_metrics):
    """Plot entropy distributions for correct vs incorrect across models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Qwen
    ax = axes[0]
    if qwen_metrics['entropy_correct']:
        ax.hist(qwen_metrics['entropy_correct'], bins=20, alpha=0.7,
                label=f"Correct (n={len(qwen_metrics['entropy_correct'])})", color='green')
    if qwen_metrics['entropy_incorrect']:
        ax.hist(qwen_metrics['entropy_incorrect'], bins=20, alpha=0.7,
                label=f"Incorrect (n={len(qwen_metrics['entropy_incorrect'])})", color='red')
    ax.set_xlabel('Attention Entropy')
    ax.set_ylabel('Count')
    ax.set_title('Qwen3-0.6B: Entropy Distribution')
    ax.legend()

    # DeepSeek
    ax = axes[1]
    if deepseek_metrics['entropy_correct']:
        ax.hist(deepseek_metrics['entropy_correct'], bins=20, alpha=0.7,
                label=f"Correct (n={len(deepseek_metrics['entropy_correct'])})", color='green')
    if deepseek_metrics['entropy_incorrect']:
        ax.hist(deepseek_metrics['entropy_incorrect'], bins=20, alpha=0.7,
                label=f"Incorrect (n={len(deepseek_metrics['entropy_incorrect'])})", color='red')
    ax.set_xlabel('Attention Entropy')
    ax.set_ylabel('Count')
    ax.set_title('DeepSeek-R1-1.5B: Entropy Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'entropy_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'entropy_comparison.png'}")


def plot_verification_comparison(qwen_metrics, deepseek_metrics):
    """Plot verification score distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot comparison
    data = []
    labels = []

    if qwen_metrics['verification_scores']:
        # Filter out None values
        qwen_correct = [v for v, p in zip(qwen_metrics['verification_scores'],
                                          qwen_metrics['is_positive']) if p and v is not None]
        qwen_incorrect = [v for v, p in zip(qwen_metrics['verification_scores'],
                                            qwen_metrics['is_positive']) if not p and v is not None]
        if qwen_correct:
            data.append(qwen_correct)
            labels.append('Qwen\nCorrect')
        if qwen_incorrect:
            data.append(qwen_incorrect)
            labels.append('Qwen\nIncorrect')

    if deepseek_metrics['verification_scores']:
        # Filter out None values
        ds_correct = [v for v, p in zip(deepseek_metrics['verification_scores'],
                                        deepseek_metrics['is_positive']) if p and v is not None]
        ds_incorrect = [v for v, p in zip(deepseek_metrics['verification_scores'],
                                          deepseek_metrics['is_positive']) if not p and v is not None]
        if ds_correct:
            data.append(ds_correct)
            labels.append('DeepSeek\nCorrect')
        if ds_incorrect:
            data.append(ds_incorrect)
            labels.append('DeepSeek\nIncorrect')

    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        colors = ['lightgreen', 'lightcoral', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)

        ax.set_ylabel('Verification Score')
        ax.set_title('Verification Score Distribution by Model and Correctness')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'verification_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'verification_comparison.png'}")


def plot_error_types(qwen_metrics, deepseek_metrics):
    """Plot error type comparison between models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Qwen errors
    ax = axes[0]
    if qwen_metrics['error_types']:
        errors = dict(qwen_metrics['error_types'])
        ax.barh(list(errors.keys()), list(errors.values()), color='steelblue')
        ax.set_xlabel('Count')
        ax.set_title('Qwen3-0.6B: Error Types')
    else:
        ax.text(0.5, 0.5, 'No errors detected', ha='center', va='center')
        ax.set_title('Qwen3-0.6B: Error Types')

    # DeepSeek errors
    ax = axes[1]
    if deepseek_metrics['error_types']:
        errors = dict(deepseek_metrics['error_types'])
        ax.barh(list(errors.keys()), list(errors.values()), color='darkorange')
        ax.set_xlabel('Count')
        ax.set_title('DeepSeek-R1-1.5B: Error Types')
    else:
        ax.text(0.5, 0.5, 'No errors detected', ha='center', va='center')
        ax.set_title('DeepSeek-R1-1.5B: Error Types')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'error_types_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'error_types_comparison.png'}")


def statistical_tests(qwen_metrics, deepseek_metrics):
    """Run statistical tests comparing models."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    # Test 1: Entropy difference between correct/incorrect within each model
    print("\n--- Entropy: Correct vs Incorrect (within model) ---")

    if qwen_metrics['entropy_correct'] and qwen_metrics['entropy_incorrect']:
        t_stat, p_val = stats.ttest_ind(qwen_metrics['entropy_correct'],
                                         qwen_metrics['entropy_incorrect'])
        print(f"Qwen3: t={t_stat:.3f}, p={p_val:.4f}")
        print(f"  Correct mean: {np.mean(qwen_metrics['entropy_correct']):.4f}")
        print(f"  Incorrect mean: {np.mean(qwen_metrics['entropy_incorrect']):.4f}")
    else:
        print("Qwen3: Insufficient data for test")

    if deepseek_metrics['entropy_correct'] and deepseek_metrics['entropy_incorrect']:
        t_stat, p_val = stats.ttest_ind(deepseek_metrics['entropy_correct'],
                                         deepseek_metrics['entropy_incorrect'])
        print(f"DeepSeek: t={t_stat:.3f}, p={p_val:.4f}")
        print(f"  Correct mean: {np.mean(deepseek_metrics['entropy_correct']):.4f}")
        print(f"  Incorrect mean: {np.mean(deepseek_metrics['entropy_incorrect']):.4f}")
    else:
        print("DeepSeek: Insufficient data for test")

    # Test 2: Cross-model comparison of entropy on correct computations
    print("\n--- Entropy: Qwen vs DeepSeek (correct only) ---")
    if qwen_metrics['entropy_correct'] and deepseek_metrics['entropy_correct']:
        t_stat, p_val = stats.ttest_ind(qwen_metrics['entropy_correct'],
                                         deepseek_metrics['entropy_correct'])
        print(f"t={t_stat:.3f}, p={p_val:.4f}")
        print(f"  Qwen mean: {np.mean(qwen_metrics['entropy_correct']):.4f}")
        print(f"  DeepSeek mean: {np.mean(deepseek_metrics['entropy_correct']):.4f}")
    else:
        print("Insufficient data for cross-model comparison")

    # Test 3: Verification score correlation with correctness
    print("\n--- Verification Score as Correctness Predictor ---")

    for name, metrics in [("Qwen3", qwen_metrics), ("DeepSeek", deepseek_metrics)]:
        if metrics['verification_scores'] and metrics['is_positive']:
            # Filter out None values from verification scores
            correct_scores = [v for v, p in zip(metrics['verification_scores'],
                                                metrics['is_positive']) if p and v is not None]
            incorrect_scores = [v for v, p in zip(metrics['verification_scores'],
                                                  metrics['is_positive']) if not p and v is not None]

            if correct_scores and incorrect_scores:
                t_stat, p_val = stats.ttest_ind(correct_scores, incorrect_scores)
                print(f"{name}: t={t_stat:.3f}, p={p_val:.4f}")
                print(f"  Correct mean: {np.mean(correct_scores):.4f}")
                print(f"  Incorrect mean: {np.mean(incorrect_scores):.4f}")
            else:
                print(f"{name}: Insufficient data")


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cross_model_comparison(qwen_anchors, deepseek_anchors):
    """
    Paper-inspired cross-model comparison.

    Compares reasoning patterns between models using multiple metrics
    with effect size calculations for robust interpretation.
    """
    print("\n" + "="*60)
    print("CROSS-MODEL COMPARISON (Paper-inspired)")
    print("="*60)

    if not qwen_anchors or not deepseek_anchors:
        print("Need data from both models for comparison")
        return

    # 1. Prob Delta Comparison (reasoning impact)
    print("\n--- Prob Delta (Reasoning Impact) ---")
    qwen_deltas = [a.get('prob_delta', 0) for a in qwen_anchors]
    ds_deltas = [a.get('prob_delta', 0) for a in deepseek_anchors]

    t, p = stats.ttest_ind(qwen_deltas, ds_deltas)
    d = cohens_d(qwen_deltas, ds_deltas)
    print(f"Qwen mean: {np.mean(qwen_deltas):+.4f} +/- {np.std(qwen_deltas):.4f}")
    print(f"DeepSeek mean: {np.mean(ds_deltas):+.4f} +/- {np.std(ds_deltas):.4f}")
    print(f"t={t:.3f}, p={p:.4f}, Cohen's d={d:.3f}")

    # 2. Reasoning Trace Length (verbosity)
    print("\n--- Reasoning Trace Length ---")
    qwen_trace_lens = [len(a.get('full_reasoning_trace', '')) for a in qwen_anchors]
    ds_trace_lens = [len(a.get('full_reasoning_trace', '')) for a in deepseek_anchors]

    t, p = stats.ttest_ind(qwen_trace_lens, ds_trace_lens)
    d = cohens_d(qwen_trace_lens, ds_trace_lens)
    print(f"Qwen mean: {np.mean(qwen_trace_lens):.0f} chars")
    print(f"DeepSeek mean: {np.mean(ds_trace_lens):.0f} chars")
    print(f"t={t:.3f}, p={p:.4f}, Cohen's d={d:.3f}")

    # 3. Anchors Per Query (branch complexity)
    print("\n--- Branch Complexity (Anchors per Query) ---")
    qwen_by_query = defaultdict(int)
    ds_by_query = defaultdict(int)

    for a in qwen_anchors:
        qwen_by_query[a.get('query', '')[:100]] += 1
    for a in deepseek_anchors:
        ds_by_query[a.get('query', '')[:100]] += 1

    qwen_counts = list(qwen_by_query.values())
    ds_counts = list(ds_by_query.values())

    print(f"Qwen: {len(qwen_by_query)} queries, {np.mean(qwen_counts):.1f} anchors/query")
    print(f"DeepSeek: {len(ds_by_query)} queries, {np.mean(ds_counts):.1f} anchors/query")

    # 4. Sentence Position Distribution (where anchors occur)
    print("\n--- Anchor Position in Reasoning ---")
    qwen_positions = [a.get('sentence_id', 0) for a in qwen_anchors]
    ds_positions = [a.get('sentence_id', 0) for a in deepseek_anchors]

    t, p = stats.ttest_ind(qwen_positions, ds_positions)
    print(f"Qwen mean position: {np.mean(qwen_positions):.1f}")
    print(f"DeepSeek mean position: {np.mean(ds_positions):.1f}")
    print(f"t={t:.3f}, p={p:.4f}")

    # 5. Accuracy by position (early vs late anchors)
    print("\n--- Accuracy by Anchor Position ---")
    for name, anchors in [("Qwen", qwen_anchors), ("DeepSeek", deepseek_anchors)]:
        positions = [a.get('sentence_id', 0) for a in anchors]
        if not positions:
            continue
        median_pos = np.median(positions)

        early = [a for a in anchors if a.get('sentence_id', 0) <= median_pos]
        late = [a for a in anchors if a.get('sentence_id', 0) > median_pos]

        early_acc = sum(1 for a in early if a.get('is_positive', True)) / len(early) if early else 0
        late_acc = sum(1 for a in late if a.get('is_positive', True)) / len(late) if late else 0

        print(f"{name}: Early={early_acc*100:.1f}% (n={len(early)}), Late={late_acc*100:.1f}% (n={len(late)})")


def plot_cross_model_summary(qwen_anchors, deepseek_anchors):
    """Create a summary comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Accuracy comparison
    ax = axes[0, 0]
    qwen_acc = sum(1 for a in qwen_anchors if a.get('is_positive', True)) / len(qwen_anchors) * 100 if qwen_anchors else 0
    ds_acc = sum(1 for a in deepseek_anchors if a.get('is_positive', True)) / len(deepseek_anchors) * 100 if deepseek_anchors else 0

    bars = ax.bar(['Qwen3-0.6B', 'DeepSeek-R1-1.5B'], [qwen_acc, ds_acc], color=['steelblue', 'darkorange'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Reasoning Accuracy')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, [qwen_acc, ds_acc]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%', ha='center')

    # 2. Entropy comparison (box plot)
    ax = axes[0, 1]
    qwen_entropy = [a.get('attention_entropy') for a in qwen_anchors if a.get('attention_entropy') is not None]
    ds_entropy = [a.get('attention_entropy') for a in deepseek_anchors if a.get('attention_entropy') is not None]

    # Filter valid values
    qwen_entropy = [e for e in qwen_entropy if e is not None and not np.isnan(float(e))]
    ds_entropy = [e for e in ds_entropy if e is not None and not np.isnan(float(e))]

    if qwen_entropy and ds_entropy:
        bp = ax.boxplot([qwen_entropy, ds_entropy], tick_labels=['Qwen3-0.6B', 'DeepSeek-R1-1.5B'], patch_artist=True)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('darkorange')
    ax.set_ylabel('Attention Entropy')
    ax.set_title('Attention Entropy Distribution')

    # 3. Prob delta distribution
    ax = axes[1, 0]
    qwen_deltas = [a.get('prob_delta', 0) for a in qwen_anchors]
    ds_deltas = [a.get('prob_delta', 0) for a in deepseek_anchors]

    ax.hist(qwen_deltas, bins=20, alpha=0.7, label='Qwen3-0.6B', color='steelblue')
    ax.hist(ds_deltas, bins=20, alpha=0.7, label='DeepSeek-R1-1.5B', color='darkorange')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Prob Delta')
    ax.set_ylabel('Count')
    ax.set_title('Reasoning Impact Distribution')
    ax.legend()

    # 4. Model comparison summary table
    ax = axes[1, 1]
    ax.axis('off')

    summary_data = [
        ['Metric', 'Qwen3-0.6B', 'DeepSeek-R1-1.5B'],
        ['Parameters', '0.6B', '1.5B'],
        ['Accuracy', f'{qwen_acc:.1f}%', f'{ds_acc:.1f}%'],
        ['Mean Entropy', f'{np.mean(qwen_entropy):.3f}' if qwen_entropy else 'N/A',
                        f'{np.mean(ds_entropy):.3f}' if ds_entropy else 'N/A'],
        ['Mean |Delta|', f'{np.mean(np.abs(qwen_deltas)):.3f}' if qwen_deltas else 'N/A',
                        f'{np.mean(np.abs(ds_deltas)):.3f}' if ds_deltas else 'N/A'],
        ['Total Anchors', str(len(qwen_anchors)), str(len(deepseek_anchors))],
    ]

    table = ax.table(cellText=summary_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#404040')
        table[(0, i)].set_text_props(color='white', weight='bold')

    ax.set_title('Model Comparison Summary', pad=20)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cross_model_summary.png', dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'cross_model_summary.png'}")


def print_summary(qwen_metrics, deepseek_metrics):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for name, metrics in [("Qwen3-0.6B", qwen_metrics), ("DeepSeek-R1-1.5B", deepseek_metrics)]:
        print(f"\n--- {name} ---")
        n_correct = len(metrics['entropy_correct'])
        n_incorrect = len(metrics['entropy_incorrect'])
        total = n_correct + n_incorrect

        print(f"Total anchors: {total}")
        print(f"  Correct: {n_correct} ({100*n_correct/total:.1f}%)" if total > 0 else "  Correct: 0")
        print(f"  Incorrect: {n_incorrect} ({100*n_incorrect/total:.1f}%)" if total > 0 else "  Incorrect: 0")

        if metrics['entropy_correct']:
            print(f"Entropy (correct): {np.mean(metrics['entropy_correct']):.4f} +/- {np.std(metrics['entropy_correct']):.4f}")
        if metrics['entropy_incorrect']:
            print(f"Entropy (incorrect): {np.mean(metrics['entropy_incorrect']):.4f} +/- {np.std(metrics['entropy_incorrect']):.4f}")

        if metrics['error_types']:
            print(f"Error types: {dict(metrics['error_types'])}")


def main():
    """Main analysis function."""
    print("Cross-Model Reasoning Circuit Analysis")
    print("="*60)

    # Check data files exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    qwen_exists = QWEN_FILE.exists()
    deepseek_exists = DEEPSEEK_FILE.exists()

    if not qwen_exists and not deepseek_exists:
        print(f"\nNo data files found!")
        print(f"Expected:")
        print(f"  - {QWEN_FILE}")
        print(f"  - {DEEPSEEK_FILE}")
        print(f"\nGenerate data using PTS CLI (see README.md)")
        return

    # Load data
    qwen_anchors = load_anchors(QWEN_FILE) if qwen_exists else []
    deepseek_anchors = load_anchors(DEEPSEEK_FILE) if deepseek_exists else []

    print(f"\nLoaded:")
    print(f"  Qwen3: {len(qwen_anchors)} anchors")
    print(f"  DeepSeek: {len(deepseek_anchors)} anchors")

    # Extract metrics
    qwen_metrics = extract_metrics(qwen_anchors) if qwen_anchors else {
        'entropy_correct': [], 'entropy_incorrect': [],
        'focus_correct': [], 'focus_incorrect': [],
        'verification_scores': [], 'is_positive': [],
        'error_types': defaultdict(int), 'prob_deltas': []
    }
    deepseek_metrics = extract_metrics(deepseek_anchors) if deepseek_anchors else {
        'entropy_correct': [], 'entropy_incorrect': [],
        'focus_correct': [], 'focus_incorrect': [],
        'verification_scores': [], 'is_positive': [],
        'error_types': defaultdict(int), 'prob_deltas': []
    }

    # Print summary
    print_summary(qwen_metrics, deepseek_metrics)

    # Statistical tests
    statistical_tests(qwen_metrics, deepseek_metrics)

    # Paper-inspired cross-model comparison
    cross_model_comparison(qwen_anchors, deepseek_anchors)

    # Generate plots
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)

    plot_entropy_comparison(qwen_metrics, deepseek_metrics)
    plot_verification_comparison(qwen_metrics, deepseek_metrics)
    plot_error_types(qwen_metrics, deepseek_metrics)
    plot_cross_model_summary(qwen_anchors, deepseek_anchors)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated figures:")
    print(f"  - {FIGURES_DIR / 'entropy_comparison.png'}")
    print(f"  - {FIGURES_DIR / 'verification_comparison.png'}")
    print(f"  - {FIGURES_DIR / 'error_types_comparison.png'}")
    print(f"  - {FIGURES_DIR / 'cross_model_summary.png'}")
    print("\nFor deeper analysis, run: python research/reasoning_circuits/scripts/deep_analysis.py")


if __name__ == "__main__":
    main()
