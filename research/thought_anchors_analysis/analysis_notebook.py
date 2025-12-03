#!/usr/bin/env python3
"""
Comprehensive Analysis Notebook: Thought Anchors Comparative Study
================================================================

This notebook contains a detailed analysis of thought anchors from Qwen3 and DeepSeek-R1,
revealing insights about how these models reason differently.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load the analysis results
with open('data/analysis_results.json', 'r') as f:
    results = json.load(f)

print("=" * 60)
print("THOUGHT ANCHORS COMPARATIVE ANALYSIS")
print("Qwen3 vs DeepSeek-R1: Understanding Model Reasoning Evolution")
print("=" * 60)

# ===== EXECUTIVE SUMMARY =====
print("\n📊 EXECUTIVE SUMMARY")
print("-" * 40)

qwen3_stats = results['basic_stats']['qwen3']
deepseek_stats = results['basic_stats']['deepseek']

print(f"Dataset Sizes:")
print(f"  • Qwen3: {qwen3_stats['total_anchors']} thought anchors from {qwen3_stats['unique_queries']} queries")
print(f"  • DeepSeek-R1: {deepseek_stats['total_anchors']} thought anchors from {deepseek_stats['unique_queries']} queries")

print(f"\nKey Insight: SURPRISING COUNTER-INTUITIVE FINDING!")
print(f"  • DeepSeek-R1 has HIGHER average prob_delta: {deepseek_stats['avg_prob_delta']:.3f}")
print(f"  • Qwen3 has LOWER average prob_delta: {qwen3_stats['avg_prob_delta']:.3f}")
print(f"  • This suggests DeepSeek-R1 may have more impactful reasoning steps")

print(f"\nReasoning Quality Distribution:")
print(f"  • Qwen3 positive ratio: {qwen3_stats['positive_ratio']:.3f}")
print(f"  • DeepSeek-R1 positive ratio: {deepseek_stats['positive_ratio']:.3f}")
print(f"  • DeepSeek-R1 shows more consistently positive impacts")

# ===== FAILURE MODE ANALYSIS =====
print("\n🚨 FAILURE MODE ANALYSIS")
print("-" * 40)

qwen3_failures = results['failure_modes']['qwen3']
deepseek_failures = results['failure_modes']['deepseek']

print("Failure Mode Diversity:")
print(f"  • Qwen3 failure types: {len(qwen3_failures)} types")
print(f"    - logical_error: {qwen3_failures.get('logical_error', 0)}")
print(f"    - computational_mistake: {qwen3_failures.get('computational_mistake', 0)}")
print(f"    - missing_step: {qwen3_failures.get('missing_step', 0)}")

print(f"  • DeepSeek-R1 failure types: {len(deepseek_failures)} types")
print(f"    - logical_error: {deepseek_failures.get('logical_error', 0)}")

print(f"\nKey Insight: FAILURE MODE COMPLEXITY")
print(f"  • Qwen3 shows more diverse failure modes (3 types vs 1 type)")
print(f"  • This could indicate more complex reasoning attempts")
print(f"  • DeepSeek-R1 failures are more focused on logical errors")

# ===== REASONING QUALITY DEEP DIVE =====
print("\n🧠 REASONING QUALITY DEEP DIVE")
print("-" * 40)

qwen3_quality = results['reasoning_quality']['qwen3']
deepseek_quality = results['reasoning_quality']['deepseek']

print("Distribution Statistics:")
print(f"  Qwen3:")
print(f"    - Mean: {qwen3_quality['mean']:.3f}")
print(f"    - Median: {qwen3_quality['median']:.3f}")
print(f"    - Q25: {qwen3_quality['q25']:.3f}")
print(f"    - Q75: {qwen3_quality['q75']:.3f}")

print(f"  DeepSeek-R1:")
print(f"    - Mean: {deepseek_quality['mean']:.3f}")
print(f"    - Median: {deepseek_quality['median']:.3f}")
print(f"    - Q25: {deepseek_quality['q25']:.3f}")
print(f"    - Q75: {deepseek_quality['q75']:.3f}")

print(f"\nKey Insight: REASONING STRATEGY DIFFERENCES")
print(f"  • DeepSeek-R1 has higher Q25 (0.5 vs -0.5) - more consistent positive impact")
print(f"  • Qwen3 has wider distribution - more extreme positive and negative impacts")
print(f"  • This suggests different reasoning strategies:")
print(f"    - DeepSeek-R1: More conservative, consistent reasoning")
print(f"    - Qwen3: More experimental, varied reasoning attempts")

# ===== CALCULATE ADVANCED METRICS =====
print("\n📈 ADVANCED METRICS")
print("-" * 40)

# Calculate risk-adjusted reasoning quality
qwen3_dist = np.array(qwen3_quality['distribution'])
deepseek_dist = np.array(deepseek_quality['distribution'])

def calculate_risk_metrics(distribution, name):
    """Calculate risk-adjusted metrics for reasoning quality."""
    dist = np.array(distribution)
    
    # Basic metrics
    mean_impact = np.mean(dist)
    volatility = np.std(dist)
    
    # Risk metrics
    downside_risk = np.std(dist[dist < 0]) if len(dist[dist < 0]) > 0 else 0
    upside_potential = np.mean(dist[dist > 0.5]) if len(dist[dist > 0.5]) > 0 else 0
    
    # Sharpe-like ratio for reasoning (higher is better)
    risk_adjusted_quality = mean_impact / volatility if volatility > 0 else 0
    
    print(f"  {name}:")
    print(f"    • Mean Impact: {mean_impact:.3f}")
    print(f"    • Volatility: {volatility:.3f}")
    print(f"    • Downside Risk: {downside_risk:.3f}")
    print(f"    • Upside Potential: {upside_potential:.3f}")
    print(f"    • Risk-Adjusted Quality: {risk_adjusted_quality:.3f}")
    
    return {
        'mean_impact': mean_impact,
        'volatility': volatility,
        'downside_risk': downside_risk,
        'upside_potential': upside_potential,
        'risk_adjusted_quality': risk_adjusted_quality
    }

qwen3_risk = calculate_risk_metrics(qwen3_dist, "Qwen3")
deepseek_risk = calculate_risk_metrics(deepseek_dist, "DeepSeek-R1")

print(f"\nRisk-Adjusted Reasoning Quality Comparison:")
if qwen3_risk['risk_adjusted_quality'] > deepseek_risk['risk_adjusted_quality']:
    print(f"  🏆 Qwen3 wins on risk-adjusted basis ({qwen3_risk['risk_adjusted_quality']:.3f} vs {deepseek_risk['risk_adjusted_quality']:.3f})")
else:
    print(f"  🏆 DeepSeek-R1 wins on risk-adjusted basis ({deepseek_risk['risk_adjusted_quality']:.3f} vs {qwen3_risk['risk_adjusted_quality']:.3f})")

# ===== MECHANISTIC INSIGHTS =====
print("\n🔬 MECHANISTIC INTERPRETABILITY INSIGHTS")
print("-" * 40)

print("What This Tells Us About Model Cognition:")
print(f"  1. REASONING APPROACH EVOLUTION:")
print(f"     • DeepSeek-R1: More conservative, consistent reasoning")
print(f"     • Qwen3: More experimental, higher variance approach")

print(f"  2. FAILURE MODE SOPHISTICATION:")
print(f"     • DeepSeek-R1: Focused failures (logical errors)")
print(f"     • Qwen3: Diverse failures (logical, computational, missing steps)")
print(f"     • Suggests Qwen3 attempts more complex reasoning chains")

print(f"  3. COGNITIVE ARCHITECTURE DIFFERENCES:")
print(f"     • DeepSeek-R1: Shorter average sentences ({deepseek_stats['avg_sentence_length']:.1f} chars)")
print(f"     • Qwen3: Longer average sentences ({qwen3_stats['avg_sentence_length']:.1f} chars)")
print(f"     • Indicates different reasoning granularity")

print(f"  4. IMPLICATIONS FOR AI SAFETY:")
print(f"     • DeepSeek-R1's consistency may be safer for deployment")
print(f"     • Qwen3's experimentation may lead to more breakthroughs")
print(f"     • Both approaches have merits for different use cases")

# ===== PRACTICAL RECOMMENDATIONS =====
print("\n💡 PRACTICAL RECOMMENDATIONS")
print("-" * 40)

print("For AI Practitioners:")
print(f"  • Use DeepSeek-R1 for: Reliable, consistent reasoning tasks")
print(f"  • Use Qwen3 for: Creative problem-solving, exploration tasks")
print(f"  • Monitor failure modes: Qwen3 needs more diverse safety measures")

print("For Researchers:")
print(f"  • Study DeepSeek-R1's consistency mechanisms")
print(f"  • Investigate Qwen3's experimental reasoning patterns")
print(f"  • Develop hybrid approaches combining both strengths")

print("For Future Model Development:")
print(f"  • Balance consistency vs. exploration in reasoning")
print(f"  • Design failure mode detection for diverse error types")
print(f"  • Consider reasoning granularity in architecture design")

print("\n=" * 60)
print("ANALYSIS COMPLETE - See images/ directory for visualizations")
print("=" * 60)

# ===== GENERATE DETAILED CASE STUDIES =====
print("\n📖 DETAILED CASE STUDIES")
print("-" * 40)

# Load original datasets for case studies
print("Loading original datasets for case studies...")
qwen3_dataset = load_dataset("codelion/Qwen3-0.6B-pts-thought-anchors")
deepseek_dataset = load_dataset("codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts-thought-anchors")

qwen3_df = pd.DataFrame(qwen3_dataset['train'])
deepseek_df = pd.DataFrame(deepseek_dataset['train'])

# Find most impactful positive and negative examples
print("\nFinding most impactful examples...")

# Most positive impact examples
qwen3_best = qwen3_df.loc[qwen3_df['prob_delta'].idxmax()]
deepseek_best = deepseek_df.loc[deepseek_df['prob_delta'].idxmax()]

print(f"\n🌟 MOST IMPACTFUL POSITIVE EXAMPLES:")
print(f"  Qwen3 (prob_delta = {qwen3_best['prob_delta']:.3f}):")
print(f"    Query: {qwen3_best['query'][:100]}...")
print(f"    Critical sentence: {qwen3_best['sentence'][:100]}...")
print(f"    Failure mode: {qwen3_best.get('failure_mode', 'None')}")

print(f"  DeepSeek-R1 (prob_delta = {deepseek_best['prob_delta']:.3f}):")
print(f"    Query: {deepseek_best['query'][:100]}...")
print(f"    Critical sentence: {deepseek_best['sentence'][:100]}...")
print(f"    Failure mode: {deepseek_best.get('failure_mode', 'None')}")

# Most negative impact examples
qwen3_worst = qwen3_df.loc[qwen3_df['prob_delta'].idxmin()]
deepseek_worst = deepseek_df.loc[deepseek_df['prob_delta'].idxmin()]

print(f"\n⚠️  MOST IMPACTFUL NEGATIVE EXAMPLES:")
print(f"  Qwen3 (prob_delta = {qwen3_worst['prob_delta']:.3f}):")
print(f"    Query: {qwen3_worst['query'][:100]}...")
print(f"    Critical sentence: {qwen3_worst['sentence'][:100]}...")
print(f"    Failure mode: {qwen3_worst.get('failure_mode', 'None')}")

print(f"  DeepSeek-R1 (prob_delta = {deepseek_worst['prob_delta']:.3f}):")
print(f"    Query: {deepseek_worst['query'][:100]}...")
print(f"    Critical sentence: {deepseek_worst['sentence'][:100]}...")
print(f"    Failure mode: {deepseek_worst.get('failure_mode', 'None')}")

print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
print(f"Next step: Create HuggingFace blog article with these insights")