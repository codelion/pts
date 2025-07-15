# Thought Anchors Comparative Analysis: Qwen3 vs DeepSeek-R1

## Overview

This repository contains a comprehensive comparative study of reasoning patterns in Qwen3 and DeepSeek-R1 language models using the thought anchors methodology. The analysis reveals surprising insights about how different models approach reasoning tasks.

## Key Findings

### 🎯 Counter-Intuitive Discovery
**DeepSeek-R1 outperforms Qwen3** on reasoning consistency metrics:
- Higher average probability delta (0.408 vs 0.278)
- More consistent positive impacts (82.7% vs 71.6%)
- Superior risk-adjusted quality score (0.845 vs 0.466)

### 🧠 Reasoning Strategy Differences
- **DeepSeek-R1**: Conservative, consistent, focused failures
- **Qwen3**: Experimental, varied, diverse failure modes

### 🔍 Mechanistic Insights
- Different cognitive granularity (sentence lengths)
- Distinct risk-reward profiles
- Unique failure mode patterns

## Files Structure

```
thought_anchors_analysis/
├── README.md                     # This file
├── huggingface_blog_article.md   # Complete HuggingFace blog article
├── data_analysis.py              # Main analysis script
├── analysis_notebook.py          # Detailed analysis notebook
├── data/
│   └── analysis_results.json     # Comprehensive analysis results
└── images/
    ├── probability_delta_comparison.png
    ├── failure_mode_comparison.png
    ├── embedding_clusters.png
    └── reasoning_quality_evolution.png
```

## Quick Start

1. **Install dependencies:**
```bash
pip install datasets matplotlib seaborn plotly umap-learn scikit-learn
```

2. **Run the analysis:**
```bash
python data_analysis.py
```

3. **View detailed insights:**
```bash
python analysis_notebook.py
```

## Key Visualizations

### 📊 Probability Delta Distributions
Shows how the models differ in reasoning impact patterns.

### 🚨 Failure Mode Comparison
Reveals that Qwen3 has more diverse failure types (3 vs 1).

### 🎯 Reasoning Quality Evolution
Demonstrates DeepSeek-R1's superior consistency.

### 🔮 Embedding Clusters
UMAP visualization showing distinct reasoning patterns.

## Datasets Used

- **Qwen3**: [codelion/Qwen3-0.6B-pts-thought-anchors](https://huggingface.co/datasets/codelion/Qwen3-0.6B-pts-thought-anchors)
- **DeepSeek-R1**: [codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts-thought-anchors](https://huggingface.co/datasets/codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts-thought-anchors)

## Analysis Methodology

1. **Data Collection**: Using PTS library for thought anchor generation
2. **Statistical Analysis**: Probability delta distributions, failure modes
3. **Semantic Analysis**: 384-dimensional embeddings and clustering
4. **Risk Assessment**: Risk-adjusted quality metrics
5. **Case Studies**: Detailed examination of impactful examples