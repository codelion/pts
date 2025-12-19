---
title: PTS Visualizer
emoji: 🔍
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - pts
  - pivotal-tokens
  - thought-anchors
  - llm-interpretability
  - reasoning
  - visualization
---

# PTS Visualizer

Interactive visualization platform for exploring **Pivotal Tokens**, **Thought Anchors**, and **Reasoning Circuits** in language models.

Inspired by [Neuronpedia](https://neuronpedia.org/), this tool helps researchers and practitioners understand how language models reason through complex tasks.

## Features

### 📊 Overview Dashboard
- Dataset statistics and distributions
- Quick summary of positive/negative impacts
- Category and pattern analysis

### 🔍 Token Explorer
- Highlight pivotal tokens in context
- Visualize probability changes before/after tokens
- Explore token-level impacts on success

### 🕸️ Reasoning Graph
- Interactive dependency graph for thought anchors
- Visualize causal relationships between reasoning steps
- Color-coded by impact (green = positive, red = negative)
- Node size indicates importance

### 🗺️ Embedding Space
- t-SNE visualization of sentence/token embeddings
- Color by category, pattern, or impact
- Explore clusters and patterns in reasoning

### ⚡ Circuit Tracer
- Step-by-step walkthrough of reasoning traces
- Probability progression chart
- Verification scores and error detection

## Supported Datasets

Load from HuggingFace Hub:
- `codelion/Qwen3-0.6B-pts` - Pivotal tokens
- `codelion/Qwen3-0.6B-pts-thought-anchors` - Thought anchors
- `codelion/Qwen3-0.6B-pts-steering-vectors` - Steering vectors
- `codelion/Qwen3-0.6B-pts-dpo-pairs` - DPO training pairs
- `codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts-thought-anchors`

Or upload your own JSONL files!

## How to Use

1. **Select a data source**: Choose HuggingFace Hub or upload a local file
2. **Load the dataset**: Click "Load Dataset"
3. **Explore**: Navigate through the tabs to visualize different aspects

## Local Development

```bash
# Clone the repository
git clone https://github.com/codelion/pts
cd pts/visualizer

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## Related Resources

- [PTS GitHub Repository](https://github.com/codelion/pts)
- [Pivotal Token Search Collection](https://huggingface.co/collections/codelion/pivotal-token-search)
- [OptiLLM](https://github.com/codelion/optillm) - Inference optimization library

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{pts,
  title = {PTS: Pivotal Token Search},
  author = {Asankhaya Sharma},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/codelion/pts}
}
```
