---
title: PTS Visualizer
emoji: 🔬
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
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
datasets:
  - codelion/Qwen3-0.6B-pts
  - codelion/Qwen3-0.6B-pts-thought-anchors
  - codelion/Qwen3-0.6B-pts-steering-vectors
  - codelion/Qwen3-0.6B-pts-dpo-pairs
  - codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts
  - codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts-thought-anchors
  - codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts-steering-vectors
  - codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts-dpo-pairs
---

# PTS Visualizer — Pivotal Token Search

Interactive visualization for **Pivotal Token Search**: the reasoning events that
shift a language model's probability of solving a task, at three representational
scales — latent workspace meta-tokens, emitted pivotal tokens, and sentence-level
thought anchors — as a single kind of object.

## Views

### Overview
Per-scale event counts, causal-link count, and separate distributions for
emitted probability deltas and latent readout scores (they are different
quantities and are never binned together).

### Event Explorer
Every event in context, filterable by scale, category, valence, score, and (for
latent events) layer. Latent readouts are shown as readout scores, never as
probability deltas.

### Causal Event Graph
The latent → token → sentence → outcome graph for a query, with nodes shaped by
scale and edges from the recorded causal links.

### Embedding Space
t-SNE of event embeddings, colored by category or impact.

### Reasoning Timeline
All scales on one shared generation axis: latent meta-tokens, pivotal tokens,
thought-anchor sentences, and the resulting success-probability curve — plus a
workspace heatmap of meta-token readout scores by position.

## Supported datasets

Loads any PTS dataset from the Hub (and legacy pivotal-token / thought-anchor /
steering-vector files, which are upgraded on the fly):

- `codelion/Qwen3-0.6B-pts`
- `codelion/Qwen3-0.6B-pts-thought-anchors`
- `codelion/Qwen3-0.6B-pts-steering-vectors`
- `codelion/Qwen3-0.6B-pts-dpo-pairs`
- `codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts`
- `codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts-thought-anchors`
- `codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts-steering-vectors`
- `codelion/DeepSeek-R1-Distill-Qwen-1.5B-pts-dpo-pairs`

Or upload your own JSONL files.

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
