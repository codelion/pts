# PTS: Pivotal Token Search

A tool for discovering pivotal tokens in large language model generations and creating DPO datasets and steering vectors from them.

## Features

- Identifies pivotal tokens in language model generations
- Supports various dataset formats including GSM8k, MATH, and custom datasets
- Handles chain-of-thought reasoning output with `<think></think>` tags
- Extracts answers from common formats like GSM8k's #### pattern and LaTeX's \boxed{} notation

## What is Pivotal Token Search?

Pivotal Token Search (PTS) is a technique described in the [Phi-4 Technical Report](https://arxiv.org/abs/2412.08905) that identifies tokens in a language model's generation that significantly impact the probability of success for the task at hand. These "pivotal tokens" are decision points where the model's choice can dramatically alter the course of the solution.

Key features:
- Identifies tokens that significantly increase or decrease the probability of a successful generation
- Generates DPO (Direct Preference Optimization) pairs for fine-tuning
- Creates steering vectors for activation-based steering during inference

## Installation

```bash
git clone https://github.com/codelion/pts.git
cd pts
pip install -e .
```

## Quick Start

```bash
# Find pivotal tokens in a dataset and save to file
pts run --model="Qwen/Qwen3-0.6B" --dataset="codelion/optillmbench" --output-path="pivotal_tokens.jsonl"

# Convert pivotal tokens to DPO dataset
pts export --input-path="pivotal_tokens.jsonl" --format="dpo" --output-path="dpo_dataset.jsonl" --model="Qwen/Qwen3-0.6B" --find-rejected-tokens

# Convert pivotal tokens to steering vectors
pts export --input-path="pivotal_tokens.jsonl" --format="steering" --output-path="steering_vectors.jsonl" --model="Qwen/Qwen3-0.6B"

# Push dataset to Hugging Face (creates README by default)
pts push --input-path="dpo_dataset.jsonl" --hf-repo="codelion/pts-dpo-dataset" --model="Qwen/Qwen3-0.6B"
```

## Try Now

| Use Case | Dataset | Link |
|----------|----------|-------|
| Fine-tuning the model | dpo dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FggA9EQ1eFBjE0Qbsl0-EFzyWIxpdhlH?usp=sharing) |
| Optimizing the inference | steering vectors | [optillm](https://github.com/codelion/optillm) |

You can also check out the [datasets](https://huggingface.co/datasets?other=pts) and [models](https://huggingface.co/models?other=pts) created with pts.

## Core Concepts

### Pivotal Tokens

A pivotal token significantly changes the probability of success when it appears in a model's generation. By identifying these tokens, we can:
1. Understand where the model makes critical decisions
2. Create preference pairs for DPO fine-tuning
3. Extract activation vectors for steering during inference

### DPO Datasets

PTS creates high-quality DPO datasets by isolating the specific token-level choices that lead to success or failure. This allows for more targeted and effective fine-tuning compared to using entire sequences.

**Important:** When exporting to DPO format, you must provide a model using the `--model` parameter and enable the `--find-rejected-tokens` flag. This is necessary because DPO pairs require both a chosen token (the pivotal token that increases success probability) and a rejected token (an alternative token that decreases success probability).

### Steering Vectors

The activation patterns associated with pivotal tokens can be used to guide models during generation, encouraging them to follow successful reasoning paths.

## Dataset Field Customization

Different datasets use different field names for questions and answers. PTS automatically detects appropriate field names for common datasets, but you can also specify them manually:

```bash
pts run --model="Qwen/Qwen3-0.6B" --dataset="your-dataset" --query-key="question" --answer-key="answer"
```

For example:
- `codelion/optillmbench`: Uses "question" and "answer" fields
- Other datasets may use fields like:
  - "instruction"/"output"
  - "problem"/"solution" 
  - "prompt"/"canonical_solution"

If not specified, PTS will attempt to automatically detect the appropriate fields based on common naming patterns.

## Command Reference

### `pts run`

Find pivotal tokens in a dataset:

```bash
pts run --model="MODEL_NAME" --dataset="DATASET_NAME" [options]
```

Options:
- `--model`: Model to use for generation
- `--dataset`: Dataset to search (default: "codelion/optillmbench")
- `--config`: Dataset configuration name (if applicable, e.g., "main" for openai/gsm8k)
- `--output-path`: Path to save pivotal tokens (default: "pivotal_tokens.jsonl")
- `--query-key`: Key for question/instruction field in dataset (auto-detected if not specified)
- `--answer-key`: Key for answer/output field in dataset (auto-detected if not specified)
- `--prob-threshold`: Probability threshold for pivotal tokens (default: 0.2)
- `--temperature`: Sampling temperature (default: 0.6)
- `--top-p`: Top-p (nucleus) sampling parameter (default: 0.95)
- `--top-k`: Top-k sampling parameter (default: 20)
- `--min-p`: Min-p sampling parameter (default: 0.0)
- `--num-samples`: Number of samples for probability estimation (default: 10)
- `--max-pairs`: Maximum number of pairs to generate (default: 1000)

### `pts export`

Export pivotal tokens to different formats:

```bash
pts export --input-path="TOKENS_PATH" --format="FORMAT" [options]
```

Options:
- `--input-path`: Path to pivotal tokens file
- `--format`: Export format ("dpo" or "steering")
- `--output-path`: Path to save exported data
- `--model`: Model to use for extracting steering vectors (required for "steering" format)

### `pts push`

Push dataset to Hugging Face:

```bash
pts push --input-path="FILE_PATH" --hf-repo="USERNAME/REPO_NAME" [options]
```

Options:
- `--input-path`: Path to file to push
- `--hf-repo`: Hugging Face repository name
- `--private`: Make the repository private (default: False)
- `--no-readme`: Skip creating a README file (a README is created by default)
- `--model`: Model name to include in the README (optional)

## Examples

### Finding Pivotal Tokens with OptillmBench

```bash
pts run --model="Qwen/Qwen3-0.6B" \
    --dataset="codelion/optillmbench" \
    --output-path="optillm_pivotal_tokens.jsonl" \
    --prob-threshold=0.2 \
    --temperature=0.6 \
    --top-p=0.95 \
    --top-k=20 \
    --min-p=0.0
```

### Working with a Custom Dataset

```bash
pts run --model="Qwen/Qwen3-0.6B" \
    --dataset="my-custom-dataset" \
    --query-key="input_text" \
    --answer-key="target_text" \
    --output-path="custom_pivotal_tokens.jsonl" \
    --prob-threshold=0.2 \
    --temperature=0.6 \
    --top-p=0.95 \
    --top-k=20 \
    --min-p=0.0
```

### Working with a Dataset Requiring Configuration

```bash
pts run --model="Qwen/Qwen3-0.6B" \
    --dataset="openai/gsm8k" \
    --config="main" \
    --split="train" \
    --output-path="gsm8k_pivotal_tokens.jsonl" \
    --prob-threshold=0.2 \
    --temperature=0.6 \
    --max-examples=10
```

### Creating a DPO Dataset

```bash
# First find pivotal tokens
pts run --model="Qwen/Qwen3-0.6B" \
    --dataset="codelion/optillmbench" \
    --output-path="optillm_pivotal_tokens.jsonl" \
    --temperature=0.6 \
    --top-p=0.95 \
    --top-k=20 \
    --min-p=0.0

# Then export to DPO format - MUST provide a model and find-rejected-tokens flag
pts export --input-path="optillm_pivotal_tokens.jsonl" \
    --format="dpo" \
    --output-path="optillm_dpo_dataset.jsonl" \
    --model="Qwen/Qwen3-0.6B" \
    --find-rejected-tokens \
    --min-prob-delta=0.1
```

### Extracting Steering Vectors

```bash
pts export --input-path="pivotal_tokens.jsonl" \
    --format="steering" \
    --output-path="steering_vectors.jsonl" \
    --model="Qwen/Qwen3-0.6B" \
    --layer-nums=19,23,27
```

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
