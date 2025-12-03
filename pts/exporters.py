"""
Exporter module for Pivotal Token Search.

This module provides utilities to export pivotal tokens into various formats
for downstream uses like DPO training and steering vectors.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Set, Union, Tuple
import random
import torch
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import defaultdict

from .storage import TokenStorage
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def generate_readme_content(file_type, model_name=None, dataset_info=None):
    """
    Generate README content for Hugging Face repositories.
    
    Args:
        file_type: Type of file ('dpo', 'steering', or 'tokens')
        model_name: Name of the model used
        dataset_info: Additional dataset information
        
    Returns:
        README content as a string
    """
    if file_type == "steering":
        content = f"""# PTS Steering Vectors Dataset

A dataset of activation-based steering vectors created using the Pivotal Token Search (PTS) technique.

## Details

- **Source:** Generated using the [PTS](https://github.com/codelion/pts) tool
- **Model:** {model_name or "Unknown"}

## Dataset Structure

This dataset contains:
- `steering_vectors.jsonl`: The main file with token-level steering vectors

## Usage

These steering vectors can be used for activation-based steering during inference to guide language models toward particular reasoning patterns.

### OptiLLM Integration

You can use these steering vectors with the open-source [OptiLLM](https://github.com/codelion/optillm) library for optimizing inference via the `autothink` approach. OptiLLM provides a proxy server that can apply steering techniques to improve model performance.

### Example Python Code

```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("{model_name or 'MODEL_NAME'}")
tokenizer = AutoTokenizer.from_pretrained("{model_name or 'MODEL_NAME'}")

# Load steering vectors directly from Hugging Face
from datasets import load_dataset
dataset = load_dataset("USERNAME/REPO_NAME")
vectors = [json.loads(example) for example in dataset["train"]]

# Define a hook to apply steering
def steering_hook(module, input, output):
    # Add steering vector to activation
    # Implementation depends on your specific use case
    return output

# Register hook on appropriate layer
model.transformer.h[LAYER_NUM].register_forward_hook(steering_hook)

# Generate text with steering
input_text = "Your prompt here"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=100)
result = tokenizer.decode(output[0])
print(result)
```
"""
    elif file_type == "thought_anchors":
        content = f"""---
language:
- en
tags:
- pts
- thought-anchors
- reasoning
- llm-analysis
- sentence-level-analysis
- pivotal-token-search
size_categories:
- n<1K
task_categories:
- other
pretty_name: "PTS Thought Anchors Dataset"
dataset_info:
  config_name: default
  features:
  - name: model_id
    dtype: string
  - name: query
    dtype: string
  - name: sentence
    dtype: string
  - name: sentence_id
    dtype: int64
  - name: prefix_context
    dtype: string
  - name: prob_with_sentence
    dtype: float64
  - name: prob_without_sentence
    dtype: float64
  - name: prob_delta
    dtype: float64
  - name: task_type
    dtype: string
  - name: suffix_context
    dtype: string
  - name: full_reasoning_trace
    dtype: string
  - name: sentence_embedding
    sequence: float64
  - name: alternatives_embeddings
    sequence:
      sequence: float64
  - name: causal_dependencies
    sequence: int64
  - name: causal_dependents
    sequence: int64
  - name: logical_relationship
    dtype: string
  - name: failure_mode
    dtype: string
  - name: error_type
    dtype: string
  - name: correction_suggestion
    dtype: string
  - name: verification_score
    dtype: float64
  - name: verification_method
    dtype: string
  - name: arithmetic_errors
    sequence:
      struct:
        expression: string
        stated: float64
        correct: float64
  - name: attention_entropy
    dtype: float64
  - name: attention_focus_score
    dtype: float64
  - name: importance_score
    dtype: float64
  - name: is_positive
    dtype: bool
  - name: sentence_category
    dtype: string
  - name: alternatives_tested
    sequence: string
  - name: dependency_sentences
    sequence: int64
  - name: dataset_id
    dtype: string
  - name: dataset_item_id
    dtype: string
  - name: timestamp
    dtype: timestamp[s]
---

# PTS Thought Anchors Dataset

A dataset of thought anchors - critical reasoning steps - identified using the Thought Anchors technique from the PTS tool.

## Details

- **Source:** Generated using the [PTS](https://github.com/codelion/pts) tool
- **Model:** {model_name or "Unknown"}
- **Tags:** `pts`, `thought-anchors`, `reasoning`, `llm-analysis`

## Dataset Structure

This dataset contains thought anchors identified from reasoning traces. Each anchor represents a sentence that significantly impacts the success probability of the reasoning process.

## Fields

Each thought anchor contains:

### Core Fields
- `model_id`: The model used to generate the reasoning trace
- `query`: The original problem/question that was processed
- `sentence`: The actual sentence that serves as a thought anchor
- `sentence_id`: Position of the sentence in the reasoning trace
- `prob_with_sentence`: Success probability when this sentence is included
- `prob_without_sentence`: Success probability when this sentence is replaced/removed
- `prob_delta`: Change in success probability (with - without)
- `importance_score`: Absolute impact score of this anchor
- `is_positive`: Whether this sentence helps (true) or hurts (false) success

### Context Fields
- `prefix_context`: All sentences that come before this one
- `suffix_context`: All sentences that come after this one
- `full_reasoning_trace`: Complete reasoning trace for context

### Semantic Analysis
- `sentence_embedding`: 384-dimensional vector representation of the sentence
- `alternatives_embeddings`: Vector representations of alternative sentences tested
- `alternatives_tested`: List of alternative sentences that were tested

### Dependency Analysis
- `causal_dependencies`: Sentence IDs this sentence logically depends on
- `causal_dependents`: Sentence IDs that depend on this sentence
- `logical_relationship`: Type of logical relationship ("premise", "conclusion", etc.)

### Failure Analysis (for negative anchors)
- `failure_mode`: Type of failure ("logical_error", "computational_mistake", etc.)
- `error_type`: More specific error classification
- `correction_suggestion`: How to improve the sentence

### Verification Fields (CRV-inspired)
- `verification_score`: Combined verification score (0.0 = wrong, 1.0 = correct)
- `verification_method`: Method used ("ground_truth", "attention", "combined")
- `arithmetic_errors`: List of detected arithmetic errors with expression, stated, and correct values
- `attention_entropy`: Attention distribution entropy (lower = more focused/confident)
- `attention_focus_score`: How concentrated the attention is on operands

### Classification
- `sentence_category`: Type of reasoning step ("plan_generation", "active_computation", etc.)
- `task_type`: Type of task being solved
- `dataset_id`: Source dataset identifier
- `dataset_item_id`: Specific item ID from the dataset
- `timestamp`: When the anchor was generated

## Usage

These thought anchors can be used for:
- Understanding which reasoning steps matter most
- Guiding inference systems to focus on critical reasoning steps
- Analyzing reasoning patterns in language models
- Building better reasoning evaluation metrics

### Example Usage

```python
from datasets import load_dataset

# Load thought anchors from Hugging Face
dataset = load_dataset("codelion/Qwen3-0.6B-pts-thought-anchors")
anchors = dataset['train']

# Find high-impact positive anchors
positive_anchors = anchors.filter(
    lambda x: x["is_positive"] and x["importance_score"] > 0.3
)

# Find planning-related sentences
planning_anchors = anchors.filter(
    lambda x: x["sentence_category"] == "plan_generation"
)

# Analyze failure modes for negative anchors
failure_analysis = {{}}
negative_anchors = anchors.filter(lambda x: not x["is_positive"] and x["failure_mode"])
for anchor in negative_anchors:
    mode = anchor["failure_mode"]
    failure_analysis[mode] = failure_analysis.get(mode, 0) + 1

print("Failure modes:", failure_analysis)
print(f"Found {{len(positive_anchors)}} positive anchors")
print(f"Found {{len(planning_anchors)}} planning anchors")

# Example: Access embeddings for similarity search
sample_anchor = anchors[0]
embedding = sample_anchor["sentence_embedding"]  # 384-dim vector
print(f"Embedding dimension: {{len(embedding)}}")
```

### Integration with Inference Systems

Thought anchors can be used to:
1. **Focus attention** on critical reasoning steps during generation
2. **Validate reasoning** by checking for presence of important anchor patterns
3. **Guide beam search** to prefer paths containing high-value anchor sentences
4. **Improve CoT prompting** by incorporating successful anchor patterns

### OptiLLM Integration

You can use these thought anchors with the open-source [OptiLLM](https://github.com/codelion/optillm) library for optimizing inference via the autothink approach. OptiLLM provides a proxy server that can apply thought anchor techniques to improve model reasoning performance by:

- **Guided reasoning**: Using positive anchors as templates for better reasoning paths
- **Quality monitoring**: Detecting negative anchor patterns to avoid poor reasoning
- **Adaptive prompting**: Incorporating successful anchor patterns into prompts
- **Real-time optimization**: Applying anchor insights during model inference

"""
    elif file_type == "dpo":
        content = f"""# PTS DPO Dataset

A Direct Preference Optimization (DPO) dataset created using the Pivotal Token Search (PTS) technique.

## Details

- **Source:** Generated using the [PTS](https://github.com/codelion/pts) tool
- **Model:** {model_name or "Unknown"}

## Format

Each example in the dataset consists of:
- `prompt`: The context leading up to the pivotal token
- `chosen`: The preferred token that increases success probability
- `rejected`: The alternative token that decreases success probability
- `metadata`: Additional information about the example

## Usage

This dataset can be used for fine-tuning language models with Direct Preference Optimization (DPO).

For a quick start, you can use our Google Colab notebook to fine-tune a model using this DPO dataset:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FggA9EQ1eFBjE0Qbsl0-EFzyWIxpdhlH?usp=sharing)

```python
from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("USERNAME/REPO_NAME")

# Use with your favorite DPO implementation
# Example with TRL library:
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    args=training_args,
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
    # ... other parameters
)

trainer.train()
```
"""
    else:  # tokens or default
        content = f"""# PTS Pivotal Tokens Dataset

A dataset of pivotal tokens discovered using the Pivotal Token Search (PTS) technique.

## Details

- **Source:** Generated using the [PTS](https://github.com/codelion/pts) tool
- **Model:** {model_name or "Unknown"}

## Format

Each token in the dataset includes:
- `query`: The original query that was processed
- `pivot_context`: The context leading up to the pivotal token
- `pivot_token`: The actual token that impacts success probability
- `prob_delta`: The change in success probability caused by the token
- Other metadata about the token

## Usage

These pivotal tokens can be used for creating DPO datasets or extracting steering vectors:

```bash
# Export to DPO format
pts export --input-path="pivotal_tokens.jsonl" --format="dpo" --output-path="dpo_dataset.jsonl" --model="MODEL_NAME" --find-rejected-tokens

# Extract steering vectors
pts export --input-path="pivotal_tokens.jsonl" --format="steering" --output-path="steering_vectors.jsonl" --model="MODEL_NAME"
```
"""
        
    # Add custom dataset info if provided
    if dataset_info:
        content += f"\n## Additional Information\n\n{dataset_info}\n"
        
    return content


class TokenExporter:
    """
    Exporter for pivotal tokens to various formats.
    """
    
    def __init__(self, token_storage: Optional[TokenStorage] = None, searcher=None):
        """
        Initialize the token exporter.
        
        Args:
            token_storage: Optional TokenStorage instance to export from
            searcher: Optional PivotalTokenSearcher instance for finding rejected tokens
        """
        self.token_storage = token_storage or TokenStorage()
        self.searcher = searcher
        
    def export_dpo_dataset(
        self,
        output_path: str,
        filter_criteria: Optional[Dict[str, Any]] = None,
        min_prob_delta: float = 0.1,
        balance_positive_negative: bool = True,
        max_pairs: Optional[int] = None,
        seed: int = 42,
        model_name: Optional[str] = None,
        save_tokens: bool = True,
        tokens_output_path: Optional[str] = None,
        num_candidates: int = 10,
        find_rejected_tokens: bool = True,
        hf_push: bool = False,
        hf_repo_id: Optional[str] = None,
        private: bool = False,
    ):
        """
        Export pivotal tokens as DPO training pairs.
        
        Args:
            output_path: Path to save the DPO pairs
            filter_criteria: Criteria to filter tokens
            min_prob_delta: Minimum probability delta for inclusion
            balance_positive_negative: Whether to balance positive and negative examples
            max_pairs: Maximum number of pairs to export
            seed: Random seed for shuffling
            model_name: Model name for finding rejected tokens
            save_tokens: Whether to save updated tokens
            tokens_output_path: Path to save updated tokens
            num_candidates: Number of candidate tokens to consider
            find_rejected_tokens: Whether to find rejected tokens
            hf_push: Whether to push to Hugging Face
            hf_repo_id: Hugging Face repository ID
            private: Whether to make the repository private
        """
        # Filter tokens
        filtered_storage = self.token_storage
        if filter_criteria or min_prob_delta:
            filtered_storage = self.token_storage.filter(
                criteria=filter_criteria,
                min_prob_delta=min_prob_delta
            )
            
        # Split into positive and negative tokens
        positive_tokens = [t for t in filtered_storage if t.get('prob_delta', 0) > 0]
        negative_tokens = [t for t in filtered_storage if t.get('prob_delta', 0) < 0]
        
        logger.info(f"Found {len(positive_tokens)} positive and {len(negative_tokens)} negative tokens")
        
        # Balance positive and negative if requested
        if balance_positive_negative:
            min_count = min(len(positive_tokens), len(negative_tokens))
            random.seed(seed)
            
            if len(positive_tokens) > min_count:
                positive_tokens = random.sample(positive_tokens, min_count)
            if len(negative_tokens) > min_count:
                negative_tokens = random.sample(negative_tokens, min_count)
                
            logger.info(f"Balanced to {len(positive_tokens)} pairs of each type")
            
        # Combine and shuffle
        all_tokens = positive_tokens + negative_tokens
        random.seed(seed)
        random.shuffle(all_tokens)
        
        # Limit to max pairs if specified
        if max_pairs and len(all_tokens) > max_pairs:
            all_tokens = all_tokens[:max_pairs]
            
        # Check if we need to create a searcher for finding rejected tokens
        if find_rejected_tokens and not self.searcher and model_name:
            try:
                # Import here to avoid circular imports
                from .core import PivotalTokenSearcher
                from .oracle import DummyOracle
                
                logger.info(f"Creating searcher with model {model_name} for finding rejected tokens")
                # Create a searcher with a dummy oracle to avoid the "Oracle must be provided" error
                self.searcher = PivotalTokenSearcher(
                    model_name=model_name,
                    oracle=DummyOracle(),  # Use a dummy oracle that always returns success
                    prob_threshold=min_prob_delta
                )
                logger.info(f"Searcher created successfully")
            except Exception as e:
                logger.error(f"Error creating searcher: {e}")
                logger.warning("Will continue without finding rejected tokens")
        
        # Create DPO pairs
        pairs = []
        modified_tokens = []
        
        for token in all_tokens:
            # Get basic fields
            query = token.get('query', '')
            pivot_context = token.get('pivot_context', '')
            pivot_token = token.get('pivot_token', '')
            rejected_token = token.get('rejected_token')
            rejected_token_id = token.get('rejected_token_id')
            prob_delta = token.get('prob_delta', 0)
            token_id = token.get('pivot_token_id')
            
            # If no rejected token and we have a searcher, try to find one
            if find_rejected_tokens and self.searcher and ((prob_delta > 0 and not rejected_token) or (prob_delta < 0)):
                try:
                    # For positive tokens, find a rejected token that decreases probability
                    if prob_delta > 0:
                        # Create a PivotalToken object for the searcher
                        from .core import PivotalToken
                        pivot_token_obj = PivotalToken(
                            query=query,
                            pivot_context=pivot_context,
                            pivot_token=pivot_token,
                            pivot_token_id=token_id,
                            prob_before=token.get('prob_before', 0),
                            prob_after=token.get('prob_after', 0),
                            prob_delta=prob_delta,
                            model_id=token.get('model_id', model_name or 'unknown'),
                            task_type=token.get('task_type', 'unknown')
                        )
                        
                        # Find a rejected token
                        result = self.searcher.find_rejected_tokens(pivot_token_obj, num_candidates=num_candidates)
                        if result:
                            rejected_token, rejected_token_id, _ = result
                            
                            # Skip if the rejected token is the same as the pivot token
                            if rejected_token == pivot_token:
                                logger.warning(f"Skipping as rejected token '{rejected_token}' is the same as pivot token '{pivot_token}'")
                                continue
                                
                            logger.info(f"Found rejected token '{rejected_token}' for positive token '{pivot_token}'")                            
                            # Update the token with the rejected token
                            token['rejected_token'] = rejected_token
                            token['rejected_token_id'] = rejected_token_id
                            modified_tokens.append(token)
                    
                    # For negative tokens, find a better token for comparison
                    else:
                        # Get the most likely token as the accepted token
                        with torch.no_grad():
                            context_ids = self.searcher.tokenizer.encode(pivot_context, return_tensors="pt").to(self.searcher.device)
                            outputs = self.searcher.model(context_ids)
                            logits = outputs.logits[0, -1, :]
                            
                            # Get top most likely tokens 
                            top_tokens = torch.topk(torch.softmax(logits, dim=0), k=5)
                            
                            for i in range(5):
                                next_token_id = top_tokens.indices[i].item()
                                next_token_str = self.searcher.tokenizer.decode([next_token_id])
                                
                                # Skip if the token is the same as the pivot token
                                if next_token_str == pivot_token:
                                    continue
                                
                                # Use it as the rejected token
                                rejected_token = next_token_str
                                rejected_token_id = next_token_id
                                logger.info(f"Using token '{rejected_token}' as alternative for negative token '{pivot_token}'")
                                
                                # Update the token
                                token['rejected_token'] = rejected_token
                                token['rejected_token_id'] = rejected_token_id
                                modified_tokens.append(token)
                                break
                except Exception as e:
                    logger.error(f"Error finding rejected token: {e}")
            
            # Skip if still no rejected token for positive examples
            if prob_delta > 0 and not rejected_token:
                logger.warning(f"No rejected token found for token: {pivot_token}, skipping")
                continue
                
            # Prepare pair - only include relevant fields
            if prob_delta > 0:
                # Positive token with rejected alternative
                if rejected_token and rejected_token != pivot_token:
                    pair = {
                        "prompt": pivot_context,
                        "chosen": pivot_token,
                        "rejected": rejected_token,
                        "metadata": {
                            "original_query": query,
                            "prob_delta": prob_delta,
                            "task_type": token.get('task_type', 'unknown')
                        }
                    }
                    pairs.append(pair)
            else:
                # Negative token (treat as rejected)
                # Find an alternative token if available
                if rejected_token and rejected_token != pivot_token:
                    pair = {
                        "prompt": pivot_context,
                        "chosen": rejected_token,
                        "rejected": pivot_token,
                        "metadata": {
                            "original_query": query,
                            "prob_delta": abs(prob_delta),
                            "task_type": token.get('task_type', 'unknown')
                        }
                    }
                    pairs.append(pair)
        
        # Save to file
        with open(output_path, 'w') as f:
            for pair in pairs:
                f.write(json.dumps(pair) + '\n')
                
        logger.info(f"Exported {len(pairs)} DPO pairs to {output_path}")
        
        # Save updated tokens if requested
        if save_tokens and modified_tokens:
            tokens_path = tokens_output_path or self.token_storage.filepath
            if tokens_path:
                logger.info(f"Saving {len(modified_tokens)} updated tokens to {tokens_path}")
                # Create a new storage for the updated tokens
                updated_storage = TokenStorage()
                for token in self.token_storage.tokens:
                    # Find if this token has been modified
                    modified = next(
                        (m for m in modified_tokens if 
                         m.get('pivot_context') == token.get('pivot_context') and 
                         m.get('pivot_token') == token.get('pivot_token')),
                        None
                    )
                    
                    # If modified, add the modified version, otherwise add the original
                    updated_storage.add_token(modified or token)
                
                # Save the updated tokens
                updated_storage.save(tokens_path)
                logger.info(f"Saved {len(updated_storage)} tokens to {tokens_path}")
        
        # Push to Hugging Face if requested
        if hf_push and hf_repo_id:
            try:
                from huggingface_hub import create_repo, upload_file
                
                # Create repo if it doesn't exist
                create_repo(
                    hf_repo_id,
                    private=private,
                    repo_type="dataset",
                    exist_ok=True
                )
                
                # Upload file
                upload_file(
                    path_or_fileobj=output_path,
                    path_in_repo="dpo_dataset.jsonl",
                    repo_id=hf_repo_id,
                    repo_type="dataset"
                )
                
                logger.info(f"Pushed DPO dataset to Hugging Face: {hf_repo_id}")
                
                # Create README if it doesn't exist
                readme_content = generate_readme_content(
                    file_type="dpo",
                    model_name=model_name,
                    dataset_info=f"- **Dataset size:** {len(pairs)} pairs\n- **Minimum probability delta:** {min_prob_delta}"
                )
                
                # Create temporary README file
                readme_path = "README.md.tmp"
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                
                # Upload README
                upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=hf_repo_id,
                    repo_type="dataset"
                )
                
                # Delete temporary file
                os.remove(readme_path)
                
            except Exception as e:
                logger.error(f"Error pushing to Hugging Face: {e}")
    
    def export_steering_vectors(
        self,
        output_path: str,
        model_name: str,
        layer_nums: List[int] = [19, 23, 27],
        num_clusters: int = 10,
        pca_components: int = 50,
        batch_size: int = 4,
        filter_criteria: Optional[Dict[str, Any]] = None,
        min_prob_delta: float = 0.2,
        select_layer: Optional[int] = None,
        hf_push: bool = False,
        hf_repo_id: Optional[str] = None,
        private: bool = False,
    ):
        """
        Export steering vectors for activation-based steering.
        
        Args:
            output_path: Path to save the steering vectors
            model_name: Model name for extracting activations
            layer_nums: List of layer numbers to extract activations from
            num_clusters: Number of clusters for K-means
            pca_components: Number of PCA components for dimensionality reduction
            batch_size: Batch size for processing
            filter_criteria: Criteria to filter tokens
            min_prob_delta: Minimum probability delta for inclusion
            select_layer: Layer to use for steering vectors (defaults to first layer in layer_nums)
            hf_push: Whether to push to Hugging Face
            hf_repo_id: Hugging Face repository ID
            private: Whether to make the repository private
        """
        # Filter tokens
        filtered_storage = self.token_storage
        if filter_criteria or min_prob_delta:
            filtered_storage = self.token_storage.filter(
                criteria=filter_criteria,
                min_prob_delta=min_prob_delta
            )
        
        if len(filtered_storage) == 0:
            logger.warning("No tokens to export")
            return
            
        # Load model and tokenizer
        logger.info(f"Loading model {model_name} for extracting activations...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model.to(device)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Extract contexts and token data
        contexts = []
        token_data = []
        
        for token in filtered_storage:
            contexts.append(token.get("pivot_context", ""))
            token_data.append(token)
            
        logger.info(f"Extracting activations for {len(contexts)} contexts...")
        
        # Extract activations
        activations = {}
        hooks = {}
        
        # Register hooks
        def get_activation(layer_num):
            def hook(module, input, output):
                # Handle different output formats
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                # Store activations
                if layer_num not in activations:
                    activations[layer_num] = []
                activations[layer_num].append(hidden_states.detach().cpu())
            return hook
        
        # Register hooks for each layer
        for layer_num in layer_nums:
            # Get the appropriate transformer layer based on model architecture
            if hasattr(model, 'transformer'):
                module = model.transformer.h[layer_num]
            elif hasattr(model, 'model'):
                if hasattr(model.model, 'layers'):
                    module = model.model.layers[layer_num]
                else:
                    module = model.model.decoder.layers[layer_num]
            else:
                module = model.layers[layer_num]
            
            # Register hook
            hooks[layer_num] = module.register_forward_hook(get_activation(layer_num))
        
        # Process in batches
        all_activations = {layer: [] for layer in layer_nums}
        
        for i in tqdm(range(0, len(contexts), batch_size), desc="Extracting activations"):
            batch_contexts = contexts[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(batch_contexts, return_tensors="pt", padding=True).to(device)
            
            # Clear activations for next batch
            for layer in layer_nums:
                activations[layer] = []
            
            # Forward pass
            with torch.no_grad():
                model(**inputs)
            
            # Collect activations for this batch
            for layer in layer_nums:
                for batch_idx, seq_len in enumerate(inputs['attention_mask'].sum(dim=1)):
                    # Get activation of final token
                    final_token_idx = seq_len - 1
                    activation = activations[layer][0][batch_idx, final_token_idx, :]
                    all_activations[layer].append(activation)
        
        # Remove hooks
        for hook in hooks.values():
            hook.remove()
        
        # Stack activations
        for layer in layer_nums:
            all_activations[layer] = torch.stack(all_activations[layer])
        
        # Select layer for steering vectors
        select_layer = select_layer if select_layer in layer_nums else layer_nums[0]
        
        # Process activations
        layer_activations = all_activations[select_layer]
        
        # Apply PCA for dimensionality reduction
        if len(layer_activations) > pca_components:
            logger.info(f"Applying PCA to reduce dimensions from {layer_activations.shape[1]} to {pca_components}")
            pca = PCA(n_components=pca_components)
            activations_np = layer_activations.numpy()
            reduced_activations = pca.fit_transform(activations_np)
        else:
            reduced_activations = layer_activations.numpy()
        
        # Cluster activations
        logger.info(f"Clustering into {num_clusters} clusters...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        clusters = kmeans.fit_predict(reduced_activations)
        
        # Group indices by cluster
        cluster_indices = defaultdict(list)
        for i, cluster in enumerate(clusters):
            cluster_indices[int(cluster)].append(i)
        
        # Compute cluster means
        cluster_vectors = {}
        for cluster, indices in cluster_indices.items():
            # Compute mean activation for this cluster
            cluster_mean = layer_activations[indices].mean(dim=0)
            
            # Get tokens in this cluster
            cluster_tokens = [token_data[i] for i in indices]
            
            # Compute percentage of positive tokens in cluster
            positive_count = sum(1 for token in cluster_tokens if token.get("prob_delta", 0) > 0)
            positive_ratio = positive_count / len(cluster_tokens) if cluster_tokens else 0
            
            # Store cluster information
            cluster_vectors[cluster] = {
                "vector": cluster_mean.tolist(),
                "indices": indices,
                "tokens": [t.get("pivot_token", "") for t in cluster_tokens],
                "positive_ratio": positive_ratio,
                "size": len(indices)
            }
        
        # Assign reasoning pattern labels to clusters
        reasoning_patterns = {
            "depth_and_thoroughness": [
                "therefore", "alternatively", "however", "wait", "let's", "so", "think",
                "analyze", "additional", "furthermore", "moreover", "deeper",
                "detailed", "comprehensive", "examine", "investigate", "explore", "consider",
                "important", "significant", "critical", "careful", "precise",
                "nuanced", "full", "complete", "exhaustive", "rigorous"
            ],
            "numerical_accuracy": [
                "calculate", "compute", "equation", "correct", "check", "verify", "math", 
                "number", "calculation", "result", "answer", "precision",
                "formula", "computation", "sum", "total", "value", "exact", "accurate",
                "integer", "decimal", "fraction", "multiply", "divide"
            ],
            "self_correction": [
                "mistake", "incorrect", "wrong", "error", "let me reconsider", "actually", 
                "revise", "correction", "revising", "mistaken", "fix", "adjust", "rectify",
                "amend", "correct", "misunderstood", "misinterpreted", "miscalculated"
            ],
            "exploration": [
                "alternative", "approach", "method", "strategy", "consider", "explore", 
                "possibility", "different", "solution", "examine", "investigate", "option",
                "alternatives", "pathway", "direction", "route", "perspective", "viewpoint"
            ],
            "organization": [
                "first", "second", "next", "finally", "step", "organize", "list", "sequence", 
                "order", "structure", "outline", "categorize", "classify", "group", "arrange",
                "prioritize", "rank", "sort", "divide", "section", "segment"
            ]
        }
        
        # Count keyword occurrences in each cluster
        pattern_scores = {}
        for cluster_id, cluster_info in cluster_vectors.items():
            pattern_scores[cluster_id] = {}
            
            # Get all tokens in this cluster as a single string
            all_tokens = " ".join(cluster_info["tokens"]).lower()
            
            # Count occurrences of keywords for each pattern
            for pattern, keywords in reasoning_patterns.items():
                score = sum(all_tokens.count(keyword.lower()) for keyword in keywords)
                pattern_scores[cluster_id][pattern] = score
        
        # Ensure every cluster has a pattern assignment
        cluster_to_pattern = {}
        for cluster_id in cluster_vectors.keys():
            # Get scores for this cluster across all patterns
            scores_for_cluster = pattern_scores[cluster_id]
            
            # Find the pattern with the highest score
            if scores_for_cluster:
                best_pattern = max(scores_for_cluster, key=scores_for_cluster.get)
                best_score = scores_for_cluster[best_pattern]
                
                if best_score > 0:
                    cluster_to_pattern[cluster_id] = best_pattern
                else:
                    # If no patterns matched, assign based on position
                    patterns = list(reasoning_patterns.keys())
                    fallback_pattern = patterns[int(cluster_id) % len(patterns)]
                    cluster_to_pattern[cluster_id] = fallback_pattern
            else:
                # Fallback assignment
                patterns = list(reasoning_patterns.keys())
                fallback_pattern = patterns[int(cluster_id) % len(patterns)]
                cluster_to_pattern[cluster_id] = fallback_pattern
        
        # Create steering vectors
        steering_tokens = []
        
        for i, token in enumerate(token_data):
            # Create a copy of the token data
            steering_token = dict(token)
            
            # Find cluster for this token
            for cluster_id, cluster_info in cluster_vectors.items():
                if i in cluster_info["indices"]:
                    # Get the reasoning pattern for this cluster
                    reasoning_pattern = cluster_to_pattern.get(cluster_id, "unknown")
                    
                    # Add only necessary steering data
                    steering_token = {
                        "query": token.get("query", ""),
                        "pivot_context": token.get("pivot_context", ""),
                        "pivot_token": token.get("pivot_token", ""),
                        "pivot_token_id": token.get("pivot_token_id", 0),
                        "prob_before": token.get("prob_before", 0),
                        "prob_after": token.get("prob_after", 0),
                        "prob_delta": token.get("prob_delta", 0),
                        "model_id": token.get("model_id", ""),
                        "task_type": token.get("task_type", "unknown"),
                        "steering_vector": layer_activations[i].tolist(),
                        "cluster_id": int(cluster_id),
                        "reasoning_pattern": reasoning_pattern,
                        "cluster_vector": cluster_vectors[cluster_id]["vector"],
                        "steering_layer": select_layer
                    }
                    
                    break
            
            steering_tokens.append(steering_token)
        
        # Save to file
        logger.info(f"Saving {len(steering_tokens)} steering tokens to {output_path}")
        with open(output_path, 'w') as f:
            for token in steering_tokens:
                f.write(json.dumps(token) + '\n')
        
        # Generate metadata summary for README
        metadata_summary = f"- **Layer used:** {select_layer}\n"
        metadata_summary += f"- **Number of clusters:** {num_clusters}\n"
        metadata_summary += f"- **Number of vectors:** {len(steering_tokens)}\n"
        metadata_summary += "- **Clusters:**\n"
        
        # Add information about each cluster
        for cluster_id, pattern_name in cluster_to_pattern.items():
            cluster_size = len(cluster_vectors[cluster_id]["indices"])
            positive_ratio = cluster_vectors[cluster_id]["positive_ratio"]
            metadata_summary += f"  - Cluster {cluster_id}: {pattern_name} ({cluster_size} vectors, {positive_ratio:.2f} positive ratio)\n"
        
        # Push to Hugging Face if requested
        if hf_push and hf_repo_id:
            try:
                from huggingface_hub import create_repo, upload_file
                
                # Create repo if it doesn't exist
                create_repo(
                    hf_repo_id,
                    private=private,
                    repo_type="dataset",
                    exist_ok=True
                )
                
                # Upload files
                upload_file(
                    path_or_fileobj=output_path,
                    path_in_repo="steering_vectors.jsonl",
                    repo_id=hf_repo_id,
                    repo_type="dataset"
                )
                
                logger.info(f"Pushed steering vectors to Hugging Face: {hf_repo_id}")
                
                # Create README
                readme_content = generate_readme_content(
                    file_type="steering",
                    model_name=model_name,
                    dataset_info=metadata_summary
                )
                
                # Create temporary README file
                readme_path = "README.md.tmp"
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                
                # Upload README
                upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=hf_repo_id,
                    repo_type="dataset"
                )
                
                # Delete temporary file
                os.remove(readme_path)
                
            except Exception as e:
                logger.error(f"Error pushing to Hugging Face: {e}")
    def export_thought_anchors(
        self,
        output_path: str,
        filter_criteria: Optional[Dict[str, Any]] = None,
        min_importance_score: float = 0.1,
        max_anchors: Optional[int] = None,
        sort_by_importance: bool = True,
        include_alternatives: bool = True,
        hf_push: bool = False,
        hf_repo_id: Optional[str] = None,
        private: bool = False,
        model_name: Optional[str] = None
    ):
        """
        Export thought anchors for use in inference systems.
        
        Args:
            output_path: Path to save the thought anchors dataset
            filter_criteria: Criteria to filter anchors
            min_importance_score: Minimum importance score for inclusion
            max_anchors: Maximum number of anchors to export
            sort_by_importance: Whether to sort by importance score
            include_alternatives: Whether to include alternative sentences tested
            hf_push: Whether to push to Hugging Face
            hf_repo_id: Hugging Face repository ID
            private: Whether to make the repository private
            model_name: Model name for metadata
        """
        logger.info(f"Exporting thought anchors to {output_path}")
        
        # Load thought anchors from storage
        if hasattr(self.token_storage, 'tokens'):
            anchors = self.token_storage.tokens
        else:
            logger.error("No thought anchors found in storage")
            return
            
        if not anchors:
            logger.warning("No thought anchors to export")
            return
            
        logger.info(f"Found {len(anchors)} thought anchors in storage")
        
        # Apply filters
        filtered_anchors = []
        for anchor in anchors:
            # Check minimum importance score
            importance_score = anchor.get("importance_score", 0)
            if importance_score < min_importance_score:
                continue
                
            # Apply any custom filter criteria
            if filter_criteria:
                skip_anchor = False
                for key, value in filter_criteria.items():
                    if key not in anchor or anchor[key] != value:
                        skip_anchor = True
                        break
                if skip_anchor:
                    continue
                    
            filtered_anchors.append(anchor)
        
        logger.info(f"Filtered to {len(filtered_anchors)} thought anchors")
        
        # Sort by importance if requested
        if sort_by_importance:
            filtered_anchors.sort(key=lambda x: x.get("importance_score", 0), reverse=True)
        
        # Limit number of anchors if specified
        if max_anchors:
            filtered_anchors = filtered_anchors[:max_anchors]
            logger.info(f"Limited to top {len(filtered_anchors)} thought anchors")
        
        # Prepare export format optimized for inference
        export_data = []
        for anchor in filtered_anchors:
            export_item = {
                # Core identification
                "anchor_id": f"{anchor.get('dataset_id', 'unknown')}_{anchor.get('dataset_item_id', 'unknown')}_{anchor.get('sentence_id', 0)}",
                "query": anchor.get("query", ""),
                "sentence": anchor.get("sentence", ""),
                "sentence_id": anchor.get("sentence_id", 0),
                
                # Context and positioning
                "prefix_context": anchor.get("prefix_context", ""),
                "sentence_category": anchor.get("sentence_category", "unknown"),
                
                # Importance metrics
                "importance_score": anchor.get("importance_score", 0),
                "prob_delta": anchor.get("prob_delta", 0),
                "prob_with_sentence": anchor.get("prob_with_sentence", 0),
                "prob_without_sentence": anchor.get("prob_without_sentence", 0),
                "is_positive": anchor.get("is_positive", False),
                
                # Metadata for inference systems
                "task_type": anchor.get("task_type", "generic"),
                "model_id": anchor.get("model_id", model_name or "unknown"),
                "timestamp": anchor.get("timestamp", ""),

                # Reasoning patterns (useful for inference guidance)
                "reasoning_pattern": self._classify_reasoning_pattern(anchor),

                # Verification fields (CRV-inspired)
                "verification_score": anchor.get("verification_score"),
                "verification_method": anchor.get("verification_method"),
                "arithmetic_errors": anchor.get("arithmetic_errors", []),
                "attention_entropy": anchor.get("attention_entropy"),
                "attention_focus_score": anchor.get("attention_focus_score"),
            }

            # Optionally include alternatives tested
            if include_alternatives and "alternatives_tested" in anchor:
                export_item["alternatives_tested"] = anchor["alternatives_tested"]
                export_item["num_alternatives"] = len(anchor.get("alternatives_tested", []))
            
            export_data.append(export_item)
        
        # Generate summary statistics
        summary_stats = self._generate_anchor_summary(export_data)
        
        # Save to JSONL format
        with open(output_path, 'w') as f:
            for item in export_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Exported {len(export_data)} thought anchors to {output_path}")
        logger.info(f"Summary: {summary_stats}")
        
        # Push to Hugging Face if requested
        if hf_push and hf_repo_id:
            try:
                from huggingface_hub import create_repo, upload_file
                
                # Create repo if it doesn't exist
                create_repo(
                    hf_repo_id,
                    private=private,
                    repo_type="dataset",
                    exist_ok=True
                )
                
                # Upload thought anchors file
                filename = os.path.basename(output_path)
                upload_file(
                    path_or_fileobj=output_path,
                    path_in_repo=filename,
                    repo_id=hf_repo_id,
                    repo_type="dataset"
                )
                
                logger.info(f"Pushed thought anchors to Hugging Face: {hf_repo_id}")
                
                # Create README
                readme_content = generate_readme_content(
                    file_type="thought_anchors",
                    model_name=model_name,
                    dataset_info=summary_stats
                )
                
                # Create temporary README file
                readme_path = "README.md.tmp"
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                
                # Upload README
                upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=hf_repo_id,
                    repo_type="dataset"
                )
                
                # Delete temporary file
                os.remove(readme_path)
                
            except Exception as e:
                logger.error(f"Error pushing to Hugging Face: {e}")
    
    def _classify_reasoning_pattern(self, anchor: Dict[str, Any]) -> str:
        """Classify the reasoning pattern of a thought anchor."""
        sentence = anchor.get("sentence", "").lower()
        category = anchor.get("sentence_category", "")
        
        # Pattern classification based on the paper's taxonomy
        if category == "plan_generation":
            return "planning"
        elif category == "uncertainty_management":
            if any(word in sentence for word in ["wait", "actually", "hmm", "reconsider"]):
                return "backtracking"
            else:
                return "uncertainty_management"
        elif category == "self_checking":
            return "verification"
        elif category == "active_computation":
            return "computation"
        elif category == "fact_retrieval":
            return "knowledge_access"
        elif category == "result_consolidation":
            return "consolidation"
        elif category == "final_answer_emission":
            return "answer_emission"
        elif category == "problem_setup":
            return "problem_understanding"
        else:
            return "other"
    
    def _generate_anchor_summary(self, anchors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for exported thought anchors."""
        if not anchors:
            return {"total_anchors": 0}
        
        # Count by category and pattern
        category_counts = {}
        pattern_counts = {}
        importance_scores = []
        
        for anchor in anchors:
            category = anchor.get("sentence_category", "unknown")
            pattern = anchor.get("reasoning_pattern", "unknown")
            importance = anchor.get("importance_score", 0)
            
            category_counts[category] = category_counts.get(category, 0) + 1
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            importance_scores.append(importance)
        
        return {
            "total_anchors": len(anchors),
            "category_distribution": category_counts,
            "reasoning_pattern_distribution": pattern_counts,
            "average_importance": sum(importance_scores) / len(importance_scores) if importance_scores else 0,
            "max_importance": max(importance_scores) if importance_scores else 0,
            "min_importance": min(importance_scores) if importance_scores else 0,
            "positive_anchors": sum(1 for a in anchors if a.get("is_positive", False)),
            "negative_anchors": sum(1 for a in anchors if not a.get("is_positive", False))
        }