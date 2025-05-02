"""
Dataset utilities for Pivotal Token Search.

This module provides functionality for loading and managing datasets for PTS.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datasets import load_dataset as hf_load_dataset
import random

logger = logging.getLogger(__name__)


def load_dataset(
    dataset_name: str = "codelion/optillmbench",
    split: str = "train",
    sample_size: Optional[int] = None,
    filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    seed: int = 42,
    query_key: Optional[str] = None,
    answer_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load a dataset from Hugging Face or a local path.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face or local path
        split: Dataset split to use
        sample_size: Number of examples to sample (None for all)
        filter_fn: Optional function to filter examples
        seed: Random seed for sampling
        query_key: Key for query/question in the dataset (None for auto-detect)
        answer_key: Key for answer/output in the dataset (None for auto-detect)
        
    Returns:
        List of dataset examples
    """
    logger.info(f"Loading dataset {dataset_name} (split: {split})")
    
    try:
        # Load the dataset
        dataset = hf_load_dataset(dataset_name, split=split)
        logger.info(f"Loaded {len(dataset)} examples")
        
        # Convert to list of dictionaries
        examples = [dict(example) for example in dataset]
        
        # Filter examples if a filter function is provided
        if filter_fn:
            examples = [example for example in examples if filter_fn(example)]
            logger.info(f"Filtered to {len(examples)} examples")
        
        # Sample examples if a sample size is provided
        if sample_size and sample_size < len(examples):
            random.seed(seed)
            examples = random.sample(examples, sample_size)
            logger.info(f"Sampled {len(examples)} examples")
        
        # Auto-detect query and answer keys if not provided
        if not query_key or not answer_key:
            # Get a list of all keys in the first example
            if examples:
                available_keys = list(examples[0].keys())
                logger.info(f"Available keys in dataset: {available_keys}")
                
                # Set defaults based on dataset name
                if dataset_name == "codelion/optillmbench":
                    default_query_key = "question"
                    default_answer_key = "answer"
                else:
                    # Default fallbacks
                    possible_query_keys = ["question", "query", "instruction", "problem", "prompt"]
                    possible_answer_keys = ["answer", "output", "solution", "response", "canonical_solution"]
                    
                    # Find the first matching key
                    default_query_key = next((k for k in possible_query_keys if k in available_keys), "question")
                    default_answer_key = next((k for k in possible_answer_keys if k in available_keys), "answer")
                
                # Use provided keys or defaults
                query_key = query_key or default_query_key
                answer_key = answer_key or default_answer_key
                
                logger.info(f"Using keys: query_key='{query_key}', answer_key='{answer_key}'")
        
        # Format examples for PTS
        formatted_examples = []
        for i, example in enumerate(examples):
            # Check if required keys are present
            if query_key not in example:
                logger.warning(f"Example {i} missing '{query_key}' key, skipping")
                continue
                
            if answer_key and answer_key not in example:
                logger.warning(f"Example {i} missing '{answer_key}' key")
            
            # Create formatted example
            formatted = {
                "query": example[query_key],
                "answer": example.get(answer_key) if answer_key else None,
                "dataset_id": dataset_name,
                "item_id": str(i)
            }
            
            # Add all original fields as metadata
            formatted["metadata"] = example
            
            formatted_examples.append(formatted)
        
        return formatted_examples
        
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        return []


def create_oracle_from_dataset(
    examples: List[Dict[str, Any]]
) -> Any:
    """
    Create an appropriate oracle from dataset examples.
    
    Args:
        examples: List of dataset examples
        
    Returns:
        Oracle instance for the dataset
    """
    try:
        # Check if it's optillmbench by looking at metadata
        is_optillmbench = False
        has_categories = False
        
        if examples and examples[0]["dataset_id"] == "codelion/optillmbench":
            is_optillmbench = True
            if examples[0]["metadata"] and "category" in examples[0]["metadata"]:
                has_categories = True
        
        # Use the specialized OptiBenchOracle for optillmbench dataset
        if is_optillmbench and has_categories:
            from .oracle import OptiBenchOracle
            
            # Create answer dictionary with categories
            examples_with_categories = {}
            
            for example in examples:
                if example.get("query") and example.get("answer"):
                    category = example["metadata"].get("category", "general")
                    if category not in examples_with_categories:
                        examples_with_categories[category] = {}
                        
                    examples_with_categories[category][example["query"]] = example["answer"]
            
            return OptiBenchOracle(examples_with_categories=examples_with_categories)
            
        else:
            # Default to QAOracle for other datasets
            from .oracle import QAOracle
            
            # Create answer dictionary
            answers = {}
            for example in examples:
                if example.get("query") and example.get("answer"):
                    answers[example["query"]] = example["answer"]
            
            return QAOracle(answers=answers)
            
    except Exception as e:
        logger.error(f"Error creating oracle: {e}")
        from .oracle import DummyOracle
        return DummyOracle()
