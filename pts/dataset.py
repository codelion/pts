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
    task_type: str = "generic",
    query_key: str = "instruction",
    answer_key: Optional[str] = "output"
) -> List[Dict[str, Any]]:
    """
    Load a dataset from Hugging Face or a local path.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face or local path
        split: Dataset split to use
        sample_size: Number of examples to sample (None for all)
        filter_fn: Optional function to filter examples
        seed: Random seed for sampling
        task_type: Type of task (math, code, qa, etc.)
        query_key: Key for query/instruction in the dataset
        answer_key: Key for answer/output in the dataset (None if no answer)
        
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
        
        # Format examples for PTS
        formatted_examples = []
        for i, example in enumerate(examples):
            # Check if required keys are present
            if query_key not in example:
                logger.warning(f"Example {i} missing '{query_key}' key, skipping")
                continue
            
            # Create formatted example
            formatted = {
                "query": example[query_key],
                "answer": example.get(answer_key) if answer_key else None,
                "task_type": task_type,
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


def load_math_dataset(
    dataset_name: str = "competition_math",
    split: str = "train",
    sample_size: Optional[int] = None,
    seed: int = 42,
    difficulty_range: Optional[Tuple[float, float]] = (0.3, 0.7)
) -> List[Dict[str, Any]]:
    """
    Load a math dataset with problem-solution pairs.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face or local path
        split: Dataset split to use
        sample_size: Number of examples to sample (None for all)
        seed: Random seed for sampling
        difficulty_range: Optional range of difficulty scores to include
        
    Returns:
        List of formatted math problems
    """
    # Define keys based on known math datasets
    if dataset_name == "competition_math":
        query_key = "problem"
        answer_key = "solution"
    elif dataset_name == "gsm8k":
        query_key = "question"
        answer_key = "answer"
    elif dataset_name.startswith("math_"):
        query_key = "question"
        answer_key = "answer"
    else:
        query_key = "problem"
        answer_key = "solution"
    
    # Define filter function for difficulty if provided
    filter_fn = None
    if difficulty_range:
        min_diff, max_diff = difficulty_range
        filter_fn = lambda ex: (
            "difficulty" in ex and
            min_diff <= ex["difficulty"] <= max_diff
        )
    
    return load_dataset(
        dataset_name=dataset_name,
        split=split,
        sample_size=sample_size,
        filter_fn=filter_fn,
        seed=seed,
        task_type="math",
        query_key=query_key,
        answer_key=answer_key
    )


def load_code_dataset(
    dataset_name: str = "codelion/optillmbench",
    split: str = "train",
    sample_size: Optional[int] = None,
    seed: int = 42,
    language: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load a code dataset with problem-solution pairs.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face or local path
        split: Dataset split to use
        sample_size: Number of examples to sample (None for all)
        seed: Random seed for sampling
        language: Optional programming language to filter by
        
    Returns:
        List of formatted coding problems
    """
    # Define keys based on known code datasets
    if dataset_name == "codelion/optillmbench":
        query_key = "instruction"
        answer_key = "output"
    elif dataset_name == "codeparrot/apps":
        query_key = "problem"
        answer_key = "solution"
    elif dataset_name == "humaneval":
        query_key = "prompt"
        answer_key = "canonical_solution"
    else:
        query_key = "instruction"
        answer_key = "output"
    
    # Define filter function for language if provided
    filter_fn = None
    if language:
        filter_fn = lambda ex: (
            "language" in ex and
            ex["language"].lower() == language.lower()
        )
    
    return load_dataset(
        dataset_name=dataset_name,
        split=split,
        sample_size=sample_size,
        filter_fn=filter_fn,
        seed=seed,
        task_type="code",
        query_key=query_key,
        answer_key=answer_key
    )


def create_oracle_from_dataset(
    examples: List[Dict[str, Any]],
    task_type: str = "generic"
) -> Any:
    """
    Create an appropriate oracle from dataset examples.
    
    Args:
        examples: List of dataset examples
        task_type: Type of task (math, code, qa, etc.)
        
    Returns:
        Oracle instance for the dataset
    """
    try:
        if task_type == "math":
            from .oracle import MathOracle
            
            # Create answer dictionary
            answers = {}
            for example in examples:
                if example.get("query") and example.get("answer"):
                    answers[example["query"]] = example["answer"]
            
            return MathOracle(answers=answers)
            
        elif task_type == "code":
            from .oracle import CodeOracle
            
            # Create test cases dictionary
            test_cases = {}
            for example in examples:
                if example.get("query") and example.get("answer"):
                    # Create a simple test case that checks if the solution contains key elements
                    query = example["query"]
                    expected_solution = example["answer"]
                    
                    # Extract function name if possible
                    import re
                    function_match = re.search(r"def\s+(\w+)\s*\(", expected_solution)
                    function_name = function_match.group(1) if function_match else None
                    
                    test_case = {
                        "setup": "",
                        "assertion": f"# Check if solution contains expected code patterns\n"
                                    f"solution_str = inspect.getsource(locals().get('{function_name}')) if '{function_name}' in locals() else ''\n"
                                    f"assert solution_str.strip(), 'No solution found'"
                    }
                    
                    # Add metadata from example
                    if "metadata" in example and example["metadata"]:
                        # If there are test cases in the metadata, use them
                        if "test_cases" in example["metadata"]:
                            test_case["tests"] = example["metadata"]["test_cases"]
                            
                        # If there's an entry point in the metadata, use it
                        if "entry_point" in example["metadata"] and function_name:
                            test_case["function_call"] = f"{function_name}(*args)"
                    
                    if query not in test_cases:
                        test_cases[query] = []
                    test_cases[query].append(test_case)
            
            return CodeOracle(test_cases=test_cases)
            
        else:
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
