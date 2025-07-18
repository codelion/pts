"""
Command-line interface for Pivotal Token Search.

This module provides a command-line interface for using PTS to find pivotal tokens
and export them to different formats.
"""

import argparse
import logging
import sys
import os
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import torch
from tqdm import tqdm

from .core import PivotalTokenSearcher
from .storage import TokenStorage
from .exporters import TokenExporter
from .dataset import load_dataset, create_oracle_from_dataset
from .thought_anchors import ThoughtAnchorSearcher, ThoughtAnchorStorage


def setup_logging(log_level: str = "INFO"):
    """Set up logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("pts.log")
        ]
    )


def run_pts(args):
    """Run the Pivotal Token Search algorithm."""
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running PTS with model {args.model}")
    
    # Load dataset
    examples = load_dataset(
        dataset_name=args.dataset,
        split=args.split,
        config=args.config,
        sample_size=args.sample_size,
        seed=args.seed,
        query_key=args.query_key,
        answer_key=args.answer_key
    )
    
    if not examples:
        logger.error(f"No examples loaded from dataset {args.dataset}")
        return
    
    logger.info(f"Loaded {len(examples)} examples from {args.dataset}")
    
    # Create oracle from dataset
    oracle = create_oracle_from_dataset(examples, debug_mode=args.debug)
    
    # Create storage for pivotal tokens
    storage = TokenStorage(filepath=args.output_path)
    
    # Create searcher
    searcher = PivotalTokenSearcher(
        model_name=args.model,
        oracle=oracle,
        device=args.device,
        prob_threshold=args.prob_threshold,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        token_storage=storage,
        log_level=getattr(logging, args.log_level.upper()),
        debug_mode=args.debug
    )
    
    # Process each example
    successful_searches = 0
    total_pivotal_tokens = 0
    
    for i, example in enumerate(tqdm(examples[:args.max_examples], desc="Processing examples")):
        if i >= args.max_examples:
            break
            
        logger.info(f"Processing example {i+1}/{min(len(examples), args.max_examples)}")
        
        query = example["query"]
        
        # Skip empty queries
        if not query.strip():
            logger.warning(f"Skipping empty query in example {i}")
            continue
            
        # Search for pivotal tokens
        query_pivotal_tokens = list(searcher.search_pivotal_tokens(
            query=query,
            system_prompt=args.system_prompt,
            task_type="generic",  # Single task type
            dataset_id=args.dataset,
            item_id=example.get("item_id", str(i)),
            max_generations=args.max_generations,
            min_prob=args.min_prob,
            max_prob=args.max_prob,
            category=example.get("metadata", {}).get("category", None)
        ))
        
        if query_pivotal_tokens:
            # Force save each token to storage explicitly
            for token in query_pivotal_tokens:
                storage.add_token(token)
                
            successful_searches += 1
            total_pivotal_tokens += len(query_pivotal_tokens)
            logger.info(f"Found {len(query_pivotal_tokens)} pivotal tokens for example {i}")
        else:
            logger.info(f"No pivotal tokens found for example {i}")
    
    # Save the tokens
    logger.info(f"Found pivotal tokens in {successful_searches}/{args.max_examples} examples")
    logger.info(f"Total pivotal tokens found: {total_pivotal_tokens}")
    
    # Make sure to save even if the tokens were added directly by the searcher
    if total_pivotal_tokens > 0:
        storage.save()
        logger.info(f"Saved tokens to {args.output_path}")
    else:
        logger.info(f"No tokens found, no file saved")
        
    logger.info(f"Finished processing {args.max_examples} examples")


def run_thought_anchors(args):
    """Run the Thought Anchor Search algorithm."""
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running Thought Anchor Search with model {args.model}")
    
    # Load dataset
    examples = load_dataset(
        dataset_name=args.dataset,
        split=args.split,
        config=args.config,
        sample_size=args.sample_size,
        seed=args.seed,
        query_key=args.query_key,
        answer_key=args.answer_key
    )
    
    if not examples:
        logger.error(f"No examples loaded from dataset {args.dataset}")
        return
    
    logger.info(f"Loaded {len(examples)} examples from {args.dataset}")
    
    # Create oracle from dataset
    oracle = create_oracle_from_dataset(examples, debug_mode=args.debug)
    
    # Pass skip_embeddings flag through oracle
    if hasattr(args, 'skip_embeddings') and args.skip_embeddings:
        oracle.skip_embeddings = True
        logger.info("Embeddings will be skipped for faster processing")
    
    # Create storage for thought anchors
    storage = ThoughtAnchorStorage(filepath=args.output_path)
    
    # Create thought anchor searcher
    searcher = ThoughtAnchorSearcher(
        model_name=args.model,
        oracle=oracle,
        device=args.device,
        prob_threshold=args.prob_threshold,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        log_level=getattr(logging, args.log_level.upper()),
        debug_mode=args.debug
    )
    
    # Pass storage to searcher for periodic saving
    searcher.storage = storage
    
    # Process each example
    successful_searches = 0
    total_thought_anchors = 0
    
    for i, example in enumerate(tqdm(examples[:args.max_examples], desc="Processing examples")):
        if i >= args.max_examples:
            break
            
        logger.info(f"Processing example {i+1}/{min(len(examples), args.max_examples)}")
        
        query = example["query"]
        
        # Skip empty queries
        if not query.strip():
            logger.warning(f"Skipping empty query in example {i}")
            continue
        
        # Generate a reasoning trace first
        logger.info(f"Generating reasoning trace for analysis...")
        
        # Format the prompt
        if args.system_prompt:
            messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": query}
            ]
            prompt = searcher.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Use category-specific formatting if available 
            category = example.get("metadata", {}).get("category", None)
            if category and hasattr(oracle, 'get_prompt_for_category'):
                prompt = oracle.get_prompt_for_category(query, category)
            else:
                prompt = query
        
        # Generate a reasoning trace
        tokenized = searcher.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = tokenized.input_ids.to(searcher.device)
        attention_mask = tokenized.attention_mask.to(searcher.device) if "attention_mask" in tokenized else None
        
        try:
            with torch.no_grad():
                outputs = searcher.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    pad_token_id=searcher.tokenizer.pad_token_id,
                    return_dict_in_generate=True
                )
            
            # Extract the reasoning trace
            reasoning_trace = searcher.tokenizer.decode(
                outputs.sequences[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            if not reasoning_trace.strip():
                logger.warning(f"Empty reasoning trace for example {i}, skipping")
                continue
            
            logger.info(f"Generated reasoning trace with {len(reasoning_trace)} characters")
            
            # Search for thought anchors in the reasoning trace
            query_thought_anchors = list(searcher.search_thought_anchors(
                query=query,
                reasoning_trace=reasoning_trace,
                system_prompt=args.system_prompt,
                task_type="generic",
                dataset_id=args.dataset,
                item_id=example.get("item_id", str(i)),
                min_prob=args.min_prob,
                max_prob=args.max_prob,
                category=example.get("metadata", {}).get("category", None)
            ))
            
            if query_thought_anchors:
                # Add thought anchors to storage
                for anchor in query_thought_anchors:
                    storage.add_thought_anchor(anchor)
                    
                successful_searches += 1
                total_thought_anchors += len(query_thought_anchors)
                logger.info(f"Found {len(query_thought_anchors)} thought anchors for example {i}")
            else:
                logger.info(f"No thought anchors found for example {i}")
                
        except Exception as e:
            logger.error(f"Error processing example {i}: {e}")
            continue
    
    # Save the thought anchors
    logger.info(f"Found thought anchors in {successful_searches}/{args.max_examples} examples")
    logger.info(f"Total thought anchors found: {total_thought_anchors}")
    
    if total_thought_anchors > 0:
        storage.save()
        logger.info(f"Saved thought anchors to {args.output_path}")
        
        # Generate and print summary
        summary = storage.get_anchor_summary()
        logger.info(f"Thought Anchor Summary:")
        logger.info(f"  Total anchors: {summary['total_anchors']}")
        logger.info(f"  Positive anchors: {summary['positive_anchors']}")
        logger.info(f"  Negative anchors: {summary['negative_anchors']}")
        logger.info(f"  Average importance: {summary['average_importance']:.3f}")
        logger.info(f"  Category distribution: {summary['category_distribution']}")
    else:
        logger.info(f"No thought anchors found, no file saved")
        
    logger.info(f"Finished processing {args.max_examples} examples")


def export_tokens(args):
    """Export pivotal tokens to different formats."""
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Exporting tokens from {args.input_path} to {args.output_path}")
    
    # Load storage
    storage = TokenStorage(filepath=args.input_path)
    
    # Create exporter
    exporter = TokenExporter(token_storage=storage)
    
    # Export based on format
    if args.format == "dpo":
        logger.info(f"Exporting to DPO format")
        exporter.export_dpo_dataset(
            output_path=args.output_path,
            min_prob_delta=args.min_prob_delta,
            balance_positive_negative=args.balance,
            max_pairs=args.max_pairs,
            seed=args.seed,
            model_name=args.model,
            save_tokens=args.save_tokens,
            tokens_output_path=args.tokens_output_path,
            num_candidates=args.num_candidates,
            find_rejected_tokens=args.find_rejected_tokens,
            hf_push=args.hf_push,
            hf_repo_id=args.hf_repo_id,
            private=args.private
        )
    elif args.format == "steering":
        logger.info(f"Exporting to steering vectors format")
        exporter.export_steering_vectors(
            output_path=args.output_path,
            model_name=args.model,
            layer_nums=args.layer_nums,
            num_clusters=args.num_clusters,
            pca_components=args.pca_components,
            batch_size=args.batch_size,
            min_prob_delta=args.min_prob_delta,
            select_layer=args.select_layer,
            hf_push=args.hf_push,
            hf_repo_id=args.hf_repo_id,
            private=args.private
        )
    elif args.format == "thought_anchors":
        logger.info(f"Exporting to thought anchors format")
        exporter.export_thought_anchors(
            output_path=args.output_path,
            min_importance_score=getattr(args, 'min_prob_delta', 0.1),
            max_anchors=args.max_pairs,
            sort_by_importance=True,
            include_alternatives=True,
            hf_push=args.hf_push,
            hf_repo_id=args.hf_repo_id,
            private=args.private,
            model_name=args.model
        )
    else:
        logger.error(f"Unsupported export format: {args.format}")
        sys.exit(1)
    
    logger.info(f"Export completed successfully")


def push_to_hf(args):
    """Push a file to Hugging Face."""
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Pushing {args.input_path} to Hugging Face: {args.hf_repo_id}")
    
    try:
        from huggingface_hub import create_repo, upload_file
        
        # Create repo if it doesn't exist
        create_repo(
            args.hf_repo_id,
            private=args.private,
            repo_type="dataset",
            exist_ok=True
        )
        
        # Upload file
        filename = os.path.basename(args.input_path)
        upload_file(
            path_or_fileobj=args.input_path,
            path_in_repo=filename,
            repo_id=args.hf_repo_id,
            repo_type="dataset"
        )
        
        logger.info(f"Pushed {args.input_path} to Hugging Face: {args.hf_repo_id}")
        
        # Create README by default unless --no-readme flag is specified
        if not args.no_readme:
            # Determine file type by filename and content
            file_type = "tokens"  # default
            
            if filename.endswith("_vectors.jsonl") or "steering" in filename:
                file_type = "steering"
            elif "dpo" in filename:
                file_type = "dpo"
            elif "thought_anchor" in filename or "anchor" in filename:
                file_type = "thought_anchors"
            else:
                # Check file content to detect thought anchors
                try:
                    with open(args.input_path, 'r') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            import json
                            data = json.loads(first_line)
                            # Check for thought anchor specific fields
                            if "sentence_embedding" in data or "prob_with_sentence" in data or "suffix_context" in data:
                                file_type = "thought_anchors"
                            elif "chosen" in data and "rejected" in data:
                                file_type = "dpo"
                            elif "steering_vector" in data or "activation_vector" in data:
                                file_type = "steering"
                except Exception:
                    pass  # Fall back to default if content check fails
                
            # Import the README generation function from exporters
            from .exporters import generate_readme_content
            
            # Generate README content
            readme_content = generate_readme_content(
                file_type=file_type,
                model_name=args.model
            )
            
            # Create temporary README file
            readme_path = "README.md.tmp"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            # Upload README
            upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=args.hf_repo_id,
                repo_type="dataset"
            )
            
            # Delete temporary file
            os.remove(readme_path)
            
            logger.info(f"Created README for {args.hf_repo_id}")
        else:
            logger.info(f"Skipped README creation (--no-readme flag used)")
        
    except Exception as e:
        logger.error(f"Error pushing to Hugging Face: {e}")
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pivotal Token Search (PTS)")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")
    
    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run PTS on a dataset")
    run_parser.add_argument("--model", type=str, required=True, help="Model name or path")
    run_parser.add_argument("--dataset", type=str, default="codelion/optillmbench", help="Dataset name or path")
    run_parser.add_argument("--config", type=str, default=None, help="Dataset configuration name (if applicable)")
    run_parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    run_parser.add_argument("--output-path", type=str, default="pivotal_tokens.jsonl", help="Output file path")
    run_parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda, cpu)")
    run_parser.add_argument("--prob-threshold", type=float, default=0.2, help="Probability threshold for pivotal tokens")
    run_parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling")
    run_parser.add_argument("--top-p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter")
    run_parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling parameter")
    run_parser.add_argument("--min-p", type=float, default=0.0, help="Min-p sampling parameter")
    run_parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    run_parser.add_argument("--num-samples", type=int, default=50, help="Number of samples for probability estimation")
    run_parser.add_argument("--batch-size", type=int, default=5, help="Batch size for generation")
    run_parser.add_argument("--sample-size", type=int, default=None, help="Number of examples to sample from dataset")
    run_parser.add_argument("--max-examples", type=int, default=100, help="Maximum number of examples to process")
    run_parser.add_argument("--max-generations", type=int, default=10, help="Maximum number of generations per example")
    run_parser.add_argument("--min-prob", type=float, default=0.2, help="Minimum initial success probability")
    run_parser.add_argument("--max-prob", type=float, default=0.8, help="Maximum initial success probability")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--system-prompt", type=str, default=None, help="System prompt for chat models")
    run_parser.add_argument("--query-key", type=str, default=None, 
                           help="Key for query field in dataset (e.g., 'question', 'instruction', 'problem'). Auto-detected if not specified.")
    run_parser.add_argument("--answer-key", type=str, default=None, 
                           help="Key for answer field in dataset (e.g., 'answer', 'output', 'solution'). Auto-detected if not specified.")
    run_parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    run_parser.add_argument("--debug", action="store_true", help="Enable debug mode to print questions and responses")
    run_parser.add_argument("--generate-thought-anchors", action="store_true", help="Generate thought anchors dataset instead of pivotal tokens")
    run_parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation for faster processing (thought anchors only)")
    
    # Export subcommand
    export_parser = subparsers.add_parser("export", help="Export tokens to different formats")
    export_parser.add_argument("--input-path", type=str, required=True, help="Input tokens file path")
    export_parser.add_argument("--output-path", type=str, required=True, help="Output file path")
    export_parser.add_argument("--format", type=str, required=True, choices=["dpo", "steering", "thought_anchors"], help="Export format")
    export_parser.add_argument("--model", type=str, default=None, help="Model name or path (required for steering)")
    export_parser.add_argument("--min-prob-delta", type=float, default=0.1, help="Minimum probability delta")
    export_parser.add_argument("--balance", action="store_true", help="Balance positive and negative examples")
    export_parser.add_argument("--max-pairs", type=int, default=None, help="Maximum number of pairs to export")
    export_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    export_parser.add_argument("--save-tokens", action="store_true", help="Save updated tokens")
    export_parser.add_argument("--tokens-output-path", type=str, default=None, help="Path to save updated tokens")
    export_parser.add_argument("--num-candidates", type=int, default=10, help="Number of candidate tokens")
    export_parser.add_argument("--find-rejected-tokens", action="store_true", help="Find rejected tokens")
    export_parser.add_argument("--layer-nums", type=int, nargs="+", default=[19, 23, 27], help="Layer numbers for steering")
    export_parser.add_argument("--num-clusters", type=int, default=10, help="Number of clusters for steering")
    export_parser.add_argument("--pca-components", type=int, default=50, help="Number of PCA components")
    export_parser.add_argument("--batch-size", type=int, default=4, help="Batch size for steering")
    export_parser.add_argument("--select-layer", type=int, default=None, help="Layer to use for steering")
    export_parser.add_argument("--hf-push", action="store_true", help="Push to Hugging Face")
    export_parser.add_argument("--hf-repo-id", type=str, default=None, help="Hugging Face repository ID")
    export_parser.add_argument("--private", action="store_true", help="Make Hugging Face repository private")
    export_parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    
    # Push subcommand
    push_parser = subparsers.add_parser("push", help="Push a file to Hugging Face")
    push_parser.add_argument("--input-path", type=str, required=True, help="Input file path")
    push_parser.add_argument("--hf-repo-id", type=str, required=True, help="Hugging Face repository ID")
    push_parser.add_argument("--private", action="store_true", help="Make Hugging Face repository private")
    push_parser.add_argument("--no-readme", action="store_true", help="Skip creating a README file")
    push_parser.add_argument("--model", type=str, default=None, help="Model name for README")
    push_parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    if args.command == "run":
        if hasattr(args, 'generate_thought_anchors') and args.generate_thought_anchors:
            run_thought_anchors(args)
        else:
            run_pts(args)
    elif args.command == "export":
        export_tokens(args)
    elif args.command == "push":
        push_to_hf(args)
    else:
        print("Please specify a command: run, export, or push")
        sys.exit(1)


if __name__ == "__main__":
    main()
