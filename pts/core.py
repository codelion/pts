"""
Core implementation of Pivotal Token Search (PTS).

This module implements the binary search algorithm for finding pivotal tokens
in language model generations, as described in the Phi-4 Technical Report.
"""

import time
import logging
from typing import List, Dict, Tuple, Any, Optional, Callable, Generator
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field

from .oracle import Oracle
from .storage import TokenStorage

logger = logging.getLogger(__name__)


@dataclass
class PivotalToken:
    """Represents a pivotal token with its context and impact on success probability."""
    
    # Core fields
    query: str
    pivot_context: str  # Prefix before the pivotal token
    pivot_token: str  # The actual token that changes success probability
    pivot_token_id: int  # Token ID of the pivot token
    
    # Probability metrics
    prob_before: float  # Success probability before the pivotal token
    prob_after: float  # Success probability after the pivotal token
    prob_delta: float  # Change in success probability (after - before)
    
    # Metadata
    model_id: str
    task_type: str
    dataset_id: Optional[str] = None
    dataset_item_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    
    # Optional additional info
    token_logprob: Optional[float] = None
    rejected_token: Optional[str] = None
    rejected_token_id: Optional[int] = None
    notes: Optional[str] = None

    def is_positive(self) -> bool:
        """Return True if this token increases success probability."""
        return self.prob_delta > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the pivotal token to a dictionary for storage."""
        return {
            "model_id": self.model_id,
            "query": self.query,
            "pivot_context": self.pivot_context,
            "pivot_token": self.pivot_token,
            "pivot_token_id": self.pivot_token_id,
            "prob_before": self.prob_before,
            "prob_after": self.prob_after,
            "prob_delta": self.prob_delta,
            "is_positive": self.is_positive(),
            "task_type": self.task_type,
            "dataset_id": self.dataset_id,
            "dataset_item_id": self.dataset_item_id,
            "timestamp": self.timestamp,
            "token_logprob": self.token_logprob,
            "rejected_token": self.rejected_token,
            "rejected_token_id": self.rejected_token_id,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PivotalToken':
        """Create a PivotalToken instance from a dictionary."""
        return cls(
            model_id=data["model_id"],
            query=data["query"],
            pivot_context=data["pivot_context"],
            pivot_token=data["pivot_token"],
            pivot_token_id=data["pivot_token_id"],
            prob_before=data["prob_before"],
            prob_after=data["prob_after"],
            prob_delta=data["prob_delta"],
            task_type=data["task_type"],
            dataset_id=data.get("dataset_id"),
            dataset_item_id=data.get("dataset_item_id"),
            timestamp=data.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
            token_logprob=data.get("token_logprob"),
            rejected_token=data.get("rejected_token"),
            rejected_token_id=data.get("rejected_token_id"),
            notes=data.get("notes")
        )


class PivotalTokenSearcher:
    """
    Implements the Pivotal Token Search (PTS) algorithm to find tokens 
    that significantly impact success probability in language model generations.
    """
    
    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        oracle: Optional[Oracle] = None,
        device: str = None,
        prob_threshold: float = 0.2,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        max_new_tokens: int = 512,
        num_samples: int = 10,
        batch_size: int = 5,
        trust_remote_code: bool = True,
        token_storage: Optional[TokenStorage] = None,
        log_level: int = logging.INFO,
        debug_mode: bool = False
    ):
        """
        Initialize the PivotalTokenSearcher.
        
        Args:
            model_name: HuggingFace model ID or path to local model
            tokenizer_name: HuggingFace tokenizer ID (if different from model)
            oracle: Oracle instance for evaluating success
            device: Device to run the model on ('cpu', 'cuda', etc.)
            prob_threshold: Minimum change in probability to be considered pivotal
            temperature: Temperature for sampling from the model
            max_new_tokens: Maximum number of new tokens to generate
            num_samples: Number of samples to estimate success probability
            batch_size: Batch size for generating multiple samples
            trust_remote_code: Whether to trust remote code in model loading
            token_storage: TokenStorage instance for saving pivotal tokens
            log_level: Logging level (e.g., logging.INFO)
        """
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)
        
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        # Determine the best available device if none provided
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        self.prob_threshold = prob_threshold
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self.oracle = oracle
        self.token_storage = token_storage or TokenStorage()
        self.debug_mode = debug_mode
        
        # Check if flash attention is available
        use_flash_attention = False
        try:
            import flash_attn
            self.logger.info("Flash Attention 2 is available and will be used")
            use_flash_attention = True
        except ImportError:
            self.logger.info("Flash Attention 2 is not available, using standard attention")
        
        # Add flash attention to config if available
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": self.device,
        }
        
        if use_flash_attention:
            # Flash Attention requires either float16 or bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                # Use bfloat16 for Ampere or newer GPUs (compute capability 8.0+)
                model_kwargs["torch_dtype"] = torch.bfloat16
                self.logger.info("Using bfloat16 precision with Flash Attention")
            else:
                # Use float16 for older GPUs
                model_kwargs["torch_dtype"] = torch.float16
                self.logger.info("Using float16 precision with Flash Attention")
                
            model_kwargs["attn_implementation"] = "flash_attention_2"
            
        # Load model and tokenizer
        self.logger.info(f"Loading model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, 
            trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Cache for memoizing probability estimates
        self.prob_cache = {}
    
    def estimate_success_probability(
        self, 
        query: str, 
        prefix: str = "",
        num_samples: Optional[int] = None,
        system_prompt: Optional[str] = None,
        category: Optional[str] = None
    ) -> float:
        """
        Estimate the probability of success by generating multiple completions 
        and checking with the oracle.
        
        Args:
            query: The original query/prompt
            prefix: Token sequence prefix to include in each sample
            num_samples: Number of samples to generate (defaults to self.num_samples)
            system_prompt: Optional system prompt for chat models
            
        Returns:
            Estimated probability of success (0.0 to 1.0)
        """
        if self.oracle is None:
            raise ValueError("Oracle must be provided to estimate success probability")
        
        # Check cache first
        cache_key = (query, prefix, num_samples, system_prompt)
        if cache_key in self.prob_cache:
            return self.prob_cache[cache_key]
        
        num_samples = num_samples or self.num_samples
        self.logger.debug(f"Estimating success probability with {num_samples} samples")
        
        # Format the prompt with the prefix
        if system_prompt:
            # Apply chat template if system prompt is provided
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Use category-specific formatting if available 
            if category and hasattr(self.oracle, 'get_prompt_for_category'):
                prompt = self.oracle.get_prompt_for_category(query, category)
            else:
                prompt = query
            
        if prefix:
            prompt = prompt + prefix
            
        # Tokenize prompt
        tokenized = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device) if "attention_mask" in tokenized else None
        
        # Generate completions in batches
        success_count = 0
        remaining_samples = num_samples
        
        with tqdm(total=num_samples, desc="Generating samples", disable=num_samples < 10) as pbar:
            while remaining_samples > 0:
                current_batch_size = min(self.batch_size, remaining_samples)
                
                # Generate completions
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        num_return_sequences=current_batch_size,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        min_p=self.min_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True
                    )
                
                # Check success for each completion
                for seq in outputs.sequences:
                    completion = self.tokenizer.decode(seq[input_ids.shape[1]:], skip_special_tokens=True)
                    full_response = prompt + completion
                    
                    # Check success with oracle
                    success = self.oracle.check_success(query, full_response)
                    if success:
                        success_count += 1
                    
                    # Print debug information if debug mode is enabled
                    if self.debug_mode:
                        print("\n" + "=" * 50)
                        print(f"QUERY: {query}")
                        if category:
                            print(f"CATEGORY: {category}")
                        print("-" * 50)
                        print(f"COMPLETION {pbar.n}/{num_samples} (SUCCESS: {success}):\n{completion}")
                        print("=" * 50)
                    
                    pbar.update(1)
                    
                remaining_samples -= current_batch_size
        
        # Calculate success probability
        success_prob = success_count / num_samples
        self.logger.debug(f"Success probability: {success_prob:.4f}")
        
        # Cache the result
        self.prob_cache[cache_key] = success_prob
        
        return success_prob
    
    def subdivide_sequence(
        self,
        query: str,
        sequence: List[int],
        prefix: List[int] = None,
        system_prompt: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[List[int]]:
        """
        Recursively divide a token sequence into segments where each segment 
        either has only one token or does not significantly change success probability.
        
        Args:
            query: Original query
            sequence: List of token IDs to subdivide
            prefix: Token sequence context before the current segment
            system_prompt: Optional system prompt
            
        Returns:
            List of token sequence segments
        """
        prefix = prefix or []
        
        # Base case 1: Single token or empty sequence
        if len(sequence) <= 1:
            return [sequence]
            
        # Get the probability before and after the sequence
        prefix_str = self.tokenizer.decode(prefix, skip_special_tokens=False)
        full_seq_str = self.tokenizer.decode(prefix + sequence, skip_special_tokens=False)
        
        prob_before = self.estimate_success_probability(query, prefix_str, system_prompt=system_prompt, category=category)
        prob_after = self.estimate_success_probability(query, full_seq_str, system_prompt=system_prompt, category=category)
        
        # Base case 2: No significant change in probability
        if abs(prob_after - prob_before) < self.prob_threshold:
            return [sequence]
            
        # Split the sequence for recursive processing
        mid = len(sequence) // 2
        left = sequence[:mid]
        right = sequence[mid:]
        
        # Recursively subdivide left side
        left_segments = self.subdivide_sequence(query, left, prefix, system_prompt, category)
        
        # Update prefix for right side by concatenating prefix and left side
        new_prefix = prefix + left
        
        # Recursively subdivide right side
        right_segments = self.subdivide_sequence(query, right, new_prefix, system_prompt, category)
        
        # Combine all segments
        return left_segments + right_segments
    
    def search_pivotal_tokens(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        task_type: str = "generic",
        dataset_id: Optional[str] = None,
        item_id: Optional[str] = None,
        max_generations: int = 10,
        min_prob: float = 0.2,
        max_prob: float = 0.8,
        category: Optional[str] = None
    ) -> Generator[PivotalToken, None, None]:
        """
        Search for pivotal tokens in responses to the given query.
        
        Args:
            query: The query/prompt to process
            system_prompt: Optional system prompt for chat models
            task_type: Type of task (math, code, qa, etc.)
            dataset_id: Dataset identifier
            item_id: Item identifier within the dataset
            max_generations: Maximum number of full sequences to analyze
            min_prob/max_prob: Only search queries with success probabilities in this range
            
        Yields:
            PivotalToken instances representing found pivotal tokens
        """
        # First, check if the query is in the right difficulty range
        init_prob = self.estimate_success_probability(
            query, 
            system_prompt=system_prompt,
            category=category
        )
        
        if not (min_prob <= init_prob <= max_prob):
            message = f"Query success probability ({init_prob:.2f}) outside target range [{min_prob:.2f}, {max_prob:.2f}], skipping"
            self.logger.info(message)
            
            # Print debug info if debug mode enabled
            if self.debug_mode:
                print("\n" + "#" * 70)
                print(f"DEBUG: {message}")
                print(f"QUERY: {query}")
                if category:
                    print(f"CATEGORY: {category}")
                print("#" * 70)
            return
        
        self.logger.info(f"Initial success probability: {init_prob:.4f}")
        
        # Format the full prompt
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = query
            
        tokenized = self.tokenizer(prompt, return_tensors="pt", padding=True)
        tokenized_prompt = tokenized.input_ids[0].to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device) if "attention_mask" in tokenized else None
        prompt_len = len(tokenized_prompt)
        
        # Generate multiple full sequences to analyze
        for i in range(max_generations):
            self.logger.info(f"Generating sequence {i+1}/{max_generations}")
            
            # Generate a single sequence
            with torch.no_grad():
                output = self.model.generate(
                    tokenized_prompt.unsqueeze(0),
                    attention_mask=attention_mask,
                    do_sample=True,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    min_p=self.min_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True
                )
                
            full_sequence = output.sequences[0][prompt_len:]
            full_sequence_list = full_sequence.tolist()
            
            # Skip very short generations
            if len(full_sequence_list) < 5:
                self.logger.info("Generated sequence too short, skipping")
                continue
                
            # Subdivide the sequence to find pivotal tokens
            subdivided = self.subdivide_sequence(
                query, 
                full_sequence_list, 
                prefix=tokenized_prompt.tolist(),
                system_prompt=system_prompt,
                category=category
            )
            
            # Process each segment to identify pivotal tokens
            current_prefix = tokenized_prompt.tolist()
            
            for segment in subdivided:
                # Skip empty segments
                if not segment:
                    continue
                    
                # Check if this is a single token that causes a significant probability change
                if len(segment) == 1:
                    token_id = segment[0]
                    token_str = self.tokenizer.decode([token_id])
                    
                    # Get probability before and after this token
                    prefix_str = self.tokenizer.decode(current_prefix, skip_special_tokens=False)
                    prefix_plus_token_str = self.tokenizer.decode(
                        current_prefix + segment, 
                        skip_special_tokens=False
                    )
                    
                    prob_before = self.estimate_success_probability(
                        query, 
                        prefix_str,
                        system_prompt=system_prompt,
                        category=category
                    )
                    
                    prob_after = self.estimate_success_probability(
                        query, 
                        prefix_plus_token_str,
                        system_prompt=system_prompt,
                        category=category
                    )
                    
                    prob_delta = prob_after - prob_before
                    
                    # Check if this is a pivotal token
                    if abs(prob_delta) >= self.prob_threshold:
                        # Create pivotal token object
                        pivotal_token = PivotalToken(
                            query=query,
                            pivot_context=prefix_str,
                            pivot_token=token_str,
                            pivot_token_id=token_id,
                            prob_before=prob_before,
                            prob_after=prob_after,
                            prob_delta=prob_delta,
                            model_id=self.model_name,
                            task_type=task_type,
                            dataset_id=dataset_id,
                            dataset_item_id=item_id
                        )
                        
                        # Add to storage and yield result
                        try:
                            if self.token_storage:
                                self.token_storage.add_token(pivotal_token)
                                self.logger.debug(f"Added token to storage: {pivotal_token.pivot_token}")
                        except Exception as e:
                            self.logger.error(f"Error adding token to storage: {e}")
                            
                        yield pivotal_token
                
                # Update the prefix for the next segment
                current_prefix = current_prefix + segment
    
    def find_rejected_tokens(
        self, 
        pivotal_token: PivotalToken,
        num_candidates: int = 10,
        category: Optional[str] = None
    ) -> Optional[Tuple[str, int, float]]:
        """
        For a positive pivotal token, find a rejected token that would decrease 
        the success probability.
        
        Args:
            pivotal_token: The pivotal token to find a rejected token for
            num_candidates: Number of candidate tokens to consider
            
        Returns:
            Tuple of (rejected_token, rejected_token_id, prob_after) or None
        """
        if not pivotal_token.is_positive():
            return None
            
        # Prepare the context (everything before the pivotal token)
        context_ids = self.tokenizer.encode(pivotal_token.pivot_context, return_tensors="pt").to(self.device)
        
        # Get the model's logits for the next token
        with torch.no_grad():
            outputs = self.model(context_ids)
            logits = outputs.logits[0, -1, :]
            
        # Get top candidate tokens by probability
        next_token_probs = torch.softmax(logits, dim=0)
        top_tokens = torch.topk(next_token_probs, k=num_candidates)
        
        # Test each candidate
        for i in range(num_candidates):
            token_id = top_tokens.indices[i].item()
            token_str = self.tokenizer.decode([token_id])
            
            # Skip the pivotal token itself
            if token_id == pivotal_token.pivot_token_id:
                continue
                
            # Check if this token decreases success probability
            full_context = pivotal_token.pivot_context + token_str
            prob_after = self.estimate_success_probability(
                pivotal_token.query, 
                full_context,
                category=category
            )
            
            # If this token decreases probability significantly, return it
            if pivotal_token.prob_before - prob_after >= self.prob_threshold:
                return (token_str, token_id, prob_after)
                
        return None
    
    def generate_dpo_pairs(
        self, 
        pivotal_token: PivotalToken,
        force_find_rejected: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a DPO pair from a pivotal token.
        
        Args:
            pivotal_token: The pivotal token to create a DPO pair from
            force_find_rejected: If True, always find a rejected token even for negative pivotal tokens
            
        Returns:
            Dictionary with query, chosen, rejected completions or None if not possible
        """
        # For positive pivotal tokens, we need to find a rejected token
        if pivotal_token.is_positive():
            if pivotal_token.rejected_token is None or force_find_rejected:
                rejected = self.find_rejected_tokens(pivotal_token)
                if rejected is None:
                    self.logger.warning("Could not find rejected token")
                    return None
                    
                rejected_token, rejected_token_id, rejected_prob = rejected
            else:
                rejected_token = pivotal_token.rejected_token
                rejected_token_id = pivotal_token.rejected_token_id
                rejected_prob = pivotal_token.prob_before - pivotal_token.prob_delta
                
            return {
                "prompt": pivotal_token.pivot_context,
                "chosen": pivotal_token.pivot_token,
                "rejected": rejected_token,
                "metadata": {
                    "original_query": pivotal_token.query,
                    "prob_chosen": pivotal_token.prob_after,
                    "prob_rejected": rejected_prob,
                    "task_type": pivotal_token.task_type
                }
            }
            
        # For negative pivotal tokens, the token itself is the rejected token
        else:
            # Find an accepted token (which could be the most likely next token)
            with torch.no_grad():
                context_ids = self.tokenizer.encode(pivotal_token.pivot_context, return_tensors="pt").to(self.device)
                outputs = self.model(context_ids)
                logits = outputs.logits[0, -1, :]
                
            # Get most likely next token
            next_token_id = torch.argmax(logits).item()
            next_token_str = self.tokenizer.decode([next_token_id])
            
            # Check its success probability
            full_context = pivotal_token.pivot_context + next_token_str
            prob_after = self.estimate_success_probability(
                pivotal_token.query, 
                full_context
            )
            
            # If it doesn't improve probability enough, return None
            if prob_after - pivotal_token.prob_before < self.prob_threshold:
                if not force_find_rejected:
                    return None
                    
                # If forced, take the best token anyway
                return {
                    "prompt": pivotal_token.pivot_context,
                    "chosen": next_token_str,
                    "rejected": pivotal_token.pivot_token,
                    "metadata": {
                        "original_query": pivotal_token.query,
                        "prob_chosen": prob_after,
                        "prob_rejected": pivotal_token.prob_after,
                        "task_type": pivotal_token.task_type
                    }
                }
                
            return {
                "prompt": pivotal_token.pivot_context,
                "chosen": next_token_str,
                "rejected": pivotal_token.pivot_token,
                "metadata": {
                    "original_query": pivotal_token.query,
                    "prob_chosen": prob_after,
                    "prob_rejected": pivotal_token.prob_after,
                    "task_type": pivotal_token.task_type
                }
            }
