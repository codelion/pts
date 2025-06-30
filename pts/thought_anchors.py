"""
Thought Anchors implementation using Pivotal Token Search principles.

This module implements sentence-level analysis to find "thought anchors" - 
critical reasoning steps that have outsized importance in model reasoning traces.
"""

import time
import logging
import re
from typing import List, Dict, Tuple, Any, Optional, Callable, Generator
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
import numpy as np

from .oracle import Oracle
from .storage import TokenStorage

logger = logging.getLogger(__name__)


@dataclass
class ThoughtAnchor:
    """Represents a thought anchor - a sentence that significantly impacts success probability."""
    
    # Core fields (required)
    query: str
    sentence: str  # The full sentence text
    sentence_id: int  # Position in the reasoning trace
    prefix_context: str  # All sentences before this one
    
    # Probability metrics (required)
    prob_with_sentence: float  # Success probability with this sentence
    prob_without_sentence: float  # Success probability when sentence is resampled with different meaning
    prob_delta: float  # Change in success probability (with - without)
    
    # Metadata (required)
    model_id: str
    task_type: str
    
    # Optional fields with defaults
    sentence_category: Optional[str] = None  # e.g., "plan_generation", "uncertainty_management"
    alternatives_tested: List[str] = field(default_factory=list)  # Alternative sentences tested
    dependency_sentences: List[int] = field(default_factory=list)  # Other sentence IDs this depends on
    dataset_id: Optional[str] = None
    dataset_item_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def is_positive(self) -> bool:
        """Return True if this sentence increases success probability."""
        return self.prob_delta > 0
    
    def importance_score(self) -> float:
        """Return the absolute importance of this thought anchor."""
        return abs(self.prob_delta)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the thought anchor to a dictionary for storage."""
        return {
            "model_id": self.model_id,
            "query": self.query,
            "sentence": self.sentence,
            "sentence_id": self.sentence_id,
            "prefix_context": self.prefix_context,
            "prob_with_sentence": self.prob_with_sentence,
            "prob_without_sentence": self.prob_without_sentence,
            "prob_delta": self.prob_delta,
            "importance_score": self.importance_score(),
            "is_positive": self.is_positive(),
            "sentence_category": self.sentence_category,
            "alternatives_tested": self.alternatives_tested,
            "dependency_sentences": self.dependency_sentences,
            "task_type": self.task_type,
            "dataset_id": self.dataset_id,
            "dataset_item_id": self.dataset_item_id,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThoughtAnchor':
        """Create a ThoughtAnchor instance from a dictionary."""
        return cls(
            model_id=data["model_id"],
            query=data["query"],
            sentence=data["sentence"],
            sentence_id=data["sentence_id"],
            prefix_context=data["prefix_context"],
            prob_with_sentence=data["prob_with_sentence"],
            prob_without_sentence=data["prob_without_sentence"],
            prob_delta=data["prob_delta"],
            sentence_category=data.get("sentence_category"),
            alternatives_tested=data.get("alternatives_tested", []),
            dependency_sentences=data.get("dependency_sentences", []),
            task_type=data["task_type"],
            dataset_id=data.get("dataset_id"),
            dataset_item_id=data.get("dataset_item_id"),
            timestamp=data.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        )


class SentenceSegmenter:
    """Utility class for segmenting reasoning traces into sentences."""
    
    def __init__(self):
        """Initialize the sentence segmenter."""
        pass
    
    def segment_reasoning_trace(self, text: str) -> List[str]:
        """
        Segment a reasoning trace into sentences.
        
        Args:
            text: The reasoning trace text
            
        Returns:
            List of sentences
        """
        # Remove thinking tags if present
        text = self._extract_final_response(text)
        
        # Enhanced sentence splitting for math problems
        # Split on periods, exclamation, question marks, and newlines
        # Also split on common math conclusion patterns
        sentences = re.split(r'[.!?]+|\n+|(?:Therefore)|(?:Thus)|(?:So,)|(?:Hence)', text)
        
        # Also split on equation lines that start a new thought
        additional_splits = []
        for sentence in sentences:
            # Split on lines that start with "=" or contain standalone equations
            parts = re.split(r'\n(?=[=])|(?<=[0-9])\s*\n(?=[0-9])', sentence)
            additional_splits.extend(parts)
        sentences = additional_splits
        
        # Clean and filter sentences with more lenient criteria for math
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip empty sentences
            if not sentence:
                continue
                
            # Skip purely formatting sentences
            if re.match(r'^[=\-#*]+$', sentence.strip()):
                continue
                
            # For math problems, keep shorter sentences that contain numbers or operations
            if re.search(r'\d|[+\-*/=]|\\boxed|answer|result', sentence, re.IGNORECASE):
                # Keep math-related sentences even if short
                if len(sentence) >= 5:  # Very minimal length requirement
                    cleaned_sentences.append(sentence)
            elif len(sentence) >= 15 and len(sentence.split()) >= 3:
                # For non-math sentences, use original criteria
                cleaned_sentences.append(sentence)
        
        # Don't merge sentences for math problems to preserve step structure
        # merged_sentences = self._merge_incomplete_sentences(cleaned_sentences)
        
        return cleaned_sentences
    
    def _extract_final_response(self, response: str) -> str:
        """Extract the final response after thinking tags if present."""
        think_match = re.search(r'</think>(.*)', response, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()
        return response
    
    def _merge_incomplete_sentences(self, sentences: List[str]) -> List[str]:
        """Merge sentences that appear to be fragments."""
        if not sentences:
            return sentences
            
        merged = []
        current = sentences[0]
        
        for i in range(1, len(sentences)):
            next_sentence = sentences[i]
            
            # Check if current sentence seems incomplete
            if self._is_incomplete_sentence(current):
                current = current + ". " + next_sentence
            else:
                merged.append(current)
                current = next_sentence
        
        # Add the last sentence
        merged.append(current)
        
        return merged
    
    def _is_incomplete_sentence(self, sentence: str) -> bool:
        """Check if a sentence appears to be incomplete."""
        # Very short sentences are likely incomplete
        if len(sentence) < 20:
            return True
            
        # Sentences ending with conjunctions or prepositions
        incomplete_endings = ['and', 'or', 'but', 'with', 'for', 'in', 'on', 'at', 'by', 'to']
        words = sentence.split()
        if words and words[-1].lower() in incomplete_endings:
            return True
            
        # Sentences that don't end with proper punctuation
        if not sentence.strip().endswith(('.', '!', '?', ':', ';')):
            return True
            
        return False


class SentenceClassifier:
    """Utility class for classifying sentence types in reasoning traces."""
    
    def __init__(self):
        """Initialize the sentence classifier."""
        # Define sentence patterns based on the Thought Anchors paper taxonomy
        self.category_patterns = {
            "problem_setup": [
                r"(?:i need to|let me|the problem|given|we have|the question)",
                r"(?:find|solve|determine|calculate)",
                r"(?:understand|parse|interpret)"
            ],
            "plan_generation": [
                r"(?:i'll|let's|my approach|strategy|plan|method)",
                r"(?:first|next|then|finally|step)",
                r"(?:alternatively|instead|different approach)"
            ],
            "fact_retrieval": [
                r"(?:i (?:know|remember) that|recall that|it's known that)",
                r"(?:the formula|the rule|the principle)",
                r"(?:according to|based on)"
            ],
            "active_computation": [
                r"(?:calculate|compute|solving|substituting)",
                r"(?:\d+\s*[+\-*/]\s*\d+|equals|=)",
                r"(?:the result is|this gives us)"
            ],
            "uncertainty_management": [
                r"(?:wait|hmm|actually|hold on)",
                r"(?:i'm not sure|uncertain|confused)",
                r"(?:let me reconsider|rethink|double-check)"
            ],
            "result_consolidation": [
                r"(?:therefore|so|thus|hence)",
                r"(?:in conclusion|to summarize|overall)",
                r"(?:the answer is|final result)"
            ],
            "self_checking": [
                r"(?:let me verify|check|confirm)",
                r"(?:is this correct|does this make sense)",
                r"(?:reviewing|validating|testing)"
            ],
            "final_answer_emission": [
                r"(?:the final answer|answer:|solution:)",
                r"(?:boxed|therefore.*answer)"
            ]
        }
    
    def classify_sentence(self, sentence: str) -> Optional[str]:
        """
        Classify a sentence into one of the reasoning categories.
        
        Args:
            sentence: The sentence to classify
            
        Returns:
            Category name or None if no match
        """
        sentence_lower = sentence.lower()
        
        # Check each category
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    return category
        
        return None


class ThoughtAnchorSearcher:
    """
    Implements Thought Anchor Search - sentence-level adaptation of PTS
    to find critical reasoning steps in model traces.
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
        num_samples: int = 20,
        batch_size: int = 3,
        similarity_threshold: float = 0.8,
        trust_remote_code: bool = True,
        log_level: int = logging.INFO,
        debug_mode: bool = False
    ):
        """
        Initialize the ThoughtAnchorSearcher.
        
        Args:
            model_name: HuggingFace model ID or path to local model
            tokenizer_name: HuggingFace tokenizer ID (if different from model)
            oracle: Oracle instance for evaluating success
            device: Device to run the model on ('cpu', 'cuda', etc.)
            prob_threshold: Minimum change in probability to be considered a thought anchor
            temperature: Temperature for sampling from the model
            max_new_tokens: Maximum number of new tokens to generate
            num_samples: Number of samples to estimate success probability
            batch_size: Batch size for generating multiple samples
            similarity_threshold: Threshold for semantic similarity (0.8 = median from paper)
            trust_remote_code: Whether to trust remote code in model loading
            log_level: Logging level (e.g., logging.INFO)
            debug_mode: Whether to enable debug output
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
        self.similarity_threshold = similarity_threshold
        self.trust_remote_code = trust_remote_code
        self.oracle = oracle
        self.debug_mode = debug_mode
        
        # Initialize utilities
        self.segmenter = SentenceSegmenter()
        self.classifier = SentenceClassifier()
        
        # Load sentence embedding model for semantic similarity
        self.logger.info("Loading sentence embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
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
    
    def compute_sentence_similarity(self, sentence1: str, sentence2: str) -> float:
        """
        Compute semantic similarity between two sentences using embeddings.
        
        Args:
            sentence1: First sentence
            sentence2: Second sentence
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            embeddings = self.embedding_model.encode([sentence1, sentence2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            self.logger.warning(f"Error computing similarity: {e}")
            return 0.0
    
    def estimate_success_probability_with_sentences(
        self, 
        query: str, 
        sentences: List[str],
        num_samples: Optional[int] = None,
        system_prompt: Optional[str] = None,
        category: Optional[str] = None
    ) -> float:
        """
        Estimate success probability when model completes from a given sentence sequence.
        
        Args:
            query: The original query/prompt
            sentences: List of sentences to use as prefix
            num_samples: Number of samples to generate
            system_prompt: Optional system prompt for chat models
            category: Optional category for specialized prompting
            
        Returns:
            Estimated probability of success (0.0 to 1.0)
        """
        if self.oracle is None:
            raise ValueError("Oracle must be provided to estimate success probability")
        
        # Combine sentences into a prefix
        sentence_prefix = " ".join(sentences) if sentences else ""
        
        # Check cache first
        cache_key = (query, sentence_prefix, num_samples, system_prompt)
        if cache_key in self.prob_cache:
            return self.prob_cache[cache_key]
        
        num_samples = num_samples or self.num_samples
        self.logger.debug(f"Estimating success probability with {num_samples} samples")
        
        # Format the prompt
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
        
        # Add sentence prefix if available
        if sentence_prefix:
            prompt = prompt + " " + sentence_prefix
            
        # Tokenize prompt
        tokenized = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = tokenized.input_ids.to(self.device)
        
        # Ensure attention mask is properly set
        if "attention_mask" in tokenized:
            attention_mask = tokenized.attention_mask.to(self.device)
        else:
            # Create attention mask manually if not provided
            attention_mask = torch.ones_like(input_ids, device=self.device)
        
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
                        print(f"SENTENCE PREFIX: {sentence_prefix}")
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
    
    def generate_alternative_sentence(
        self, 
        original_sentence: str, 
        prefix_sentences: List[str],
        query: str,
        max_attempts: int = 5
    ) -> Optional[str]:
        """
        Generate an alternative sentence that has different semantic meaning.
        
        Args:
            original_sentence: The original sentence to replace
            prefix_sentences: Sentences that come before this one
            query: The original query
            max_attempts: Maximum attempts to generate a different sentence
            
        Returns:
            Alternative sentence or None if unable to generate one
        """
        # Create context for generation
        context = " ".join(prefix_sentences) if prefix_sentences else ""
        
        # Add some instruction to generate a different approach
        generation_prompts = [
            f"{context} Let me try a different approach:",
            f"{context} Alternatively, I could:",
            f"{context} Another way to think about this:",
            f"{context} Instead, let me:",
            f"{context} Actually, maybe I should:",
        ]
        
        for attempt in range(max_attempts):
            try:
                # Use a different generation prompt each time
                prompt_to_use = generation_prompts[attempt % len(generation_prompts)]
                
                # Tokenize and generate
                tokenized = self.tokenizer(prompt_to_use, return_tensors="pt", padding=True)
                input_ids = tokenized.input_ids.to(self.device)
                
                # Ensure attention mask is properly set
                if "attention_mask" in tokenized:
                    attention_mask = tokenized.attention_mask.to(self.device)
                else:
                    attention_mask = torch.ones_like(input_ids, device=self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        max_new_tokens=50,  # Shorter for single sentence
                        temperature=self.temperature * 1.2,  # Slightly higher temperature for diversity
                        top_p=self.top_p,
                        top_k=self.top_k,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True
                    )
                
                # Extract the generated text
                generated = self.tokenizer.decode(outputs.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
                
                # Extract the first sentence
                generated_sentences = self.segmenter.segment_reasoning_trace(generated)
                if not generated_sentences:
                    continue
                
                alternative = generated_sentences[0]
                
                # Check if it's semantically different
                similarity = self.compute_sentence_similarity(original_sentence, alternative)
                if similarity < self.similarity_threshold:
                    return alternative
                    
            except Exception as e:
                self.logger.debug(f"Error generating alternative sentence: {e}")
                continue
        
        return None
    
    def search_thought_anchors(
        self,
        query: str,
        reasoning_trace: str,
        system_prompt: Optional[str] = None,
        task_type: str = "generic",
        dataset_id: Optional[str] = None,
        item_id: Optional[str] = None,
        min_prob: float = 0.2,
        max_prob: float = 0.8,
        category: Optional[str] = None
    ) -> Generator[ThoughtAnchor, None, None]:
        """
        Search for thought anchors in a reasoning trace.
        
        Args:
            query: The original query/prompt
            reasoning_trace: The model's reasoning trace to analyze
            system_prompt: Optional system prompt for chat models
            task_type: Type of task (math, code, qa, etc.)
            dataset_id: Dataset identifier
            item_id: Item identifier within the dataset
            min_prob/max_prob: Only search queries with success probabilities in this range
            category: Optional category for specialized prompting
            
        Yields:
            ThoughtAnchor instances representing found thought anchors
        """
        # First, check if the query is in the right difficulty range
        init_prob = self.estimate_success_probability_with_sentences(
            query, [],  # No prefix sentences
            system_prompt=system_prompt,
            category=category
        )
        
        if not (min_prob <= init_prob <= max_prob):
            message = f"Query success probability ({init_prob:.2f}) outside target range [{min_prob:.2f}, {max_prob:.2f}], skipping"
            self.logger.info(message)
            
            if self.debug_mode:
                print("\n" + "#" * 70)
                print(f"DEBUG: {message}")
                print(f"QUERY: {query}")
                if category:
                    print(f"CATEGORY: {category}")
                print("#" * 70)
            return
        
        self.logger.info(f"Initial success probability: {init_prob:.4f}")
        
        # Segment the reasoning trace into sentences
        sentences = self.segmenter.segment_reasoning_trace(reasoning_trace)
        
        if len(sentences) < 2:
            self.logger.info("Too few sentences in reasoning trace, skipping")
            return
        
        self.logger.info(f"Analyzing {len(sentences)} sentences for thought anchors")
        
        # Analyze each sentence
        thought_anchors_found = []
        
        for i, sentence in enumerate(sentences):
            self.logger.info(f"Analyzing sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
            
            # Get prefix sentences (all sentences before this one)
            prefix_sentences = sentences[:i]
            
            # Estimate success probability WITH this sentence
            prob_with = self.estimate_success_probability_with_sentences(
                query,
                sentences[:i+1],  # Include this sentence
                system_prompt=system_prompt,
                category=category
            )
            
            # Generate alternative sentences and test them
            alternatives_tested = []
            prob_without_list = []
            
            for attempt in range(3):  # Try 3 alternative sentences
                alternative = self.generate_alternative_sentence(
                    sentence, prefix_sentences, query
                )
                
                if alternative:
                    # Check semantic similarity
                    similarity = self.compute_sentence_similarity(sentence, alternative)
                    
                    if similarity < self.similarity_threshold:
                        # Test success probability with alternative
                        alternative_sentences = prefix_sentences + [alternative]
                        prob_with_alternative = self.estimate_success_probability_with_sentences(
                            query,
                            alternative_sentences,
                            system_prompt=system_prompt,
                            category=category
                        )
                        
                        alternatives_tested.append(alternative)
                        prob_without_list.append(prob_with_alternative)
            
            if prob_without_list:
                # Use average probability across alternatives
                prob_without = sum(prob_without_list) / len(prob_without_list)
            else:
                # Fallback: estimate probability without this sentence
                prob_without = self.estimate_success_probability_with_sentences(
                    query,
                    prefix_sentences,  # Just the prefix, skip this sentence
                    system_prompt=system_prompt,
                    category=category
                )
                alternatives_tested = ["<no semantic alternative found>"]
            
            # Calculate probability delta
            prob_delta = prob_with - prob_without
            
            # Check if this is a thought anchor
            if abs(prob_delta) >= self.prob_threshold:
                # Classify the sentence
                sentence_category = self.classifier.classify_sentence(sentence)
                
                # Create thought anchor object
                thought_anchor = ThoughtAnchor(
                    query=query,
                    sentence=sentence,
                    sentence_id=i,
                    prefix_context=" ".join(prefix_sentences),
                    prob_with_sentence=prob_with,
                    prob_without_sentence=prob_without,
                    prob_delta=prob_delta,
                    sentence_category=sentence_category,
                    alternatives_tested=alternatives_tested,
                    dependency_sentences=[],  # Could be enhanced with dependency analysis
                    model_id=self.model_name,
                    task_type=task_type,
                    dataset_id=dataset_id,
                    dataset_item_id=item_id
                )
                
                thought_anchors_found.append(thought_anchor)
                
                self.logger.info(f"Found thought anchor: {sentence[:50]}... (Δp={prob_delta:.3f})")
                
                yield thought_anchor
        
        self.logger.info(f"Found {len(thought_anchors_found)} thought anchors for query")


class ThoughtAnchorStorage(TokenStorage):
    """
    Storage class specifically for thought anchors.
    Extends TokenStorage with thought anchor specific functionality.
    """
    
    def __init__(self, filepath: Optional[str] = None):
        """Initialize the thought anchor storage."""
        super().__init__(filepath)
    
    def add_thought_anchor(self, anchor: ThoughtAnchor):
        """Add a thought anchor to the storage."""
        self.add_token(anchor)
    
    def get_anchors_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all thought anchors of a specific category."""
        return [anchor for anchor in self.tokens if anchor.get("sentence_category") == category]
    
    def get_most_important_anchors(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the N most important thought anchors by importance score."""
        sorted_anchors = sorted(
            self.tokens, 
            key=lambda x: x.get("importance_score", 0), 
            reverse=True
        )
        return sorted_anchors[:n]
    
    def get_anchor_summary(self) -> Dict[str, Any]:
        """Get a summary of the thought anchors in storage."""
        if not self.tokens:
            return {"total_anchors": 0}
        
        # Count by category
        category_counts = {}
        total_positive = 0
        total_negative = 0
        importance_scores = []
        
        for anchor in self.tokens:
            category = anchor.get("sentence_category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
            
            if anchor.get("is_positive", False):
                total_positive += 1
            else:
                total_negative += 1
                
            importance_scores.append(anchor.get("importance_score", 0))
        
        return {
            "total_anchors": len(self.tokens),
            "positive_anchors": total_positive,
            "negative_anchors": total_negative,
            "category_distribution": category_counts,
            "average_importance": sum(importance_scores) / len(importance_scores) if importance_scores else 0,
            "max_importance": max(importance_scores) if importance_scores else 0,
            "min_importance": min(importance_scores) if importance_scores else 0
        }
