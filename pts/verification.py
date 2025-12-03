"""
CoT Verification module for Pivotal Token Search.

This module implements arithmetic verification using ground truth computation
and attention pattern analysis, inspired by the CRV (Circuit-based Reasoning
Verification) paper.
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Optional
import torch

logger = logging.getLogger(__name__)

# Try to import math_verify for robust expression parsing
try:
    from math_verify import parse, verify as math_verify_check
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    MATH_VERIFY_AVAILABLE = True
    logger.info("math_verify library available for robust expression verification")
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    logger.warning("math_verify not available, falling back to regex-based verification")


class ArithmeticVerifier:
    """
    Verify arithmetic operations using ground truth + attention analysis.

    Inspired by CRV (Circuit-based Reasoning Verification) paper which shows
    that correct vs incorrect reasoning has distinct "structural fingerprints"
    in internal model states.

    This verifier:
    1. Extracts arithmetic expressions from sentences
    2. Computes ground truth to verify correctness
    3. Analyzes attention patterns for interpretability research
    4. Uses math_verify for robust LaTeX/expression parsing when available
    """

    def __init__(self, model, tokenizer, device: str):
        """
        Initialize the ArithmeticVerifier.

        Args:
            model: The language model (for attention analysis)
            tokenizer: The tokenizer for the model
            device: Device to run computations on ('cuda', 'cpu', 'mps')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Regex for simple arithmetic expressions like "5 + 3 = 8"
        # Handles integers and decimals
        self.arithmetic_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'
        )

        # Additional patterns for more complex expressions
        self.multi_op_pattern = re.compile(
            r'(\d+(?:\.\d+)?(?:\s*[+\-*/]\s*\d+(?:\.\d+)?)+)\s*=\s*(\d+(?:\.\d+)?)'
        )

        # math_verify extraction configs if available
        if MATH_VERIFY_AVAILABLE:
            self.latex_config = LatexExtractionConfig()
            self.expr_config = ExprExtractionConfig()

        logger.info(f"ArithmeticVerifier initialized on device: {device}")

    def extract_arithmetic_ops(self, sentence: str) -> List[Dict[str, Any]]:
        """
        Extract arithmetic operations from a sentence.

        Args:
            sentence: The sentence to extract operations from

        Returns:
            List of dictionaries containing operation details
        """
        ops = []

        # Extract simple binary operations (a op b = c)
        for match in self.arithmetic_pattern.finditer(sentence):
            ops.append({
                "operand1": float(match.group(1)),
                "operator": match.group(2),
                "operand2": float(match.group(3)),
                "stated_result": float(match.group(4)),
                "expression": match.group(0),
                "span": match.span(),
                "type": "binary"
            })

        return ops

    def compute_ground_truth(self, op: Dict[str, Any]) -> float:
        """
        Compute the correct result for an arithmetic operation.

        Args:
            op: Dictionary containing operand1, operator, operand2

        Returns:
            The correct result of the operation
        """
        a, b = op["operand1"], op["operand2"]
        operator = op["operator"]

        if operator == "+":
            return a + b
        elif operator == "-":
            return a - b
        elif operator == "*":
            return a * b
        elif operator == "/":
            return a / b if b != 0 else float('inf')
        else:
            return float('nan')

    def verify_with_math_verify(
        self,
        expression: str,
        expected_result: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Use math_verify for robust expression verification.

        Args:
            expression: The mathematical expression
            expected_result: The expected result

        Returns:
            Tuple of (is_correct, error_message)
        """
        if not MATH_VERIFY_AVAILABLE:
            return False, "math_verify not available"

        try:
            parsed_expr = parse(expression, [self.latex_config, self.expr_config])
            parsed_expected = parse(expected_result, [self.latex_config, self.expr_config])
            is_correct = math_verify_check(parsed_expr, parsed_expected)
            return is_correct, None
        except Exception as e:
            logger.debug(f"math_verify parsing failed: {e}")
            return False, str(e)

    def compute_attention_metrics(
        self,
        sentence: str,
        prefix_context: str = ""
    ) -> Tuple[float, float]:
        """
        Compute attention entropy and focus score for interpretability research.

        CRV insight: Models show distinctive attention patterns when correctly
        computing vs. when hallucinating results. Lower entropy suggests more
        focused, confident computation.

        Args:
            sentence: The sentence to analyze
            prefix_context: Context preceding the sentence

        Returns:
            Tuple of (attention_entropy, focus_score)
        """
        full_text = f"{prefix_context} {sentence}" if prefix_context else sentence

        try:
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)

            # Get attention weights from model
            attentions = outputs.attentions

            if attentions is None or len(attentions) == 0:
                logger.warning("No attention weights available from model")
                return 0.0, 0.0

            # Use last 4 layers (most task-relevant per CRV findings)
            num_layers = len(attentions)
            layers_to_use = min(4, num_layers)
            last_layers = attentions[-layers_to_use:]

            # Average attention across selected layers and heads
            # Each attention tensor shape: (batch, heads, seq_len, seq_len)
            avg_attn = torch.stack([a.mean(dim=(0, 1)) for a in last_layers]).mean(dim=0)

            # Convert to float32 to avoid float16 precision issues with log
            avg_attn = avg_attn.float()

            # Compute entropy (lower = more focused)
            # Use clamp to avoid log(0) - works better than epsilon with float16 source
            avg_attn_clamped = torch.clamp(avg_attn, min=1e-10)
            entropy = -(avg_attn * torch.log(avg_attn_clamped)).sum().item()

            # Normalize entropy by sequence length for comparability
            seq_len = avg_attn.shape[0]
            max_entropy = seq_len * torch.log(torch.tensor(seq_len, dtype=torch.float32)).item()
            entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Compute focus score (how concentrated attention is on key tokens)
            # Use the Gini coefficient of attention weights as focus measure
            # Higher Gini = more unequal distribution = more focused attention
            sorted_attn, _ = torch.sort(avg_attn.flatten())
            n = sorted_attn.shape[0]
            cumsum = torch.cumsum(sorted_attn, dim=0)
            # Gini coefficient: measures inequality in attention distribution
            gini = (2 * torch.arange(1, n + 1, device=avg_attn.device).float() - n - 1).dot(sorted_attn) / (n * sorted_attn.sum())
            focus_score = gini.item()
            # Normalize to 0-1 range (Gini is already roughly 0-1)
            focus_score = max(0.0, min(1.0, focus_score))

            return entropy, focus_score

        except Exception as e:
            logger.warning(f"Error computing attention metrics: {e}")
            return 0.0, 0.0

    def verify_sentence(
        self,
        sentence: str,
        prefix_context: str = ""
    ) -> Dict[str, Any]:
        """
        Verify arithmetic in a sentence using ground truth + attention analysis.

        This is the main entry point for verification. It:
        1. Extracts arithmetic operations from the sentence
        2. Verifies each operation against ground truth
        3. Computes attention metrics for interpretability
        4. Returns a combined verification score

        Args:
            sentence: The sentence to verify
            prefix_context: Context preceding the sentence

        Returns:
            Dictionary containing verification results:
            - has_arithmetic: Whether arithmetic was found
            - verification_score: Combined score (0.0 = wrong, 1.0 = correct)
            - verification_method: Method used for verification
            - arithmetic_errors: List of detected errors
            - attention_entropy: Entropy of attention distribution
            - attention_focus_score: How focused the attention is
        """
        ops = self.extract_arithmetic_ops(sentence)

        # No arithmetic found
        if not ops:
            return {
                "has_arithmetic": False,
                "verification_score": None,
                "verification_method": None,
                "arithmetic_errors": [],
                "attention_entropy": None,
                "attention_focus_score": None
            }

        # Verify each operation
        errors = []
        for op in ops:
            correct_result = self.compute_ground_truth(op)

            # Check if stated result matches ground truth (with tolerance)
            if abs(op["stated_result"] - correct_result) > 1e-6:
                errors.append({
                    "expression": op["expression"],
                    "stated": op["stated_result"],
                    "correct": correct_result,
                    "operator": op["operator"]
                })

        # Compute attention metrics for interpretability research
        entropy, focus = self.compute_attention_metrics(sentence, prefix_context)

        # Combine verification score:
        # - 70% weight on ground truth correctness
        # - 30% weight on attention focus (normalized)
        ground_truth_score = 0.0 if errors else 1.0

        # Entropy is now normalized to 0-1 range (lower = more focused = better)
        # Focus score is already 0-1 (higher = more focused = better)
        # Combine them: (1 - entropy) gives higher score for lower entropy
        attention_confidence = (1.0 - entropy + focus) / 2.0 if entropy > 0 else 0.5

        verification_score = (
            ground_truth_score * 0.7 +
            attention_confidence * 0.3
        )

        return {
            "has_arithmetic": True,
            "verification_score": verification_score,
            "verification_method": "combined",
            "arithmetic_errors": errors,
            "attention_entropy": entropy,
            "attention_focus_score": focus
        }

    def batch_verify(
        self,
        sentences: List[str],
        prefix_contexts: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Verify multiple sentences in batch.

        Args:
            sentences: List of sentences to verify
            prefix_contexts: Optional list of contexts for each sentence

        Returns:
            List of verification results for each sentence
        """
        if prefix_contexts is None:
            prefix_contexts = [""] * len(sentences)

        results = []
        for sentence, context in zip(sentences, prefix_contexts):
            result = self.verify_sentence(sentence, context)
            results.append(result)

        return results
