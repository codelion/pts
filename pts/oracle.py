"""
Oracle implementations for Pivotal Token Search.

This module defines oracles that determine whether a model response is successful
for a given query or task.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union, Callable
import subprocess
import tempfile
import os
import json

logger = logging.getLogger(__name__)


class Oracle:
    """Base class for oracles that determine success of model responses."""
    
    def check_success(self, query: str, response: str) -> bool:
        """
        Check if a response is successful for a given query.
        
        Args:
            query: The original query
            response: The model's response
            
        Returns:
            Boolean indicating success
        """
        raise NotImplementedError("Subclasses must implement this method")


class DummyOracle(Oracle):
    """A dummy oracle that always returns success. Useful for export operations where actual checking isn't needed."""
    
    def __init__(self, success_rate: float = 0.5):
        """
        Initialize the dummy oracle.
        
        Args:
            success_rate: The fixed success rate to return
        """
        self.success_rate = success_rate
        
    def check_success(self, query: str, response: str) -> bool:
        """
        Always returns success based on the fixed success rate.
        
        Args:
            query: The original query
            response: The model's response
            
        Returns:
            Always returns True
        """
        return True


class MathOracle(Oracle):
    """Oracle for mathematical problem-solving tasks."""
    
    def __init__(
        self, 
        answers: Dict[str, Union[str, List[str]]] = None, 
        extract_answer_regex: str = r'(?:answer(?:\s+)?(?:is|:)?(?:\s+)?|=(?:\s+)?)(?P<answer>[^.,]+)',
        extract_boxed_regex: str = r'\\boxed{([^}]+)}',
        tolerance: float = 1e-6,
        exact_match: bool = False
    ):
        """
        Initialize the math oracle.
        
        Args:
            answers: Dictionary mapping queries to expected answers
            extract_answer_regex: Regex pattern to extract answers from responses
            extract_boxed_regex: Regex pattern to extract LaTeX boxed answers
            tolerance: Floating point comparison tolerance
            exact_match: Whether to require exact string matches for non-numeric answers
        """
        self.answers = answers or {}
        self.extract_answer_regex = extract_answer_regex
        self.extract_boxed_regex = extract_boxed_regex
        self.tolerance = tolerance
        self.exact_match = exact_match
        
    def add_answer(self, query: str, answer: Union[str, List[str]]):
        """Add or update an answer for a query."""
        self.answers[query] = answer
    
    def extract_answer(self, response: str) -> Optional[str]:
        """Extract the answer from a response using regular expressions."""
        # First check for boxed answers (common in LaTeX math)
        boxed_match = re.search(self.extract_boxed_regex, response, re.IGNORECASE)
        if boxed_match:
            return boxed_match.group(1).strip()
            
        # Then check for answers with standard patterns
        answer_match = re.search(self.extract_answer_regex, response, re.IGNORECASE)
        if answer_match:
            return answer_match.group('answer').strip()
            
        # If no match, return the last non-empty line as a fallback
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            return lines[-1]
            
        return None
        
    def normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        if answer is None:
            return ""
            
        # Remove spaces, convert to lowercase
        normalized = answer.lower().replace(" ", "")
        
        # Remove common mathematical decorators
        normalized = re.sub(r'\\(?:math)?(?:bf|text|rm|cal)', '', normalized)
        
        # Replace decimal commas with decimal points
        normalized = re.sub(r'(\d),(\d)', r'\1.\2', normalized)
        
        return normalized
        
    def _compare_numeric(self, extracted: str, expected: str) -> bool:
        """Compare numeric answers with tolerance."""
        try:
            extracted_float = float(extracted)
            expected_float = float(expected)
            return abs(extracted_float - expected_float) < self.tolerance
        except ValueError:
            return False
            
    def _is_numeric(self, s: str) -> bool:
        """Check if a string represents a numeric value."""
        try:
            float(s)
            return True
        except ValueError:
            return False
            
    def compare_answers(self, extracted: str, expected: Union[str, List[str]]) -> bool:
        """
        Compare extracted answer with expected answer(s).
        
        Args:
            extracted: The extracted answer
            expected: Expected answer or list of valid answers
            
        Returns:
            Boolean indicating whether the answers match
        """
        if extracted is None:
            return False
            
        # Normalize the extracted answer
        norm_extracted = self.normalize_answer(extracted)
        
        # Handle list of valid answers
        if isinstance(expected, list):
            for exp in expected:
                norm_expected = self.normalize_answer(exp)
                
                # Try numeric comparison for numeric answers
                if self._is_numeric(norm_extracted) and self._is_numeric(norm_expected):
                    if self._compare_numeric(norm_extracted, norm_expected):
                        return True
                        
                # String comparison
                elif self.exact_match:
                    if norm_extracted == norm_expected:
                        return True
                else:
                    if norm_extracted in norm_expected or norm_expected in norm_extracted:
                        return True
                        
            return False
            
        # Single expected answer
        else:
            norm_expected = self.normalize_answer(expected)
            
            # Try numeric comparison for numeric answers
            if self._is_numeric(norm_extracted) and self._is_numeric(norm_expected):
                return self._compare_numeric(norm_extracted, norm_expected)
                
            # String comparison
            elif self.exact_match:
                return norm_extracted == norm_expected
            else:
                return norm_extracted in norm_expected or norm_expected in norm_extracted
    
    def check_success(self, query: str, response: str) -> bool:
        """
        Check if a math response has the correct answer.
        
        Args:
            query: The original math problem
            response: The model's response including working and answer
            
        Returns:
            Boolean indicating success
        """
        # Must have an expected answer for the query
        if query not in self.answers:
            logger.warning(f"No expected answer for query: {query}")
            # Print available answers for debugging
            if self.answers:
                logger.info(f"Available queries: {list(self.answers.keys())[:3]}{'...' if len(self.answers) > 3 else ''}")
            return False
            
        expected = self.answers[query]
        extracted = self.extract_answer(response)
        
        if extracted is None:
            logger.debug(f"Could not extract answer from response: {response}")
            return False
            
        result = self.compare_answers(extracted, expected)
        logger.debug(f"Query: {query}\nExtracted: {extracted}\nExpected: {expected}\nResult: {result}")
        return result


class CodeOracle(Oracle):
    """Oracle for code-related tasks using execution-based validation."""
    
    def __init__(
        self, 
        test_cases: Dict[str, List[Dict[str, Any]]] = None,
        language: str = "python",
        timeout: int = 5,
        extract_code_regex: str = r'```(?:\w+)?\s*\n([\s\S]*?)\n```'
    ):
        """
        Initialize the code oracle.
        
        Args:
            test_cases: Dictionary mapping queries to lists of test cases
            language: Programming language (currently supports 'python')
            timeout: Execution timeout in seconds
            extract_code_regex: Regex pattern to extract code from responses
        """
        self.test_cases = test_cases or {}
        self.language = language
        self.timeout = timeout
        self.extract_code_regex = extract_code_regex
        
    def add_test_case(self, query: str, test_case: Dict[str, Any]):
        """Add a test case for a query."""
        if query not in self.test_cases:
            self.test_cases[query] = []
        self.test_cases[query].append(test_case)
        
    def extract_code(self, response: str) -> Optional[str]:
        """Extract code from a response."""
        # Look for markdown code blocks
        code_match = re.search(self.extract_code_regex, response)
        if code_match:
            return code_match.group(1)
            
        # If no markdown block, try to extract indented code blocks
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if in_code:
                if not line.strip():  # Empty line might end a code block
                    code_lines.append(line)
                elif line.startswith('    ') or line.startswith('\t'):
                    code_lines.append(line)
                else:
                    in_code = False
            elif line.startswith('    ') or line.startswith('\t'):
                in_code = True
                code_lines.append(line)
                
        if code_lines:
            return '\n'.join(code_lines)
            
        # If neither approach finds code, assume the entire response is code
        return response
    
    def run_python_test(self, code: str, test_case: Dict[str, Any]) -> bool:
        """Run a Python test case."""
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
            f.write(code + '\n\n')
            
            # Write test code
            if 'setup' in test_case:
                f.write(test_case['setup'] + '\n')
                
            if 'function_call' in test_case:
                f.write(f"result = {test_case['function_call']}\n")
                
            if 'expected_output' in test_case:
                expected = test_case['expected_output']
                if isinstance(expected, str):
                    f.write(f"assert result == '{expected}', f'Got {{result}} instead of {expected}'\n")
                else:
                    f.write(f"assert result == {expected}, f'Got {{result}} instead of {expected}'\n")
                    
            if 'assertion' in test_case:
                f.write(test_case['assertion'] + '\n')
                
            f.write("print('TEST_PASSED')")
            
            file_path = f.name
            
        try:
            result = subprocess.run(
                ['python', file_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            os.unlink(file_path)  # Delete the temporary file
            
            # Check if the test passed
            return result.returncode == 0 and 'TEST_PASSED' in result.stdout
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Test execution timed out after {self.timeout} seconds")
            os.unlink(file_path)
            return False
        except Exception as e:
            logger.error(f"Error running test: {e}")
            os.unlink(file_path)
            return False
    
    def check_success(self, query: str, response: str) -> bool:
        """
        Check if a code response passes all test cases.
        
        Args:
            query: The original coding problem
            response: The model's response containing code
            
        Returns:
            Boolean indicating success
        """
        # Must have test cases for the query
        if query not in self.test_cases:
            logger.warning(f"No test cases for query: {query}")
            return False
            
        # Extract code from the response
        code = self.extract_code(response)
        if not code:
            logger.debug("Could not extract code from response")
            return False
            
        # Run all test cases
        for test_case in self.test_cases[query]:
            if self.language == 'python':
                if not self.run_python_test(code, test_case):
                    return False
            else:
                logger.warning(f"Unsupported language: {self.language}")
                return False
                
        # All tests passed
        return True


class QAOracle(Oracle):
    """Oracle for question answering tasks."""
    
    def __init__(
        self, 
        answers: Dict[str, Union[str, List[str]]] = None,
        extract_answer_regex: str = None,
        fuzzy_match: bool = True,
        similarity_threshold: float = 0.8
    ):
        """
        Initialize the QA oracle.
        
        Args:
            answers: Dictionary mapping queries to expected answers
            extract_answer_regex: Optional regex to extract answers
            fuzzy_match: Whether to use fuzzy string matching
            similarity_threshold: Threshold for fuzzy matching
        """
        self.answers = answers or {}
        self.extract_answer_regex = extract_answer_regex
        self.fuzzy_match = fuzzy_match
        self.similarity_threshold = similarity_threshold
        
    def add_answer(self, query: str, answer: Union[str, List[str]]):
        """Add or update an answer for a query."""
        self.answers[query] = answer
    
    def extract_answer(self, response: str) -> str:
        """Extract the answer from a response."""
        if self.extract_answer_regex:
            match = re.search(self.extract_answer_regex, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
                
        # Default approach: use the full response
        return response.strip()
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
            
        # Convert to lowercase, remove punctuation
        normalized = text.lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple word overlap similarity
        words1 = set(self.normalize_text(text1).split())
        words2 = set(self.normalize_text(text2).split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def check_success(self, query: str, response: str) -> bool:
        """
        Check if a QA response contains the correct answer.
        
        Args:
            query: The original question
            response: The model's response
            
        Returns:
            Boolean indicating success
        """
        # Must have an expected answer for the query
        if query not in self.answers:
            logger.warning(f"No expected answer for query: {query}")
            return False
            
        expected = self.answers[query]
        extracted = self.extract_answer(response)
        
        # Handle list of valid answers
        if isinstance(expected, list):
            if self.fuzzy_match:
                return any(
                    self.compute_similarity(extracted, exp) >= self.similarity_threshold
                    for exp in expected
                )
            else:
                return any(
                    self.normalize_text(extracted) == self.normalize_text(exp)
                    for exp in expected
                )
        # Single expected answer
        else:
            if self.fuzzy_match:
                similarity = self.compute_similarity(extracted, expected)
                return similarity >= self.similarity_threshold
            else:
                return self.normalize_text(extracted) == self.normalize_text(expected)
