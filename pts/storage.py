"""
Storage module for Pivotal Token Search.

This module provides storage utilities for saving, loading, and managing 
pivotal tokens discovered through the PTS algorithm.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Set, Union, Callable
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class TokenStorage:
    """
    Storage class for managing collections of pivotal tokens.
    Provides functionality to save, load, filter, and update token collections.
    """
    
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize the token storage.
        
        Args:
            filepath: Optional path to a JSONL file containing pivotal tokens
        """
        self.tokens = []
        self.filepath = filepath
        
        # If a filepath is provided, load tokens from it
        if filepath and os.path.exists(filepath):
            self.load(filepath)
            
    def add_token(self, token: Any):
        """
        Add a pivotal token to the storage.
        
        Args:
            token: PivotalToken instance or dictionary representation
        """
        # Convert to dictionary if it's not already
        try:
            if hasattr(token, 'to_dict'):
                token_dict = token.to_dict()
            else:
                token_dict = token
                
            # Add timestamp if not present
            if 'timestamp' not in token_dict:
                token_dict['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")
                
            self.tokens.append(token_dict)
            logger.debug(f"Added token to storage: {token_dict.get('pivot_token', 'unknown')}")
        except Exception as e:
            logger.error(f"Error adding token to storage: {e}")
            logger.error(f"Problematic token: {token}")
        
    def add_tokens(self, tokens: List[Any]):
        """
        Add multiple pivotal tokens to the storage.
        
        Args:
            tokens: List of PivotalToken instances or dictionaries
        """
        for token in tokens:
            self.add_token(token)
            
    def save(self, filepath: Optional[str] = None):
        """
        Save tokens to a JSONL file.
        
        Args:
            filepath: Path to save to (defaults to self.filepath)
        """
        filepath = filepath or self.filepath
        if not filepath:
            raise ValueError("No filepath specified for saving")
        
        # Debug information
        logger.info(f"Saving {len(self.tokens)} tokens to {filepath}")
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(os.path.abspath(filepath)) 
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Save tokens with detailed error handling
        try:
            with open(filepath, 'w') as f:
                for i, token in enumerate(self.tokens):
                    try:
                        json_str = json.dumps(token)
                        f.write(json_str + '\n')
                    except Exception as e:
                        logger.error(f"Error saving token {i}: {e}")
                        logger.error(f"Problematic token: {token}")
                        
            logger.info(f"Successfully saved {len(self.tokens)} tokens to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save tokens to {filepath}: {e}")
            raise
        
    def load(self, filepath: Optional[str] = None):
        """
        Load tokens from a JSONL file.
        
        Args:
            filepath: Path to load from (defaults to self.filepath)
        """
        filepath = filepath or self.filepath
        if not filepath:
            raise ValueError("No filepath specified for loading")
            
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return
            
        # Clear existing tokens
        self.tokens = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        token = json.loads(line)
                        self.tokens.append(token)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON line: {line}")
                        
        logger.info(f"Loaded {len(self.tokens)} tokens from {filepath}")
        self.filepath = filepath
        
    def filter(self, 
               criteria: Optional[Dict[str, Any]] = None,
               min_prob_delta: Optional[float] = None,
               max_prob_delta: Optional[float] = None,
               is_positive: Optional[bool] = None,
               custom_filter: Optional[Callable[[Dict[str, Any]], bool]] = None) -> 'TokenStorage':
        """
        Filter tokens based on various criteria.
        
        Args:
            criteria: Dictionary of field-value pairs to match
            min_prob_delta: Minimum absolute probability delta
            max_prob_delta: Maximum absolute probability delta
            is_positive: Whether the token should be positive (increase success probability)
            custom_filter: Custom filtering function
            
        Returns:
            New TokenStorage instance with filtered tokens
        """
        result = TokenStorage()
        
        for token in self.tokens:
            # Apply criteria filters
            if criteria:
                matches_criteria = True
                for key, value in criteria.items():
                    if key not in token or token[key] != value:
                        matches_criteria = False
                        break
                if not matches_criteria:
                    continue
                    
            # Apply probability delta filters
            if min_prob_delta is not None:
                if abs(token.get('prob_delta', 0)) < min_prob_delta:
                    continue
                    
            if max_prob_delta is not None:
                if abs(token.get('prob_delta', 0)) > max_prob_delta:
                    continue
                    
            # Apply positive/negative filter
            if is_positive is not None:
                token_is_positive = token.get('prob_delta', 0) > 0
                if token_is_positive != is_positive:
                    continue
                    
            # Apply custom filter
            if custom_filter and not custom_filter(token):
                continue
                
            # Token passes all filters
            result.add_token(token)
            
        return result
    
    def __len__(self):
        """Return the number of tokens in storage."""
        return len(self.tokens)
    
    def __getitem__(self, index):
        """Get token at the specified index."""
        return self.tokens[index]
    
    def __iter__(self):
        """Iterate through tokens."""
        return iter(self.tokens)
