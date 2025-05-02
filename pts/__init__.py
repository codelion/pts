"""
Pivotal Token Search (PTS) - Find pivotal tokens in language model generations.

This module implements the Pivotal Token Search algorithm from the Phi-4 Technical Report,
which identifies tokens that significantly impact the probability of generating a successful response.
"""

from .core import PivotalToken, PivotalTokenSearcher
from .oracle import Oracle, MathOracle, CodeOracle, QAOracle
from .dataset import load_dataset
from .exporters import TokenExporter
from .storage import TokenStorage

__version__ = "0.1.0"
