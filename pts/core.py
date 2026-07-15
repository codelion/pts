"""
Compatibility shim for the v1 ``pts.core`` module.

Token PTS now lives in ``pts.searchers.token``. This module re-exports it under
the legacy names, so existing imports keep resolving::

    from pts.core import PivotalToken, PivotalTokenSearcher   # still works

What changed, and what that means for legacy code (see ``docs/migration.md``):

* ``PivotalToken(...)`` is now a *constructor function* that returns a
  ``CausalReasoningEvent``. It takes the v1 keyword arguments unchanged, so
  construction is source-compatible.
* The returned object stores the token under ``.label`` and its context under
  ``.context`` (not ``.pivot_token`` / ``.pivot_context``), and ``is_positive``
  is a bool attribute rather than a method. Use
  ``pts.events.to_legacy_pivotal_token(event)`` to get the old dict shape back.
* v1 *JSONL files* load unchanged -- ``EventStorage`` migrates them on read --
  which is the compatibility that actually matters for published datasets.
"""

from typing import Any, Optional

from .events import CausalReasoningEvent, make_token_event, to_legacy_pivotal_token
from .searchers.token import PivotalTokenSearcher, TokenPTSSearcher

__all__ = [
    "PivotalToken",
    "PivotalTokenSearcher",
    "TokenPTSSearcher",
    "to_legacy_pivotal_token",
]


def PivotalToken(
    query: str,
    pivot_context: str,
    pivot_token: str,
    pivot_token_id: int,
    prob_before: float,
    prob_after: float,
    model_id: str,
    task_type: str,
    prob_delta: Optional[float] = None,
    dataset_id: Optional[str] = None,
    dataset_item_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    **kwargs: Any,
) -> CausalReasoningEvent:
    """v1 ``PivotalToken`` constructor, returning a unified event.

    ``prob_delta`` is accepted for signature compatibility but recomputed from
    ``prob_after - prob_before``, which is what the legacy format did anyway.
    """
    event = make_token_event(
        query=query,
        context=pivot_context,
        token=pivot_token,
        token_id=pivot_token_id,
        prob_before=prob_before,
        prob_after=prob_after,
        model_id=model_id,
        task_type=task_type,
        dataset_id=dataset_id,
        dataset_item_id=dataset_item_id,
        metadata=kwargs or None,
    )
    if timestamp:
        event.timestamp = timestamp
    return event
