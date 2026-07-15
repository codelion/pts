"""
Compatibility shim for the v1 ``pts.storage`` module.

``TokenStorage`` now subclasses ``EventStorage`` and holds unified events. It
keeps the v1 surface -- ``add_token``, ``add_tokens``, ``save``, ``load``,
``filter``, ``.tokens`` -- and reads legacy JSONL files unchanged, migrating each
record on the way in.

One behavioural change worth knowing about: ``add_token`` is now idempotent by
``event_id``. the legacy code let the searcher and the CLI each write the same token, which
silently doubled every dataset produced through the CLI.
"""

from .event_storage import EventStorage, TokenStorage

__all__ = ["TokenStorage", "EventStorage"]
