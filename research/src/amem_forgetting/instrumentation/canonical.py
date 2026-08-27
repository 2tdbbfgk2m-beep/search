"""Canonical serialization and state hashing.

Every hash in the event log is taken over ``canonical_json`` output so
that online state and replayed state can be compared byte-for-byte.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping


def canonical_json(obj: Any) -> str:
    """Serialize deterministically: sorted keys, no spaces, full unicode."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def sha256_hex(text: str | bytes) -> str:
    """Return the ``sha256:...`` prefixed digest used across the project."""
    if isinstance(text, str):
        text = text.encode("utf-8")
    return "sha256:" + sha256(text).hexdigest()


# Fields of a MemoryNote that constitute the observable memory state.
# ``links`` stays positional (list of indices) because that is the faithful
# upstream representation; resolved link ids live in the registry.
_SNAPSHOT_FIELDS = (
    "content",
    "keywords",
    "tags",
    "context",
    "links",
    "timestamp",
    "last_accessed",
    "retrieval_count",
    "importance_score",
    "category",
    "evolution_history",
)


def note_snapshot(note: Any) -> dict:
    """Project a MemoryNote-like object onto the hashable snapshot fields."""
    snapshot: dict[str, Any] = {}
    for field in _SNAPSHOT_FIELDS:
        value = getattr(note, field, None)
        if isinstance(value, list):
            value = list(value)
        snapshot[field] = value
    return snapshot


def memory_state_hash(snapshots: Mapping[str, dict]) -> str:
    """Hash the full memory state (id -> note snapshot), order-independent."""
    return sha256_hex(canonical_json(dict(snapshots)))
