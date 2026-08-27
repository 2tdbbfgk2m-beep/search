"""Instrumentation primitives for STAGE 1.

The modules here wrap an unmodified ``AgenticMemorySystem`` and record:
provenance (source turns), stable memory ids, content/embedding hashes,
serialized sizes, evolution before/after versions, and an append-only
hash-chained event log that can replay the full memory state.
"""

from amem_forgetting.instrumentation.canonical import (
    canonical_json,
    sha256_hex,
    note_snapshot,
    memory_state_hash,
)
from amem_forgetting.instrumentation.events import Event, EventLog
from amem_forgetting.instrumentation.registry import MemoryRecord, MemoryRegistry
from amem_forgetting.instrumentation.evolution import (
    EvolutionRecord,
    EvolutionTracker,
)
from amem_forgetting.instrumentation.wrapper import (
    InstrumentedMemorySystem,
    LLMCallRecorder,
    retriever_document,
)
from amem_forgetting.instrumentation.replay import EventReplayer

__all__ = [
    "canonical_json",
    "sha256_hex",
    "note_snapshot",
    "memory_state_hash",
    "Event",
    "EventLog",
    "MemoryRecord",
    "MemoryRegistry",
    "EvolutionRecord",
    "EvolutionTracker",
    "InstrumentedMemorySystem",
    "LLMCallRecorder",
    "retriever_document",
    "EventReplayer",
]
