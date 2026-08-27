"""Memory registry: the plan section 3.9 record for every memory note.

The registry lives beside (not inside) the upstream MemoryNote objects, so
the original A-MEM classes are never modified. Records are the join point
between source turns, the link graph, and the evolution DAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Mapping

from amem_forgetting.instrumentation.canonical import sha256_hex


@dataclass
class MemoryRecord:
    memory_id: str
    source_turn_ids: list
    created_step: int
    raw_content: str
    keywords: list
    tags: list
    context: str
    embedding_model: str
    embedding_hash: str
    link_ids: list  # resolved stable ids (deduplicated, insertion order)
    parent_memory_ids: list  # link-graph parents, mirrors link_ids
    evolution_ancestors: list  # trigger ids, transitively, cycle-safe
    version: int
    content_hash: str
    token_size: int
    byte_size: int
    active_state: str = "active"
    raw_links: list = field(default_factory=list)  # faithful positional links
    unresolved_link_count: int = 0
    doc_hash: str = ""  # hash of the exact retriever document string
    tokenizer: str = "whitespace"

    def to_dict(self) -> dict:
        return asdict(self)


def count_tokens(text: str) -> tuple[int, str]:
    """Token count for budget accounting; falls back to whitespace split.

    Returns (count, tokenizer_name) so runs can report which tokenizer
    produced the serialized-token numbers.
    """
    try:
        from nltk.tokenize import word_tokenize

        return len(word_tokenize(text)), "nltk.word_tokenize"
    except Exception:
        return len(text.split()), "whitespace"


class MemoryRegistry:
    """Bookkeeping from stable memory ids to provenance and lineage."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.records: dict[str, MemoryRecord] = {}

    def register(
        self,
        *,
        memory_id: str,
        note: Any,
        source_turn_ids: Iterable[str],
        step: int,
        retriever_doc: str,
        link_ids: list,
        raw_links: list,
        unresolved_link_count: int,
    ) -> MemoryRecord:
        if not source_turn_ids:
            raise ValueError(
                f"source_turn_ids is required for memory {memory_id}: "
                "every memory must be traceable to its source turn(s)"
            )
        if memory_id in self.records:
            raise ValueError(f"duplicate memory id {memory_id}")
        token_size, tokenizer = count_tokens(retriever_doc)
        record = MemoryRecord(
            memory_id=memory_id,
            source_turn_ids=list(source_turn_ids),
            created_step=step,
            raw_content=note.content,
            keywords=list(note.keywords),
            tags=list(note.tags),
            context=note.context,
            embedding_model=self.embedding_model,
            embedding_hash=sha256_hex(retriever_doc),
            link_ids=list(link_ids),
            parent_memory_ids=list(link_ids),
            evolution_ancestors=[],
            version=0,
            content_hash=sha256_hex(note.content),
            token_size=token_size,
            byte_size=len(retriever_doc.encode("utf-8")),
            raw_links=list(raw_links),
            unresolved_link_count=unresolved_link_count,
            doc_hash=sha256_hex(retriever_doc),
            tokenizer=tokenizer,
        )
        self.records[memory_id] = record
        return record

    def apply_link_resolution(self, memory_id: str, link_ids: list, raw_links: list, unresolved: int) -> None:
        record = self.records[memory_id]
        record.link_ids = list(link_ids)
        record.parent_memory_ids = list(link_ids)
        record.raw_links = list(raw_links)
        record.unresolved_link_count = unresolved

    def apply_evolution(self, memory_id: str, *, trigger_memory_id: str, ancestors_of_trigger: Iterable[str]) -> None:
        """Bump the version and extend the causal ancestor set."""
        record = self.records[memory_id]
        record.version += 1
        new_ancestors = {trigger_memory_id, *ancestors_of_trigger}
        new_ancestors.discard(memory_id)  # never self-ancestor
        merged = list(dict.fromkeys(record.evolution_ancestors + sorted(new_ancestors)))
        record.evolution_ancestors = merged

    def update_fields(self, memory_id: str, *, context: str, tags: list) -> None:
        record = self.records[memory_id]
        record.context = context
        record.tags = list(tags)

    def turn_to_memories(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for record in self.records.values():
            for turn in record.source_turn_ids:
                mapping.setdefault(turn, []).append(record.memory_id)
        return mapping

    def total_tokens(self, *, active_only: bool = True) -> int:
        return sum(
            record.token_size
            for record in self.records.values()
            if (not active_only or record.active_state == "active")
        )

    def total_bytes(self, *, active_only: bool = True) -> int:
        return sum(
            record.byte_size
            for record in self.records.values()
            if (not active_only or record.active_state == "active")
        )
