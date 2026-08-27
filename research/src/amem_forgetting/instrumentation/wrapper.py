"""Minimal-invasion instrumentation wrapper around AgenticMemorySystem.

The wrapper never modifies upstream classes: it calls the original
``add_note`` / retrieval methods as a black box, snapshots observable
state before and after, and derives provenance records, link
resolutions, and evolution before/after diffs from those snapshots.

Operation granularity: one wrapper-level operation (one ``add_note``,
one retrieval, one question-answer) becomes one op in the event log.
The upstream applies a whole ``add_note`` atomically, so events inside
an op carry the op's boundary state hashes rather than fabricated
intermediate states.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from amem_forgetting.instrumentation.canonical import (
    memory_state_hash,
    note_snapshot,
    sha256_hex,
)
from amem_forgetting.instrumentation.events import Event, EventLog
from amem_forgetting.instrumentation.evolution import EvolutionRecord, EvolutionTracker
from amem_forgetting.instrumentation.registry import MemoryRecord, MemoryRegistry


def retriever_document(note: Any) -> str:
    """Reconstruct the exact document string the retriever embeds.

    Mirrors the format in ``AgenticMemorySystem.add_note`` on the
    audited upstream commit; the document is frozen at write time (the
    upstream only rebuilds it during periodic consolidation), so
    token/byte sizes and the embedding input hash are recorded once,
    at registration.
    """
    return (
        "content:" + note.content
        + " context:" + note.context
        + " keywords: " + ", ".join(note.keywords)
        + " tags: " + ", ".join(note.tags)
    )


class LLMCallRecorder:
    """Record prompt/response hashes of LLM calls without touching call sites."""

    ANALYSIS_MARKER = "Generate a structured analysis"
    EVOLUTION_MARKER = "memory evolution agent"

    def __init__(self, controller: Any):
        self._controller = controller
        self._original = controller.get_completion
        self.calls: list[dict] = []
        self._current_op: str | None = None
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._controller.get_completion = self._recorded  # type: ignore[method-assign]
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        self._controller.get_completion = self._original  # type: ignore[method-assign]
        self._installed = False

    def _classify(self, prompt: str) -> str:
        if self.EVOLUTION_MARKER in prompt:
            return "evolution"
        if self.ANALYSIS_MARKER in prompt:
            return "note_analysis"
        return "other"

    def _recorded(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        response = self._original(prompt, *args, **kwargs)
        if self._current_op is not None:
            prompt_text = prompt if isinstance(prompt, str) else str(prompt)
            response_text = response if isinstance(response, str) else str(response)
            self.calls.append(
                {
                    "op_id": self._current_op,
                    "seq": len(self.calls) + 1,
                    "purpose": self._classify(prompt_text),
                    "prompt_hash": sha256_hex(prompt_text),
                    "response_hash": sha256_hex(response_text),
                    "prompt_chars": len(prompt_text),
                    "response_chars": len(response_text),
                    "prompt_preview": prompt_text[:80],
                }
            )
        return response

    def begin_op(self, op_id: str) -> None:
        self._current_op = op_id

    def end_op(self) -> list[dict]:
        """Return the calls recorded for the current op and close it."""
        calls = [call for call in self.calls if call["op_id"] == self._current_op]
        self._current_op = None
        return calls

    def calls_for(self, op_id: str, purpose: str | None = None) -> list[dict]:
        return [
            call
            for call in self.calls
            if call["op_id"] == op_id and (purpose is None or call["purpose"] == purpose)
        ]


class InstrumentedMemorySystem:
    """Event-sourced provenance layer over an unmodified A-MEM system."""

    def __init__(
        self,
        base: Any,
        run_id: str,
        *,
        policy: str = "KeepAll",
        event_log: EventLog | None = None,
    ):
        self.base = base
        self.log = event_log or EventLog(run_id=run_id, policy=policy)
        self.registry = MemoryRegistry(
            embedding_model=getattr(
                getattr(base, "retriever", None),
                "model_name",
                "all-MiniLM-L6-v2",
            )
            or "all-MiniLM-L6-v2"
        )
        self.tracker = EvolutionTracker()
        self.recorder = LLMCallRecorder(base.llm_controller.llm)
        self.recorder.install()

    # ------------------------------------------------------------------
    # state hashing

    def snapshots(self) -> dict[str, dict]:
        return {
            memory_id: note_snapshot(note)
            for memory_id, note in self.base.memories.items()
        }

    def state_hash(self) -> str:
        return memory_state_hash(self.snapshots())

    def _memory_ids(self) -> list[str]:
        return list(self.base.memories.keys())

    def _ids_at(self, indices: Iterable[int]) -> list[str]:
        ids = self._memory_ids()
        return [ids[index] for index in indices]

    def _next_memory_id(self) -> str:
        return f"m{len(self.registry.records) + 1:06d}"

    # ------------------------------------------------------------------
    # write path

    def add_note(
        self,
        content: str,
        time: str | None = None,
        *,
        source_turn_ids: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> str:
        if source_turn_ids is None:
            raise ValueError(
                "source_turn_ids is required: every memory must be traceable "
                "to the conversation turn(s) that produced it"
            )
        source_turn_ids = list(source_turn_ids)
        if not source_turn_ids:
            raise ValueError(
                "source_turn_ids is required: every memory must be traceable "
                "to the conversation turn(s) that produced it"
            )
        snapshots_before = self.snapshots()
        hash_before = memory_state_hash(snapshots_before)
        step, op_id = self.log.next_op()
        self.recorder.begin_op(op_id)
        try:
            memory_id = self._next_memory_id()
            returned_id = self.base.add_note(content, time, id=memory_id, **kwargs)
        finally:
            llm_calls = self.recorder.end_op()
        if returned_id != memory_id:
            raise RuntimeError(
                f"upstream returned id {returned_id!r} instead of the injected {memory_id!r}"
            )
        note = self.base.memories[memory_id]

        # Evolution diff: which existing notes did this write change?
        evolution_records: list[EvolutionRecord] = []
        evolution_prompt_hashes = [
            call["prompt_hash"] for call in llm_calls if call["purpose"] == "evolution"
        ]
        prompt_hash = evolution_prompt_hashes[-1] if evolution_prompt_hashes else ""
        for existing_id, before in snapshots_before.items():
            current = self.base.memories[existing_id]
            before_fields = {"context": before["context"], "tags": list(before["tags"])}
            after_fields = {"context": current.context, "tags": list(current.tags)}
            if after_fields == before_fields:
                continue
            record = self.tracker.add(
                op_id=op_id,
                step=step,
                trigger_memory_id=memory_id,
                evolved_memory_id=existing_id,
                old_version=self.registry.records[existing_id].version,
                before=before_fields,
                after=after_fields,
                prompt_hash=prompt_hash,
            )
            self.registry.apply_evolution(
                existing_id,
                trigger_memory_id=memory_id,
                ancestors_of_trigger=self.tracker.ancestors(memory_id),
            )
            self.registry.update_fields(
                existing_id, context=current.context, tags=list(current.tags)
            )
            evolution_records.append(record)

        # Register the new note (its own links/tags may have been set by
        # the strengthen action inside process_memory).
        document = retriever_document(note)
        resolved, unresolved = self._resolve_links(note.links)
        link_ids = list(dict.fromkeys(resolved))
        registry_record = self.registry.register(
            memory_id=memory_id,
            note=note,
            source_turn_ids=source_turn_ids,
            step=step,
            retriever_doc=document,
            link_ids=link_ids,
            raw_links=list(note.links),
            unresolved_link_count=unresolved,
        )

        snapshots_after = self.snapshots()
        hash_after = memory_state_hash(snapshots_after)

        self.log.emit(
            "write",
            op_id=op_id,
            step=step,
            memory_ids=[memory_id],
            details={
                "record": registry_record.to_dict(),
                "snapshot": snapshots_after[memory_id],
                "source_turn_ids": source_turn_ids,
                "llm_calls": llm_calls,
            },
            state_hash_before=hash_before,
            state_hash_after=hash_after,
        )
        if list(note.links):
            self.log.emit(
                "link",
                op_id=op_id,
                step=step,
                memory_ids=[memory_id],
                details={
                    "memory_id": memory_id,
                    "raw_links": list(note.links),
                    "link_ids": link_ids,
                    "unresolved_link_count": unresolved,
                },
                state_hash_before=hash_before,
                state_hash_after=hash_after,
            )
        for record in evolution_records:
            self.log.emit(
                "evolve",
                op_id=op_id,
                step=step,
                memory_ids=[record.evolved_memory_id],
                details={"evolution": record.to_dict()},
                state_hash_before=hash_before,
                state_hash_after=hash_after,
            )
        return memory_id

    def _resolve_links(self, raw_links: Sequence[Any]) -> tuple[list[str], int]:
        ids = self._memory_ids()
        resolved: list[str] = []
        unresolved = 0
        for raw in raw_links:
            try:
                index = int(raw)
            except (TypeError, ValueError):
                unresolved += 1
                continue
            if 0 <= index < len(ids):
                resolved.append(ids[index])
            else:
                unresolved += 1
        return resolved, unresolved

    # ------------------------------------------------------------------
    # retrieval path

    def find_related_memories(self, query: str, k: int = 5) -> tuple:
        """Flat top-k retrieval (candidate generation, no link expansion)."""
        hash_before = self.state_hash()
        step, op_id = self.log.next_op()
        memory_str, indices = self.base.find_related_memories(query, k)
        retrieved_ids = self._ids_at(indices)
        self.log.emit(
            "retrieve",
            op_id=op_id,
            step=step,
            memory_ids=retrieved_ids,
            details={
                "query": query,
                "k": k,
                "mode": "flat_topk",
                "raw_indices": list(indices),
                "retrieved_ids": retrieved_ids,
            },
            state_hash_before=hash_before,
            state_hash_after=hash_before,  # retrieval does not mutate state
        )
        return memory_str, indices

    def _raw_traversal_ids(self, query: str, k: int) -> tuple[list[str], list[str], list[str]]:
        """Mirror ``find_related_memories_raw`` and return its id sequence.

        Includes the upstream quirk that up to k+1 neighbors per seed are
        appended (the bound is checked after appending), and that a
        memory may appear more than once (as a seed and again as some
        other seed's link neighbor); duplicates are preserved verbatim.
        """
        indices = self.base.retriever.search(query, k)
        all_memories = list(self.base.memories.values())
        seed_ids = self._ids_at(indices)
        memory_ids = self._memory_ids()
        prompt_ids: list[str] = []
        expanded_entries: list[str] = []
        for index in indices:
            prompt_ids.append(memory_ids[index])
            neighborhood = all_memories[index].links
            j = 0
            for neighbor in neighborhood:
                neighbor_id = memory_ids[neighbor]  # IndexError is faithful
                prompt_ids.append(neighbor_id)
                expanded_entries.append(neighbor_id)
                if j >= k:
                    break
                j += 1
        return seed_ids, expanded_entries, prompt_ids

    def retrieve_raw(self, query: str, k: int = 5) -> tuple[list[str], list[str], list[str]]:
        """Instrumented version of the QA-time raw retrieval (1-hop expansion)."""
        hash_before = self.state_hash()
        step, op_id = self.log.next_op()
        seed_ids, expanded_ids, prompt_ids = self._raw_traversal_ids(query, k)
        self.log.emit(
            "retrieve",
            op_id=op_id,
            step=step,
            memory_ids=prompt_ids,
            details={
                "query": query,
                "k": k,
                "mode": "raw_1hop_expansion",
                "seed_ids": seed_ids,
                "expanded_ids": expanded_ids,
                "prompt_ids": prompt_ids,
            },
            state_hash_before=hash_before,
            state_hash_after=hash_before,
        )
        return seed_ids, expanded_ids, prompt_ids

    # ------------------------------------------------------------------
    # question-answer path (mock answerer in STAGE 1 wiring runs)

    def answer(
        self,
        question: str,
        k: int = 5,
        answerer: Callable[[str, list[str]], str] | None = None,
    ) -> dict:
        """Question -> retrieval -> answer, recorded as one op.

        The default answerer is a deterministic mock used only to validate
        event wiring; real runs must pass an answerer backed by the frozen
        track model (recorded in the run manifest).
        """
        hash_before = self.state_hash()
        step, op_id = self.log.next_op()
        seed_ids, expanded_ids, prompt_ids = self._raw_traversal_ids(question, k)
        if answerer is None:
            answer_text = "mock:" + sha256_hex(question)[7:15]
        else:
            answer_text = answerer(question, prompt_ids)
        self.log.emit(
            "query",
            op_id=op_id,
            step=step,
            memory_ids=[],
            details={"question": question, "k": k},
            state_hash_before=hash_before,
            state_hash_after=hash_before,
        )
        self.log.emit(
            "retrieve",
            op_id=op_id,
            step=step,
            memory_ids=prompt_ids,
            details={
                "query": question,
                "k": k,
                "mode": "raw_1hop_expansion",
                "seed_ids": seed_ids,
                "expanded_ids": expanded_ids,
                "prompt_ids": prompt_ids,
            },
            state_hash_before=hash_before,
            state_hash_after=hash_before,
        )
        self.log.emit(
            "answer",
            op_id=op_id,
            step=step,
            memory_ids=prompt_ids,
            details={
                "question": question,
                "answer": answer_text,
                "answerer": "mock_deterministic" if answerer is None else "custom",
                "seed_ids": seed_ids,
                "expanded_ids": expanded_ids,
                "prompt_ids": prompt_ids,
            },
            state_hash_before=hash_before,
            state_hash_after=hash_before,
        )
        return {
            "question": question,
            "answer": answer_text,
            "seed_ids": seed_ids,
            "expanded_ids": expanded_ids,
            "prompt_ids": prompt_ids,
        }

    # ------------------------------------------------------------------

    def save_event_log(self, path: str) -> int:
        return self.log.save(path)

    @property
    def run_id(self) -> str:
        return self.log.run_id
