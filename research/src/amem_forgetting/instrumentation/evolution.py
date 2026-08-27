"""Evolution lineage: before/after versions and the evolution DAG.

Upstream A-MEM overwrites a neighbor's context/tags when a new memory
triggers ``update_neighbor``. The tracker snapshots the overwritten
values so no version is physically lost (plan section 5.3), and keeps
the causal DAG separate from the link graph. Cycles are legal in
practice (A evolves B, later B evolves A); they are flagged, never
silently merged away.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Iterable

from amem_forgetting.instrumentation.canonical import canonical_json, sha256_hex


@dataclass
class EvolutionRecord:
    op_id: str
    trigger_memory_id: str
    evolved_memory_id: str
    old_version: int
    new_version: int
    changed_fields: list
    before: dict  # field -> value before the evolution
    after: dict  # field -> value after the evolution
    before_hash: str
    after_hash: str
    created_step: int
    prompt_hash: str
    creates_cycle: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class EvolutionTracker:
    """Causal DAG of ``trigger -> evolved`` edges with cycle flags."""

    def __init__(self):
        self.records: list[EvolutionRecord] = []
        # evolved_id -> list of trigger ids (in evolution order)
        self._parents: dict[str, list[str]] = {}
        self.cycle_edges: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------

    def _is_reachable(self, start: str, target: str) -> bool:
        """True if ``target`` is reachable from ``start`` over causal edges."""
        stack = [start]
        visited = set()
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            for trigger in self._parents.get(current, []):
                stack.append(trigger)
        return False

    def add(
        self,
        *,
        op_id: str,
        step: int,
        trigger_memory_id: str,
        evolved_memory_id: str,
        old_version: int,
        before: dict,
        after: dict,
        prompt_hash: str,
    ) -> EvolutionRecord:
        if trigger_memory_id == evolved_memory_id:
            raise ValueError("a memory cannot evolve itself")
        changed = sorted(
            key for key in set(before) | set(after) if before.get(key) != after.get(key)
        )
        # An edge trigger -> evolved closes a cycle iff evolved already
        # (transitively) triggered trigger's own evolutions.
        creates_cycle = self._is_reachable(evolved_memory_id, trigger_memory_id)
        record = EvolutionRecord(
            op_id=op_id,
            trigger_memory_id=trigger_memory_id,
            evolved_memory_id=evolved_memory_id,
            old_version=old_version,
            new_version=old_version + 1,
            changed_fields=changed,
            before=dict(before),
            after=dict(after),
            before_hash=sha256_hex(canonical_json(before)),
            after_hash=sha256_hex(canonical_json(after)),
            created_step=step,
            prompt_hash=prompt_hash,
            creates_cycle=creates_cycle,
        )
        self.records.append(record)
        self._parents.setdefault(evolved_memory_id, []).append(trigger_memory_id)
        if creates_cycle:
            self.cycle_edges.add((trigger_memory_id, evolved_memory_id))
        return record

    # ------------------------------------------------------------------

    def ancestors(self, memory_id: str) -> list[str]:
        """Transitive causal ancestors, cycle-safe, in discovery order."""
        result: list[str] = []
        seen = {memory_id}
        queue = list(self._parents.get(memory_id, []))
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            result.append(current)
            queue.extend(self._parents.get(current, []))
        return result

    def descendants(self, memory_id: str) -> list[str]:
        children: dict[str, list[str]] = {}
        for evolved, triggers in self._parents.items():
            for trigger in triggers:
                children.setdefault(trigger, []).append(evolved)
        result: list[str] = []
        seen = {memory_id}
        queue = list(children.get(memory_id, []))
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            result.append(current)
            queue.extend(children.get(current, []))
        return result

    def versions(self, memory_id: str) -> list[EvolutionRecord]:
        return [record for record in self.records if record.evolved_memory_id == memory_id]

    def edges(self) -> list[tuple[str, str]]:
        return [(record.trigger_memory_id, record.evolved_memory_id) for record in self.records]
