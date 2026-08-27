"""Snapshot replay: rebuild any point-in-time state from the event log.

Plan section 5.2 requires that given the event log, the memory set, link
graph, and versions can be reconstructed at any step, and that replayed
state hashes match the online state 100% at event boundaries.
"""

from __future__ import annotations

from typing import Any

from amem_forgetting.instrumentation.canonical import memory_state_hash
from amem_forgetting.instrumentation.events import Event, EventLog


class ReplayedState:
    """Plain-dict reconstruction of the instrumented system state."""

    def __init__(self):
        self.memories: dict[str, dict] = {}  # id -> note snapshot
        self.registry: dict[str, dict] = {}  # id -> memory record
        self.evolution_parents: dict[str, list[str]] = {}
        self.evolution_records: list[dict] = []
        self.cycle_edges: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------

    def _apply_write(self, event: Event) -> None:
        record = dict(event.details["record"])
        snapshot = dict(event.details["snapshot"])
        memory_id = record["memory_id"]
        if memory_id in self.memories:
            raise ValueError(f"replay: duplicate write of {memory_id}")
        self.registry[memory_id] = record
        self.memories[memory_id] = snapshot

    def _apply_link(self, event: Event) -> None:
        details = event.details
        memory_id = details["memory_id"]
        record = self.registry[memory_id]
        record["link_ids"] = list(details["link_ids"])
        record["parent_memory_ids"] = list(details["link_ids"])
        record["raw_links"] = list(details["raw_links"])
        record["unresolved_link_count"] = details["unresolved_link_count"]

    def _ancestors(self, memory_id: str) -> list[str]:
        result: list[str] = []
        seen = {memory_id}
        queue = list(self.evolution_parents.get(memory_id, []))
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            result.append(current)
            queue.extend(self.evolution_parents.get(current, []))
        return result

    def _is_reachable(self, start: str, target: str) -> bool:
        stack = [start]
        visited = set()
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self.evolution_parents.get(current, []))
        return False

    def _apply_evolve(self, event: Event) -> None:
        evolution = dict(event.details["evolution"])
        trigger = evolution["trigger_memory_id"]
        evolved = evolution["evolved_memory_id"]
        if evolved not in self.memories:
            raise ValueError(f"replay: evolution of unknown memory {evolved}")
        snapshot = self.memories[evolved]
        for field, value in evolution["after"].items():
            snapshot[field] = value
        record = self.registry[evolved]
        record["version"] = evolution["new_version"]
        for field, value in evolution["after"].items():
            if field in record:
                record[field] = value
        new_ancestors = {trigger, *self._ancestors(trigger)}
        new_ancestors.discard(evolved)
        record["evolution_ancestors"] = list(
            dict.fromkeys(record["evolution_ancestors"] + sorted(new_ancestors))
        )
        if self._is_reachable(evolved, trigger):
            self.cycle_edges.add((trigger, evolved))
        self.evolution_parents.setdefault(evolved, []).append(trigger)
        self.evolution_records.append(evolution)

    def apply(self, event: Event) -> None:
        if event.event_type == "write":
            self._apply_write(event)
        elif event.event_type == "link":
            self._apply_link(event)
        elif event.event_type == "evolve":
            self._apply_evolve(event)
        # retrieve / query / answer do not mutate memory state

    def state_hash(self) -> str:
        return memory_state_hash(self.memories)

    def turn_to_memories(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for memory_id, record in self.registry.items():
            for turn in record["source_turn_ids"]:
                mapping.setdefault(turn, []).append(memory_id)
        return mapping


class EventReplayer:
    """Replay an EventLog and check per-op consistency against its hashes."""

    def __init__(self, log: EventLog | Any):
        self.log = log

    def replay(self, upto_step: int | None = None) -> ReplayedState:
        state = ReplayedState()
        for op_events in self.log.ops():
            if upto_step is not None and op_events[0].step > upto_step:
                break
            for event in op_events:
                state.apply(event)
        return state

    def verify_boundaries(self) -> dict:
        """Replay op by op and compare state hashes at every boundary."""
        state = ReplayedState()
        ops_checked = 0
        boundary_matches = 0
        mismatches: list[dict] = []
        for op_events in self.log.ops():
            expected_before = op_events[0].state_hash_before
            expected_after = op_events[-1].state_hash_after
            actual_before = state.state_hash()
            for event in op_events:
                state.apply(event)
            actual_after = state.state_hash()
            ops_checked += 1
            ok_before = actual_before == expected_before
            ok_after = actual_after == expected_after
            if ok_before and ok_after:
                boundary_matches += 1
            else:
                mismatches.append(
                    {
                        "op_id": op_events[0].op_id,
                        "before_ok": ok_before,
                        "after_ok": ok_after,
                    }
                )
        return {
            "ops_checked": ops_checked,
            "boundary_matches": boundary_matches,
            "consistency_rate": boundary_matches / ops_checked if ops_checked else 1.0,
            "mismatches": mismatches,
        }

    def verify_against_online(self, wrapper: Any) -> dict:
        """Compare the fully replayed state with the live wrapper state."""
        state = self.replay()
        online_hash = wrapper.state_hash()
        replay_hash = state.state_hash()
        online_ids = set(wrapper.base.memories.keys())
        replay_ids = set(state.memories.keys())
        field_mismatches: list[str] = []
        for memory_id in sorted(online_ids & replay_ids):
            from amem_forgetting.instrumentation.canonical import note_snapshot

            if note_snapshot(wrapper.base.memories[memory_id]) != state.memories[memory_id]:
                field_mismatches.append(memory_id)
        return {
            "state_hash_match": online_hash == replay_hash,
            "online_state_hash": online_hash,
            "replay_state_hash": replay_hash,
            "id_sets_match": online_ids == replay_ids,
            "field_mismatches": field_mismatches,
            "registry_ids_match": set(wrapper.registry.records.keys())
            == set(state.registry.keys()),
        }
