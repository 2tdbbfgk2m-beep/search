"""Append-only, hash-chained event log (plan section 3.9).

Events are emitted in operation granularity groups: one ``add_note`` call
produces a ``write`` event, then optional ``link`` and ``evolve`` events
that share the same ``op_id``/``step``. The state hashes recorded on an
event are the hashes at the enclosing operation's boundaries, because the
upstream implementation applies the whole operation atomically; the event
log never fabricates intermediate states that did not exist online.

The hash chain covers every event individually, in emission order.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from amem_forgetting.instrumentation.canonical import canonical_json, sha256_hex

GENESIS_HASH = "0" * 64

EVENT_TYPES = ("write", "link", "evolve", "retrieve", "query", "answer")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Event:
    event_id: str
    run_id: str
    step: int
    op_id: str
    event_type: str
    timestamp: str
    memory_ids: list
    policy: str
    score: float | None
    budget: float | None
    state_hash_before: str
    state_hash_after: str
    details: dict
    prev_event_hash: str
    event_hash: str
    future_label: Any = None
    credit_status: str = "resolved"

    def to_dict(self) -> dict:
        return asdict(self)


class EventLog:
    """Hash-chained, append-only event storage for one run."""

    def __init__(self, run_id: str, policy: str = "KeepAll", events: Iterable[Event] | None = None):
        self.run_id = run_id
        self.policy = policy
        self.events: list[Event] = list(events or [])
        self._persisted = 0  # number of events already written to disk
        self._step = 0

    # ------------------------------------------------------------------
    # emission

    def next_op(self) -> tuple[int, str]:
        """Advance to the next operation; returns (step, op_id)."""
        self._step += 1
        return self._step, f"op{self._step:06d}"

    @property
    def current_step(self) -> int:
        return self._step

    def emit(
        self,
        event_type: str,
        *,
        op_id: str,
        step: int,
        memory_ids: list,
        details: Mapping[str, Any],
        state_hash_before: str,
        state_hash_after: str,
        score: float | None = None,
        budget: float | None = None,
        future_label: Any = None,
        credit_status: str = "resolved",
        timestamp: str | None = None,
    ) -> Event:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event_type {event_type!r}")
        prev_hash = self.events[-1].event_hash if self.events else GENESIS_HASH
        seq = len(self.events) + 1
        payload = {
            "event_id": f"e{seq:06d}",
            "run_id": self.run_id,
            "step": step,
            "op_id": op_id,
            "event_type": event_type,
            "timestamp": timestamp or _utc_now_iso(),
            "memory_ids": list(memory_ids),
            "policy": self.policy,
            "score": score,
            "budget": budget,
            "state_hash_before": state_hash_before,
            "state_hash_after": state_hash_after,
            "details": json.loads(json.dumps(dict(details), default=str, allow_nan=False)),
            "prev_event_hash": prev_hash,
            "future_label": future_label,
            "credit_status": credit_status,
        }
        event = Event(**payload, event_hash=sha256_hex(canonical_json(payload)))
        self.events.append(event)
        return event

    # ------------------------------------------------------------------
    # verification

    @staticmethod
    def _event_hash(event: Event) -> str:
        payload = event.to_dict()
        payload.pop("event_hash")
        return sha256_hex(canonical_json(payload))

    def verify(self) -> bool:
        """Recompute the full chain: ids, ordering, and hashes."""
        prev_hash = GENESIS_HASH
        for index, event in enumerate(self.events):
            if event.event_id != f"e{index + 1:06d}":
                return False
            if event.prev_event_hash != prev_hash:
                return False
            if event.event_hash != self._event_hash(event):
                return False
            prev_hash = event.event_hash
        return True

    def verify_chronology(self) -> bool:
        """Steps are non-decreasing and op groups are contiguous."""
        last_step = 0
        seen_ops: dict[str, int] = {}
        for event in self.events:
            if event.step < last_step:
                return False
            last_step = event.step
            if event.op_id in seen_ops and seen_ops[event.op_id] != event.step:
                return False
            if event.op_id not in seen_ops:
                seen_ops[event.op_id] = event.step
        return True

    # ------------------------------------------------------------------
    # persistence (append-only)

    def save(self, path: str | Path) -> int:
        """Append the not-yet-persisted events to ``path`` (JSONL).

        Returns the number of events appended. Existing file content is
        never rewritten: if the file exists it must contain exactly the
        already-persisted prefix of this log.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and self._persisted == 0:
            # A fresh in-memory log saving onto an existing file is a
            # provenance error: raw results are append-only.
            raise ValueError(f"refusing to overwrite existing event log {path}")
        if path.exists():
            existing = self._count_lines(path)
            if existing != self._persisted:
                raise ValueError(
                    f"event log {path} has {existing} events, expected {self._persisted}"
                )
        new_events = self.events[self._persisted:]
        if new_events:
            with open(path, "a", encoding="utf-8") as handle:
                for event in new_events:
                    handle.write(canonical_json(event.to_dict()) + "\n")
            self._persisted += len(new_events)
        return len(new_events)

    @staticmethod
    def _count_lines(path: Path) -> int:
        with open(path, "r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    @classmethod
    def load(cls, path: str | Path) -> "EventLog":
        """Load and verify a JSONL event log; tampering raises ValueError."""
        path = Path(path)
        events: list[Event] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise ValueError(f"blank event at line {line_number}")
            try:
                events.append(Event(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid event at line {line_number}") from error
        log = cls(run_id=events[0].run_id if events else "", policy=events[0].policy if events else "KeepAll", events=events)
        log._step = events[-1].step if events else 0
        log._persisted = len(events)
        if events and not log.verify():
            raise ValueError("event hash chain verification failed")
        return log

    # ------------------------------------------------------------------

    def ops(self) -> list[list[Event]]:
        """Group events into operation groups (contiguous by op_id)."""
        groups: list[list[Event]] = []
        for event in self.events:
            if not groups or groups[-1][0].op_id != event.op_id:
                groups.append([event])
            else:
                groups[-1].append(event)
        return groups
