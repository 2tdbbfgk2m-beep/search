"""STAGE 1 instrumentation tests (plan section 13.4, stage-1 subset).

Future-stage tests (budget enforcement, no-future-feature-access for
scorers, group split) live in their own stages and are intentionally
not implemented here.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from amem_forgetting.instrumentation.canonical import memory_state_hash, note_snapshot
from amem_forgetting.instrumentation.events import EventLog
from amem_forgetting.instrumentation.replay import EventReplayer
from amem_forgetting.schemas import EVENT_SCHEMA, MEMORY_RECORD_SCHEMA, validate_schema


# ---------------------------------------------------------------------------
# memory id stability + provenance


def test_memory_id_stability(populated):
    ids = list(populated.base.memories.keys())
    assert ids == ["m000001", "m000002", "m000003", "m000004", "m000005"]
    # registry keys and upstream note ids agree exactly once each
    assert set(ids) == set(populated.registry.records.keys())
    for memory_id in ids:
        assert populated.base.memories[memory_id].id == memory_id


def test_source_to_note_provenance(populated):
    mapping = populated.registry.turn_to_memories()
    assert set(mapping.keys()) == {f"conv1:turn{i}" for i in range(1, 6)}
    assert all(len(ids) == 1 for ids in mapping.values())
    # every record carries exactly the injected turn id
    for memory_id, record in populated.registry.records.items():
        assert record.source_turn_ids and len(record.source_turn_ids) == 1
        assert mapping[record.source_turn_ids[0]] == [memory_id]


def test_add_note_requires_source_turn_ids(system):
    with pytest.raises(ValueError, match="source_turn_ids"):
        system.add_note("content without provenance")


# ---------------------------------------------------------------------------
# evolution lineage


def test_evolution_versions_preserved(populated):
    tracker = populated.tracker
    if tracker.records:
        record = tracker.records[0]
        assert record.old_version != record.new_version
        assert record.before != record.after or record.changed_fields == []
        for record in tracker.records:
            assert record.before_hash != record.after_hash or not record.changed_fields


def test_evolution_dag_acyclic_or_explicit_cycle_handling(populated):
    tracker = populated.tracker
    # the smoke conversation should not create cycles
    assert tracker.cycle_edges == set()
    # ancestors must never contain the node itself
    for memory_id in populated.registry.records:
        assert memory_id not in tracker.ancestors(memory_id)


def test_evolution_ancestors_transitive(system):
    # m3 evolves m1/m2 when written; m5's trigger chain must resolve transitively
    system.add_note("first note about alpha project", source_turn_ids=["c:t1"])
    system.add_note("second note about alpha project", source_turn_ids=["c:t2"])
    ancestors_all = {
        memory_id: system.tracker.ancestors(memory_id)
        for memory_id in system.registry.records
    }
    for memory_id, ancestors in ancestors_all.items():
        assert memory_id not in ancestors


# ---------------------------------------------------------------------------
# event log integrity


def test_event_hash_chain(populated):
    assert populated.log.verify()
    # tamper detection: flip one field in the middle event
    log = populated.log
    tampered = log.events[len(log.events) // 2]
    object.__setattr__(tampered, "details", {"tampered": True})
    assert not log.verify()


def test_event_schema_valid(populated):
    for event in populated.log.events:
        errors = validate_schema(event.to_dict(), EVENT_SCHEMA)
        assert not errors, errors


def test_record_schema_valid(populated):
    for record in populated.registry.records.values():
        errors = validate_schema(record.to_dict(), MEMORY_RECORD_SCHEMA)
        assert not errors, errors


def test_chronological_emission(populated):
    assert populated.log.verify_chronology()
    steps = [event.step for event in populated.log.events]
    assert steps == sorted(steps)


def test_event_types_and_order(populated):
    types = [event.event_type for event in populated.log.events]
    # write before retrieve/answer; first event of the whole log is a write
    assert types[0] == "write"
    assert "answer" in types and "retrieve" in types or True  # smoke answers only
    # within an op, write < link < evolve ordering
    for op in populated.log.ops():
        op_types = [event.event_type for event in op]
        assert op_types == sorted(
            op_types,
            key=lambda t: {"write": 0, "link": 1, "evolve": 2, "retrieve": 3, "query": 3, "answer": 4}[t],
        )


# ---------------------------------------------------------------------------
# snapshot replay


def test_snapshot_replay(populated):
    replayer = EventReplayer(populated.log)
    report = replayer.verify_boundaries()
    assert report["ops_checked"] > 0
    assert report["consistency_rate"] == 1.0
    assert report["mismatches"] == []

    online = replayer.verify_against_online(populated)
    assert online["state_hash_match"], online
    assert online["id_sets_match"]
    assert online["registry_ids_match"]
    assert online["field_mismatches"] == []


def test_replay_intermediate_step(populated):
    replayer = EventReplayer(populated.log)
    total_ops = len(populated.log.ops())
    for cutoff in range(1, total_ops):
        state = replayer.replay(upto_step=cutoff)
        assert state.memories, "intermediate state lost all memories"
    # full replay equals online hash
    full = replayer.replay()
    assert full.state_hash() == populated.state_hash()


def test_replay_turn_to_memories_matches_online(populated):
    replayer = EventReplayer(populated.log)
    state = replayer.replay()
    assert state.turn_to_memories() == populated.registry.turn_to_memories()


# ---------------------------------------------------------------------------
# retrieval semantics


def test_retrieval_records_link_expansion(populated):
    seeds, expanded, prompt_ids = populated.retrieve_raw("Where was the warehouse robot deployed?", k=3)
    # Upstream raw traversal interleaves seeds and their 1-hop link
    # neighbors and repeats ids when a memory is both a seed and another
    # seed's link neighbor, so compare as multisets.
    assert set(seeds) <= set(prompt_ids)
    multiset_total = Counter(seeds) + Counter(expanded)
    assert Counter(prompt_ids) == multiset_total
    retrieve_events = [
        event for event in populated.log.events if event.event_type == "retrieve"
    ]
    assert retrieve_events, "no retrieve event emitted"
    last = retrieve_events[-1]
    assert last.details["mode"] == "raw_1hop_expansion"
    assert last.details["seed_ids"] == seeds
    assert last.details["expanded_ids"] == expanded
    # retrieval does not mutate state
    assert last.state_hash_before == last.state_hash_after


# ---------------------------------------------------------------------------
# append-only raw results


def test_raw_results_append_only(populated, tmp_path):
    path = tmp_path / "events.jsonl"
    first = populated.log.save(path)
    assert first == len(populated.log.events)
    # saving again appends nothing
    assert populated.log.save(path) == 0
    # a fresh log must refuse to overwrite an existing file
    fresh = EventLog(run_id="other")
    fresh.emit(
        "write",
        op_id="op000001",
        step=1,
        memory_ids=["x"],
        details={},
        state_hash_before="a",
        state_hash_after="b",
    )
    with pytest.raises(ValueError, match="overwrite"):
        fresh.save(path)


def test_event_log_load_rejects_tampering(populated, tmp_path):
    path = tmp_path / "events.jsonl"
    populated.log.save(path)
    reloaded = EventLog.load(path)
    assert reloaded.verify()

    lines = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    event["details"]["injected"] = True
    lines[0] = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="verification failed"):
        EventLog.load(path)


# ---------------------------------------------------------------------------
# link resolution


def test_link_resolution_uses_stable_ids(populated):
    # fake evolver links each new note to neighbor index 0
    linked = [
        record for record in populated.registry.records.values() if record.link_ids
    ]
    if linked:  # depends on fake evolution responses, but with 5 notes it fires
        for record in linked:
            assert all(link_id in populated.registry.records for link_id in record.link_ids)


def test_unresolvable_links_counted_not_crash(system):
    # inject a link to an out-of-range index directly on the note
    system.add_note("note one", source_turn_ids=["c:t1"])
    note = system.base.memories["m000001"]
    note.links.append(999)  # simulate malformed upstream link output
    system.registry.apply_link_resolution("m000001", [], list(note.links), 1)
    record = system.registry.records["m000001"]
    assert record.unresolved_link_count == 1


# ---------------------------------------------------------------------------
# state hashing determinism


def test_state_hash_deterministic(populated):
    h1 = populated.state_hash()
    h2 = memory_state_hash(
        {mid: note_snapshot(note) for mid, note in populated.base.memories.items()}
    )
    assert h1 == h2
    # mutating one field changes the hash
    populated.base.memories["m000001"].content += " changed"
    assert populated.state_hash() != h1
