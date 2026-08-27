#!/usr/bin/env python3
"""STAGE 1 mock smoke run: wiring validation without any LLM.

Drives the real upstream ``AgenticMemorySystem`` through
write -> link -> evolve -> retrieve -> answer using a deterministic
fake LLM and stubbed embedding model, records everything through the
instrumentation layer, and produces the full run-directory contract
(run_manifest.json, config.yaml, git_commit.txt, stdout/stderr logs,
results.jsonl, metrics.csv, gate.json).

This validates event wiring, provenance, and replay consistency only;
it is NOT a reference-track reproduction and produces no paper numbers.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../work/A-MEM
SRC_ROOT = REPO_ROOT / "research" / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from amem_forgetting.testsupport.stubs import install_import_stubs  # noqa: E402

STUBBED = install_import_stubs()

import memory_layer  # noqa: E402  (real upstream module, stubs if deps missing)

from amem_forgetting.instrumentation import (  # noqa: E402
    EventReplayer,
    InstrumentedMemorySystem,
)
from amem_forgetting.runsupport import RunDirectory, git_commit  # noqa: E402
from amem_forgetting.schemas import validate_schema  # noqa: E402
from amem_forgetting.schemas import EVENT_SCHEMA, MEMORY_RECORD_SCHEMA  # noqa: E402
from amem_forgetting.testsupport.fakes import FakeLLMController  # noqa: E402

CONVERSATION = [
    ("conv1:turn1", "Alice moved to Berlin in 2019 and started working at a robotics startup."),
    ("conv1:turn2", "Bob visited Alice in Berlin and they toured the museum of technology together."),
    ("conv1:turn3", "Alice's startup launched a warehouse robot that reduces picking errors."),
    ("conv1:turn4", "Bob now lives in Vienna and studies computational linguistics."),
    ("conv1:turn5", "The warehouse robot from Alice's startup was deployed in Vienna last month."),
]

QUESTIONS = [
    ("q1", "Where does Alice live?", "Berlin"),
    ("q2", "What did Alice and Bob do together in Berlin?", "museum"),
    ("q3", "Where was the warehouse robot deployed?", "Vienna"),
]


def build_config() -> dict:
    return {
        "project": {"name": "amem_graph_forgetting", "track": "extension", "run_root": "research/runs/raw"},
        "models": {
            "writer": "FakeLLMController(deterministic)",
            "linker": "FakeLLMController(deterministic)",
            "evolver": "FakeLLMController(deterministic)",
            "answerer": "mock_deterministic",
            "embedding": "stub-hash-encoder" if "sentence_transformers" in STUBBED else "all-MiniLM-L6-v2",
            "temperature": 0.0,
        },
        "amem": {
            "enable_link_generation": True,
            "enable_memory_evolution": True,
            "top_k_link": 5,
            "top_k_retrieval": 5,
            "retrieval_mode": "faithful_flat_with_1hop_link_expansion",
        },
        "notes": {
            "llm": "fake deterministic; wiring validation only, no paper numbers",
            "embedding": "stubbed" if STUBBED else "real",
            "retrieval": "real SimpleEmbeddingRetriever over stubbed/real encoder",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default=None, help="run id (default: stage1_mock_smoke_<timestamp>)")
    parser.add_argument("--runs_root", default=str(REPO_ROOT / "research" / "runs" / "raw"))
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    run_id = args.run_id or "stage1_mock_smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = RunDirectory(run_id, args.runs_root)
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    exit_code = 0

    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        try:
            base = memory_layer.AgenticMemorySystem(
                model_name="all-MiniLM-L6-v2",
                llm_backend="ollama",  # constructor only wires controllers; we replace the inner one
                llm_model="fake",
            )
            fake = FakeLLMController()
            base.llm_controller.llm = fake

            system = InstrumentedMemorySystem(base, run_id=run_id)

            # ---- write -> link -> evolve ---------------------------------
            for offset, (turn_id, text) in enumerate(CONVERSATION, start=1):
                system.add_note(
                    text,
                    source_turn_ids=[turn_id],
                    time="2025060112" + f"{(offset - 1) * 10:02d}",  # upstream %Y%m%d%H%M
                )

            # ---- retrieve + answer --------------------------------------
            for question_id, question, _gold in QUESTIONS:
                result = system.answer(question, k=args.k)
                system.log.events[-1]  # answer event exists; keep linter quiet
                run.append_result(
                    {
                        "run_id": run_id,
                        "question_id": question_id,
                        "question": question,
                        "answer": result["answer"],
                        "seed_ids": result["seed_ids"],
                        "expanded_ids": result["expanded_ids"],
                        "prompt_ids": result["prompt_ids"],
                    }
                )

            # ---- replay consistency -------------------------------------
            replayer = EventReplayer(system.log)
            boundary_report = replayer.verify_boundaries()
            online_report = replayer.verify_against_online(system)

            # ---- schema + chain checks ----------------------------------
            event_schema_errors = []
            for event in system.log.events:
                event_schema_errors.extend(validate_schema(event.to_dict(), EVENT_SCHEMA))
            record_schema_errors = []
            for record in system.registry.records.values():
                record_schema_errors.extend(validate_schema(record.to_dict(), MEMORY_RECORD_SCHEMA))
            chain_ok = system.log.verify()
            chronology_ok = system.log.verify_chronology()
            persisted = system.log.save(run.path / "events.jsonl")
            reloaded_ok = False
            if persisted:
                from amem_forgetting.instrumentation.events import EventLog

                reloaded = EventLog.load(run.path / "events.jsonl")
                reloaded_ok = reloaded.verify() and len(reloaded.events) == len(system.log.events)

            # ---- provenance checks --------------------------------------
            provenance = {
                turn: ids for turn, ids in sorted(system.registry.turn_to_memories().items())
            }
            all_memories_have_source = all(
                record.source_turn_ids for record in system.registry.records.values()
            )

            metrics_rows = [
                {
                    "metric": "memories",
                    "value": len(system.registry.records),
                },
                {"metric": "evolution_records", "value": len(system.tracker.records)},
                {"metric": "cycle_edges", "value": len(system.tracker.cycle_edges)},
                {"metric": "events", "value": len(system.log.events)},
                {"metric": "replay_boundary_consistency", "value": boundary_report["consistency_rate"]},
                {"metric": "replay_online_hash_match", "value": float(online_report["state_hash_match"])},
                {"metric": "event_hash_chain_ok", "value": float(chain_ok)},
                {"metric": "event_chronology_ok", "value": float(chronology_ok)},
                {"metric": "reload_verification_ok", "value": float(reloaded_ok)},
                {"metric": "event_schema_errors", "value": len(event_schema_errors)},
                {"metric": "record_schema_errors", "value": len(record_schema_errors)},
                {"metric": "all_memories_have_source", "value": float(all_memories_have_source)},
            ]

            criteria = {
                "event_hash_chain": {"value": chain_ok, "threshold": True, "pass": chain_ok},
                "chronological_emission": {"value": chronology_ok, "threshold": True, "pass": chronology_ok},
                "reload_roundtrip": {"value": reloaded_ok, "threshold": True, "pass": reloaded_ok},
                "replay_boundary_consistency": {
                    "value": boundary_report["consistency_rate"],
                    "threshold": 1.0,
                    "pass": boundary_report["consistency_rate"] == 1.0,
                },
                "replay_online_state_match": {
                    "value": online_report["state_hash_match"],
                    "threshold": True,
                    "pass": online_report["state_hash_match"],
                },
                "replay_registry_match": {
                    "value": online_report["registry_ids_match"],
                    "threshold": True,
                    "pass": online_report["registry_ids_match"],
                },
                "event_schema_valid": {
                    "value": len(event_schema_errors),
                    "threshold": 0,
                    "pass": not event_schema_errors,
                },
                "record_schema_valid": {
                    "value": len(record_schema_errors),
                    "threshold": 0,
                    "pass": not record_schema_errors,
                },
                "every_memory_traceable_to_source": {
                    "value": all_memories_have_source,
                    "threshold": True,
                    "pass": all_memories_have_source,
                },
            }
            status = "PASS" if all(item["pass"] for item in criteria.values()) else "FAIL"
            missing = run.missing_required_files()

            run.write_manifest(
                {
                    "stage": "stage1",
                    "kind": "mock_smoke_wiring",
                    "git_commit": git_commit(REPO_ROOT),
                    "stubbed_dependencies": STUBBED,
                    "llm": "FakeLLMController (deterministic, no network)",
                    "conversation_turns": len(CONVERSATION),
                    "questions": len(QUESTIONS),
                    "top_k_retrieval": args.k,
                }
            )
            run.write_config_mapping(build_config())
            run.write_git_commit(git_commit(REPO_ROOT))
            run.write_metrics(metrics_rows)
            run.write_gate(
                {
                    "stage": "stage1",
                    "status": status,
                    "run_ids": [run_id],
                    "preconditions": {"stage0_status": "BLOCKED_on_LLM_env", "mock_wiring_only": True},
                    "criteria": criteria,
                    "uncertainties": [
                        "links 参与检索（raw 路径 1-hop 扩展），faithful_flat 与 graph_expansion 分支将在 reference/extension 正式运行中分轨",
                        "嵌入为 stub 或真实模型时检索结果不同；wiring 校验与两者无关",
                    ],
                    "blocking_issues": [
                        "无可用 LLM 后端（无 OPENAI_API_KEY/Ollama/vLLM）：端到端 reference smoke 待环境就绪"
                    ] if status == "PASS" else missing,
                    "next_stage_allowed": False,  # wiring-only gate; real STAGE 1 requires the LLM env
                    "scope_note": "mock smoke validates instrumentation wiring; the stage1 PASS gate itself requires a real end-to-end run",
                }
            )
            print(f"run_id: {run_id}")
            print(f"status: {status}")
            print(f"criteria: {json.dumps({k: v['pass'] for k, v in criteria.items()}, indent=2)}")
            print(f"provenance: {json.dumps(provenance)}")
            print(f"missing_required_files: {missing}")
            if event_schema_errors or record_schema_errors:
                print(f"schema errors: {event_schema_errors[:3]} {record_schema_errors[:3]}")
        except Exception:
            exit_code = 1
            traceback.print_exc(file=sys.stderr)

    # Persist logs after the redirected block so they capture everything.
    logs = run.log_paths()
    logs["stdout"].write_text(stdout_buffer.getvalue(), encoding="utf-8")
    logs["stderr"].write_text(stderr_buffer.getvalue(), encoding="utf-8")
    print(f"run directory: {run.path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
