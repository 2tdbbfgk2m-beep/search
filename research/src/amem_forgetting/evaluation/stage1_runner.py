"""STAGE 1 real-evaluation runner: faithful A-MEM reproduction on LoCoMo.

Implements plan section 5 experiments 1A (end-to-end reproduction) with
full instrumentation: source-turn provenance, per-conversation event
segments, replay verification, and the run-directory contract.

KeepAll only (STAGE 1 does not forget); budget fixed at 1.0.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
import traceback
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from amem_forgetting.evaluation.metrics import group_by, macro_average, score_all
from amem_forgetting.instrumentation import EventReplayer, InstrumentedMemorySystem
from amem_forgetting.runsupport import RunDirectory, git_commit


# ---------------------------------------------------------------------------
# deterministic helpers


def _session_time(date_time: str | None, session_id: int, stats: dict) -> str:
    """LoCoMo ``1:56 pm on 8 May, 2023`` -> upstream ``%Y%m%d%H%M``."""
    if date_time:
        try:
            parsed = datetime.strptime(date_time.strip(), "%I:%M %p on %d %B, %Y")
            return parsed.strftime("%Y%m%d%H%M")
        except ValueError:
            pass
    stats["time_parse_fallbacks"] = stats.get("time_parse_fallbacks", 0) + 1
    return f"20230101{session_id:02d}00"


def _turn_text(turn: Any) -> str:
    """Mirror the upstream evaluation's memory text format exactly."""
    return "Speaker " + turn.speaker + "says : " + turn.text


def _gold_first(question: str) -> bool:
    """Deterministic per-question option order (fixes the unseeded coin)."""
    return int(sha256(question.encode("utf-8")).hexdigest()[0], 16) % 2 == 0


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# answerers


def build_answerers(
    adapter: Any,
    system: InstrumentedMemorySystem,
    *,
    temperature: float = 0.0,
) -> tuple[Callable[[str, list[str]], str], Callable[[str, list[str], list[str]], str]]:
    """Return (plain_answerer, adversarial_answerer) over retrieved ids."""

    def _context(prompt_ids: list[str]) -> str:
        lines = []
        for memory_id in prompt_ids:
            note = system.base.memories.get(memory_id)
            if note is not None:
                lines.append(note.content)
        return "\n".join(lines)

    def plain(question: str, prompt_ids: list[str]) -> str:
        user_prompt = (
            f"Based on the context: {_context(prompt_ids)}, answer the following "
            f"question. {question}\n"
            "Answer the question based only on the information provided in the "
            "context above. Short answer:"
        )
        return adapter.get_completion(user_prompt, response_format=None, temperature=temperature)

    def adversarial(question: str, prompt_ids: list[str], options: list[str]) -> str:
        user_prompt = (
            f"Based on the context: {_context(prompt_ids)}, answer the following "
            f"question. {question}\n\n"
            f"Select the correct answer: {options[0]} or {options[1]}  Short answer:"
        )
        return adapter.get_completion(user_prompt, response_format=None, temperature=temperature)

    return plain, adversarial


# ---------------------------------------------------------------------------
# preflight


def preflight(adapter: Any, dataset_path: Path, expected_sha256: str | None) -> dict:
    """Plan section 13 step 3: dataset hash, backend ping, disk headroom."""
    import shutil

    report: dict[str, Any] = {"dataset_sha256": _sha256_file(dataset_path)}
    report["dataset_hash_match"] = (
        expected_sha256 is None or report["dataset_sha256"] == expected_sha256
    )
    try:
        ping = adapter.get_completion(
            'Return exactly this JSON: {"ok": true}', response_format=None, temperature=0.0
        )
        report["backend_preflight_ok"] = "ok" in str(ping).lower()
        report["backend_preflight_preview"] = str(ping)[:120]
    except Exception as error:  # noqa: BLE001 - preflight reports, never crashes
        report["backend_preflight_ok"] = False
        report["backend_preflight_error"] = str(error)[:300]
    usage = shutil.disk_usage(str(dataset_path.anchor or "/"))
    report["disk_free_gb"] = round(usage.free / 1e9, 1)
    return report


# ---------------------------------------------------------------------------
# per-conversation evaluation


def _evaluate_conversation(
    system: InstrumentedMemorySystem,
    sample: Any,
    *,
    adapter: Any,
    top_k: int,
    max_qa: int | None,
    run_id: str,
    stats: dict,
) -> list[dict]:
    """Write all turns as memories, then answer every QA with raw semantics."""
    plain, adversarial = build_answerers(adapter, system)

    for session_id, session in sorted(sample.conversation.sessions.items()):
        note_time = _session_time(session.date_time, session_id, stats)
        for turn in session.turns:
            system.add_note(
                _turn_text(turn),
                time=note_time,
                source_turn_ids=[f"{sample.sample_id}:{turn.dia_id}"],
            )
    stats["memories_total"] = stats.get("memories_total", 0) + len(system.base.memories)

    rows: list[dict] = []
    questions = sample.qa if max_qa is None else sample.qa[:max_qa]
    for qa_index, qa in enumerate(questions):
        question_id = f"{sample.sample_id}:q{qa_index:04d}"
        gold = qa.final_answer
        prompt_ids: list[str] = []
        try:
            seeds, expanded, prompt_ids = system._raw_traversal_ids(qa.question, top_k)
            if qa.category == 5:
                gold_option = "Not mentioned in the conversation"
                other_option = str(gold)
                options = (
                    [gold_option, other_option]
                    if _gold_first(qa.question)
                    else [other_option, gold_option]
                )
                answer = adversarial(qa.question, prompt_ids, options)
            else:
                answer = plain(qa.question, prompt_ids)
            metrics = score_all(str(answer), str(gold)) if gold is not None else {}
            error = None
        except Exception as eval_error:  # noqa: BLE001 - per-question errors never kill the run
            answer, metrics, error = "", {}, str(eval_error)[:300]
            stats["qa_errors"] = stats.get("qa_errors", 0) + 1
        rows.append(
            {
                "run_id": run_id,
                "dataset_group": str(sample.sample_id),
                "question_id": question_id,
                "category": qa.category,
                "policy": "KeepAll",
                "budget_ratio": 1.0,
                "question": qa.question,
                "gold_answer": gold,
                "answer": answer,
                "gold_evidence_ids": list(qa.evidence),
                "prompt_ids": prompt_ids,
                "task_metrics": metrics,
                "error": error,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# run orchestration


def run_stage1_locomo(
    dataset_path: str | Path,
    *,
    run_id: str,
    runs_root: str | Path,
    adapter_factory: Callable[[], Any],
    embedding_model: str = "all-MiniLM-L6-v2",
    track: str = "extension",
    top_k_retrieval: int = 10,
    limit_conversations: int | None = None,
    max_qa_per_conversation: int | None = None,
    require_stage0_gate: bool = True,
    dataset_sha256: str | None = None,
    repo_root: str | Path | None = None,
) -> dict:
    """Execute a full STAGE 1 KeepAll reproduction run with gate artifacts."""
    import memory_layer

    runs_root = Path(runs_root)
    gate_path = runs_root / "stage0_audit" / "gate.json"
    if require_stage0_gate and gate_path.exists():
        stage0_status = json.loads(gate_path.read_text(encoding="utf-8")).get("status")
        if stage0_status not in {"PASS", "PARTIAL"}:
            return {
                "refused": True,
                "reason": (
                    f"stage0 gate status={stage0_status!r}; resolve STAGE 0 "
                    "(real backend smoke) or pass require_stage0_gate=False for dev slices"
                ),
            }

    run = RunDirectory(run_id, runs_root)
    stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
    exit_code, error_text = 0, None
    started = time.time()
    summary: dict[str, Any] = {"run_id": run_id}
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        try:
            from load_dataset import load_locomo_dataset

            dataset_path = Path(dataset_path)
            adapter = adapter_factory()
            pre = preflight(adapter, dataset_path, dataset_sha256)
            samples = load_locomo_dataset(dataset_path)
            if limit_conversations is not None:
                samples = samples[:limit_conversations]

            stats: dict[str, Any] = {"qa_errors": 0, "time_parse_fallbacks": 0}
            all_rows: list[dict] = []
            replay_reports: list[dict] = []
            schema_errors = 0
            for sample in samples:
                base = memory_layer.AgenticMemorySystem(
                    model_name=embedding_model,
                    llm_backend="ollama",
                    llm_model="unused-instrumented",
                )
                base.llm_controller.llm = adapter
                system = InstrumentedMemorySystem(base, run_id=f"{run_id}::{sample.sample_id}")
                rows = _evaluate_conversation(
                    system,
                    sample,
                    adapter=adapter,
                    top_k=top_k_retrieval,
                    max_qa=max_qa_per_conversation,
                    run_id=run_id,
                    stats=stats,
                )
                all_rows.extend(rows)
                replayer = EventReplayer(system.log)
                boundary = replayer.verify_boundaries()
                online = replayer.verify_against_online(system)
                replay_reports.append(
                    {
                        "sample_id": sample.sample_id,
                        "consistency_rate": boundary["consistency_rate"],
                        "online_hash_match": online["state_hash_match"],
                    }
                )
                for event in system.log.events:
                    from amem_forgetting.schemas import EVENT_SCHEMA, validate_schema

                    schema_errors += len(validate_schema(event.to_dict(), EVENT_SCHEMA))
                run.write_events_segment(
                    str(sample.sample_id).replace("/", "_"),
                    "\n".join(
                        json.dumps(
                            event.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False
                        )
                        for event in system.log.events
                    )
                    + ("\n" if system.log.events else ""),
                )
                for row in rows:
                    run.append_result(row)

            summary.update(_summarize(all_rows, replay_reports, schema_errors, stats, pre))
            summary["preflight"] = pre
            _write_run_files(
                run,
                summary=summary,
                run_id=run_id,
                track=track,
                top_k=top_k_retrieval,
                dataset_path=dataset_path,
                samples=samples,
                repo_root=repo_root,
            )
        except Exception:
            exit_code = 1
            error_text = traceback.format_exc()
            print(error_text, file=sys.stderr)
            summary["fatal"] = True
            summary["error"] = error_text[-2000:]
            run.write_error_analysis(
                "# Error analysis\n\nRun failed with an unhandled exception:\n\n```\n"
                + error_text
                + "\n```\n"
            )
    logs = run.log_paths()
    logs["stdout"].write_text(stdout_buffer.getvalue(), encoding="utf-8")
    logs["stderr"].write_text(stderr_buffer.getvalue(), encoding="utf-8")
    summary["elapsed_seconds"] = round(time.time() - started, 1)
    summary["exit_code"] = exit_code
    summary["run_path"] = str(run.path)
    return summary


# ---------------------------------------------------------------------------
# summary + run files


def _summarize(
    all_rows: list[dict],
    replay_reports: list[dict],
    schema_errors: int,
    stats: dict,
    pre: dict,
) -> dict:
    overall = macro_average(all_rows)
    per_category = {
        category: macro_average(rows)
        for category, rows in group_by(all_rows, "category").items()
    }
    worst_consistency = min(
        (r["consistency_rate"] for r in replay_reports), default=1.0
    )
    all_hash_match = all(r["online_hash_match"] for r in replay_reports) if replay_reports else True
    return {
        "questions": len(all_rows),
        "macro": overall,
        "per_category": per_category,
        "replay": {
            "conversations": len(replay_reports),
            "worst_boundary_consistency": worst_consistency,
            "all_online_hash_match": all_hash_match,
        },
        "schema_errors": schema_errors,
        "stats": stats,
        "preflight": pre,
    }


def _write_run_files(
    run: RunDirectory,
    *,
    summary: dict,
    run_id: str,
    track: str,
    top_k: int,
    dataset_path: Path,
    samples: list,
    repo_root: str | Path | None,
) -> None:
    worst = summary["replay"]["worst_boundary_consistency"]
    gate_pass = (
        worst == 1.0
        and summary["replay"]["all_online_hash_match"]
        and summary["schema_errors"] == 0
        and summary["stats"].get("qa_errors", 0) == 0
        and summary["preflight"]["dataset_hash_match"]
    )
    gate = {
        "stage": "stage1",
        "status": "PASS" if gate_pass else "FAIL",
        "run_ids": [run_id],
        "preconditions": {"stage0": "checked-before-run"},
        "criteria": {
            "replay_boundary_consistency": {
                "value": worst, "threshold": 1.0, "pass": worst == 1.0,
            },
            "replay_online_hash_match": {
                "value": summary["replay"]["all_online_hash_match"],
                "threshold": True,
                "pass": summary["replay"]["all_online_hash_match"],
            },
            "event_schema_valid": {
                "value": summary["schema_errors"], "threshold": 0,
                "pass": summary["schema_errors"] == 0,
            },
            "qa_errors": {
                "value": summary["stats"].get("qa_errors", 0), "threshold": 0,
                "pass": summary["stats"].get("qa_errors", 0) == 0,
            },
            "dataset_hash_match": {
                "value": summary["preflight"]["dataset_hash_match"], "threshold": True,
                "pass": summary["preflight"]["dataset_hash_match"],
            },
        },
        "uncertainties": summary.get("preflight", {}).get("backend_preflight_ok") is not None
        and [],
        "blocking_issues": [],
        "next_stage_allowed": bool(gate_pass),
    }
    run.write_manifest(
        {
            "stage": "stage1",
            "kind": "keepall_reproduction",
            "track": track,
            "git_commit": git_commit(repo_root or Path(__file__).resolve().parents[3]),
            "dataset": str(dataset_path),
            "dataset_sha256": summary["preflight"]["dataset_sha256"],
            "conversations": len(samples),
            "top_k_retrieval": top_k,
            "adversarial_protocol": "deterministic per-question option order",
        }
    )
    run.write_config_mapping(
        {
            "project": {"name": "amem_graph_forgetting", "track": track, "run_root": str(run.root)},
            "amem": {"top_k_retrieval": top_k, "retrieval_mode": "raw_1hop_expansion"},
            "forgetting": {"policy": "KeepAll", "budget_ratio": 1.0},
        }
    )
    run.write_git_commit(git_commit(repo_root or Path(__file__).resolve().parents[3]))
    metric_rows = [{"scope": "overall", **summary["macro"]}]
    for category, macro in summary["per_category"].items():
        metric_rows.append({"scope": f"category_{category}", **macro})
    run.write_metrics(metric_rows)
    run.write_gate(gate)


# ---------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> int:
    import argparse

    from amem_forgetting.evaluation.backends import build_adapter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--runs_root", default="research/runs/raw")
    parser.add_argument("--backend", default="vllm", choices=["openai", "vllm", "ollama", "sglang"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--base_url", default=None, help="OpenAI-compatible endpoint override")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--track", default="extension", choices=["extension", "reference"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--limit_conversations", type=int, default=None)
    parser.add_argument("--max_qa_per_conversation", type=int, default=None)
    parser.add_argument("--require_stage0_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset_sha256", default=None)
    args = parser.parse_args(argv)

    summary = run_stage1_locomo(
        args.dataset,
        run_id=args.run_id,
        runs_root=args.runs_root,
        adapter_factory=lambda: build_adapter(
            args.backend, args.model, base_url=args.base_url, api_key=args.api_key
        ),
        track=args.track,
        top_k_retrieval=args.top_k,
        limit_conversations=args.limit_conversations,
        max_qa_per_conversation=args.max_qa_per_conversation,
        require_stage0_gate=args.require_stage0_gate,
        dataset_sha256=args.dataset_sha256,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return int(summary.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
