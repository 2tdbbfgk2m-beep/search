"""STAGE 1 runner end-to-end tests on a synthetic LoCoMo slice.

The fake adapter answers deterministically; these tests validate the
plumbing (provenance, event segments, replay, metrics, run contract,
gate) — never paper numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (REPO_ROOT / "research" / "src", REPO_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from amem_forgetting.testsupport.stubs import install_import_stubs

install_import_stubs()

from amem_forgetting.evaluation import stage1_runner  # noqa: E402
from amem_forgetting.evaluation.stage1_runner import (  # noqa: E402
    _gold_first,
    _session_time,
    run_stage1_locomo,
)
from amem_forgetting.testsupport.fakes import FakeLLMController  # noqa: E402


class EchoAdapter(FakeLLMController):
    """Fake backend whose answers quote retrieved context deterministically."""

    def get_completion(self, prompt, response_format=None, temperature=0.7, **_):
        # Answer with the first words of the retrieved-context snippet —
        # deterministic and traceable to retrieved context.
        marker = "Based on the context: "
        if marker in prompt:
            snippet = prompt.split(marker, 1)[1].split(",", 1)[0]
            return " ".join(snippet.split()[:3])
        return super().get_completion(prompt, response_format, temperature)


def _synthetic_dataset() -> list[dict]:
    conversation = {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1": [
            {"speaker": "Alice", "dia_id": "D1:1", "text": "Alice moved to Berlin in 2019."},
            {"speaker": "Bob", "dia_id": "D1:2", "text": "Bob visited the technology museum."},
        ],
        "session_2": [
            {"speaker": "Alice", "dia_id": "D2:1", "text": "Alice started a robotics startup."},
        ],
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_2_date_time": "3:30 pm on 9 June, 2023",
    }
    qa = [
        {
            "question": "Where does Alice live?",
            "answer": "Berlin",
            "evidence": ["D1:1"],
            "category": 4,
        },
        {
            "question": "When did Alice relocate?",
            "answer": "2019",
            "evidence": ["D1:1"],
            "category": 2,
        },
        {
            "question": "What did Alice launch?",
            "answer": "robotics startup",
            "evidence": ["D2:1"],
            "category": 1,
        },
        {
            "question": "Did Alice ever visit Paris?",
            "answer": "No",
            "adversarial_answer": "Not mentioned in the conversation",
            "evidence": [],
            "category": 5,
        },
    ]
    return [{"sample_id": "0", "conversation": conversation, "qa": qa,
             "event_summary": {}, "observation": {}, "session_summary": {}}]


@pytest.fixture()
def dataset_file(tmp_path):
    path = tmp_path / "mini_locomo.json"
    path.write_text(json.dumps(_synthetic_dataset()), encoding="utf-8")
    return path


@pytest.fixture()
def runs_root(tmp_path):
    return tmp_path / "runs"


def test_session_time_parsing():
    stats = {}
    assert _session_time("1:56 pm on 8 May, 2023", 1, stats) == "202305081356"
    assert _session_time(None, 2, stats) == "202301010200"
    assert _session_time("garbage date", 3, stats) == "202301010300"
    assert stats["time_parse_fallbacks"] == 2


def test_gold_first_is_deterministic():
    assert _gold_first("Where does Alice live?") == _gold_first("Where does Alice live?")
    values = {_gold_first(f"question {i}") for i in range(64)}
    assert values == {True, False}  # balanced pseudo-random distribution


def test_runner_refuses_when_stage0_blocked(dataset_file, runs_root):
    runs_root.mkdir(parents=True)
    (runs_root / "stage0_audit").mkdir()
    (runs_root / "stage0_audit" / "gate.json").write_text(
        json.dumps({"status": "BLOCKED"}), encoding="utf-8"
    )
    result = run_stage1_locomo(
        dataset_file,
        run_id="refused_test",
        runs_root=runs_root,
        adapter_factory=EchoAdapter,
        require_stage0_gate=True,
    )
    assert result["refused"] is True
    assert "stage0 gate status" in result["reason"]


def test_runner_end_to_end_dev_slice(dataset_file, runs_root):
    dataset_sha = stage1_runner._sha256_file(dataset_file)
    summary = run_stage1_locomo(
        dataset_file,
        run_id="e2e_dev_slice",
        runs_root=runs_root,
        adapter_factory=EchoAdapter,
        require_stage0_gate=False,  # dev slice: no real backend, gate bypass recorded
        dataset_sha256=dataset_sha,
        repo_root=REPO_ROOT,
    )
    assert summary.get("exit_code") == 0, summary.get("error")
    assert summary["questions"] == 4
    assert summary["replay"]["worst_boundary_consistency"] == 1.0
    assert summary["replay"]["all_online_hash_match"] is True
    assert summary["schema_errors"] == 0

    run_dir = Path(summary["run_path"])
    gate = json.loads((run_dir / "gate.json").read_text(encoding="utf-8"))
    assert gate["stage"] == "stage1"
    assert gate["status"] == "PASS"
    assert gate["next_stage_allowed"] is True

    results = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(results) == 4
    adversarial_rows = [row for row in results if row["category"] == 5]
    assert len(adversarial_rows) == 1
    assert adversarial_rows[0]["gold_answer"] == "Not mentioned in the conversation"

    # provenance: every memory traces to a dia_id of the synthetic conversation
    segment = run_dir / "events_0.jsonl"
    events = [json.loads(line) for line in segment.read_text(encoding="utf-8").splitlines()]
    writes = [e for e in events if e["event_type"] == "write"]
    assert len(writes) == 3  # 3 turns
    source_turns = {t for w in writes for t in w["details"]["source_turn_ids"]}
    assert source_turns == {"0:D1:1", "0:D1:2", "0:D2:1"}

    # run contract files all present
    missing = sorted(
        name for name in
        ["run_manifest.json", "config.yaml", "git_commit.txt", "stdout.log",
         "stderr.log", "results.jsonl", "metrics.csv", "gate.json"]
        if not (run_dir / name).exists()
    )
    assert missing == []


def test_runner_refuses_reused_run_id(dataset_file, runs_root):
    kwargs = dict(
        runs_root=runs_root,
        adapter_factory=EchoAdapter,
        require_stage0_gate=False,
        repo_root=REPO_ROOT,
    )
    first = run_stage1_locomo(dataset_file, run_id="dup_check", **kwargs)
    assert first.get("exit_code") == 0
    with pytest.raises(ValueError, match="append-only"):
        run_stage1_locomo(dataset_file, run_id="dup_check", **kwargs)


def test_runner_records_qa_errors_but_completes(dataset_file, runs_root):
    class ExplodingAdapter(EchoAdapter):
        def get_completion(self, prompt, response_format=None, temperature=0.7, **_):
            if "Based on the context:" in prompt and "Select the correct answer" not in prompt:
                raise RuntimeError("backend exploded")
            return super().get_completion(prompt, response_format, temperature)

    summary = run_stage1_locomo(
        dataset_file,
        run_id="partial_failure",
        runs_root=runs_root,
        adapter_factory=ExplodingAdapter,
        require_stage0_gate=False,
        repo_root=REPO_ROOT,
    )
    assert summary.get("exit_code") == 0  # per-question errors don't kill the run
    assert summary["stats"]["qa_errors"] == 3  # 3 non-adversarial questions fail
    gate = json.loads((Path(summary["run_path"]) / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "FAIL"
    assert gate["criteria"]["qa_errors"]["pass"] is False
