"""Run scaffolding: manifest, config snapshot, gate, required run files.

Implements the run-directory contract from plan section 3.1 (every run
contains run_manifest.json, config.yaml, git_commit.txt, stdout.log,
stderr.log, results.jsonl, metrics.csv, gate.json).

Security posture: run ids are validated against a strict pattern and
every written file name must be in a fixed allowlist, so no caller
input can influence the location of a write.
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from amem_forgetting.schemas import GATE_SCHEMA, validate_schema

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SEGMENT_STEM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,80}$")

# The only file names a run directory may ever contain.
_ALLOWED_FILES = frozenset(
    {
        "run_manifest.json",
        "config.yaml",
        "git_commit.txt",
        "stdout.log",
        "stderr.log",
        "results.jsonl",
        "metrics.csv",
        "gate.json",
        "error_analysis.md",
    }
)


def _segment_file_name(stem: str) -> str:
    """Map a validated stem to ``events_<stem>.jsonl``; stems stay constrained."""
    if not isinstance(stem, str) or not _SEGMENT_STEM_PATTERN.match(stem):
        raise ValueError(f"unsafe event segment stem: {stem!r}")
    return f"events_{stem}.jsonl"


def git_commit(repo_root: Path | str) -> str:
    """Return the short commit hash, or 'unknown' outside a git work tree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _is_allowed_file_name(name: str) -> bool:
    if name in _ALLOWED_FILES:
        return True
    if name.startswith("events_") and name.endswith(".jsonl"):
        return bool(_SEGMENT_STEM_PATTERN.match(name[len("events_") : -len(".jsonl")]))
    return False


def _atomic_write(directory: Path, filename: str, content: str) -> None:
    """Write ``directory/filename`` atomically; ``filename`` must be allowlisted."""
    if not _is_allowed_file_name(filename):
        raise ValueError(f"unexpected run file name: {filename!r}")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{filename}.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(directory / filename)


class RunDirectory:
    """Create and populate a run directory under ``runs/raw``."""

    def __init__(self, run_id: str, root: Path | str):
        if not isinstance(run_id, str) or not _RUN_ID_PATTERN.match(run_id):
            raise ValueError(
                "run_id must match [A-Za-z0-9][A-Za-z0-9._-]* and contain no path separators"
            )
        self.run_id = run_id
        self.root = Path(root)
        self.path = self.root / run_id
        self.path.mkdir(parents=True, exist_ok=True)
        existing = [entry.name for entry in self.path.iterdir()]
        if existing:
            raise ValueError(
                f"run directory {self.path} already contains {len(existing)} file(s); "
                "raw results are append-only — use a fresh run_id"
            )
        self.files_written: list[str] = []

    # ------------------------------------------------------------------

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        payload = dict(manifest)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
        self._write_json("run_manifest.json", payload)

    def write_config(self, config_text: str) -> None:
        _atomic_write(self.path, "config.yaml", config_text)
        self.files_written.append("config.yaml")

    def write_git_commit(self, commit: str) -> None:
        _atomic_write(self.path, "git_commit.txt", commit + "\n")
        self.files_written.append("git_commit.txt")

    def write_config_mapping(self, config: Mapping[str, Any]) -> None:
        try:
            import yaml

            text = yaml.safe_dump(dict(config), sort_keys=True, allow_unicode=True)
        except ImportError:
            text = json.dumps(dict(config), indent=2, ensure_ascii=False)
        self.write_config(text)

    def append_result(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(dict(record), sort_keys=True, ensure_ascii=False, allow_nan=False, default=str)
        if "results.jsonl" not in self.files_written:
            self.files_written.append("results.jsonl")
            _atomic_write(self.path, "results.jsonl", line + "\n")
        else:
            if "results.jsonl" not in _ALLOWED_FILES:  # pragma: no cover - invariant
                raise ValueError("results.jsonl not allowlisted")
            with open(self.path / "results.jsonl", "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def write_metrics(self, rows: list[Mapping[str, Any]]) -> None:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        _atomic_write(self.path, "metrics.csv", buffer.getvalue())
        self.files_written.append("metrics.csv")

    def write_gate(self, gate: Mapping[str, Any]) -> dict:
        payload = dict(gate)
        payload.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
        errors = validate_schema(payload, GATE_SCHEMA)
        if errors:
            raise ValueError(f"gate.json schema violations: {errors}")
        self._write_json("gate.json", payload)
        return payload

    def write_error_analysis(self, text: str) -> None:
        _atomic_write(self.path, "error_analysis.md", text)
        self.files_written.append("error_analysis.md")

    def write_events_segment(self, stem: str, text: str) -> str:
        """Persist one conversation's event-log segment (per-sample provenance)."""
        name = _segment_file_name(stem)
        _atomic_write(self.path, name, text)
        self.files_written.append(name)
        return name

    def log_paths(self) -> dict[str, Path]:
        return {
            "stdout": self.path / "stdout.log",
            "stderr": self.path / "stderr.log",
        }

    def missing_required_files(self) -> list[str]:
        required = [
            "run_manifest.json", "config.yaml", "git_commit.txt",
            "stdout.log", "stderr.log", "results.jsonl", "metrics.csv", "gate.json",
        ]
        return [name for name in required if not (self.path / name).exists()]

    # ------------------------------------------------------------------

    def _write_json(self, name: str, payload: Mapping[str, Any]) -> None:
        text = json.dumps(dict(payload), indent=2, ensure_ascii=False, allow_nan=False, default=str)
        _atomic_write(self.path, name, text)
        self.files_written.append(name)
