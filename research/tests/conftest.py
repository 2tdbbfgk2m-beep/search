"""Shared test fixtures: stub heavy deps before importing memory_layer."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (REPO_ROOT / "research" / "src", REPO_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from amem_forgetting.testsupport.stubs import install_import_stubs

STUBBED = install_import_stubs()

import pytest

import memory_layer  # noqa: E402  (requires stubs on machines without the ML stack)

from amem_forgetting.instrumentation import (  # noqa: E402
    EventReplayer,
    InstrumentedMemorySystem,
)
from amem_forgetting.testsupport.fakes import FakeLLMController  # noqa: E402


@pytest.fixture()
def system():
    """A fully wired instrumented A-MEM with a deterministic fake LLM."""
    base = memory_layer.AgenticMemorySystem(
        model_name="all-MiniLM-L6-v2",
        llm_backend="ollama",
        llm_model="fake",
    )
    base.llm_controller.llm = FakeLLMController()
    return InstrumentedMemorySystem(base, run_id="test_run")


@pytest.fixture()
def populated(system):
    """System with a 5-turn conversation loaded, evolutions triggered.

    Explicit timestamps make the fake-LLM evolution decisions (which key
    off prompt text, including talk start times) fully deterministic.
    """
    turns = [
        ("conv1:turn1", "Alice moved to Berlin in 2019 and joined a robotics startup."),
        ("conv1:turn2", "Bob visited Alice in Berlin and they toured the technology museum."),
        ("conv1:turn3", "Alice's startup launched a warehouse robot that cuts picking errors."),
        ("conv1:turn4", "Bob relocated to Vienna for a linguistics degree."),
        ("conv1:turn5", "The warehouse robot was deployed in a Vienna logistics hub."),
    ]
    for offset, (turn_id, text) in enumerate(turns, start=1):
        system.add_note(
            text,
            source_turn_ids=[turn_id],
            time="2025060112" + f"{(offset - 1) * 10:02d}",  # upstream %Y%m%d%H%M
        )
    return system
