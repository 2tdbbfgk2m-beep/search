"""Deterministic fake LLM controller for wiring tests and mock smoke runs.

It implements the same ``get_completion(prompt, response_format,
temperature)`` interface as the upstream inner controllers and answers
the two upstream prompts (note analysis, memory evolution) with valid,
content-derived JSON. Responses depend only on the prompt text, so
runs are byte-reproducible when note timestamps are fixed.
"""

from __future__ import annotations

import hashlib
import json
import re


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeLLMController:
    """Drop-in replacement for ``system.llm_controller.llm``."""

    def __init__(self, *, evolve_probability: float = 1.0):
        self.evolve_probability = evolve_probability
        self.calls: list[dict] = []

    # ------------------------------------------------------------------

    def get_completion(self, prompt: str, response_format=None, temperature: float = 0.7, **_kwargs) -> str:
        self.calls.append({"prompt_chars": len(prompt)})
        if "memory evolution agent" in prompt:
            response = self._evolution_response(prompt)
        elif "Generate a structured analysis" in prompt:
            response = self._analysis_response(prompt)
        else:
            response = json.dumps({"keywords": [], "context": "General", "tags": []})
        return json.dumps(response) if not isinstance(response, str) else response

    # ------------------------------------------------------------------

    @staticmethod
    def _content_of_analysis_prompt(prompt: str) -> str:
        marker = "Content for analysis:"
        return prompt.split(marker, 1)[1].strip() if marker in prompt else prompt.strip()

    def _analysis_response(self, prompt: str) -> dict:
        content = self._content_of_analysis_prompt(prompt)
        words = re.findall(r"[A-Za-z0-9]+", content)
        keywords = sorted(set(words), key=lambda w: (-len(w), w))[:3] or ["note"]
        context = " ".join(words[:8]) + ("..." if len(words) > 8 else "")
        digest = _digest(content)[:6]
        return {
            "keywords": keywords,
            "context": context or "General",
            "tags": [f"tag-{digest}", "synthetic"],
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _neighbor_indices(prompt: str) -> list[int]:
        return [int(match) for match in re.findall(r"memory index:(\d+)", prompt)]

    @staticmethod
    def _original_contexts(prompt: str) -> list[str]:
        """Parse the original context of each neighbor from the prompt."""
        contexts: list[str] = []
        for line in prompt.splitlines():
            match = re.search(r"memory context:(.*?)(?:\t| memory keywords:)", line)
            contexts.append(match.group(1).strip() if match else "")
        return contexts

    def _evolution_response(self, prompt: str) -> dict:
        neighbors = self._neighbor_indices(prompt)
        if not neighbors:
            return {
                "should_evolve": False,
                "actions": [],
                "suggested_connections": [],
                "tags_to_update": [],
                "new_context_neighborhood": [],
                "new_tags_neighborhood": [],
            }
        digest = _digest(prompt)[:6]
        # Odd-positioned neighbors keep their original context (when it can
        # be parsed back) so that changed_fields varies between evolutions.
        original_contexts = self._original_contexts(prompt)
        new_contexts: list[str] = []
        for position in range(len(neighbors)):
            if position % 2 == 1 and position < len(original_contexts) and original_contexts[position]:
                new_contexts.append(original_contexts[position])
            else:
                new_contexts.append(f"evolved-context-{digest}-{position}")
        return {
            "should_evolve": True,
            "actions": ["strengthen", "update_neighbor"],
            "suggested_connections": [neighbors[0]],
            "tags_to_update": [f"strengthened-{digest}"],
            "new_context_neighborhood": new_contexts,
            "new_tags_neighborhood": [[f"evolved-tag-{digest}-{position}"] for position in range(len(neighbors))],
        }
