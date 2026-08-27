"""LLM backend adapters for real STAGE 1 runs.

One OpenAI-compatible adapter covers OpenAI, vLLM, and Ollama servers
(all expose ``/chat/completions``); the upstream SGLang controller is
reused for SGLang-served models. Anything that satisfies
``get_completion(prompt, response_format=None, temperature=0.0) -> str``
can be injected instead (tests use FakeLLMController).
"""

from __future__ import annotations

import json
import time
from typing import Any


class ChatCompatAdapter:
    """OpenAI-compatible chat completions client (OpenAI / vLLM / Ollama)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:30000/v1",
        api_key: str | None = None,
        timeout: int = 120,
        max_retries: int = 2,
    ):
        import requests

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout
        self.max_retries = max_retries
        self._requests = requests

    def get_completion(self, prompt: str, response_format: dict | None = None, temperature: float = 0.7) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        if response_format and "json_schema" in response_format:
            if self.base_url.startswith("https://api.openai.com"):
                # Reference track: OpenAI supports json_schema natively.
                payload["response_format"] = response_format
            else:
                # vLLM guided decoding accepts this top-level parameter.
                payload["guided_json"] = response_format["json_schema"]["schema"]
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"backend HTTP {response.status_code}: {response.text[:200]}")
                return response.json()["choices"][0]["message"]["content"]
            except Exception as error:  # noqa: BLE001 - retry any transport/parse failure
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"chat completion failed after retries: {last_error}")


class SGLangAdapter:
    """Thin pass-through over the upstream SGLangController interface."""

    def __init__(self, model: str, *, host: str = "http://localhost", port: int = 30000):
        from memory_layer import SGLangController

        self._inner = SGLangController(model, host, port)

    def get_completion(self, prompt: str, response_format: dict | None = None, temperature: float = 0.7) -> str:
        if response_format is None:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "r", "schema": {"type": "object"}, "strict": True},
            }
        return self._inner.get_completion(prompt, response_format, temperature)


def build_adapter(
    backend: str,
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Return an adapter exposing the uniform ``get_completion`` interface."""
    if backend in {"openai", "openai_compat", "vllm", "ollama"}:
        default_urls = {
            "openai": "https://api.openai.com/v1",
            "openai_compat": base_url or "http://localhost:30000/v1",
            "vllm": base_url or "http://localhost:8000/v1",
            "ollama": base_url or "http://localhost:11434/v1",
        }
        key_env_hint = {"openai": "OPENAI_API_KEY"}
        if backend == "openai" and api_key is None:
            import os

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(f"{key_env_hint['openai']} not set for reference track")
        return ChatCompatAdapter(model, base_url=default_urls[backend], api_key=api_key)
    if backend == "sglang":
        host, _, port_part = (base_url or "http://localhost:30000").rstrip("/").rsplit(":", 1)
        return SGLangAdapter(model, host=host or "http://localhost", port=int(port_part))
    raise ValueError(f"unknown backend {backend!r}; expected openai|vllm|ollama|sglang|mock")
