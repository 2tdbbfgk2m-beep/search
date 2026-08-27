"""Import-time stubs for the heavy dependencies of ``memory_layer``.

The upstream module imports rank_bm25, sentence_transformers,
litellm, and transformers at module level. When those packages are
absent, ``install_import_stubs()`` inserts minimal stand-ins so the
real upstream code can still be imported and exercised.

The fake sentence encoder is a deterministic bag-of-hashed-words
embedding: cosine similarity approximates token overlap, which keeps
retrieval meaningful (not random) while remaining byte-reproducible.
"""

from __future__ import annotations

import hashlib
import importlib
import re
import sys
import types

import numpy as np


class _FakeBM25Okapi:
    """Never used by AgenticMemorySystem; satisfies the module import."""

    def __init__(self, corpus):
        self.corpus = list(corpus)

    def get_scores(self, query):
        return [0.0] * len(self.corpus)

    def add_document(self, document):
        self.corpus.append(document)


class _FakeSentenceTransformer:
    """Deterministic hash-based encoder standing in for all-MiniLM-L6-v2."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    def get_config_dict(self):
        return {"model_name": self.model_name}

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(64, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            index = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % 64
            vector[index] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def encode(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return np.array([self._vector(text) for text in texts], dtype=np.float32)


def _litellm_completion_stub(*_args, **_kwargs):
    raise RuntimeError(
        "litellm is stubbed: no real LLM backend is available in this environment"
    )


def _make_module(name: str, attributes: dict) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def install_import_stubs() -> list[str]:
    """Insert stub modules for missing heavy dependencies.

    Returns the list of module names that were stubbed. Real packages,
    when present, always win; already-imported modules are never replaced.
    """
    specs = {
        "rank_bm25": {"BM25Okapi": _FakeBM25Okapi},
        "sentence_transformers": {"SentenceTransformer": _FakeSentenceTransformer},
        "litellm": {"completion": _litellm_completion_stub},
        "transformers": {"AutoModel": object, "AutoTokenizer": object},
    }
    stubbed = []
    for name, attributes in specs.items():
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
            continue  # real package available
        except ImportError:
            sys.modules[name] = _make_module(name, attributes)
            stubbed.append(name)
    return stubbed
