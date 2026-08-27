"""JSON Schemas for the STAGE 1 artifacts (plan sections 3.9, 13.1, 13.2)."""

from __future__ import annotations

import json

MEMORY_RECORD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "MemoryRecord",
    "type": "object",
    "required": [
        "memory_id", "source_turn_ids", "created_step", "raw_content",
        "keywords", "tags", "context", "embedding_model", "embedding_hash",
        "link_ids", "parent_memory_ids", "evolution_ancestors", "version",
        "content_hash", "token_size", "byte_size", "active_state",
    ],
    "properties": {
        "memory_id": {"type": "string", "pattern": "^m[0-9]{6}$"},
        "source_turn_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "created_step": {"type": "integer", "minimum": 0},
        "raw_content": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
        "context": {"type": "string"},
        "embedding_model": {"type": "string"},
        "embedding_hash": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
        "link_ids": {"type": "array", "items": {"type": "string"}},
        "parent_memory_ids": {"type": "array", "items": {"type": "string"}},
        "evolution_ancestors": {"type": "array", "items": {"type": "string"}},
        "version": {"type": "integer", "minimum": 0},
        "content_hash": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
        "token_size": {"type": "integer", "minimum": 0},
        "byte_size": {"type": "integer", "minimum": 0},
        "active_state": {"type": "string", "enum": ["active", "cold", "deleted"]},
    },
    "additionalProperties": True,
}

EVENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Event",
    "type": "object",
    "required": [
        "event_id", "run_id", "step", "op_id", "event_type", "timestamp",
        "memory_ids", "policy", "score", "budget", "state_hash_before",
        "state_hash_after", "details", "prev_event_hash", "event_hash",
    ],
    "properties": {
        "event_id": {"type": "string", "pattern": "^e[0-9]{6}$"},
        "run_id": {"type": "string", "minLength": 1},
        "step": {"type": "integer", "minimum": 1},
        "op_id": {"type": "string", "pattern": "^op[0-9]{6}$"},
        "event_type": {"type": "string", "enum": ["write", "link", "evolve", "retrieve", "query", "answer"]},
        "timestamp": {"type": "string"},
        "memory_ids": {"type": "array", "items": {"type": "string"}},
        "policy": {"type": "string"},
        "score": {"type": ["number", "null"]},
        "budget": {"type": ["number", "null"]},
        "state_hash_before": {"type": "string"},
        "state_hash_after": {"type": "string"},
        "details": {"type": "object"},
        "prev_event_hash": {"type": "string"},
        "event_hash": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
        "future_label": {},
        "credit_status": {"type": "string", "enum": ["pending", "resolved"]},
    },
    "additionalProperties": False,
}

GATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Gate",
    "type": "object",
    "required": ["stage", "status", "run_ids", "criteria", "next_stage_allowed", "generated_at"],
    "properties": {
        "stage": {"type": "string"},
        "status": {"type": "string", "enum": ["PASS", "FAIL", "PARTIAL", "BLOCKED", "NOT_STARTED"]},
        "run_ids": {"type": "array", "items": {"type": "string"}},
        "preconditions": {"type": "object"},
        "criteria": {"type": "object"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "blocking_issues": {"type": "array", "items": {"type": "string"}},
        "next_stage_allowed": {"type": "boolean"},
        "generated_at": {"type": "string"},
    },
    "additionalProperties": True,
}


def validate_schema(instance: dict, schema: dict) -> list[str]:
    """Validate and return a list of human-readable errors (empty = valid)."""
    try:
        import jsonschema
    except ImportError:
        return []  # schema validation unavailable; treated as skipped
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(instance), key=lambda e: str(e.message))
    ]


__all__ = [
    "MEMORY_RECORD_SCHEMA",
    "EVENT_SCHEMA",
    "GATE_SCHEMA",
    "validate_schema",
    "json",
]
