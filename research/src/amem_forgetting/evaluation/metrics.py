"""Task-quality metrics for STAGE 1 reproduction (plan section 3.8).

Pure-Python implementations: Token-F1 and EM follow SQuAD-style
normalization; BLEU-1 is reported as unigram precision with brevity
penalty omitted but length ratio recorded separately via metrics rows;
ROUGE-L uses LCS F-measure. METEOR/SBERT are reference-track extras and
deliberately not implemented here (documented limitation).
"""

from __future__ import annotations

import re
from typing import Sequence

_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def normalize_text(text: str) -> str:
    return " ".join(_PUNCT_RE.sub(" ", str(text).lower()).split())


def tokenize_for_metric(text: str) -> list[str]:
    tokens = normalize_text(text).split()
    return [token for token in tokens if token not in _ARTICLES]


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = tokenize_for_metric(prediction)
    gold_tokens = tokenize_for_metric(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = sum(min(count, gold_tokens.count(token)) for token, count in common.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_text(prediction) == normalize_text(gold))


def bleu1(prediction: str, gold: str) -> float:
    """Unigram precision against the gold reference (clipped counts)."""
    pred_tokens = tokenize_for_metric(prediction)
    gold_tokens = tokenize_for_metric(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    clipped = sum(min(pred_tokens.count(token), gold_tokens.count(token)) for token in set(pred_tokens))
    return clipped / len(pred_tokens)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        token_a = a[i - 1]
        for j in range(1, len(b) + 1):
            if token_a == b[j - 1]:
                current[j] = previous[j - 1] + 1
            elif current[j - 1] >= previous[j]:
                current[j] = current[j - 1]
            else:
                current[j] = previous[j]
        previous = current
    return previous[-1]


def rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L F-measure over token sequences."""
    pred_tokens = tokenize_for_metric(prediction)[:300]
    gold_tokens = tokenize_for_metric(gold)[:300]
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


TASK_METRICS = {
    "token_f1": token_f1,
    "exact_match": exact_match,
    "bleu1": bleu1,
    "rouge_l": rouge_l,
}


def score_all(prediction: str, gold: str) -> dict[str, float]:
    return {name: round(fn(prediction, gold), 6) for name, fn in TASK_METRICS.items()}


def macro_average(rows: list[dict]) -> dict[str, float]:
    """Macro average over result rows that carry task_metrics dicts."""
    summary: dict[str, float] = {}
    if not rows:
        return summary
    metric_names = sorted({name for row in rows for name in row.get("task_metrics", {})})
    for name in metric_names:
        values = [
            row["task_metrics"][name]
            for row in rows
            if name in row.get("task_metrics", {})
        ]
        summary[f"macro_{name}"] = round(sum(values) / len(values), 6)
    return summary


def group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "unknown")), []).append(row)
    return grouped
