from __future__ import annotations

from ts_rag_agent.application.evidence_selection import tokenize_text


def token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 between a prediction and a gold answer."""

    prediction_tokens = tokenize_text(prediction)
    gold_tokens = tokenize_text(gold)
    if not prediction_tokens or not gold_tokens:
        return 0.0

    prediction_counts = _count_tokens(prediction_tokens)
    gold_counts = _count_tokens(gold_tokens)
    overlap = sum(
        min(prediction_counts[token], gold_counts[token])
        for token in prediction_counts.keys() & gold_counts.keys()
    )
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _count_tokens(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts
