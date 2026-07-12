from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median

from ts_rag_agent.application.verified_rag_evaluation import (
    VerifiedRAGEvaluationResult,
    VerifiedRAGQuestionResult,
)

NEWLY_REFUSED_BUCKET_DEFINITIONS = {
    "reasonable_refusal_unanswerable": "数据集标注为不可回答，验证层拒答，方向正确。",
    "safe_refusal_retrieval_miss": "问题可回答，但 gold 文档不在检索上下文中，拒答比强答更安全。",
    "possible_threshold_over_refusal_gold_cited": (
        "问题可回答，gold 文档在上下文中且已被引用，但证据分低于阈值，可能是阈值误杀。"
    ),
    "evidence_selection_miss_gold_available": (
        "问题可回答，gold 文档在上下文中，但原始答案没有引用它，说明证据选择仍有问题。"
    ),
    "unknown_new_refusal": "未被当前规则覆盖的新增拒答，需要人工检查。",
}


def analyze_verified_rag_quality(
    evaluation: VerifiedRAGEvaluationResult,
    min_evidence_score: float,
    sample_limit_per_bucket: int = 5,
) -> dict:
    """分析答案验证层带来的质量收益、误杀风险和剩余风险。"""

    if min_evidence_score < 0:
        raise ValueError("min_evidence_score must be non-negative")
    if sample_limit_per_bucket < 0:
        raise ValueError("sample_limit_per_bucket must be non-negative")

    question_results = evaluation.question_results
    newly_refused = [
        result
        for result in question_results
        if not result.original_answer.refused and result.verified_answer.refused
    ]
    unanswerable_still_answered = [
        result
        for result in question_results
        if not result.question.answerable and not result.verified_answer.refused
    ]
    answerable_still_answered = [
        result
        for result in question_results
        if result.question.answerable and not result.verified_answer.refused
    ]
    answerable_answered_without_gold_citation = [
        result for result in answerable_still_answered if not _gold_doc_cited(result)
    ]

    bucket_counts = Counter(_categorize_newly_refused(result) for result in newly_refused)
    return {
        "newly_refused": {
            "total": len(newly_refused),
            "answerable": sum(1 for result in newly_refused if result.question.answerable),
            "unanswerable": sum(1 for result in newly_refused if not result.question.answerable),
            "bucket_counts": _ordered_bucket_counts(bucket_counts),
            "answerable_gold_in_context": sum(
                1
                for result in newly_refused
                if result.question.answerable and _gold_doc_in_context(result)
            ),
            "answerable_gold_cited": sum(
                1
                for result in newly_refused
                if result.question.answerable and _gold_doc_cited(result)
            ),
            "near_threshold_count": _count_near_threshold(newly_refused, min_evidence_score),
            "max_evidence_score_stats": _score_stats(newly_refused),
        },
        "remaining_risks": {
            "unanswerable_still_answered": len(unanswerable_still_answered),
            "answerable_still_answered": len(answerable_still_answered),
            "answerable_answered_without_gold_citation": len(
                answerable_answered_without_gold_citation
            ),
            "answerable_answered_without_gold_citation_rate": _safe_rate(
                len(answerable_answered_without_gold_citation),
                len(answerable_still_answered),
            ),
            "unanswerable_still_answered_rate": _safe_rate(
                len(unanswerable_still_answered),
                sum(1 for result in question_results if not result.question.answerable),
            ),
        },
        "bucket_definitions": NEWLY_REFUSED_BUCKET_DEFINITIONS,
        "samples_by_newly_refused_bucket": _samples_by_bucket(
            newly_refused,
            sample_limit_per_bucket,
        ),
        "remaining_risk_samples": {
            "unanswerable_still_answered": [
                _build_sample(result)
                for result in unanswerable_still_answered[:sample_limit_per_bucket]
            ],
            "answerable_answered_without_gold_citation": [
                _build_sample(result)
                for result in answerable_answered_without_gold_citation[:sample_limit_per_bucket]
            ],
        },
    }


def _categorize_newly_refused(result: VerifiedRAGQuestionResult) -> str:
    if not result.question.answerable:
        return "reasonable_refusal_unanswerable"

    if not _gold_doc_in_context(result):
        return "safe_refusal_retrieval_miss"

    if _gold_doc_cited(result):
        return "possible_threshold_over_refusal_gold_cited"

    if result.question.answer_doc_id:
        return "evidence_selection_miss_gold_available"

    return "unknown_new_refusal"


def _ordered_bucket_counts(bucket_counts: Counter[str]) -> dict[str, int]:
    return {
        bucket_name: bucket_counts.get(bucket_name, 0)
        for bucket_name in NEWLY_REFUSED_BUCKET_DEFINITIONS
    }


def _samples_by_bucket(
    newly_refused: list[VerifiedRAGQuestionResult],
    sample_limit_per_bucket: int,
) -> dict[str, list[dict]]:
    bucketed_results: dict[str, list[VerifiedRAGQuestionResult]] = defaultdict(list)
    for result in newly_refused:
        bucketed_results[_categorize_newly_refused(result)].append(result)

    samples = {}
    for bucket_name in NEWLY_REFUSED_BUCKET_DEFINITIONS:
        bucket_results = bucketed_results.get(bucket_name, [])
        samples[bucket_name] = [
            _build_sample(result) for result in bucket_results[:sample_limit_per_bucket]
        ]
    return samples


def _build_sample(result: VerifiedRAGQuestionResult) -> dict:
    return {
        "question_id": result.question.id,
        "question_title": result.question.title,
        "question_text": _truncate(result.question.text),
        "answerable": result.question.answerable,
        "gold_answer_doc_id": result.question.answer_doc_id,
        "gold_doc_in_context": _gold_doc_in_context(result),
        "gold_doc_cited": _gold_doc_cited(result),
        "retrieved_doc_ids": [retrieval.document.id for retrieval in result.retrieval_results],
        "cited_doc_ids": [citation.document_id for citation in result.original_answer.citations],
        "max_evidence_score": _max_evidence_score(result),
        "verification_reasons": result.verification_result.reasons,
        "original_answer": _truncate(result.original_answer.answer),
        "gold_answer": _truncate(result.question.answer),
    }


def _gold_doc_in_context(result: VerifiedRAGQuestionResult) -> bool:
    if not result.question.answer_doc_id:
        return False
    return result.question.answer_doc_id in {
        retrieval.document.id for retrieval in result.retrieval_results
    }


def _gold_doc_cited(result: VerifiedRAGQuestionResult) -> bool:
    if not result.question.answer_doc_id:
        return False
    return result.question.answer_doc_id in {
        citation.document_id for citation in result.original_answer.citations
    }


def _max_evidence_score(result: VerifiedRAGQuestionResult) -> float | None:
    if not result.original_answer.citations:
        return None
    return max(citation.evidence_score for citation in result.original_answer.citations)


def _count_near_threshold(
    question_results: list[VerifiedRAGQuestionResult],
    min_evidence_score: float,
) -> int:
    threshold_floor = max(0.0, min_evidence_score - 2.0)
    return sum(
        1
        for result in question_results
        if (score := _max_evidence_score(result)) is not None
        and threshold_floor <= score < min_evidence_score
    )


def _score_stats(question_results: list[VerifiedRAGQuestionResult]) -> dict[str, float | None]:
    scores = [
        score
        for result in question_results
        if (score := _max_evidence_score(result)) is not None
    ]
    if not scores:
        return {"min": None, "median": None, "mean": None, "max": None}

    return {
        "min": round(min(scores), 4),
        "median": round(median(scores), 4),
        "mean": round(mean(scores), 4),
        "max": round(max(scores), 4),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _truncate(text: str, max_chars: int = 700) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."
