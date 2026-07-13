from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.evidence_selection import classify_question_route
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.application.text_metrics import token_f1
from ts_rag_agent.domain.dataset import PrimeQAQuestion


@dataclass(frozen=True)
class ChangedAnswerRiskVisualization:
    """One generated changed-answer risk visualization."""

    name: str
    path: str


def analyze_verified_rag_changed_answer_risk(
    baseline_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    questions: Sequence[PrimeQAQuestion],
) -> dict[str, Any]:
    """Analyze changed verified answers and unanswerable risk in two RAG reports."""

    baseline_samples = _samples_by_question_id(baseline_report)
    candidate_samples = _samples_by_question_id(candidate_report)
    question_by_id = {question.id: question for question in questions}
    shared_question_ids = sorted(set(baseline_samples) & set(candidate_samples))
    _validate_questions_available(shared_question_ids, question_by_id)
    max_citation_rank = int(candidate_report["rag"]["max_citation_rank"])

    changed_cases = [
        _build_changed_case(
            question=question_by_id[question_id],
            baseline_sample=baseline_samples[question_id],
            candidate_sample=candidate_samples[question_id],
            max_citation_rank=max_citation_rank,
        )
        for question_id in shared_question_ids
        if _verified_answer_changed(
            baseline_samples[question_id],
            candidate_samples[question_id],
        )
    ]

    unanswerable_regressions = [
        case
        for case in changed_cases
        if case["outcome"] == "unanswerable_refusal_regression"
    ]
    answerable_cases = [case for case in changed_cases if case["answerable"]]
    candidate_out_of_rank_cases = [
        case for case in changed_cases if case["candidate_citations"]["has_out_of_rank"]
    ]

    return {
        "baseline_report": _report_identity(baseline_report),
        "candidate_report": _report_identity(candidate_report),
        "sample_completeness": {
            "shared_questions": len(shared_question_ids),
            "baseline_samples": len(baseline_samples),
            "candidate_samples": len(candidate_samples),
            "baseline_total_questions": int(
                baseline_report["metrics"]["verified"]["total_questions"]
            ),
            "candidate_total_questions": int(
                candidate_report["metrics"]["verified"]["total_questions"]
            ),
        },
        "summary": {
            "changed_verified_answers": len(changed_cases),
            "changed_answerable": sum(case["answerable"] for case in changed_cases),
            "changed_unanswerable": sum(not case["answerable"] for case in changed_cases),
            "unanswerable_refusal_regressions": len(unanswerable_regressions),
            "answerable_improved": sum(
                case["outcome"] == "answerable_f1_improved" for case in changed_cases
            ),
            "answerable_regressed": sum(
                case["outcome"] == "answerable_f1_regressed" for case in changed_cases
            ),
            "answerable_tied_changed": sum(
                case["outcome"] == "answerable_f1_tied_changed"
                for case in changed_cases
            ),
            "candidate_has_out_of_rank_citation": len(candidate_out_of_rank_cases),
            "unanswerable_regression_has_out_of_rank_citation": sum(
                case["candidate_citations"]["has_out_of_rank"]
                for case in unanswerable_regressions
            ),
        },
        "route_distribution": {
            "all_changed": _counter_dict(case["question_route"] for case in changed_cases),
            "answerable_changed": _counter_dict(
                case["question_route"] for case in answerable_cases
            ),
            "unanswerable_changed": _counter_dict(
                case["question_route"] for case in changed_cases if not case["answerable"]
            ),
            "unanswerable_refusal_regressions": _counter_dict(
                case["question_route"] for case in unanswerable_regressions
            ),
        },
        "outcome_distribution": _counter_dict(case["outcome"] for case in changed_cases),
        "candidate_rank_distribution": {
            "best_rank": _counter_dict(
                _rank_bucket(case["candidate_citations"]["best_rank"])
                for case in changed_cases
            ),
            "worst_rank": _counter_dict(
                _rank_bucket(case["candidate_citations"]["worst_rank"])
                for case in changed_cases
            ),
            "out_of_rank_by_outcome": _out_of_rank_by_outcome(changed_cases),
        },
        "candidate_score_distribution": {
            "max_evidence_score": _counter_dict(
                _score_bucket(case["candidate_citations"]["max_evidence_score"])
                for case in changed_cases
            ),
            "min_evidence_score": _counter_dict(
                _score_bucket(case["candidate_citations"]["min_evidence_score"])
                for case in changed_cases
            ),
        },
        "risk_observations": _risk_observations(
            changed_cases=changed_cases,
            unanswerable_regressions=unanswerable_regressions,
            max_citation_rank=max_citation_rank,
        ),
        "changed_cases": changed_cases,
        "unanswerable_refusal_regression_cases": unanswerable_regressions,
    }


def write_changed_answer_risk_visualizations(
    analysis: Mapping[str, Any],
    output_dir: Path,
) -> list[ChangedAnswerRiskVisualization]:
    """Write compact SVG charts for changed-answer risk analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage47_changed_by_route.svg": render_horizontal_bar_chart_svg(
            title="Stage 47 changed verified answers by route",
            bars=_counter_bars(analysis["route_distribution"]["all_changed"]),
            x_label="changed verified answers",
            margin_left=330,
        ),
        "stage47_changed_by_outcome.svg": render_horizontal_bar_chart_svg(
            title="Stage 47 changed verified answers by outcome",
            bars=_counter_bars(analysis["outcome_distribution"]),
            x_label="changed verified answers",
            margin_left=330,
        ),
        "stage47_out_of_rank_by_outcome.svg": render_horizontal_bar_chart_svg(
            title="Stage 47 candidate out-of-rank citations by outcome",
            bars=[
                BarDatum(
                    label=outcome,
                    value=float(row["with_out_of_rank"]),
                    value_label=f'{row["with_out_of_rank"]}/{row["total"]}',
                )
                for outcome, row in analysis["candidate_rank_distribution"][
                    "out_of_rank_by_outcome"
                ].items()
            ],
            x_label="cases with at least one citation beyond verifier max rank",
            margin_left=330,
        ),
        "stage47_candidate_worst_rank.svg": render_horizontal_bar_chart_svg(
            title="Stage 47 candidate worst citation rank",
            bars=_counter_bars(analysis["candidate_rank_distribution"]["worst_rank"]),
            x_label="changed verified answers",
            margin_left=330,
        ),
    }

    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(ChangedAnswerRiskVisualization(name=filename, path=str(path)))
    return artifacts


def _samples_by_question_id(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    samples = report.get("samples", [])
    return {str(sample["question_id"]): sample for sample in samples}


def _validate_questions_available(
    question_ids: Sequence[str],
    question_by_id: Mapping[str, PrimeQAQuestion],
) -> None:
    missing_ids = sorted(set(question_ids) - set(question_by_id))
    if missing_ids:
        raise ValueError(f"Missing question metadata for: {', '.join(missing_ids[:10])}")


def _verified_answer_changed(
    baseline_sample: Mapping[str, Any],
    candidate_sample: Mapping[str, Any],
) -> bool:
    return (
        baseline_sample["verified_answer"]["answer"]
        != candidate_sample["verified_answer"]["answer"]
    )


def _build_changed_case(
    question: PrimeQAQuestion,
    baseline_sample: Mapping[str, Any],
    candidate_sample: Mapping[str, Any],
    max_citation_rank: int,
) -> dict[str, Any]:
    baseline_answer = baseline_sample["verified_answer"]
    candidate_answer = candidate_sample["verified_answer"]
    baseline_citations = _citation_summary(baseline_answer, max_citation_rank)
    candidate_citations = _citation_summary(candidate_answer, max_citation_rank)
    baseline_f1 = _answer_f1(baseline_sample, baseline_answer)
    candidate_f1 = _answer_f1(candidate_sample, candidate_answer)
    f1_delta = (
        round(candidate_f1 - baseline_f1, 6)
        if baseline_f1 is not None and candidate_f1 is not None
        else None
    )

    return {
        "question_id": question.id,
        "question_title": question.title,
        "answerable": question.answerable,
        "question_route": classify_question_route(question),
        "outcome": _case_outcome(
            answerable=question.answerable,
            baseline_refused=bool(baseline_answer["refused"]),
            candidate_refused=bool(candidate_answer["refused"]),
            f1_delta=f1_delta,
        ),
        "baseline_refused": bool(baseline_answer["refused"]),
        "candidate_refused": bool(candidate_answer["refused"]),
        "baseline_reasons": list(baseline_sample["verification"]["reasons"]),
        "candidate_reasons": list(candidate_sample["verification"]["reasons"]),
        "baseline_f1": round(baseline_f1, 4) if baseline_f1 is not None else None,
        "candidate_f1": round(candidate_f1, 4) if candidate_f1 is not None else None,
        "f1_delta": f1_delta,
        "baseline_cites_gold": _cites_gold_document(baseline_sample, "verified_answer"),
        "candidate_cites_gold": _cites_gold_document(candidate_sample, "verified_answer"),
        "baseline_citations": baseline_citations,
        "candidate_citations": candidate_citations,
        "gold_answer_doc_id": question.answer_doc_id if question.answerable else None,
        "baseline_answer": baseline_answer["answer"],
        "candidate_answer": candidate_answer["answer"],
    }


def _citation_summary(answer: Mapping[str, Any], max_citation_rank: int) -> dict[str, Any]:
    citations = list(answer["citations"])
    ranks = [int(citation["retrieval_rank"]) for citation in citations]
    scores = [float(citation["evidence_score"]) for citation in citations]
    return {
        "citation_count": len(citations),
        "document_ids": [str(citation["document_id"]) for citation in citations],
        "retrieval_ranks": ranks,
        "evidence_scores": [round(score, 4) for score in scores],
        "best_rank": min(ranks) if ranks else None,
        "worst_rank": max(ranks) if ranks else None,
        "min_evidence_score": round(min(scores), 4) if scores else None,
        "max_evidence_score": round(max(scores), 4) if scores else None,
        "has_out_of_rank": any(rank > max_citation_rank for rank in ranks),
        "out_of_rank_count": sum(rank > max_citation_rank for rank in ranks),
    }


def _answer_f1(sample: Mapping[str, Any], answer: Mapping[str, Any]) -> float | None:
    if not bool(sample["answerable"]) or bool(answer["refused"]):
        return None
    return token_f1(str(answer["answer"]), str(sample["gold_answer"]))


def _case_outcome(
    answerable: bool,
    baseline_refused: bool,
    candidate_refused: bool,
    f1_delta: float | None,
) -> str:
    if not answerable:
        if baseline_refused and not candidate_refused:
            return "unanswerable_refusal_regression"
        if not baseline_refused and candidate_refused:
            return "unanswerable_refusal_recovery"
        if not baseline_refused and not candidate_refused:
            return "unanswerable_answer_changed"
        return "unanswerable_refused_answer_changed"

    if baseline_refused and not candidate_refused:
        return "answerable_refusal_recovery"
    if not baseline_refused and candidate_refused:
        return "answerable_refusal_regression"
    if f1_delta is None:
        return "answerable_not_comparable"
    if f1_delta > 0:
        return "answerable_f1_improved"
    if f1_delta < 0:
        return "answerable_f1_regressed"
    return "answerable_f1_tied_changed"


def _cites_gold_document(sample: Mapping[str, Any], answer_key: str) -> bool:
    if not bool(sample["answerable"]) or bool(sample[answer_key]["refused"]):
        return False
    gold_answer_doc_id = str(sample["gold_answer_doc_id"])
    citation_doc_ids = {
        str(citation["document_id"]) for citation in sample[answer_key]["citations"]
    }
    return gold_answer_doc_id in citation_doc_ids


def _risk_observations(
    changed_cases: Sequence[Mapping[str, Any]],
    unanswerable_regressions: Sequence[Mapping[str, Any]],
    max_citation_rank: int,
) -> dict[str, Any]:
    out_of_rank_cases = [
        case for case in changed_cases if case["candidate_citations"]["has_out_of_rank"]
    ]
    changed_count = len(changed_cases)
    out_of_rank_count = len(out_of_rank_cases)
    unanswerable_regression_count = len(unanswerable_regressions)
    unanswerable_regression_out_of_rank = sum(
        case["candidate_citations"]["has_out_of_rank"]
        for case in unanswerable_regressions
    )
    candidate_all_citations_within_rank_count = changed_count - out_of_rank_count
    blocked_if_all_citations_rank_lte_max = out_of_rank_count
    return {
        "verifier_max_citation_rank": max_citation_rank,
        "changed_cases_with_candidate_out_of_rank_citation": out_of_rank_count,
        "changed_cases_with_all_candidate_citations_within_rank": (
            candidate_all_citations_within_rank_count
        ),
        "unanswerable_regressions_with_candidate_out_of_rank_citation": (
            unanswerable_regression_out_of_rank
        ),
        "unanswerable_regressions_total": unanswerable_regression_count,
        "would_block_changed_cases_if_all_citations_rank_lte_max": (
            blocked_if_all_citations_rank_lte_max
        ),
        "would_block_unanswerable_regressions_if_all_citations_rank_lte_max": (
            unanswerable_regression_out_of_rank
        ),
        "note": (
            "This is an observational risk analysis over saved report outputs, not a "
            "validated runtime gate."
        ),
    }


def _out_of_rank_by_outcome(
    changed_cases: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    totals: Counter[str] = Counter()
    out_of_rank: Counter[str] = Counter()
    for case in changed_cases:
        outcome = str(case["outcome"])
        totals[outcome] += 1
        if case["candidate_citations"]["has_out_of_rank"]:
            out_of_rank[outcome] += 1
    return {
        outcome: {
            "total": totals[outcome],
            "with_out_of_rank": out_of_rank[outcome],
        }
        for outcome in sorted(totals)
    }


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "no_citation"
    if rank <= 3:
        return f"rank_{rank}"
    if rank <= 5:
        return "rank_4_5"
    return "rank_6_plus"


def _score_bucket(score: float | None) -> str:
    if score is None:
        return "no_citation"
    if score < 15:
        return "score_lt_15"
    if score < 60:
        return "score_15_59"
    if score < 100:
        return "score_60_99"
    return "score_gte_100"


def _counter_dict(values: Sequence[str] | Any) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _counter_bars(counter: Mapping[str, int]) -> list[BarDatum]:
    return [
        BarDatum(label=label, value=float(value), value_label=str(value))
        for label, value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _report_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    rag = report["rag"]
    return {
        "split": report["split"],
        "evidence_selector": rag["evidence_selector"],
        "composition_policy": rag["composition_policy"],
        "candidate_reranker": rag.get("candidate_reranker"),
        "retrieval_top_k": rag["retrieval_top_k"],
        "max_sentences": rag["max_sentences"],
        "min_sentence_score": rag["min_sentence_score"],
        "max_candidates_per_document": rag["max_candidates_per_document"],
        "min_evidence_score": rag["min_evidence_score"],
        "max_citation_rank": rag["max_citation_rank"],
        "min_citations": rag["min_citations"],
    }
