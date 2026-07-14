from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.application.text_metrics import token_f1


@dataclass(frozen=True)
class PreservationGuardScenario:
    """One baseline/candidate report pair for guard analysis."""

    label: str
    baseline_report: Mapping[str, Any]
    candidate_report: Mapping[str, Any]


@dataclass(frozen=True)
class PreservationGuardVisualization:
    """One generated preservation-guard visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class PreservationGuardSpec:
    """Runtime-only document-preservation guard candidate."""

    label: str
    description: str
    runtime_signal: str
    protected_document_ids: Callable[[Mapping[str, Any], int], set[str]]


def analyze_gold_citation_preservation_guards(
    scenarios: Sequence[PreservationGuardScenario],
) -> dict[str, Any]:
    """Evaluate runtime-only preservation guards over saved report outputs."""

    guard_specs = _default_guard_specs()
    return {
        "guard_specs": [
            {
                "label": spec.label,
                "description": spec.description,
                "runtime_signal": spec.runtime_signal,
            }
            for spec in guard_specs
        ],
        "scenarios": [
            _analyze_scenario(scenario, guard_specs) for scenario in scenarios
        ],
        "important_boundary": (
            "Guard predicates are runtime-only, but metrics and gold-citation "
            "outcomes use gold labels for offline evaluation."
        ),
    }


def write_preservation_guard_visualizations(
    analysis: Mapping[str, Any],
    output_dir: Path,
) -> list[PreservationGuardVisualization]:
    """Write compact SVG charts for preservation-guard analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {}
    for scenario in analysis["scenarios"]:
        label = scenario["label"]
        guard_results = scenario["guard_results"]
        charts[f"stage50_{label}_f1_delta.svg"] = render_horizontal_bar_chart_svg(
            title=f"Stage 50 {label} F1 delta by guard",
            bars=[
                BarDatum(
                    label=result["guard_label"],
                    value=float(result["metric_deltas_vs_baseline"]["average_token_f1"]),
                    value_label=_signed_float(
                        result["metric_deltas_vs_baseline"]["average_token_f1"]
                    ),
                )
                for result in guard_results
            ],
            x_label="verified F1 delta vs top-k baseline",
            margin_left=360,
        )
        charts[
            f"stage50_{label}_gold_citation_delta.svg"
        ] = render_horizontal_bar_chart_svg(
            title=f"Stage 50 {label} gold citation delta by guard",
            bars=[
                BarDatum(
                    label=result["guard_label"],
                    value=float(result["metric_deltas_vs_baseline"]["gold_cited_count"]),
                    value_label=_signed_int(
                        result["metric_deltas_vs_baseline"]["gold_cited_count"]
                    ),
                )
                for result in guard_results
            ],
            x_label="verified gold citation count delta vs top-k baseline",
            margin_left=360,
        )
        charts[f"stage50_{label}_blocked_changed_count.svg"] = (
            render_horizontal_bar_chart_svg(
                title=f"Stage 50 {label} blocked changed answers by guard",
                bars=[
                    BarDatum(
                        label=result["guard_label"],
                        value=float(result["blocked_changed_answer_count"]),
                        value_label=str(result["blocked_changed_answer_count"]),
                    )
                    for result in guard_results
                ],
                x_label="blocked changed original answers",
                margin_left=360,
            )
        )

    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(PreservationGuardVisualization(name=filename, path=str(path)))
    return artifacts


def _analyze_scenario(
    scenario: PreservationGuardScenario,
    guard_specs: Sequence[PreservationGuardSpec],
) -> dict[str, Any]:
    baseline_samples = _samples_by_question_id(scenario.baseline_report)
    candidate_samples = _samples_by_question_id(scenario.candidate_report)
    shared_question_ids = sorted(set(baseline_samples) & set(candidate_samples))
    baseline_metrics = _evaluate_samples(baseline_samples, shared_question_ids)
    candidate_metrics = _evaluate_samples(candidate_samples, shared_question_ids)
    max_citation_rank = int(scenario.candidate_report["rag"]["max_citation_rank"])
    return {
        "label": scenario.label,
        "baseline_report": _report_identity(scenario.baseline_report),
        "candidate_report": _report_identity(scenario.candidate_report),
        "sample_completeness": {
            "shared_questions": len(shared_question_ids),
            "baseline_samples": len(baseline_samples),
            "candidate_samples": len(candidate_samples),
            "baseline_total_questions": int(
                scenario.baseline_report["metrics"]["verified"]["total_questions"]
            ),
            "candidate_total_questions": int(
                scenario.candidate_report["metrics"]["verified"]["total_questions"]
            ),
            "complete": (
                len(shared_question_ids)
                == int(scenario.baseline_report["metrics"]["verified"]["total_questions"])
                == int(scenario.candidate_report["metrics"]["verified"]["total_questions"])
            ),
        },
        "baseline_exact_metrics": baseline_metrics,
        "candidate_exact_metrics": candidate_metrics,
        "candidate_metric_deltas_vs_baseline": _metric_deltas(
            baseline_metrics,
            candidate_metrics,
        ),
        "guard_results": [
            _evaluate_guard(
                guard_spec=guard_spec,
                baseline_samples=baseline_samples,
                candidate_samples=candidate_samples,
                question_ids=shared_question_ids,
                max_citation_rank=max_citation_rank,
                baseline_metrics=baseline_metrics,
            )
            for guard_spec in guard_specs
        ],
    }


def _evaluate_guard(
    guard_spec: PreservationGuardSpec,
    baseline_samples: Mapping[str, Mapping[str, Any]],
    candidate_samples: Mapping[str, Mapping[str, Any]],
    question_ids: Sequence[str],
    max_citation_rank: int,
    baseline_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    blocked_question_ids = [
        question_id
        for question_id in question_ids
        if _original_answer_changed(
            baseline_samples[question_id],
            candidate_samples[question_id],
        )
        and _guard_blocks_candidate(
            guard_spec=guard_spec,
            baseline_sample=baseline_samples[question_id],
            candidate_sample=candidate_samples[question_id],
            max_citation_rank=max_citation_rank,
        )
    ]
    blocked_id_set = set(blocked_question_ids)
    simulated_samples = {
        question_id: (
            baseline_samples[question_id]
            if question_id in blocked_id_set
            else candidate_samples[question_id]
        )
        for question_id in question_ids
    }
    simulated_metrics = _evaluate_samples(simulated_samples, question_ids)
    blocked_cases = [
        _blocked_case_detail(
            question_id=question_id,
            baseline_sample=baseline_samples[question_id],
            candidate_sample=candidate_samples[question_id],
            max_citation_rank=max_citation_rank,
            guard_spec=guard_spec,
        )
        for question_id in blocked_question_ids
    ]
    outcome = _changed_answer_outcomes(
        baseline_samples=baseline_samples,
        simulated_samples=simulated_samples,
        question_ids=question_ids,
    )
    return {
        "guard_label": guard_spec.label,
        "description": guard_spec.description,
        "runtime_signal": guard_spec.runtime_signal,
        "blocked_changed_answer_count": len(blocked_question_ids),
        "blocked_question_ids": blocked_question_ids,
        "blocked_cases": blocked_cases,
        "blocked_gold_citation_loss_count": sum(
            case["candidate_lost_gold_citation"] for case in blocked_cases
        ),
        "blocked_answerable_improvement_count": sum(
            case["candidate_answerable_f1_delta"] is not None
            and case["candidate_answerable_f1_delta"] > 0
            for case in blocked_cases
        ),
        "blocked_answerable_regression_count": sum(
            case["candidate_answerable_f1_delta"] is not None
            and case["candidate_answerable_f1_delta"] < 0
            for case in blocked_cases
        ),
        "simulated_exact_metrics": simulated_metrics,
        "metric_deltas_vs_baseline": _metric_deltas(
            baseline_metrics,
            simulated_metrics,
        ),
        "changed_answer_outcomes_vs_baseline": outcome,
    }


def _default_guard_specs() -> list[PreservationGuardSpec]:
    return [
        PreservationGuardSpec(
            label="candidate_as_is",
            description="Accept the candidate report unchanged.",
            runtime_signal="No additional preservation predicate.",
            protected_document_ids=lambda _sample, _max_rank: set(),
        ),
        PreservationGuardSpec(
            label="preserve_all_baseline_docs",
            description=(
                "Block the candidate rewrite when it drops any baseline answer "
                "citation document."
            ),
            runtime_signal="baseline_original_doc_ids subset candidate_original_doc_ids",
            protected_document_ids=lambda sample, _max_rank: _original_document_ids(sample),
        ),
        PreservationGuardSpec(
            label="preserve_baseline_out_of_rank_docs",
            description=(
                "Block the candidate rewrite when it drops a baseline citation "
                "document whose retrieval rank is beyond the verifier max rank."
            ),
            runtime_signal=(
                "baseline_original_doc_ids_where_rank_gt_max_citation_rank subset "
                "candidate_original_doc_ids"
            ),
            protected_document_ids=_baseline_out_of_rank_document_ids,
        ),
        PreservationGuardSpec(
            label="preserve_baseline_out_of_rank_score_gte_60",
            description=(
                "Block the candidate rewrite when it drops a baseline citation "
                "document with rank beyond the verifier max rank and evidence score >= 60."
            ),
            runtime_signal=(
                "baseline_original_doc_ids_where_rank_gt_max_citation_rank_and_score_gte_60 "
                "subset candidate_original_doc_ids"
            ),
            protected_document_ids=_baseline_out_of_rank_score_gte_60_document_ids,
        ),
        PreservationGuardSpec(
            label="preserve_baseline_score_gte_80_docs",
            description=(
                "Block the candidate rewrite when it drops any baseline citation "
                "document with evidence score >= 80."
            ),
            runtime_signal=(
                "baseline_original_doc_ids_where_evidence_score_gte_80 subset "
                "candidate_original_doc_ids"
            ),
            protected_document_ids=_baseline_score_gte_80_document_ids,
        ),
    ]


def _guard_blocks_candidate(
    guard_spec: PreservationGuardSpec,
    baseline_sample: Mapping[str, Any],
    candidate_sample: Mapping[str, Any],
    max_citation_rank: int,
) -> bool:
    protected_docs = guard_spec.protected_document_ids(baseline_sample, max_citation_rank)
    return not protected_docs.issubset(_original_document_ids(candidate_sample))


def _blocked_case_detail(
    question_id: str,
    baseline_sample: Mapping[str, Any],
    candidate_sample: Mapping[str, Any],
    max_citation_rank: int,
    guard_spec: PreservationGuardSpec,
) -> dict[str, Any]:
    protected_docs = guard_spec.protected_document_ids(baseline_sample, max_citation_rank)
    candidate_docs = _original_document_ids(candidate_sample)
    baseline_f1 = _answer_f1(baseline_sample)
    candidate_f1 = _answer_f1(candidate_sample)
    return {
        "question_id": question_id,
        "question_title": baseline_sample["question_title"],
        "answerable": bool(baseline_sample["answerable"]),
        "gold_answer_doc_id": baseline_sample["gold_answer_doc_id"],
        "protected_document_ids": sorted(protected_docs),
        "dropped_protected_document_ids": sorted(protected_docs - candidate_docs),
        "baseline_original_citations": _citation_rows(baseline_sample["original_answer"]),
        "candidate_original_citations": _citation_rows(candidate_sample["original_answer"]),
        "baseline_cites_gold": _cites_gold_document(baseline_sample),
        "candidate_cites_gold": _cites_gold_document(candidate_sample),
        "candidate_lost_gold_citation": (
            _cites_gold_document(baseline_sample)
            and not _cites_gold_document(candidate_sample)
        ),
        "baseline_answerable_f1": round(baseline_f1, 4)
        if baseline_f1 is not None
        else None,
        "candidate_answerable_f1": round(candidate_f1, 4)
        if candidate_f1 is not None
        else None,
        "candidate_answerable_f1_delta": round(candidate_f1 - baseline_f1, 6)
        if baseline_f1 is not None and candidate_f1 is not None
        else None,
    }


def _evaluate_samples(
    samples_by_question_id: Mapping[str, Mapping[str, Any]],
    question_ids: Sequence[str],
) -> dict[str, Any]:
    answerable_count = 0
    unanswerable_count = 0
    generated_answerable_count = 0
    refused_answerable_count = 0
    refused_unanswerable_count = 0
    gold_cited_count = 0
    f1_values = []
    for question_id in question_ids:
        sample = samples_by_question_id[question_id]
        answer = sample["verified_answer"]
        if bool(sample["answerable"]):
            answerable_count += 1
            if bool(answer["refused"]):
                refused_answerable_count += 1
                continue
            generated_answerable_count += 1
            if _cites_gold_document(sample):
                gold_cited_count += 1
            f1_values.append(token_f1(answer["answer"], sample["gold_answer"]))
        else:
            unanswerable_count += 1
            if bool(answer["refused"]):
                refused_unanswerable_count += 1

    return {
        "total_questions": len(question_ids),
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "generated_answerable_questions": generated_answerable_count,
        "refused_answerable_questions": refused_answerable_count,
        "refused_unanswerable_questions": refused_unanswerable_count,
        "gold_cited_count": gold_cited_count,
        "gold_doc_citation_rate": round(
            gold_cited_count / generated_answerable_count,
            4,
        )
        if generated_answerable_count
        else 0.0,
        "answerable_refusal_rate": round(
            refused_answerable_count / answerable_count,
            4,
        )
        if answerable_count
        else 0.0,
        "unanswerable_refusal_rate": round(
            refused_unanswerable_count / unanswerable_count,
            4,
        )
        if unanswerable_count
        else 0.0,
        "average_token_f1": round(sum(f1_values) / len(f1_values), 4)
        if f1_values
        else 0.0,
    }


def _metric_deltas(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    deltas = {}
    for key, baseline_value in baseline_metrics.items():
        candidate_value = candidate_metrics[key]
        if isinstance(baseline_value, float):
            deltas[key] = round(float(candidate_value) - baseline_value, 4)
        else:
            deltas[key] = int(candidate_value) - int(baseline_value)
    return deltas


def _changed_answer_outcomes(
    baseline_samples: Mapping[str, Mapping[str, Any]],
    simulated_samples: Mapping[str, Mapping[str, Any]],
    question_ids: Sequence[str],
) -> dict[str, Any]:
    improved = []
    regressed = []
    tied = []
    changed_unanswerable = []
    for question_id in question_ids:
        baseline_sample = baseline_samples[question_id]
        simulated_sample = simulated_samples[question_id]
        if (
            baseline_sample["verified_answer"]["answer"]
            == simulated_sample["verified_answer"]["answer"]
        ):
            continue
        if not bool(baseline_sample["answerable"]):
            changed_unanswerable.append(question_id)
            continue
        baseline_f1 = _answer_f1(baseline_sample)
        simulated_f1 = _answer_f1(simulated_sample)
        if baseline_f1 is None or simulated_f1 is None:
            tied.append(question_id)
            continue
        delta = round(simulated_f1 - baseline_f1, 6)
        if delta > 0:
            improved.append(question_id)
        elif delta < 0:
            regressed.append(question_id)
        else:
            tied.append(question_id)
    return {
        "changed_verified_answers": (
            len(improved) + len(regressed) + len(tied) + len(changed_unanswerable)
        ),
        "answerable_improved": len(improved),
        "answerable_regressed": len(regressed),
        "answerable_tied_or_not_comparable": len(tied),
        "changed_unanswerable": len(changed_unanswerable),
        "answerable_improved_question_ids": improved,
        "answerable_regressed_question_ids": regressed,
        "changed_unanswerable_question_ids": changed_unanswerable,
    }


def _samples_by_question_id(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    samples = report.get("samples", [])
    return {str(sample["question_id"]): sample for sample in samples}


def _original_answer_changed(
    baseline_sample: Mapping[str, Any],
    candidate_sample: Mapping[str, Any],
) -> bool:
    return (
        baseline_sample["original_answer"]["answer"]
        != candidate_sample["original_answer"]["answer"]
    )


def _original_document_ids(sample: Mapping[str, Any]) -> set[str]:
    return {
        str(citation["document_id"])
        for citation in sample["original_answer"]["citations"]
    }


def _baseline_out_of_rank_document_ids(
    sample: Mapping[str, Any],
    max_citation_rank: int,
) -> set[str]:
    return {
        str(citation["document_id"])
        for citation in sample["original_answer"]["citations"]
        if int(citation["retrieval_rank"]) > max_citation_rank
    }


def _baseline_out_of_rank_score_gte_60_document_ids(
    sample: Mapping[str, Any],
    max_citation_rank: int,
) -> set[str]:
    return {
        str(citation["document_id"])
        for citation in sample["original_answer"]["citations"]
        if int(citation["retrieval_rank"]) > max_citation_rank
        and float(citation["evidence_score"]) >= 60
    }


def _baseline_score_gte_80_document_ids(
    sample: Mapping[str, Any],
    _max_citation_rank: int,
) -> set[str]:
    return {
        str(citation["document_id"])
        for citation in sample["original_answer"]["citations"]
        if float(citation["evidence_score"]) >= 80
    }


def _cites_gold_document(sample: Mapping[str, Any]) -> bool:
    if not bool(sample["answerable"]) or bool(sample["verified_answer"]["refused"]):
        return False
    gold_answer_doc_id = str(sample["gold_answer_doc_id"])
    return gold_answer_doc_id in {
        str(citation["document_id"])
        for citation in sample["verified_answer"]["citations"]
    }


def _answer_f1(sample: Mapping[str, Any]) -> float | None:
    if not bool(sample["answerable"]) or bool(sample["verified_answer"]["refused"]):
        return None
    return token_f1(sample["verified_answer"]["answer"], sample["gold_answer"])


def _citation_rows(answer: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "document_id": citation["document_id"],
            "retrieval_rank": citation["retrieval_rank"],
            "evidence_score": citation["evidence_score"],
        }
        for citation in answer["citations"]
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


def _signed_float(value: Any) -> str:
    return f"{float(value):+.4f}"


def _signed_int(value: Any) -> str:
    integer_value = int(value)
    if integer_value > 0:
        return f"+{integer_value}"
    return str(integer_value)
