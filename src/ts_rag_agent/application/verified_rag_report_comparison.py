from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from ts_rag_agent.application.text_metrics import token_f1


@dataclass(frozen=True)
class VerifiedRAGComparisonVisualization:
    """One generated verified-RAG comparison visualization."""

    name: str
    path: str


def compare_verified_rag_reports(
    baseline_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two verified RAG reports using metrics and full saved samples."""

    baseline_samples = _samples_by_question_id(baseline_report)
    candidate_samples = _samples_by_question_id(candidate_report)
    shared_question_ids = sorted(set(baseline_samples) & set(candidate_samples))
    sample_complete = (
        len(shared_question_ids) == _total_questions(baseline_report)
        and len(shared_question_ids) == _total_questions(candidate_report)
    )

    return {
        "baseline_report": _report_identity(baseline_report),
        "candidate_report": _report_identity(candidate_report),
        "sample_completeness": {
            "shared_questions": len(shared_question_ids),
            "baseline_samples": len(baseline_samples),
            "candidate_samples": len(candidate_samples),
            "baseline_total_questions": _total_questions(baseline_report),
            "candidate_total_questions": _total_questions(candidate_report),
            "complete": sample_complete,
        },
        "metric_deltas": _metric_deltas(baseline_report, candidate_report),
        "exact_gold_citations": _exact_gold_citation_summary(
            baseline_samples=baseline_samples,
            candidate_samples=candidate_samples,
            question_ids=shared_question_ids,
        ),
        "changed_answers": _changed_answer_summary(
            baseline_samples=baseline_samples,
            candidate_samples=candidate_samples,
            question_ids=shared_question_ids,
        ),
        "verified_answerable_f1_outcomes": _verified_answerable_f1_outcomes(
            baseline_samples=baseline_samples,
            candidate_samples=candidate_samples,
            question_ids=shared_question_ids,
        ),
    }


def write_verified_rag_comparison_visualizations(
    comparison: Mapping[str, Any],
    output_dir: Path,
) -> list[VerifiedRAGComparisonVisualization]:
    """Write compact SVG charts for a verified-RAG report comparison."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "verified_rag_metric_deltas.svg": _render_bar_chart_svg(
            title="Verified RAG metric deltas",
            bars=[
                _Bar(
                    label="verified F1",
                    value=float(comparison["metric_deltas"]["verified_average_token_f1"]),
                    value_label=_signed_float(
                        comparison["metric_deltas"]["verified_average_token_f1"]
                    ),
                ),
                _Bar(
                    label="gold citation rate",
                    value=float(
                        comparison["metric_deltas"]["verified_gold_doc_citation_rate"]
                    ),
                    value_label=_signed_float(
                        comparison["metric_deltas"]["verified_gold_doc_citation_rate"]
                    ),
                ),
                _Bar(
                    label="answerable refusals",
                    value=float(
                        comparison["metric_deltas"][
                            "verified_refused_answerable_questions"
                        ]
                    ),
                    value_label=_signed_int(
                        comparison["metric_deltas"][
                            "verified_refused_answerable_questions"
                        ]
                    ),
                ),
                _Bar(
                    label="unanswerable refusals",
                    value=float(
                        comparison["metric_deltas"][
                            "verified_refused_unanswerable_questions"
                        ]
                    ),
                    value_label=_signed_int(
                        comparison["metric_deltas"][
                            "verified_refused_unanswerable_questions"
                        ]
                    ),
                ),
            ],
            x_label="candidate minus baseline",
        ),
        "verified_rag_changed_answers.svg": _render_bar_chart_svg(
            title="Changed answer counts",
            bars=[
                _Bar(
                    label="original all",
                    value=float(comparison["changed_answers"]["original"]["all_count"]),
                    value_label=str(comparison["changed_answers"]["original"]["all_count"]),
                ),
                _Bar(
                    label="original answerable",
                    value=float(
                        comparison["changed_answers"]["original"]["answerable_count"]
                    ),
                    value_label=str(
                        comparison["changed_answers"]["original"]["answerable_count"]
                    ),
                ),
                _Bar(
                    label="verified all",
                    value=float(comparison["changed_answers"]["verified"]["all_count"]),
                    value_label=str(comparison["changed_answers"]["verified"]["all_count"]),
                ),
                _Bar(
                    label="verified answerable",
                    value=float(
                        comparison["changed_answers"]["verified"]["answerable_count"]
                    ),
                    value_label=str(
                        comparison["changed_answers"]["verified"]["answerable_count"]
                    ),
                ),
            ],
            x_label="questions",
        ),
        "verified_rag_answerable_f1_outcomes.svg": _render_bar_chart_svg(
            title="Changed verified-answer F1 outcomes",
            bars=[
                _Bar(
                    label="improved",
                    value=float(
                        comparison["verified_answerable_f1_outcomes"]["improved_count"]
                    ),
                    value_label=str(
                        comparison["verified_answerable_f1_outcomes"]["improved_count"]
                    ),
                ),
                _Bar(
                    label="regressed",
                    value=float(
                        comparison["verified_answerable_f1_outcomes"]["regressed_count"]
                    ),
                    value_label=str(
                        comparison["verified_answerable_f1_outcomes"]["regressed_count"]
                    ),
                ),
                _Bar(
                    label="changed but tied",
                    value=float(
                        comparison["verified_answerable_f1_outcomes"][
                            "changed_tied_count"
                        ]
                    ),
                    value_label=str(
                        comparison["verified_answerable_f1_outcomes"][
                            "changed_tied_count"
                        ]
                    ),
                ),
            ],
            x_label="answerable generated questions",
        ),
    }

    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            VerifiedRAGComparisonVisualization(name=filename, path=str(path))
        )
    return artifacts


def _samples_by_question_id(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    samples = report.get("samples", [])
    return {str(sample["question_id"]): sample for sample in samples}


def _total_questions(report: Mapping[str, Any]) -> int:
    return int(report["metrics"]["verified"]["total_questions"])


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


def _metric_deltas(
    baseline_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
) -> dict[str, float | int]:
    baseline_verified = baseline_report["metrics"]["verified"]
    candidate_verified = candidate_report["metrics"]["verified"]
    baseline_original = baseline_report["metrics"]["original"]
    candidate_original = candidate_report["metrics"]["original"]
    return {
        "original_average_token_f1": _float_delta(
            baseline_original["average_token_f1"],
            candidate_original["average_token_f1"],
        ),
        "verified_average_token_f1": _float_delta(
            baseline_verified["average_token_f1"],
            candidate_verified["average_token_f1"],
        ),
        "original_gold_doc_citation_rate": _float_delta(
            baseline_original["gold_doc_citation_rate"],
            candidate_original["gold_doc_citation_rate"],
        ),
        "verified_gold_doc_citation_rate": _float_delta(
            baseline_verified["gold_doc_citation_rate"],
            candidate_verified["gold_doc_citation_rate"],
        ),
        "verified_generated_answerable_questions": int(
            candidate_verified["generated_answerable_questions"]
        )
        - int(baseline_verified["generated_answerable_questions"]),
        "verified_refused_answerable_questions": int(
            candidate_verified["refused_answerable_questions"]
        )
        - int(baseline_verified["refused_answerable_questions"]),
        "verified_refused_unanswerable_questions": int(
            candidate_verified["refused_unanswerable_questions"]
        )
        - int(baseline_verified["refused_unanswerable_questions"]),
        "newly_refused": int(candidate_report["verification"]["newly_refused"])
        - int(baseline_report["verification"]["newly_refused"]),
    }


def _exact_gold_citation_summary(
    baseline_samples: Mapping[str, Mapping[str, Any]],
    candidate_samples: Mapping[str, Mapping[str, Any]],
    question_ids: Sequence[str],
) -> dict[str, Any]:
    baseline_original = _gold_citation_counts(
        baseline_samples,
        question_ids,
        answer_key="original_answer",
    )
    baseline_verified = _gold_citation_counts(
        baseline_samples,
        question_ids,
        answer_key="verified_answer",
    )
    candidate_original = _gold_citation_counts(
        candidate_samples,
        question_ids,
        answer_key="original_answer",
    )
    candidate_verified = _gold_citation_counts(
        candidate_samples,
        question_ids,
        answer_key="verified_answer",
    )
    baseline_verified_ids = _gold_cited_question_ids(
        baseline_samples,
        question_ids,
        answer_key="verified_answer",
    )
    candidate_verified_ids = _gold_cited_question_ids(
        candidate_samples,
        question_ids,
        answer_key="verified_answer",
    )
    return {
        "baseline": {
            "original": baseline_original,
            "verified": baseline_verified,
        },
        "candidate": {
            "original": candidate_original,
            "verified": candidate_verified,
        },
        "deltas": {
            "original_gold_cited_count": (
                candidate_original["gold_cited_count"]
                - baseline_original["gold_cited_count"]
            ),
            "verified_gold_cited_count": (
                candidate_verified["gold_cited_count"]
                - baseline_verified["gold_cited_count"]
            ),
            "verified_generated_answerable_count": (
                candidate_verified["generated_answerable_count"]
                - baseline_verified["generated_answerable_count"]
            ),
        },
        "verified_citation_gained_question_ids": sorted(
            candidate_verified_ids - baseline_verified_ids
        ),
        "verified_citation_lost_question_ids": sorted(
            baseline_verified_ids - candidate_verified_ids
        ),
    }


def _gold_citation_counts(
    samples: Mapping[str, Mapping[str, Any]],
    question_ids: Sequence[str],
    answer_key: str,
) -> dict[str, int]:
    answerable_count = 0
    generated_answerable_count = 0
    gold_cited_count = 0
    for question_id in question_ids:
        sample = samples[question_id]
        if not bool(sample["answerable"]):
            continue
        answerable_count += 1
        answer = sample[answer_key]
        if bool(answer["refused"]):
            continue
        generated_answerable_count += 1
        if _cites_gold_document(sample, answer_key):
            gold_cited_count += 1
    return {
        "answerable_count": answerable_count,
        "generated_answerable_count": generated_answerable_count,
        "gold_cited_count": gold_cited_count,
    }


def _gold_cited_question_ids(
    samples: Mapping[str, Mapping[str, Any]],
    question_ids: Sequence[str],
    answer_key: str,
) -> set[str]:
    return {
        question_id
        for question_id in question_ids
        if bool(samples[question_id]["answerable"])
        and not bool(samples[question_id][answer_key]["refused"])
        and _cites_gold_document(samples[question_id], answer_key)
    }


def _cites_gold_document(sample: Mapping[str, Any], answer_key: str) -> bool:
    gold_answer_doc_id = str(sample["gold_answer_doc_id"])
    cited_doc_ids = {
        str(citation["document_id"]) for citation in sample[answer_key]["citations"]
    }
    return gold_answer_doc_id in cited_doc_ids


def _changed_answer_summary(
    baseline_samples: Mapping[str, Mapping[str, Any]],
    candidate_samples: Mapping[str, Mapping[str, Any]],
    question_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "original": _changed_answers_for_key(
            baseline_samples,
            candidate_samples,
            question_ids,
            "original_answer",
        ),
        "verified": _changed_answers_for_key(
            baseline_samples,
            candidate_samples,
            question_ids,
            "verified_answer",
        ),
    }


def _changed_answers_for_key(
    baseline_samples: Mapping[str, Mapping[str, Any]],
    candidate_samples: Mapping[str, Mapping[str, Any]],
    question_ids: Sequence[str],
    answer_key: str,
) -> dict[str, Any]:
    all_ids = [
        question_id
        for question_id in question_ids
        if baseline_samples[question_id][answer_key]["answer"]
        != candidate_samples[question_id][answer_key]["answer"]
    ]
    answerable_ids = [
        question_id for question_id in all_ids if bool(baseline_samples[question_id]["answerable"])
    ]
    unanswerable_ids = [
        question_id
        for question_id in all_ids
        if not bool(baseline_samples[question_id]["answerable"])
    ]
    return {
        "all_count": len(all_ids),
        "answerable_count": len(answerable_ids),
        "unanswerable_count": len(unanswerable_ids),
        "all_question_ids": all_ids,
        "answerable_question_ids": answerable_ids,
        "unanswerable_question_ids": unanswerable_ids,
    }


def _verified_answerable_f1_outcomes(
    baseline_samples: Mapping[str, Mapping[str, Any]],
    candidate_samples: Mapping[str, Mapping[str, Any]],
    question_ids: Sequence[str],
) -> dict[str, Any]:
    improved = []
    regressed = []
    changed_tied = []
    delta_sum = 0.0
    comparable_count = 0
    for question_id in question_ids:
        baseline_sample = baseline_samples[question_id]
        candidate_sample = candidate_samples[question_id]
        if not bool(baseline_sample["answerable"]):
            continue
        baseline_answer = baseline_sample["verified_answer"]
        candidate_answer = candidate_sample["verified_answer"]
        if bool(baseline_answer["refused"]) or bool(candidate_answer["refused"]):
            continue
        comparable_count += 1
        baseline_f1 = token_f1(baseline_answer["answer"], baseline_sample["gold_answer"])
        candidate_f1 = token_f1(
            candidate_answer["answer"],
            candidate_sample["gold_answer"],
        )
        delta = round(candidate_f1 - baseline_f1, 6)
        delta_sum += delta
        if baseline_answer["answer"] == candidate_answer["answer"] and delta == 0:
            continue
        case = {
            "question_id": question_id,
            "question_title": baseline_sample["question_title"],
            "baseline_f1": round(baseline_f1, 4),
            "candidate_f1": round(candidate_f1, 4),
            "delta": delta,
        }
        if delta > 0:
            improved.append(case)
        elif delta < 0:
            regressed.append(case)
        else:
            changed_tied.append(case)

    return {
        "comparable_generated_answerable_count": comparable_count,
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "changed_tied_count": len(changed_tied),
        "delta_sum": round(delta_sum, 6),
        "average_delta_over_comparable": round(delta_sum / comparable_count, 6)
        if comparable_count
        else 0.0,
        "improved_cases": sorted(improved, key=lambda case: case["delta"], reverse=True),
        "regressed_cases": sorted(regressed, key=lambda case: case["delta"]),
        "changed_tied_cases": changed_tied,
    }


def _float_delta(baseline_value: Any, candidate_value: Any) -> float:
    return round(float(candidate_value) - float(baseline_value), 4)


def _signed_float(value: Any) -> str:
    return f"{float(value):+.4f}"


def _signed_int(value: Any) -> str:
    integer_value = int(value)
    if integer_value > 0:
        return f"+{integer_value}"
    return str(integer_value)


@dataclass(frozen=True)
class _Bar:
    label: str
    value: float
    value_label: str


def _render_bar_chart_svg(
    title: str,
    bars: Sequence[_Bar],
    x_label: str,
) -> str:
    width = 920
    margin_left = 230
    margin_right = 160
    margin_top = 56
    margin_bottom = 56
    row_height = 38
    height = margin_top + margin_bottom + max(1, len(bars)) * row_height
    plot_width = width - margin_left - margin_right
    max_abs_value = max((abs(bar.value) for bar in bars), default=0.0)
    scale_denominator = max_abs_value if max_abs_value > 0 else 1.0
    has_negative = any(bar.value < 0 for bar in bars)
    zero_x = margin_left + (plot_width // 2 if has_negative else 0)
    positive_plot_width = plot_width // 2 if has_negative else plot_width
    chart_id = _svg_id(title)

    bar_lines = []
    for index, bar in enumerate(bars):
        y = margin_top + index * row_height
        bar_width = int(round(positive_plot_width * abs(bar.value) / scale_denominator))
        if bar.value >= 0:
            x = zero_x
            value_x = x + bar_width + 8
            anchor = "start"
            fill = "#2563eb"
        else:
            x = zero_x - bar_width
            value_x = x - 8
            anchor = "end"
            fill = "#dc2626"
        label_y = y + 21
        bar_lines.append(
            "\n".join(
                [
                    f'<text x="{margin_left - 12}" y="{label_y}" text-anchor="end">'
                    f"{escape(bar.label)}</text>",
                    (
                        f'<rect x="{x}" y="{y + 6}" width="{bar_width}" '
                        f'height="22" rx="3" fill="{fill}" />'
                    ),
                    f'<text x="{value_x}" y="{label_y}" text-anchor="{anchor}">'
                    f"{escape(bar.value_label)}</text>",
                ]
            )
        )

    axis_y = height - margin_bottom + 12
    return "\n".join(
        [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
                f'aria-labelledby="{chart_id}-title {chart_id}-desc">'
            ),
            f'<title id="{chart_id}-title">{escape(title)}</title>',
            (
                f'<desc id="{chart_id}-desc">Horizontal bar chart showing '
                f'{escape(x_label)} for {len(bars)} categories.</desc>'
            ),
            '<rect width="100%" height="100%" fill="#ffffff" />',
            (
                '<style>text{font-family:Arial, sans-serif;font-size:13px;'
                'fill:#111827}.title{font-size:18px;font-weight:700}'
                '.axis{fill:#4b5563;font-size:12px}.grid{stroke:#e5e7eb}'
                '</style>'
            ),
            f'<text x="24" y="32" class="title">{escape(title)}</text>',
            (
                f'<line x1="{zero_x}" x2="{zero_x}" y1="{margin_top - 8}" '
                f'y2="{height - margin_bottom}" class="grid" />'
            ),
            *bar_lines,
            f'<text x="{margin_left}" y="{axis_y}" class="axis">{escape(x_label)}</text>',
            "</svg>",
        ]
    )


def _svg_id(title: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in title).strip("-")
