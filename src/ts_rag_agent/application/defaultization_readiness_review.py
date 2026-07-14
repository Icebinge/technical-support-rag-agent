from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg


@dataclass(frozen=True)
class SplitReviewSources:
    """All reports needed to review one split before defaultization."""

    split: str
    topk_report: Mapping[str, Any]
    rank_contained_report: Mapping[str, Any]
    candidate_report: Mapping[str, Any]
    candidate_risk_report: Mapping[str, Any]


@dataclass(frozen=True)
class ReviewVisualization:
    """One generated Stage 52 visualization."""

    name: str
    path: str


def review_defaultization_readiness(
    sources: Sequence[SplitReviewSources],
) -> dict[str, Any]:
    """Aggregate dev/train evidence and freeze the held-out test protocol."""

    split_reviews = [_review_split(source) for source in sources]
    candidate_passes = all(
        check["passed"]
        for split_review in split_reviews
        for check in split_review["candidate_readiness_checks"]
    )
    rank_contained_passes = all(
        check["passed"]
        for split_review in split_reviews
        for check in split_review["rank_contained_safety_checks"]
    )
    candidate_policy = _single_candidate_policy(split_reviews)
    return {
        "review_scope": (
            "Defaultization pre-review only. This review uses existing PrimeQA "
            "dev/train reports and changed-answer risk reports. It does not use "
            "the held-out NVIDIA TechQA-RAG-Eval data and does not change the "
            "runtime default policy."
        ),
        "candidate_policy": candidate_policy,
        "split_reviews": split_reviews,
        "overall_decision": {
            "candidate_passes_dev_train_readiness": candidate_passes,
            "rank_contained_passes_dev_train_safety": rank_contained_passes,
            "unique_heldout_candidate": candidate_policy
            if candidate_passes and not rank_contained_passes
            else None,
            "default_runtime_change": "not_allowed_before_heldout_evaluation",
            "status": _overall_status(candidate_passes, rank_contained_passes),
        },
        "heldout_test_protocol": _heldout_test_protocol(candidate_policy),
    }


def write_defaultization_review_visualizations(
    review: Mapping[str, Any],
    output_dir: Path,
) -> list[ReviewVisualization]:
    """Write compact SVG charts for the Stage 52 readiness review."""

    output_dir.mkdir(parents=True, exist_ok=True)
    split_reviews = review["split_reviews"]
    charts = {
        "stage52_verified_f1_by_policy.svg": render_horizontal_bar_chart_svg(
            title="Stage 52 verified F1 by split and policy",
            bars=[
                BarDatum(
                    label=f"{split_review['split']} {policy_label}",
                    value=float(summary["verified_average_token_f1"]),
                    value_label=f"{summary['verified_average_token_f1']:.4f}",
                )
                for split_review in split_reviews
                for policy_label, summary in [
                    ("top-k", split_review["topk_summary"]),
                    ("rank-contained", split_review["rank_contained_summary"]),
                    ("stage51", split_review["candidate_summary"]),
                ]
            ],
            x_label="verified average token F1",
            margin_left=320,
        ),
        "stage52_gold_citation_delta_vs_topk.svg": render_horizontal_bar_chart_svg(
            title="Stage 52 gold citation delta vs top-k",
            bars=[
                BarDatum(
                    label=f"{split_review['split']} {label}",
                    value=float(comparison["verified_gold_citation_delta"]),
                    value_label=_signed_int(
                        comparison["verified_gold_citation_delta"]
                    ),
                )
                for split_review in split_reviews
                for label, comparison in [
                    ("rank-contained", split_review["rank_contained_vs_topk"]),
                    ("stage51", split_review["candidate_vs_topk"]),
                ]
            ],
            x_label="verified gold citation count delta",
            margin_left=320,
        ),
        "stage52_changed_answer_risk.svg": render_horizontal_bar_chart_svg(
            title="Stage 52 Stage51 changed-answer risk",
            bars=[
                BarDatum(
                    label=f"{split_review['split']} changed answers",
                    value=float(split_review["candidate_risk_summary"]["changed_verified_answers"]),
                    value_label=str(
                        split_review["candidate_risk_summary"][
                            "changed_verified_answers"
                        ]
                    ),
                )
                for split_review in split_reviews
            ]
            + [
                BarDatum(
                    label=f"{split_review['split']} unanswerable refusal regressions",
                    value=float(
                        split_review["candidate_risk_summary"][
                            "unanswerable_refusal_regressions"
                        ]
                    ),
                    value_label=str(
                        split_review["candidate_risk_summary"][
                            "unanswerable_refusal_regressions"
                        ]
                    ),
                )
                for split_review in split_reviews
            ],
            x_label="case count",
            margin_left=360,
        ),
        "stage52_readiness_pass_count.svg": render_horizontal_bar_chart_svg(
            title="Stage 52 readiness checks passed",
            bars=[
                _readiness_pass_count_bar(split_review)
                for split_review in split_reviews
            ],
            x_label="passed checks",
            margin_left=180,
        ),
    }

    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(ReviewVisualization(name=filename, path=str(path)))
    return artifacts


def _review_split(source: SplitReviewSources) -> dict[str, Any]:
    topk_summary = _report_summary(source.topk_report)
    rank_contained_summary = _report_summary(source.rank_contained_report)
    candidate_summary = _report_summary(source.candidate_report)
    risk_summary = dict(source.candidate_risk_report["summary"])
    candidate_vs_topk = _compare_summaries(candidate_summary, topk_summary)
    rank_contained_vs_topk = _compare_summaries(
        rank_contained_summary,
        topk_summary,
    )
    candidate_vs_rank_contained = _compare_summaries(
        candidate_summary,
        rank_contained_summary,
    )
    return {
        "split": source.split,
        "topk_summary": topk_summary,
        "rank_contained_summary": rank_contained_summary,
        "candidate_summary": candidate_summary,
        "candidate_vs_topk": candidate_vs_topk,
        "candidate_vs_rank_contained": candidate_vs_rank_contained,
        "rank_contained_vs_topk": rank_contained_vs_topk,
        "candidate_risk_summary": risk_summary,
        "candidate_readiness_checks": _candidate_readiness_checks(
            candidate_vs_topk,
            risk_summary,
            candidate_summary,
        ),
        "rank_contained_safety_checks": _rank_contained_safety_checks(
            rank_contained_vs_topk,
            rank_contained_summary,
        ),
    }


def _readiness_pass_count_bar(split_review: Mapping[str, Any]) -> BarDatum:
    checks = split_review["candidate_readiness_checks"]
    passed_count = sum(check["passed"] for check in checks)
    return BarDatum(
        label=split_review["split"],
        value=float(passed_count),
        value_label=f"{passed_count}/{len(checks)}",
    )


def _report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    verified = report["metrics"]["verified"]
    sample_count = len(report.get("samples", []))
    total_questions = int(verified["total_questions"])
    samples_complete = sample_count == total_questions
    exact_gold_cited_count = (
        _exact_gold_cited_count(report) if samples_complete else None
    )
    return {
        "split": report["split"],
        "composition_policy": report["rag"]["composition_policy"],
        "candidate_reranker": report["rag"].get("candidate_reranker"),
        "verified_average_token_f1": float(verified["average_token_f1"]),
        "verified_generated_answerable_questions": int(
            verified["generated_answerable_questions"]
        ),
        "verified_gold_cited_count": exact_gold_cited_count,
        "verified_gold_doc_citation_rate": float(
            verified["gold_doc_citation_rate"]
        ),
        "verified_refused_answerable_questions": int(
            verified["refused_answerable_questions"]
        ),
        "verified_refused_unanswerable_questions": int(
            verified["refused_unanswerable_questions"]
        ),
        "newly_refused": int(report["verification"]["newly_refused"]),
        "samples_complete": samples_complete,
        "sample_count": sample_count,
        "total_questions": total_questions,
    }


def _compare_summaries(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "verified_average_token_f1_delta": round(
            candidate["verified_average_token_f1"]
            - baseline["verified_average_token_f1"],
            4,
        ),
        "verified_gold_citation_delta": _nullable_int_delta(
            candidate["verified_gold_cited_count"],
            baseline["verified_gold_cited_count"],
        ),
        "verified_generated_answerable_delta": (
            candidate["verified_generated_answerable_questions"]
            - baseline["verified_generated_answerable_questions"]
        ),
        "verified_refused_answerable_delta": (
            candidate["verified_refused_answerable_questions"]
            - baseline["verified_refused_answerable_questions"]
        ),
        "verified_refused_unanswerable_delta": (
            candidate["verified_refused_unanswerable_questions"]
            - baseline["verified_refused_unanswerable_questions"]
        ),
        "newly_refused_delta": candidate["newly_refused"]
        - baseline["newly_refused"],
    }


def _candidate_readiness_checks(
    candidate_vs_topk: Mapping[str, Any],
    risk_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _check(
            "candidate_samples_complete",
            bool(candidate_summary["samples_complete"]),
            observed=candidate_summary["sample_count"],
            threshold=candidate_summary["total_questions"],
            rule="sample_count must equal total_questions",
        ),
        _check(
            "verified_f1_delta_vs_topk_non_negative",
            candidate_vs_topk["verified_average_token_f1_delta"] >= 0,
            observed=candidate_vs_topk["verified_average_token_f1_delta"],
            threshold=0,
            rule="candidate F1 delta vs top-k must be >= 0",
        ),
        _check(
            "gold_citation_delta_vs_topk_non_negative",
            candidate_vs_topk["verified_gold_citation_delta"] is not None
            and candidate_vs_topk["verified_gold_citation_delta"] >= 0,
            observed=candidate_vs_topk["verified_gold_citation_delta"],
            threshold=0,
            rule="candidate gold citation delta vs top-k must be >= 0",
        ),
        _check(
            "newly_refused_delta_vs_topk_non_positive",
            candidate_vs_topk["newly_refused_delta"] <= 0,
            observed=candidate_vs_topk["newly_refused_delta"],
            threshold=0,
            rule="candidate must not add newly refused answers vs top-k",
        ),
        _check(
            "answerable_refusal_delta_vs_topk_non_positive",
            candidate_vs_topk["verified_refused_answerable_delta"] <= 0,
            observed=candidate_vs_topk["verified_refused_answerable_delta"],
            threshold=0,
            rule="candidate must not add answerable refusals vs top-k",
        ),
        _check(
            "unanswerable_refusal_delta_vs_topk_non_positive",
            candidate_vs_topk["verified_refused_unanswerable_delta"] <= 0,
            observed=candidate_vs_topk["verified_refused_unanswerable_delta"],
            threshold=0,
            rule="candidate must not add unanswerable refusals vs top-k",
        ),
        _check(
            "unanswerable_refusal_regressions_zero",
            int(risk_summary["unanswerable_refusal_regressions"]) == 0,
            observed=risk_summary["unanswerable_refusal_regressions"],
            threshold=0,
            rule="candidate changed-answer risk must have zero unanswerable refusal regressions",
        ),
        _check(
            "candidate_out_of_rank_citation_zero",
            int(risk_summary["candidate_has_out_of_rank_citation"]) == 0,
            observed=risk_summary["candidate_has_out_of_rank_citation"],
            threshold=0,
            rule="candidate changed answers must not include out-of-rank citations",
        ),
    ]


def _rank_contained_safety_checks(
    rank_contained_vs_topk: Mapping[str, Any],
    rank_contained_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _check(
            "rank_contained_samples_complete",
            bool(rank_contained_summary["samples_complete"]),
            observed=rank_contained_summary["sample_count"],
            threshold=rank_contained_summary["total_questions"],
            rule="sample_count must equal total_questions",
        ),
        _check(
            "rank_contained_gold_citation_delta_vs_topk_non_negative",
            rank_contained_vs_topk["verified_gold_citation_delta"] is not None
            and rank_contained_vs_topk["verified_gold_citation_delta"] >= 0,
            observed=rank_contained_vs_topk["verified_gold_citation_delta"],
            threshold=0,
            rule="rank-contained candidate must not lose gold citations vs top-k",
        ),
    ]


def _heldout_test_protocol(candidate_policy: str) -> dict[str, Any]:
    return {
        "status": "frozen_before_first_heldout_run",
        "heldout_dataset": "nvidia/TechQA-RAG-Eval",
        "local_paths": {
            "samples": "data/raw/nvidia_techqa_rag_eval/train.json",
            "corpus": "data/raw/nvidia_techqa_rag_eval/corpus.zip",
        },
        "frozen_candidate_policy": candidate_policy,
        "frozen_runtime_parameters": {
            "retrieval_top_k": 5,
            "evidence_selector": "hybrid-routing",
            "max_candidates_per_document": 3,
            "max_sentences": 3,
            "min_sentence_score": 2.0,
            "min_evidence_score": 15,
            "max_citation_rank": 3,
            "min_citations": 1,
            "composition_policy": (
                "candidate-score-rank-contained-preserve-baseline-out-of-rank-reranker"
            ),
            "candidate_reranker_dataset": (
                "artifacts/candidate_reranker_dataset_stage31_dev_train_hybrid.jsonl"
            ),
            "candidate_reranker_model": "logistic_best_candidate",
            "candidate_reranker_train_split": "train",
        },
        "required_pre_test_steps": [
            (
                "Implement or reuse a NVIDIA TechQA-RAG-Eval evaluator without "
                "changing the frozen runtime parameters."
            ),
            (
                "Run and save a leakage report comparing NVIDIA questions against "
                "all PrimeQA train/dev data used for tuning."
            ),
            (
                "Run the top-k baseline and the frozen candidate on the held-out "
                "dataset once under the same evaluator."
            ),
        ],
        "acceptance_criteria": [
            "heldout verified F1 delta vs top-k >= 0",
            "heldout verified gold citation delta vs top-k >= 0",
            "heldout newly refused delta vs top-k <= 0",
            "heldout answerable refusal delta vs top-k <= 0",
            "heldout unanswerable refusal delta vs top-k <= 0",
            "heldout unanswerable refusal regressions == 0",
            "heldout candidate out-of-rank citation count == 0",
            "leakage report has no unhandled train/dev overlap with the held-out rows",
        ],
        "prohibited_after_freeze": [
            (
                "Do not tune thresholds, prompts, selectors, reranker model choice, "
                "or guard logic after seeing held-out results."
            ),
            "Do not compare multiple new candidates on held-out results and then pick the best.",
            "Do not train or refit candidate rerankers on NVIDIA TechQA-RAG-Eval.",
            "Do not change the default runtime policy before held-out acceptance criteria pass.",
        ],
    }


def _overall_status(candidate_passes: bool, rank_contained_passes: bool) -> str:
    if candidate_passes and not rank_contained_passes:
        return "stage51_is_unique_candidate_for_single_heldout_evaluation"
    if candidate_passes and rank_contained_passes:
        return "multiple_candidates_need_user_decision_before_heldout"
    return "no_candidate_ready_for_heldout"


def _single_candidate_policy(split_reviews: Sequence[Mapping[str, Any]]) -> str:
    policies = {
        str(split_review["candidate_summary"]["composition_policy"])
        for split_review in split_reviews
    }
    if len(policies) != 1:
        raise ValueError(f"Expected one candidate policy, got: {sorted(policies)}")
    return next(iter(policies))


def _exact_gold_cited_count(report: Mapping[str, Any]) -> int:
    count = 0
    for sample in report.get("samples", []):
        if not bool(sample["answerable"]):
            continue
        verified_answer = sample["verified_answer"]
        if bool(verified_answer["refused"]):
            continue
        gold_doc_id = sample["gold_answer_doc_id"]
        if gold_doc_id in {
            citation["document_id"] for citation in verified_answer["citations"]
        }:
            count += 1
    return count


def _nullable_int_delta(
    candidate_value: int | None,
    baseline_value: int | None,
) -> int | None:
    if candidate_value is None or baseline_value is None:
        return None
    return candidate_value - baseline_value


def _check(
    name: str,
    passed: bool,
    observed: Any,
    threshold: Any,
    rule: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "observed": observed,
        "threshold": threshold,
        "rule": rule,
    }


def _signed_int(value: int) -> str:
    return f"{value:+d}"
