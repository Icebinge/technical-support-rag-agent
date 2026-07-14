from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.candidate_reranker_dataset_audit import (
    load_candidate_reranker_rows,
)
from ts_rag_agent.application.candidate_score_guarded_composition_policy import (
    fit_candidate_score_guarded_reranker_composition_policy,
)
from ts_rag_agent.application.evidence_selection import SentenceEvidenceCandidate
from ts_rag_agent.application.msqa_stage51_candidate_adapter import (
    MsqaStage51AdapterSample,
    load_msqa_stage51_adapter_samples,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.application.text_metrics import token_f1
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult

_STAGE = "Stage 64"
_CREATED_AT = "2026-07-14"
_EXPECTED_ADAPTER_STAGE = "Stage 63"
_EXPECTED_DISTRIBUTION_STAGE = "Stage 63"
_EXPECTED_ADAPTER_STATUS = "msqa_stage31_aligned_candidate_adapter_dry_run_passed"
_EXPECTED_DISTRIBUTION_STATUS = (
    "msqa_stage51_adapter_comparison_ready_for_user_confirmation"
)
_DEFAULT_MODEL_NAME = "logistic_best_candidate"
_DEFAULT_TRAIN_SPLIT = "train"
_DEFAULT_MAX_ANSWER_CANDIDATES = 3
_DEFAULT_MAX_CITATION_RANK = 3


@dataclass(frozen=True)
class MsqaCandidateEntry:
    """One Stage64 candidate row plus runtime candidate object."""

    row: dict[str, Any]
    candidate: SentenceEvidenceCandidate
    candidate_rank: int
    candidate_token_f1: float


@dataclass(frozen=True)
class MsqaStage51ComparisonCase:
    """One MSQA question-level comparison case."""

    query_question_id: str
    question_route: str
    action: str
    decision_reason: str
    baseline_candidate_ids: list[str]
    stage51_candidate_ids: list[str]
    baseline_source_row_ids: list[str]
    stage51_source_row_ids: list[str]
    model_selected_candidate_id: str | None
    model_selected_rank: int
    model_selected_candidate_score: float
    model_score_margin_vs_top_candidate: float
    baseline_top1_token_f1: float
    stage51_top1_token_f1: float
    baseline_top3_answer_token_f1: float
    stage51_top3_answer_token_f1: float
    oracle_best_single_token_f1: float
    top3_f1_delta_vs_baseline: float
    top1_f1_delta_vs_baseline: float
    baseline_gold_source_cited: bool
    stage51_gold_source_cited: bool
    citation_delta: int
    protected_baseline_out_of_rank_source_row_ids: list[str]
    dropped_protected_baseline_out_of_rank_source_row_ids: list[str]


@dataclass(frozen=True)
class MsqaStage51AdapterComparisonVisualization:
    """One generated Stage64 visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class _MsqaStage51CaseBuild:
    cases: list[MsqaStage51ComparisonCase]
    policy_name: str


def compare_msqa_stage51_capped_adapter(
    *,
    split_jsonl_path: Path,
    candidate_jsonl_path: Path,
    adapter_report_path: Path,
    distribution_report_path: Path,
    candidate_reranker_dataset_path: Path,
    stage31_summary_path: Path,
    model_name: str = _DEFAULT_MODEL_NAME,
    train_split: str = _DEFAULT_TRAIN_SPLIT,
    max_answer_candidates: int = _DEFAULT_MAX_ANSWER_CANDIDATES,
    max_citation_rank: int = _DEFAULT_MAX_CITATION_RANK,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Run one capped MSQA Stage51 adapter comparison on the Stage63 pool."""

    _validate_options(
        max_answer_candidates=max_answer_candidates,
        max_citation_rank=max_citation_rank,
        sample_limit=sample_limit,
    )
    adapter_report = _load_json(adapter_report_path)
    distribution_report = _load_json(distribution_report_path)
    candidate_rows = _load_candidate_rows(candidate_jsonl_path)
    case_build = _build_msqa_stage51_capped_adapter_case_view(
        split_jsonl_path=split_jsonl_path,
        candidate_jsonl_path=candidate_jsonl_path,
        adapter_report_path=adapter_report_path,
        distribution_report_path=distribution_report_path,
        candidate_reranker_dataset_path=candidate_reranker_dataset_path,
        stage31_summary_path=stage31_summary_path,
        model_name=model_name,
        train_split=train_split,
        max_answer_candidates=max_answer_candidates,
        max_citation_rank=max_citation_rank,
    )
    cases = case_build.cases
    stage31_summary = _load_json(stage31_summary_path)
    selector_name = str(stage31_summary["build_config"]["evidence_selector"])
    metrics = _metrics(cases)
    route_metrics = _segment_metrics(cases, segment_fn=lambda case: case.question_route)
    reason_metrics = dict(sorted(Counter(case.decision_reason for case in cases).items()))
    selected_rank_metrics = dict(
        sorted(Counter(_rank_bucket(case.model_selected_rank) for case in cases).items())
    )

    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "One capped MSQA Stage51 adapter comparison against the unchanged "
            "Stage63 candidate pool. This is an answer-source-row proxy over "
            "MSQA processed-answer sentence candidates. It is not a PrimeQA "
            "verified RAG document-citation metric, does not rebuild the "
            "candidate pool, does not fetch external pages, and does not change "
            "the default runtime."
        ),
        "source_files": {
            "split_jsonl": _fingerprint(split_jsonl_path),
            "candidate_jsonl": _fingerprint(candidate_jsonl_path),
            "adapter_report": _fingerprint(adapter_report_path),
            "distribution_report": _fingerprint(distribution_report_path),
            "candidate_reranker_dataset": _fingerprint(candidate_reranker_dataset_path),
            "stage31_summary": _fingerprint(stage31_summary_path),
        },
        "comparison_contract": {
            "question_text_source": (
                "Stage57 frozen split only, used for runtime feature computation; "
                "question text is not written back to the Stage63 candidate JSONL "
                "and is not used for retrieval indexing."
            ),
            "candidate_pool_source": str(candidate_jsonl_path),
            "candidate_pool_rebuilt": False,
            "candidate_pool_rows": len(candidate_rows),
            "candidate_pool_stage": adapter_report["stage"],
            "candidate_pool_top_k": adapter_report["adapter_contract"]["top_k"],
            "candidate_pool_max_candidates_per_source_row": adapter_report[
                "adapter_contract"
            ]["max_candidates_per_source_row"],
            "candidate_pool_effective_cap": adapter_report["adapter_contract"][
                "effective_candidate_pool_cap"
            ],
            "candidate_pool_rank_for_comparison": (
                "Within each query, Stage64 ranks the unchanged capped candidates "
                "by candidate_score descending, retrieval_rank ascending, and "
                "candidate_id ascending. This mirrors Stage31 candidate_rank as a "
                "global rank inside the already-capped candidate pool."
            ),
            "model_name": model_name,
            "train_split": train_split,
            "selector_name_runtime_feature": selector_name,
            "composition_policy": case_build.policy_name,
            "runtime_guard": (
                "candidate_score_gte_60_all_selected_citations_rank_lte_"
                "max_citation_rank_preserve_baseline_out_of_rank_docs"
            ),
            "max_answer_candidates": max_answer_candidates,
            "rank_contained_max_retrieval_rank": max_citation_rank,
            "preserve_baseline_out_of_rank_source_rows": True,
            "candidate_jsonl_rows_with_question_key": sum(
                1 for row in candidate_rows if "question" in row
            ),
            "candidate_score_boundary": (
                "MSQA candidate_score is the Stage63 dry-run adapter score. The "
                "Stage51 score>=60 guard is preserved unchanged for this one "
                "comparison, so score calibration differences must be interpreted "
                "as part of the adapter-risk evidence."
            ),
        },
        "stage63_source_availability_warning": _stage63_warning(distribution_report),
        "metrics": metrics,
        "route_metrics": route_metrics,
        "decision_reason_counts": reason_metrics,
        "model_selected_rank_distribution": selected_rank_metrics,
        "sample_cases": _sample_cases(cases, sample_limit=sample_limit),
        "decision": _decision(metrics, distribution_report),
    }


def build_msqa_stage51_capped_adapter_cases(
    *,
    split_jsonl_path: Path,
    candidate_jsonl_path: Path,
    adapter_report_path: Path,
    distribution_report_path: Path,
    candidate_reranker_dataset_path: Path,
    stage31_summary_path: Path,
    model_name: str = _DEFAULT_MODEL_NAME,
    train_split: str = _DEFAULT_TRAIN_SPLIT,
    max_answer_candidates: int = _DEFAULT_MAX_ANSWER_CANDIDATES,
    max_citation_rank: int = _DEFAULT_MAX_CITATION_RANK,
) -> list[MsqaStage51ComparisonCase]:
    """Rebuild the full Stage64 case view without modifying the capped pool."""

    return _build_msqa_stage51_capped_adapter_case_view(
        split_jsonl_path=split_jsonl_path,
        candidate_jsonl_path=candidate_jsonl_path,
        adapter_report_path=adapter_report_path,
        distribution_report_path=distribution_report_path,
        candidate_reranker_dataset_path=candidate_reranker_dataset_path,
        stage31_summary_path=stage31_summary_path,
        model_name=model_name,
        train_split=train_split,
        max_answer_candidates=max_answer_candidates,
        max_citation_rank=max_citation_rank,
    ).cases


def _build_msqa_stage51_capped_adapter_case_view(
    *,
    split_jsonl_path: Path,
    candidate_jsonl_path: Path,
    adapter_report_path: Path,
    distribution_report_path: Path,
    candidate_reranker_dataset_path: Path,
    stage31_summary_path: Path,
    model_name: str = _DEFAULT_MODEL_NAME,
    train_split: str = _DEFAULT_TRAIN_SPLIT,
    max_answer_candidates: int = _DEFAULT_MAX_ANSWER_CANDIDATES,
    max_citation_rank: int = _DEFAULT_MAX_CITATION_RANK,
) -> _MsqaStage51CaseBuild:
    """Rebuild Stage64 cases and retain the runtime policy identity."""

    _validate_options(
        max_answer_candidates=max_answer_candidates,
        max_citation_rank=max_citation_rank,
        sample_limit=0,
    )
    for path in [
        split_jsonl_path,
        candidate_jsonl_path,
        adapter_report_path,
        distribution_report_path,
        candidate_reranker_dataset_path,
        stage31_summary_path,
    ]:
        _ensure_file(path)

    adapter_report = _load_json(adapter_report_path)
    distribution_report = _load_json(distribution_report_path)
    stage31_summary = _load_json(stage31_summary_path)
    _validate_stage63_reports(adapter_report, distribution_report)
    selector_name = str(stage31_summary["build_config"]["evidence_selector"])

    samples = load_msqa_stage51_adapter_samples(split_jsonl_path)
    candidate_rows = _load_candidate_rows(candidate_jsonl_path)
    candidate_rows_by_query = _candidate_rows_by_query(candidate_rows)
    _validate_candidate_pool(samples=samples, candidate_rows_by_query=candidate_rows_by_query)

    policy = fit_candidate_score_guarded_reranker_composition_policy(
        rows=load_candidate_reranker_rows(candidate_reranker_dataset_path),
        selector_name=selector_name,
        model_name=model_name,
        train_split=train_split,
        rank_contained_max_retrieval_rank=max_citation_rank,
        preserve_baseline_out_of_rank_docs=True,
    )
    cases = [
        _compare_sample(
            sample=sample,
            rows=candidate_rows_by_query[sample.question_id],
            policy=policy,
            max_answer_candidates=max_answer_candidates,
        )
        for sample in samples
    ]
    return _MsqaStage51CaseBuild(cases=cases, policy_name=policy.name)


def summarize_msqa_stage51_comparison_cases(
    cases: Sequence[MsqaStage51ComparisonCase],
) -> dict[str, Any]:
    """Summarize full Stage64 comparison cases."""

    return _metrics(cases)


def msqa_stage51_comparison_case_to_dict(
    case: MsqaStage51ComparisonCase,
) -> dict[str, Any]:
    """Convert one Stage64 comparison case to a JSON-safe dictionary."""

    return _case_to_dict(case)


def write_msqa_stage51_adapter_comparison_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[MsqaStage51AdapterComparisonVisualization]:
    """Write SVG charts for the Stage64 MSQA adapter comparison."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage64_msqa_answer_f1.svg": render_horizontal_bar_chart_svg(
            title="Stage64 MSQA answer proxy F1",
            bars=_answer_f1_bars(report),
            x_label="average token F1",
            margin_left=280,
        ),
        "stage64_msqa_answer_f1_delta.svg": render_horizontal_bar_chart_svg(
            title="Stage64 MSQA Stage51 F1 delta",
            bars=_answer_delta_bars(report),
            x_label="delta vs capped baseline",
            margin_left=280,
        ),
        "stage64_msqa_gold_source_citation.svg": render_horizontal_bar_chart_svg(
            title="Stage64 MSQA gold-source citation counts",
            bars=_gold_source_bars(report),
            x_label="question count",
            margin_left=320,
        ),
        "stage64_msqa_decision_reasons.svg": render_horizontal_bar_chart_svg(
            title="Stage64 MSQA Stage51 decision reasons",
            bars=_decision_reason_bars(report),
            x_label="question count",
            margin_left=390,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            MsqaStage51AdapterComparisonVisualization(name=filename, path=str(path))
        )
    return artifacts


def _compare_sample(
    *,
    sample: MsqaStage51AdapterSample,
    rows: Sequence[Mapping[str, Any]],
    policy,
    max_answer_candidates: int,
) -> MsqaStage51ComparisonCase:
    ranked_rows = _rank_candidate_rows(rows)
    question = _primeqa_question_from_msqa_sample(sample)
    entries = [
        _candidate_entry(
            sample=sample,
            row=row,
            candidate_rank=rank,
        )
        for rank, row in enumerate(ranked_rows, start=1)
    ]
    candidates = [entry.candidate for entry in entries]
    selected = policy.select(
        question=question,
        candidates=candidates,
        max_sentences=max_answer_candidates,
    ).selected_candidates
    trace = policy.last_trace
    if trace is None:
        raise ValueError("Stage51 policy did not expose a decision trace")

    baseline_entries = entries[:max_answer_candidates]
    entry_by_object_id = {id(entry.candidate): entry for entry in entries}
    selected_entries = [entry_by_object_id[id(candidate)] for candidate in selected]
    model_entry = (
        entries[trace.selected_candidate_rank - 1]
        if trace.selected_candidate_rank > 0
        else None
    )
    question_route = _route(question)
    baseline_top1_f1 = baseline_entries[0].candidate_token_f1 if baseline_entries else 0.0
    stage51_top1_f1 = selected_entries[0].candidate_token_f1 if selected_entries else 0.0
    baseline_top3_f1 = token_f1(_answer_text(baseline_entries), sample.answer)
    stage51_top3_f1 = token_f1(_answer_text(selected_entries), sample.answer)
    oracle_best_single_f1 = max(
        (entry.candidate_token_f1 for entry in entries),
        default=0.0,
    )
    baseline_gold = _gold_source_cited(
        entries=baseline_entries,
        gold_source_row_id=sample.question_id,
    )
    stage51_gold = _gold_source_cited(
        entries=selected_entries,
        gold_source_row_id=sample.question_id,
    )

    return MsqaStage51ComparisonCase(
        query_question_id=sample.question_id,
        question_route=question_route,
        action=trace.action,
        decision_reason=trace.reason,
        baseline_candidate_ids=[_candidate_id(entry) for entry in baseline_entries],
        stage51_candidate_ids=[_candidate_id(entry) for entry in selected_entries],
        baseline_source_row_ids=[_source_row_id(entry) for entry in baseline_entries],
        stage51_source_row_ids=[_source_row_id(entry) for entry in selected_entries],
        model_selected_candidate_id=_candidate_id(model_entry) if model_entry else None,
        model_selected_rank=trace.selected_candidate_rank,
        model_selected_candidate_score=trace.selected_candidate_score,
        model_score_margin_vs_top_candidate=trace.model_score_margin_vs_top_candidate,
        baseline_top1_token_f1=round(baseline_top1_f1, 4),
        stage51_top1_token_f1=round(stage51_top1_f1, 4),
        baseline_top3_answer_token_f1=round(baseline_top3_f1, 4),
        stage51_top3_answer_token_f1=round(stage51_top3_f1, 4),
        oracle_best_single_token_f1=round(oracle_best_single_f1, 4),
        top3_f1_delta_vs_baseline=round(stage51_top3_f1 - baseline_top3_f1, 4),
        top1_f1_delta_vs_baseline=round(stage51_top1_f1 - baseline_top1_f1, 4),
        baseline_gold_source_cited=baseline_gold,
        stage51_gold_source_cited=stage51_gold,
        citation_delta=(int(stage51_gold) - int(baseline_gold)),
        protected_baseline_out_of_rank_source_row_ids=list(
            trace.protected_baseline_out_of_rank_document_ids
        ),
        dropped_protected_baseline_out_of_rank_source_row_ids=list(
            trace.dropped_protected_baseline_out_of_rank_document_ids
        ),
    )


def _candidate_entry(
    *,
    sample: MsqaStage51AdapterSample,
    row: Mapping[str, Any],
    candidate_rank: int,
) -> MsqaCandidateEntry:
    document = PrimeQADocument(
        id=str(row["source_row_id"]),
        title="",
        text=str(row["candidate_sentence"]),
    )
    candidate = SentenceEvidenceCandidate(
        sentence=str(row["candidate_sentence"]),
        retrieval_result=RetrievalResult(
            document=document,
            score=float(row["retrieval_score"]),
            rank=int(row["retrieval_rank"]),
        ),
        score=float(row["candidate_score"]),
        overlap_terms=tuple(str(term) for term in row.get("overlap_terms", ())),
    )
    return MsqaCandidateEntry(
        row=dict(row),
        candidate=candidate,
        candidate_rank=candidate_rank,
        candidate_token_f1=token_f1(str(row["candidate_sentence"]), sample.answer),
    )


def _primeqa_question_from_msqa_sample(sample: MsqaStage51AdapterSample) -> PrimeQAQuestion:
    return PrimeQAQuestion(
        id=sample.question_id,
        title="",
        text=sample.question,
        answer=sample.answer,
        answerable=True,
        answer_doc_id=sample.question_id,
        doc_ids=[sample.question_id],
    )


def _metrics(cases: Sequence[MsqaStage51ComparisonCase]) -> dict[str, Any]:
    question_count = len(cases)
    baseline_top1_values = [case.baseline_top1_token_f1 for case in cases]
    stage51_top1_values = [case.stage51_top1_token_f1 for case in cases]
    baseline_top3_values = [case.baseline_top3_answer_token_f1 for case in cases]
    stage51_top3_values = [case.stage51_top3_answer_token_f1 for case in cases]
    oracle_values = [case.oracle_best_single_token_f1 for case in cases]
    top3_deltas = [case.top3_f1_delta_vs_baseline for case in cases]
    top1_deltas = [case.top1_f1_delta_vs_baseline for case in cases]
    changed_cases = [
        case for case in cases if case.baseline_candidate_ids != case.stage51_candidate_ids
    ]
    citation_lost = [case for case in cases if case.citation_delta < 0]
    citation_gained = [case for case in cases if case.citation_delta > 0]
    improved = [case for case in cases if case.top3_f1_delta_vs_baseline > 0]
    regressed = [case for case in cases if case.top3_f1_delta_vs_baseline < 0]

    return {
        "question_count": question_count,
        "baseline_top1_average_token_f1": _mean(baseline_top1_values),
        "stage51_top1_average_token_f1": _mean(stage51_top1_values),
        "baseline_top3_average_answer_token_f1": _mean(baseline_top3_values),
        "stage51_top3_average_answer_token_f1": _mean(stage51_top3_values),
        "oracle_best_single_average_token_f1": _mean(oracle_values),
        "top1_average_delta_vs_baseline": _mean(top1_deltas),
        "top3_average_delta_vs_baseline": _mean(top3_deltas),
        "top3_improved_count": len(improved),
        "top3_regressed_count": len(regressed),
        "top3_tied_count": question_count - len(improved) - len(regressed),
        "changed_answer_count": len(changed_cases),
        "changed_answer_rate": _ratio(len(changed_cases), question_count),
        "replacement_count": sum(
            case.action == "replace_with_model_candidate" for case in cases
        ),
        "replacement_rate": _ratio(
            sum(case.action == "replace_with_model_candidate" for case in cases),
            question_count,
        ),
        "baseline_gold_source_citation_count": sum(
            case.baseline_gold_source_cited for case in cases
        ),
        "stage51_gold_source_citation_count": sum(
            case.stage51_gold_source_cited for case in cases
        ),
        "baseline_gold_source_citation_rate": _ratio(
            sum(case.baseline_gold_source_cited for case in cases),
            question_count,
        ),
        "stage51_gold_source_citation_rate": _ratio(
            sum(case.stage51_gold_source_cited for case in cases),
            question_count,
        ),
        "gold_source_citation_delta": sum(case.citation_delta for case in cases),
        "citation_lost_count": len(citation_lost),
        "citation_gained_count": len(citation_gained),
        "protected_out_of_rank_source_row_question_count": sum(
            bool(case.protected_baseline_out_of_rank_source_row_ids) for case in cases
        ),
        "dropped_protected_out_of_rank_source_row_question_count": sum(
            bool(case.dropped_protected_baseline_out_of_rank_source_row_ids)
            for case in cases
        ),
        "top3_delta_distribution": _distribution(top3_deltas),
        "top1_delta_distribution": _distribution(top1_deltas),
    }


def _segment_metrics(
    cases: Sequence[MsqaStage51ComparisonCase],
    segment_fn,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[MsqaStage51ComparisonCase]] = defaultdict(list)
    for case in cases:
        grouped[str(segment_fn(case))].append(case)
    metrics = []
    for segment, segment_cases in grouped.items():
        summary = _metrics(segment_cases)
        metrics.append(
            {
                "segment_name": segment,
                "question_count": summary["question_count"],
                "top3_average_delta_vs_baseline": summary[
                    "top3_average_delta_vs_baseline"
                ],
                "changed_answer_count": summary["changed_answer_count"],
                "replacement_count": summary["replacement_count"],
                "top3_improved_count": summary["top3_improved_count"],
                "top3_regressed_count": summary["top3_regressed_count"],
                "gold_source_citation_delta": summary["gold_source_citation_delta"],
            }
        )
    return sorted(
        metrics,
        key=lambda item: (
            item["top3_average_delta_vs_baseline"],
            item["question_count"],
        ),
        reverse=True,
    )


def _decision(
    metrics: Mapping[str, Any],
    distribution_report: Mapping[str, Any],
) -> dict[str, Any]:
    citation_delta = int(metrics["gold_source_citation_delta"])
    top3_delta = float(metrics["top3_average_delta_vs_baseline"])
    source_warning = _stage63_warning(distribution_report)
    if citation_delta < 0:
        status = "msqa_stage51_capped_adapter_comparison_citation_regressed"
    elif top3_delta < 0:
        status = "msqa_stage51_capped_adapter_comparison_f1_regressed"
    else:
        status = "msqa_stage51_capped_adapter_comparison_completed"
    return {
        "status": status,
        "stage51_adapter_comparison_run_performed": True,
        "can_defaultize_runtime_now": False,
        "default_runtime_policy": "unchanged",
        "candidate_pool_reused_without_rebuild": True,
        "source_availability_warning_preserved": bool(source_warning),
        "recommended_next_stage": (
            "Stage 65: review Stage64 MSQA changed cases and source-citation "
            "tradeoffs before deciding whether another external dataset or a "
            "frozen final evaluation protocol is needed"
        ),
        "reason": (
            "Stage64 is a capped MSQA answer-source proxy comparison only. It "
            "can inform risk, but cannot defaultize Stage51 without a separate "
            "final evaluation decision."
        ),
    }


def _sample_cases(
    cases: Sequence[MsqaStage51ComparisonCase],
    *,
    sample_limit: int,
) -> dict[str, list[dict[str, Any]]]:
    if sample_limit <= 0:
        return {}
    return {
        "largest_improvements": [
            _case_to_dict(case)
            for case in sorted(
                cases,
                key=lambda case: (case.top3_f1_delta_vs_baseline, case.query_question_id),
                reverse=True,
            )
            if case.top3_f1_delta_vs_baseline > 0
        ][:sample_limit],
        "largest_regressions": [
            _case_to_dict(case)
            for case in sorted(
                cases,
                key=lambda case: (case.top3_f1_delta_vs_baseline, case.query_question_id),
            )
            if case.top3_f1_delta_vs_baseline < 0
        ][:sample_limit],
        "citation_lost": [
            _case_to_dict(case) for case in cases if case.citation_delta < 0
        ][:sample_limit],
        "citation_gained": [
            _case_to_dict(case) for case in cases if case.citation_delta > 0
        ][:sample_limit],
        "changed_answers": [
            _case_to_dict(case)
            for case in cases
            if case.baseline_candidate_ids != case.stage51_candidate_ids
        ][:sample_limit],
    }


def _case_to_dict(case: MsqaStage51ComparisonCase) -> dict[str, Any]:
    return {
        "query_question_id": case.query_question_id,
        "question_route": case.question_route,
        "action": case.action,
        "decision_reason": case.decision_reason,
        "baseline_candidate_ids": case.baseline_candidate_ids,
        "stage51_candidate_ids": case.stage51_candidate_ids,
        "baseline_source_row_ids": case.baseline_source_row_ids,
        "stage51_source_row_ids": case.stage51_source_row_ids,
        "model_selected_candidate_id": case.model_selected_candidate_id,
        "model_selected_rank": case.model_selected_rank,
        "model_selected_candidate_score": case.model_selected_candidate_score,
        "model_score_margin_vs_top_candidate": case.model_score_margin_vs_top_candidate,
        "baseline_top3_answer_token_f1": case.baseline_top3_answer_token_f1,
        "stage51_top3_answer_token_f1": case.stage51_top3_answer_token_f1,
        "top3_f1_delta_vs_baseline": case.top3_f1_delta_vs_baseline,
        "baseline_gold_source_cited": case.baseline_gold_source_cited,
        "stage51_gold_source_cited": case.stage51_gold_source_cited,
        "citation_delta": case.citation_delta,
        "protected_baseline_out_of_rank_source_row_ids": (
            case.protected_baseline_out_of_rank_source_row_ids
        ),
        "dropped_protected_baseline_out_of_rank_source_row_ids": (
            case.dropped_protected_baseline_out_of_rank_source_row_ids
        ),
    }


def _load_candidate_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No MSQA candidate rows loaded from {path}")
    return rows


def _candidate_rows_by_query(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_question_id"])].append(dict(row))
    return grouped


def _rank_candidate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -float(row["candidate_score"]),
            int(row["retrieval_rank"]),
            str(row["candidate_id"]),
        ),
    )


def _validate_stage63_reports(
    adapter_report: Mapping[str, Any],
    distribution_report: Mapping[str, Any],
) -> None:
    if adapter_report.get("stage") != _EXPECTED_ADAPTER_STAGE:
        raise ValueError("Stage64 requires a Stage63 adapter report")
    if adapter_report["decision"].get("status") != _EXPECTED_ADAPTER_STATUS:
        raise ValueError("Stage63 adapter report must have passed")
    if adapter_report["decision"].get("stage51_candidate_run_performed") is not False:
        raise ValueError("Stage64 requires Stage63 not to have run Stage51")
    contract = adapter_report["adapter_contract"]
    if contract.get("top_k") != 5:
        raise ValueError("Stage64 requires Stage63 top_k=5")
    if contract.get("max_candidates_per_source_row") != 3:
        raise ValueError("Stage64 requires Stage63 max_candidates_per_source_row=3")
    if contract.get("effective_candidate_pool_cap") != 15:
        raise ValueError("Stage64 requires Stage63 effective candidate cap 15")
    if distribution_report.get("stage") != _EXPECTED_DISTRIBUTION_STAGE:
        raise ValueError("Stage64 requires a Stage63 distribution report")
    if distribution_report["decision"].get("status") != _EXPECTED_DISTRIBUTION_STATUS:
        raise ValueError("Stage63 distribution report must be ready for comparison")


def _validate_candidate_pool(
    *,
    samples: Sequence[MsqaStage51AdapterSample],
    candidate_rows_by_query: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    missing = [
        sample.question_id
        for sample in samples
        if sample.question_id not in candidate_rows_by_query
    ]
    if missing:
        raise ValueError(f"Samples missing Stage63 candidates: {missing[:5]}")
    oversized = [
        query_id
        for query_id, rows in candidate_rows_by_query.items()
        if len(rows) > 15
    ]
    if oversized:
        raise ValueError(f"Stage63 candidate pool exceeds cap for: {oversized[:5]}")
    rows_with_question_key = [
        row
        for rows in candidate_rows_by_query.values()
        for row in rows
        if "question" in row
    ]
    if rows_with_question_key:
        raise ValueError("Stage63 candidate JSONL must not contain question text")


def _validate_options(
    *,
    max_answer_candidates: int,
    max_citation_rank: int,
    sample_limit: int,
) -> None:
    if max_answer_candidates <= 0:
        raise ValueError("max_answer_candidates must be positive")
    if max_citation_rank <= 0:
        raise ValueError("max_citation_rank must be positive")
    if sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")


def _stage63_warning(distribution_report: Mapping[str, Any]) -> dict[str, Any]:
    comparison = distribution_report["candidate_pool_comparison"]
    return {
        "name": "gold_source_candidate_rate_matches_training_pool",
        "stage63_gold_source_candidate_rate": distribution_report[
            "adapter_candidate_distribution"
        ]["gold_source_candidate_rate"],
        "stage31_gold_document_candidate_rate": distribution_report[
            "stage31_candidate_distribution"
        ]["gold_document_candidate_rate"],
        "delta": comparison["gold_candidate_rate_delta_adapter_minus_stage31"],
        "interpretation": (
            "Stage63 source availability is lower than the Stage31 training "
            "reference under the top5 source-row boundary."
        ),
    }


def _route(question: PrimeQAQuestion) -> str:
    from ts_rag_agent.application.evidence_selection import classify_question_route

    return classify_question_route(question)


def _answer_text(entries: Sequence[MsqaCandidateEntry]) -> str:
    return "\n".join(entry.candidate.sentence for entry in entries)


def _gold_source_cited(
    *,
    entries: Sequence[MsqaCandidateEntry],
    gold_source_row_id: str,
) -> bool:
    return any(_source_row_id(entry) == gold_source_row_id for entry in entries)


def _candidate_id(entry: MsqaCandidateEntry | None) -> str | None:
    if entry is None:
        return None
    return str(entry.row["candidate_id"])


def _source_row_id(entry: MsqaCandidateEntry) -> str:
    return str(entry.row["source_row_id"])


def _rank_bucket(rank: int) -> str:
    if rank <= 0:
        return "missing"
    if rank == 1:
        return "rank_1"
    if rank == 2:
        return "rank_2"
    if rank == 3:
        return "rank_3"
    if rank <= 5:
        return "rank_4_5"
    if rank <= 10:
        return "rank_6_10"
    return "rank_11_15"


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "average": 0.0,
        }
    sorted_values = sorted(float(value) for value in values)
    return {
        "count": len(sorted_values),
        "min": round(sorted_values[0], 4),
        "p10": _percentile(sorted_values, 10),
        "p25": _percentile(sorted_values, 25),
        "median": round(float(statistics.median(sorted_values)), 4),
        "p75": _percentile(sorted_values, 75),
        "p90": _percentile(sorted_values, 90),
        "max": round(sorted_values[-1], 4),
        "average": _mean(sorted_values),
    }


def _percentile(sorted_values: Sequence[float], percentile: int) -> float:
    if len(sorted_values) == 1:
        return round(sorted_values[0], 4)
    index = round((percentile / 100) * (len(sorted_values) - 1))
    return round(sorted_values[index], 4)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _answer_f1_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    metrics = report["metrics"]
    return [
        BarDatum(
            "baseline_top1",
            float(metrics["baseline_top1_average_token_f1"]),
            str(metrics["baseline_top1_average_token_f1"]),
        ),
        BarDatum(
            "stage51_top1",
            float(metrics["stage51_top1_average_token_f1"]),
            str(metrics["stage51_top1_average_token_f1"]),
        ),
        BarDatum(
            "baseline_top3",
            float(metrics["baseline_top3_average_answer_token_f1"]),
            str(metrics["baseline_top3_average_answer_token_f1"]),
        ),
        BarDatum(
            "stage51_top3",
            float(metrics["stage51_top3_average_answer_token_f1"]),
            str(metrics["stage51_top3_average_answer_token_f1"]),
        ),
        BarDatum(
            "oracle_best_single",
            float(metrics["oracle_best_single_average_token_f1"]),
            str(metrics["oracle_best_single_average_token_f1"]),
        ),
    ]


def _answer_delta_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    metrics = report["metrics"]
    return [
        BarDatum(
            "top1_delta",
            float(metrics["top1_average_delta_vs_baseline"]),
            f"{metrics['top1_average_delta_vs_baseline']:+.4f}",
        ),
        BarDatum(
            "top3_delta",
            float(metrics["top3_average_delta_vs_baseline"]),
            f"{metrics['top3_average_delta_vs_baseline']:+.4f}",
        ),
    ]


def _gold_source_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    metrics = report["metrics"]
    return [
        BarDatum(
            "baseline_gold_source",
            float(metrics["baseline_gold_source_citation_count"]),
            str(metrics["baseline_gold_source_citation_count"]),
        ),
        BarDatum(
            "stage51_gold_source",
            float(metrics["stage51_gold_source_citation_count"]),
            str(metrics["stage51_gold_source_citation_count"]),
        ),
        BarDatum(
            "citation_delta",
            float(metrics["gold_source_citation_delta"]),
            f"{metrics['gold_source_citation_delta']:+d}",
        ),
    ]


def _decision_reason_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(label, float(count), str(count))
        for label, count in report["decision_reason_counts"].items()
    ]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
