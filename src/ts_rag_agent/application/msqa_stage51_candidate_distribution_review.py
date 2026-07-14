from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 62"
_CREATED_AT = "2026-07-14"
_SUPPORTED_REVIEW_STAGES = ("Stage 62", "Stage 63")
_ADAPTER_PASS_STATUSES = {
    "Stage 61": "msqa_stage51_candidate_adapter_dry_run_passed",
    "Stage 63": "msqa_stage31_aligned_candidate_adapter_dry_run_passed",
}
_PASS = "pass"
_WARN = "warn"
_BLOCKED = "blocked"
_INFO = "info"
_BLOCKER = "blocker"


@dataclass(frozen=True)
class DistributionVisualization:
    """One generated Stage 62 candidate distribution visualization."""

    name: str
    path: str


def review_msqa_stage51_candidate_distribution(
    *,
    adapter_report_path: Path,
    candidate_jsonl_path: Path,
    stage31_summary_path: Path,
    stage_name: str = _STAGE,
) -> dict[str, Any]:
    """Review whether an MSQA adapter candidate pool is fair for Stage 51."""

    for path in [adapter_report_path, candidate_jsonl_path, stage31_summary_path]:
        _ensure_file(path)
    _validate_stage_name(stage_name)
    adapter_report = _load_json(adapter_report_path)
    stage31_summary = _load_json(stage31_summary_path)
    _validate_adapter_report(adapter_report)
    _validate_stage31_summary(stage31_summary)

    candidate_stats = _scan_candidate_jsonl(candidate_jsonl_path)
    total_samples = int(adapter_report["dry_run_summary"]["evaluation_samples"])
    adapter_stage = str(adapter_report["stage"])
    stage61_distribution = _stage61_distribution(
        candidate_stats=candidate_stats,
        total_samples=total_samples,
    )
    stage31_distribution = _stage31_distribution(stage31_summary)
    comparison = _compare_candidate_pools(
        stage61=stage61_distribution,
        stage31=stage31_distribution,
    )
    checks = _fairness_checks(
        adapter_report=adapter_report,
        candidate_stats=candidate_stats,
        stage61=stage61_distribution,
        stage31=stage31_distribution,
        comparison=comparison,
    )
    decision = _decision(
        checks,
        stage_name=stage_name,
        adapter_stage=adapter_stage,
    )
    adapter_summary = {
        "stage": adapter_stage,
        "evaluation_samples": total_samples,
        "candidate_rows": int(adapter_report["dry_run_summary"]["candidate_rows"]),
        "samples_with_candidates": int(
            adapter_report["dry_run_summary"]["samples_with_candidates"]
        ),
        "samples_with_gold_source_candidate": int(
            adapter_report["dry_run_summary"]["samples_with_gold_source_candidate"]
        ),
        "adapter_contract": adapter_report.get("adapter_contract", {}),
        "source_retrieval_summary": adapter_report["source_retrieval_summary"],
        "stage51_candidate_run_performed": bool(
            adapter_report["decision"]["stage51_candidate_run_performed"]
        ),
    }
    return {
        "stage": stage_name,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            f"{stage_name} MSQA Stage 51 adapter candidate distribution review "
            f"for {adapter_stage}. This report does not run Stage 51, does not "
            "tune policies, does not change candidate rows, and does not change "
            "the default runtime."
        ),
        "source_files": {
            "adapter_report": _fingerprint(adapter_report_path),
            "candidate_jsonl": _fingerprint(candidate_jsonl_path),
            "stage31_summary": _fingerprint(stage31_summary_path),
        },
        "adapter_summary": adapter_summary,
        "stage61_adapter_summary": adapter_summary,
        "stage31_training_candidate_contract": {
            "retrieval_top_k": stage31_summary["build_config"]["retrieval_top_k"],
            "max_candidates_per_document": stage31_summary["build_config"][
                "max_candidates_per_document"
            ],
            "candidate_limit": stage31_summary["build_config"]["candidate_limit"],
            "evidence_selector": stage31_summary["build_config"][
                "evidence_selector"
            ],
            "average_rows_per_question": stage31_summary["summary"][
                "average_rows_per_question"
            ],
            "total_questions": stage31_summary["summary"]["total_questions"],
            "total_rows": stage31_summary["summary"]["total_rows"],
        },
        "adapter_candidate_distribution": stage61_distribution,
        "stage61_candidate_distribution": stage61_distribution,
        "stage31_candidate_distribution": stage31_distribution,
        "candidate_pool_comparison": comparison,
        "fairness_checks": checks,
        "decision": decision,
    }


def write_msqa_stage51_candidate_distribution_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[DistributionVisualization]:
    """Write SVG charts for MSQA candidate distribution review."""

    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = _visualization_filenames(report)
    stage_label = str(report.get("stage", _STAGE))
    adapter_label = str(report.get("adapter_summary", {}).get("stage", "adapter"))
    charts = {
        filenames["percentiles"]: render_horizontal_bar_chart_svg(
            title=f"{stage_label} MSQA candidate count percentiles",
            bars=_percentile_bars(
                report["stage61_candidate_distribution"][
                    "candidate_count_per_query"
                ],
            ),
            x_label="candidates per query",
            margin_left=180,
        ),
        filenames["comparison"]: render_horizontal_bar_chart_svg(
            title=f"{stage_label} Stage31 vs {adapter_label} candidate pool",
            bars=_comparison_bars(report),
            x_label="candidate count",
            margin_left=320,
        ),
        filenames["retrieval_rank"]: render_horizontal_bar_chart_svg(
            title=f"{stage_label} MSQA candidate rows by retrieval rank",
            bars=_retrieval_rank_bars(report),
            x_label="candidate row count",
            margin_left=180,
        ),
        filenames["fairness"]: render_horizontal_bar_chart_svg(
            title=f"{stage_label} fairness checks",
            bars=_fairness_check_bars(report),
            x_label="1 means pass",
            margin_left=390,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(DistributionVisualization(name=filename, path=str(path)))
    return artifacts


def _scan_candidate_jsonl(path: Path) -> dict[str, Any]:
    counts_by_query: Counter[str] = Counter()
    gold_counts_by_query: Counter[str] = Counter()
    source_rows_by_query: dict[str, set[str]] = defaultdict(set)
    retrieval_rank_counts: Counter[int] = Counter()
    score_values: list[float] = []
    top_candidate_by_query: dict[str, dict[str, Any]] = {}
    row_count = 0
    rows_with_question_key = 0
    rows_missing_required_distribution_fields = 0

    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        row_count += 1
        if "question" in row:
            rows_with_question_key += 1
        if _missing_distribution_field(row):
            rows_missing_required_distribution_fields += 1
            continue
        query_id = str(row["query_question_id"])
        source_row_id = str(row["source_row_id"])
        gold_source_row_id = str(row["gold_source_row_id"])
        retrieval_rank = int(row["retrieval_rank"])
        candidate_score = float(row["candidate_score"])
        counts_by_query[query_id] += 1
        source_rows_by_query[query_id].add(source_row_id)
        retrieval_rank_counts[retrieval_rank] += 1
        score_values.append(candidate_score)
        if source_row_id == gold_source_row_id:
            gold_counts_by_query[query_id] += 1
        _maybe_update_top_candidate(
            top_candidate_by_query=top_candidate_by_query,
            query_id=query_id,
            row=row,
            candidate_score=candidate_score,
            retrieval_rank=retrieval_rank,
        )

    return {
        "row_count": row_count,
        "counts_by_query": counts_by_query,
        "gold_counts_by_query": gold_counts_by_query,
        "unique_source_counts_by_query": Counter(
            {
                query_id: len(source_rows)
                for query_id, source_rows in source_rows_by_query.items()
            }
        ),
        "retrieval_rank_counts": retrieval_rank_counts,
        "score_values": score_values,
        "top_candidate_by_query": top_candidate_by_query,
        "rows_with_question_key": rows_with_question_key,
        "rows_missing_required_distribution_fields": (
            rows_missing_required_distribution_fields
        ),
    }


def _stage61_distribution(
    *,
    candidate_stats: Mapping[str, Any],
    total_samples: int,
) -> dict[str, Any]:
    candidate_counts = _counts_with_zeroes(
        candidate_stats["counts_by_query"],
        total_samples,
    )
    gold_counts = _counts_with_zeroes(
        candidate_stats["gold_counts_by_query"],
        total_samples,
    )
    unique_source_counts = _counts_with_zeroes(
        candidate_stats["unique_source_counts_by_query"],
        total_samples,
    )
    top_candidate_rows = candidate_stats["top_candidate_by_query"].values()
    top_candidate_gold_count = sum(
        1
        for row in top_candidate_rows
        if str(row["source_row_id"]) == str(row["gold_source_row_id"])
    )
    return {
        "queries_seen_in_candidate_jsonl": len(candidate_stats["counts_by_query"]),
        "candidate_jsonl_rows": candidate_stats["row_count"],
        "rows_with_question_key": candidate_stats["rows_with_question_key"],
        "rows_missing_required_distribution_fields": candidate_stats[
            "rows_missing_required_distribution_fields"
        ],
        "candidate_count_per_query": _distribution(candidate_counts),
        "unique_source_rows_per_query": _distribution(unique_source_counts),
        "gold_source_candidate_count_per_query": _distribution(gold_counts),
        "candidate_score_distribution": _distribution(
            candidate_stats["score_values"],
        ),
        "retrieval_rank_counts": {
            str(rank): candidate_stats["retrieval_rank_counts"][rank]
            for rank in sorted(candidate_stats["retrieval_rank_counts"])
        },
        "queries_with_gold_source_candidate": sum(1 for value in gold_counts if value > 0),
        "gold_source_candidate_rate": round(
            sum(1 for value in gold_counts if value > 0) / total_samples,
            4,
        ),
        "queries_with_top_candidate_from_gold_source": top_candidate_gold_count,
        "top_candidate_gold_source_rate": round(
            top_candidate_gold_count / total_samples,
            4,
        ),
    }


def _stage31_distribution(stage31_summary: Mapping[str, Any]) -> dict[str, Any]:
    question_summaries = stage31_summary["question_summaries"]
    candidate_counts = [int(row["candidate_count"]) for row in question_summaries]
    gold_counts = [
        int(row["gold_document_candidate_count"]) for row in question_summaries
    ]
    total_questions = int(stage31_summary["summary"]["total_questions"])
    return {
        "candidate_count_per_question": _distribution(candidate_counts),
        "gold_document_candidate_count_per_question": _distribution(gold_counts),
        "questions_with_gold_document_candidate": sum(
            1 for value in gold_counts if value > 0
        ),
        "gold_document_candidate_rate": round(
            sum(1 for value in gold_counts if value > 0) / total_questions,
            4,
        ),
        "candidate_limit": int(stage31_summary["build_config"]["candidate_limit"]),
        "effective_max_candidates": (
            int(stage31_summary["build_config"]["retrieval_top_k"])
            * int(stage31_summary["build_config"]["max_candidates_per_document"])
        ),
    }


def _compare_candidate_pools(
    *,
    stage61: Mapping[str, Any],
    stage31: Mapping[str, Any],
) -> dict[str, Any]:
    stage61_counts = stage61["candidate_count_per_query"]
    stage31_counts = stage31["candidate_count_per_question"]
    adapter_average_ratio = round(
        stage61_counts["average"] / stage31_counts["average"],
        4,
    )
    adapter_median_ratio = round(
        stage61_counts["median"] / stage31_counts["median"],
        4,
    )
    adapter_median_exceeds_stage31_max = stage61_counts["median"] > stage31_counts["max"]
    adapter_p10_exceeds_stage31_max = stage61_counts["p10"] > stage31_counts["max"]
    gold_candidate_rate_delta = round(
        stage61["gold_source_candidate_rate"]
        - stage31["gold_document_candidate_rate"],
        4,
    )
    return {
        "average_candidate_count_ratio_adapter_vs_stage31": adapter_average_ratio,
        "median_candidate_count_ratio_adapter_vs_stage31": adapter_median_ratio,
        "adapter_median_exceeds_stage31_max": adapter_median_exceeds_stage31_max,
        "adapter_p10_exceeds_stage31_max": adapter_p10_exceeds_stage31_max,
        "gold_candidate_rate_delta_adapter_minus_stage31": gold_candidate_rate_delta,
        "average_candidate_count_ratio_stage61_vs_stage31": round(
            stage61_counts["average"] / stage31_counts["average"],
            4,
        ),
        "median_candidate_count_ratio_stage61_vs_stage31": round(
            stage61_counts["median"] / stage31_counts["median"],
            4,
        ),
        "stage61_median_exceeds_stage31_max": adapter_median_exceeds_stage31_max,
        "stage61_p10_exceeds_stage31_max": adapter_p10_exceeds_stage31_max,
        "gold_candidate_rate_delta_stage61_minus_stage31": gold_candidate_rate_delta,
    }


def _fairness_checks(
    *,
    adapter_report: Mapping[str, Any],
    candidate_stats: Mapping[str, Any],
    stage61: Mapping[str, Any],
    stage31: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> list[dict[str, Any]]:
    contract_checks = adapter_report["candidate_contract_checks"]
    all_contract_checks_passed = all(check["passed"] for check in contract_checks)
    adapter_stage = str(adapter_report["stage"])
    samples_without_candidates = int(
        adapter_report["dry_run_summary"]["samples_without_candidates"]
    )
    candidate_pool_aligned = not comparison["adapter_median_exceeds_stage31_max"]
    candidate_volume_aligned = not comparison["adapter_p10_exceeds_stage31_max"]
    gold_rate_aligned = (
        abs(comparison["gold_candidate_rate_delta_adapter_minus_stage31"]) <= 0.01
    )
    direct_comparison_fair = (
        all_contract_checks_passed
        and candidate_stats["rows_with_question_key"] == 0
        and samples_without_candidates == 0
        and candidate_pool_aligned
        and candidate_volume_aligned
    )
    return [
        _check(
            name="adapter_contract_passed",
            status=_PASS if all_contract_checks_passed else _BLOCKED,
            severity=_INFO if all_contract_checks_passed else _BLOCKER,
            evidence=(
                f"{adapter_stage} contract checks passed: "
                f"{sum(check['passed'] for check in contract_checks)} / "
                f"{len(contract_checks)}."
            ),
            decision_effect="Allows distribution review to proceed.",
        ),
        _check(
            name="candidate_jsonl_has_no_question_text_field",
            status=_PASS if candidate_stats["rows_with_question_key"] == 0 else _BLOCKED,
            severity=_INFO if candidate_stats["rows_with_question_key"] == 0 else _BLOCKER,
            evidence=(
                "Rows with question key: "
                f"{candidate_stats['rows_with_question_key']}."
            ),
            decision_effect="Confirms the Stage 60 no-question-text boundary.",
        ),
        _check(
            name="all_adapter_samples_have_candidates",
            status=(
                _PASS
                if samples_without_candidates == 0
                else _BLOCKED
            ),
            severity=_INFO,
            evidence=(
                f"Samples with candidates: "
                f"{adapter_report['dry_run_summary']['samples_with_candidates']} / "
                f"{adapter_report['dry_run_summary']['evaluation_samples']}."
            ),
            decision_effect="Adapter coverage is complete at the candidate-row level.",
        ),
        _check(
            name="gold_source_candidate_rate_matches_training_pool",
            status=_PASS if gold_rate_aligned else _WARN,
            severity=_INFO,
            evidence=(
                f"{adapter_stage} gold-source candidate rate "
                f"{stage61['gold_source_candidate_rate']} vs Stage31 "
                f"{stage31['gold_document_candidate_rate']}."
            ),
            decision_effect=_gold_rate_effect(gold_rate_aligned),
        ),
        _check(
            name="candidate_pool_size_aligned_with_stage31",
            status=_PASS if candidate_pool_aligned else _BLOCKED,
            severity=_INFO if candidate_pool_aligned else _BLOCKER,
            evidence=(
                f"{adapter_stage} median candidates/query "
                f"{stage61['candidate_count_per_query']['median']} vs Stage31 max "
                f"{stage31['candidate_count_per_question']['max']}."
            ),
            decision_effect=_candidate_pool_size_effect(
                candidate_pool_aligned,
            ),
        ),
        _check(
            name="adapter_candidate_volume_within_training_limit",
            status=_PASS if candidate_volume_aligned else _BLOCKED,
            severity=_INFO if candidate_volume_aligned else _BLOCKER,
            evidence=(
                f"{adapter_stage} p10 candidates/query "
                f"{stage61['candidate_count_per_query']['p10']} vs Stage31 max "
                f"{stage31['candidate_count_per_question']['max']}."
            ),
            decision_effect=_candidate_volume_effect(
                candidate_volume_aligned,
            ),
        ),
        _check(
            name="direct_stage51_adapter_comparison_fair_now",
            status=_PASS if direct_comparison_fair else _BLOCKED,
            severity=_INFO if direct_comparison_fair else _BLOCKER,
            evidence=_direct_comparison_evidence(
                adapter_report=adapter_report,
                candidate_pool_aligned=candidate_pool_aligned,
                candidate_volume_aligned=candidate_volume_aligned,
            ),
            decision_effect=_direct_comparison_effect(direct_comparison_fair),
        ),
    ]


def _decision(
    checks: Sequence[Mapping[str, Any]],
    *,
    stage_name: str,
    adapter_stage: str,
) -> dict[str, Any]:
    blocker_checks = [
        check["name"]
        for check in checks
        if check["status"] == _BLOCKED and check["severity"] == _BLOCKER
    ]
    ready_for_confirmation = not blocker_checks
    return {
        "status": "msqa_stage51_adapter_comparison_blocked_by_candidate_pool_mismatch"
        if blocker_checks
        else "msqa_stage51_adapter_comparison_ready_for_user_confirmation",
        "can_run_stage51_candidate_now": False,
        "can_run_stage51_candidate_next_with_user_confirmation": (
            ready_for_confirmation
        ),
        "can_defaultize_runtime_now": False,
        "default_runtime_policy": "unchanged",
        "stage51_candidate_run_performed": False,
        "blocker_checks": blocker_checks,
        "recommended_next_stage": _recommended_next_stage(
            stage_name=stage_name,
            adapter_stage=adapter_stage,
            ready_for_confirmation=ready_for_confirmation,
        ),
        "reason": _decision_reason(
            adapter_stage=adapter_stage,
            ready_for_confirmation=ready_for_confirmation,
        ),
    }


def _direct_comparison_evidence(
    *,
    adapter_report: Mapping[str, Any],
    candidate_pool_aligned: bool,
    candidate_volume_aligned: bool,
) -> str:
    adapter_stage = str(adapter_report["stage"])
    contract = adapter_report.get("adapter_contract", {})
    top_k = contract.get("top_k")
    max_candidates = contract.get("max_candidates_per_source_row")
    if candidate_pool_aligned and candidate_volume_aligned:
        return (
            f"{adapter_stage} adapter candidates use top_k={top_k} and "
            f"max_candidates_per_source_row={max_candidates}, matching the "
            "Stage31 effective candidate pool size boundary."
        )
    return (
        f"{adapter_stage} adapter candidates are not aligned with the Stage31 "
        "training candidate pool size boundary."
    )


def _direct_comparison_effect(direct_comparison_fair: bool) -> str:
    if direct_comparison_fair:
        return "Allows one capped Stage 51 adapter comparison after user confirmation."
    return "Rejects immediate Stage 51 adapter comparison."


def _gold_rate_effect(gold_rate_aligned: bool) -> str:
    if gold_rate_aligned:
        return "Gold-source availability is close to the Stage31 training pool."
    return (
        "Records a source-retrieval availability tradeoff; this does not block "
        "candidate-pool size fairness."
    )


def _candidate_pool_size_effect(candidate_pool_aligned: bool) -> str:
    if candidate_pool_aligned:
        return "Confirms candidate-pool median is within the Stage31 size boundary."
    return (
        "Blocks direct Stage 51 comparison until the MSQA candidate pool is "
        "aligned with the Stage31 training candidate contract."
    )


def _candidate_volume_effect(candidate_volume_aligned: bool) -> str:
    if candidate_volume_aligned:
        return "Confirms lower-tail candidate volume is within the Stage31 limit."
    return (
        "Shows the volume mismatch affects almost all MSQA queries, not only "
        "outliers."
    )


def _recommended_next_stage(
    *,
    stage_name: str,
    adapter_stage: str,
    ready_for_confirmation: bool,
) -> str:
    if ready_for_confirmation:
        return (
            "Stage 64: run one capped Stage 51 adapter comparison against the "
            "same capped candidate pool after user confirmation"
        )
    if stage_name == "Stage 63" or adapter_stage == "Stage 63":
        return (
            "Stage 64: inspect the capped candidate distribution blockers before "
            "running any Stage 51 comparison"
        )
    return (
        "Stage 63: design a Stage31-aligned MSQA candidate-pool cap and rerun "
        "the adapter dry run before any Stage 51 comparison"
    )


def _decision_reason(*, adapter_stage: str, ready_for_confirmation: bool) -> str:
    if ready_for_confirmation:
        return (
            f"The {adapter_stage} adapter contract passed and the candidate pool "
            "is aligned with the Stage31 training candidate size boundary. Stage "
            "51 still has not been run and the default runtime is unchanged."
        )
    return (
        f"The {adapter_stage} adapter contract passed, but the candidate pool is "
        "not yet aligned with the Stage31 training candidate pool. A direct "
        "Stage 51 adapter comparison would mix protocol effects with policy "
        "effects."
    )


def _distribution(values: Sequence[float | int]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
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
        "p95": _percentile(sorted_values, 95),
        "p99": _percentile(sorted_values, 99),
        "max": round(sorted_values[-1], 4),
        "average": round(sum(sorted_values) / len(sorted_values), 4),
    }


def _percentile(sorted_values: Sequence[float], percentile: int) -> float:
    if len(sorted_values) == 1:
        return round(sorted_values[0], 4)
    index = round((percentile / 100) * (len(sorted_values) - 1))
    return round(sorted_values[index], 4)


def _counts_with_zeroes(counter: Mapping[str, int], total_count: int) -> list[int]:
    values = list(counter.values())
    if len(values) < total_count:
        values.extend([0] * (total_count - len(values)))
    return values


def _missing_distribution_field(row: Mapping[str, Any]) -> bool:
    required = (
        "query_question_id",
        "source_row_id",
        "gold_source_row_id",
        "retrieval_rank",
        "candidate_score",
        "candidate_id",
    )
    return any(row.get(field) in (None, "") for field in required)


def _maybe_update_top_candidate(
    *,
    top_candidate_by_query: dict[str, dict[str, Any]],
    query_id: str,
    row: Mapping[str, Any],
    candidate_score: float,
    retrieval_rank: int,
) -> None:
    current = top_candidate_by_query.get(query_id)
    candidate_id = str(row["candidate_id"])
    if current is None:
        top_candidate_by_query[query_id] = dict(row)
        return
    current_score = float(current["candidate_score"])
    current_rank = int(current["retrieval_rank"])
    current_candidate_id = str(current["candidate_id"])
    if (
        candidate_score > current_score
        or (candidate_score == current_score and retrieval_rank < current_rank)
        or (
            candidate_score == current_score
            and retrieval_rank == current_rank
            and candidate_id < current_candidate_id
        )
    ):
        top_candidate_by_query[query_id] = dict(row)


def _check(
    *,
    name: str,
    status: str,
    severity: str,
    evidence: str,
    decision_effect: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "severity": severity,
        "evidence": evidence,
        "decision_effect": decision_effect,
    }


def _percentile_bars(distribution: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=key,
            value=float(distribution[key]),
            value_label=str(distribution[key]),
        )
        for key in ("p10", "p25", "median", "p75", "p90", "p95", "p99", "max")
    ]


def _comparison_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    stage61 = report["stage61_candidate_distribution"]["candidate_count_per_query"]
    stage31 = report["stage31_candidate_distribution"]["candidate_count_per_question"]
    return [
        BarDatum("Stage31 average", float(stage31["average"]), str(stage31["average"])),
        BarDatum("Stage31 median", float(stage31["median"]), str(stage31["median"])),
        BarDatum("Stage31 max", float(stage31["max"]), str(stage31["max"])),
        BarDatum("Adapter average", float(stage61["average"]), str(stage61["average"])),
        BarDatum("Adapter median", float(stage61["median"]), str(stage61["median"])),
        BarDatum("Adapter p10", float(stage61["p10"]), str(stage61["p10"])),
    ]


def _retrieval_rank_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counts = report["stage61_candidate_distribution"]["retrieval_rank_counts"]
    return [
        BarDatum(label=f"rank {rank}", value=float(count), value_label=str(count))
        for rank, count in counts.items()
    ]


def _fairness_check_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(check["name"]),
            value=1.0 if check["status"] == _PASS else 0.0,
            value_label=check["status"],
        )
        for check in report["fairness_checks"]
    ]


def _visualization_filenames(report: Mapping[str, Any]) -> dict[str, str]:
    stage_label = str(report.get("stage", _STAGE))
    if stage_label == "Stage 62":
        return {
            "percentiles": "stage62_candidate_count_percentiles.svg",
            "comparison": "stage62_stage31_vs_stage61_candidate_pool.svg",
            "retrieval_rank": "stage62_candidate_rows_by_retrieval_rank.svg",
            "fairness": "stage62_fairness_checks.svg",
        }
    stage_slug = _stage_slug(stage_label)
    return {
        "percentiles": f"{stage_slug}_candidate_count_percentiles.svg",
        "comparison": f"{stage_slug}_stage31_vs_adapter_candidate_pool.svg",
        "retrieval_rank": f"{stage_slug}_candidate_rows_by_retrieval_rank.svg",
        "fairness": f"{stage_slug}_fairness_checks.svg",
    }


def _validate_adapter_report(report: Mapping[str, Any]) -> None:
    adapter_stage = str(report.get("stage"))
    if adapter_stage not in _ADAPTER_PASS_STATUSES:
        raise ValueError(
            "Expected Stage 61 or Stage 63 adapter report, got: "
            f"{report.get('stage')!r}"
        )
    if report["decision"].get("stage51_candidate_run_performed") is not False:
        raise ValueError("Distribution review expects adapter not to have run Stage 51")
    if report["decision"].get("status") != _ADAPTER_PASS_STATUSES[adapter_stage]:
        raise ValueError(f"{adapter_stage} adapter dry run must have passed")


def _validate_stage_name(stage_name: str) -> None:
    if stage_name not in _SUPPORTED_REVIEW_STAGES:
        raise ValueError(
            "stage_name must be one of: "
            + ", ".join(repr(stage) for stage in _SUPPORTED_REVIEW_STAGES)
        )


def _validate_stage31_summary(report: Mapping[str, Any]) -> None:
    if report.get("dataset") != "PrimeQA/TechQA":
        raise ValueError("Stage 31 summary must describe PrimeQA/TechQA")
    required_build_keys = {
        "retrieval_top_k",
        "max_candidates_per_document",
        "candidate_limit",
    }
    missing = sorted(required_build_keys.difference(report.get("build_config", {})))
    if missing:
        raise ValueError(f"Stage 31 summary missing build_config keys: {missing}")
    if "question_summaries" not in report:
        raise ValueError("Stage 31 summary missing question_summaries")


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


def _stage_slug(stage_label: str) -> str:
    return "".join(stage_label.lower().split())
