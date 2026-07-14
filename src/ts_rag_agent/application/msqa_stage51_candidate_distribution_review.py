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
_PASS = "pass"
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
) -> dict[str, Any]:
    """Review whether the Stage 61 MSQA candidate pool is fair for Stage 51."""

    for path in [adapter_report_path, candidate_jsonl_path, stage31_summary_path]:
        _ensure_file(path)
    adapter_report = _load_json(adapter_report_path)
    stage31_summary = _load_json(stage31_summary_path)
    _validate_adapter_report(adapter_report)
    _validate_stage31_summary(stage31_summary)

    candidate_stats = _scan_candidate_jsonl(candidate_jsonl_path)
    total_samples = int(adapter_report["dry_run_summary"]["evaluation_samples"])
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
    decision = _decision(checks)
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_scope": (
            "MSQA Stage 51 adapter candidate distribution review. This report "
            "does not run Stage 51, does not tune policies, does not change "
            "candidate rows, and does not change the default runtime."
        ),
        "source_files": {
            "adapter_report": _fingerprint(adapter_report_path),
            "candidate_jsonl": _fingerprint(candidate_jsonl_path),
            "stage31_summary": _fingerprint(stage31_summary_path),
        },
        "stage61_adapter_summary": {
            "evaluation_samples": total_samples,
            "candidate_rows": int(adapter_report["dry_run_summary"]["candidate_rows"]),
            "samples_with_candidates": int(
                adapter_report["dry_run_summary"]["samples_with_candidates"]
            ),
            "samples_with_gold_source_candidate": int(
                adapter_report["dry_run_summary"][
                    "samples_with_gold_source_candidate"
                ]
            ),
            "source_retrieval_summary": adapter_report["source_retrieval_summary"],
            "stage51_candidate_run_performed": bool(
                adapter_report["decision"]["stage51_candidate_run_performed"]
            ),
        },
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
    """Write SVG charts for Stage 62 candidate distribution review."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage62_candidate_count_percentiles.svg": render_horizontal_bar_chart_svg(
            title="Stage 62 MSQA candidate count percentiles",
            bars=_percentile_bars(
                report["stage61_candidate_distribution"][
                    "candidate_count_per_query"
                ],
            ),
            x_label="candidates per query",
            margin_left=180,
        ),
        "stage62_stage31_vs_stage61_candidate_pool.svg": render_horizontal_bar_chart_svg(
            title="Stage 62 Stage31 vs Stage61 candidate pool",
            bars=_comparison_bars(report),
            x_label="candidate count",
            margin_left=320,
        ),
        "stage62_candidate_rows_by_retrieval_rank.svg": render_horizontal_bar_chart_svg(
            title="Stage 62 MSQA candidate rows by retrieval rank",
            bars=_retrieval_rank_bars(report),
            x_label="candidate row count",
            margin_left=180,
        ),
        "stage62_fairness_checks.svg": render_horizontal_bar_chart_svg(
            title="Stage 62 fairness checks",
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
    return {
        "average_candidate_count_ratio_stage61_vs_stage31": round(
            stage61_counts["average"] / stage31_counts["average"],
            4,
        ),
        "median_candidate_count_ratio_stage61_vs_stage31": round(
            stage61_counts["median"] / stage31_counts["median"],
            4,
        ),
        "stage61_median_exceeds_stage31_max": (
            stage61_counts["median"] > stage31_counts["max"]
        ),
        "stage61_p10_exceeds_stage31_max": (
            stage61_counts["p10"] > stage31_counts["max"]
        ),
        "gold_candidate_rate_delta_stage61_minus_stage31": round(
            stage61["gold_source_candidate_rate"]
            - stage31["gold_document_candidate_rate"],
            4,
        ),
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
    return [
        _check(
            name="stage61_adapter_contract_passed",
            status=_PASS if all_contract_checks_passed else _BLOCKED,
            severity=_INFO if all_contract_checks_passed else _BLOCKER,
            evidence=(
                "Stage 61 contract checks passed: "
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
            name="all_stage61_samples_have_candidates",
            status=(
                _PASS
                if int(adapter_report["dry_run_summary"]["samples_without_candidates"])
                == 0
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
            status=(
                _PASS
                if abs(
                    comparison["gold_candidate_rate_delta_stage61_minus_stage31"]
                )
                <= 0.01
                else _BLOCKED
            ),
            severity=_INFO,
            evidence=(
                f"Stage61 gold-source candidate rate "
                f"{stage61['gold_source_candidate_rate']} vs Stage31 "
                f"{stage31['gold_document_candidate_rate']}."
            ),
            decision_effect=(
                "Gold-source availability is close to the Stage 31 training pool."
            ),
        ),
        _check(
            name="candidate_pool_size_aligned_with_stage31",
            status=(
                _BLOCKED
                if comparison["stage61_median_exceeds_stage31_max"]
                else _PASS
            ),
            severity=(
                _BLOCKER
                if comparison["stage61_median_exceeds_stage31_max"]
                else _INFO
            ),
            evidence=(
                f"Stage61 median candidates/query "
                f"{stage61['candidate_count_per_query']['median']} vs Stage31 max "
                f"{stage31['candidate_count_per_question']['max']}."
            ),
            decision_effect=(
                "Blocks direct Stage 51 comparison until the MSQA candidate pool "
                "is aligned with the Stage 31 training candidate contract."
            ),
        ),
        _check(
            name="stage61_candidate_volume_within_training_limit",
            status=(
                _BLOCKED
                if comparison["stage61_p10_exceeds_stage31_max"]
                else _PASS
            ),
            severity=(
                _BLOCKER
                if comparison["stage61_p10_exceeds_stage31_max"]
                else _INFO
            ),
            evidence=(
                f"Stage61 p10 candidates/query "
                f"{stage61['candidate_count_per_query']['p10']} vs Stage31 max "
                f"{stage31['candidate_count_per_question']['max']}."
            ),
            decision_effect=(
                "Shows the volume mismatch affects almost all MSQA queries, not "
                "only outliers."
            ),
        ),
        _check(
            name="direct_stage51_adapter_comparison_fair_now",
            status=_BLOCKED,
            severity=_BLOCKER,
            evidence=(
                "Stage61 adapter candidates are uncapped answer sentences from "
                "top10 source rows, while Stage31 training used top5 retrieval "
                "and max3 candidates per document."
            ),
            decision_effect="Rejects immediate Stage 51 adapter comparison.",
        ),
    ]


def _decision(checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blocker_checks = [
        check["name"]
        for check in checks
        if check["status"] == _BLOCKED and check["severity"] == _BLOCKER
    ]
    return {
        "status": "msqa_stage51_adapter_comparison_blocked_by_candidate_pool_mismatch"
        if blocker_checks
        else "msqa_stage51_adapter_comparison_ready_for_user_confirmation",
        "can_run_stage51_candidate_now": False,
        "can_defaultize_runtime_now": False,
        "default_runtime_policy": "unchanged",
        "stage51_candidate_run_performed": False,
        "blocker_checks": blocker_checks,
        "recommended_next_stage": (
            "Stage 63: design a Stage31-aligned MSQA candidate-pool cap and "
            "rerun the adapter dry run before any Stage 51 comparison"
        ),
        "reason": (
            "The Stage 61 adapter contract passed, but the uncapped MSQA candidate "
            "pool is much larger than the Stage 31 training candidate pool. A "
            "direct Stage 51 adapter comparison would mix protocol effects with "
            "policy effects."
        ),
    }


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
        BarDatum(label=key, value=float(distribution[key]), value_label=str(distribution[key]))
        for key in ("p10", "p25", "median", "p75", "p90", "p95", "p99", "max")
    ]


def _comparison_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    stage61 = report["stage61_candidate_distribution"]["candidate_count_per_query"]
    stage31 = report["stage31_candidate_distribution"]["candidate_count_per_question"]
    return [
        BarDatum("Stage31 average", float(stage31["average"]), str(stage31["average"])),
        BarDatum("Stage31 median", float(stage31["median"]), str(stage31["median"])),
        BarDatum("Stage31 max", float(stage31["max"]), str(stage31["max"])),
        BarDatum("Stage61 average", float(stage61["average"]), str(stage61["average"])),
        BarDatum("Stage61 median", float(stage61["median"]), str(stage61["median"])),
        BarDatum("Stage61 p10", float(stage61["p10"]), str(stage61["p10"])),
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


def _validate_adapter_report(report: Mapping[str, Any]) -> None:
    if report.get("stage") != "Stage 61":
        raise ValueError(f"Expected Stage 61 adapter report, got: {report.get('stage')!r}")
    if report["decision"].get("stage51_candidate_run_performed") is not False:
        raise ValueError("Stage 62 expects Stage 61 not to have run Stage 51")
    if report["decision"].get("status") != "msqa_stage51_candidate_adapter_dry_run_passed":
        raise ValueError("Stage 61 adapter dry run must have passed")


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
