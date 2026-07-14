from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 76"
_CREATED_AT = "2026-07-15"
_SOURCE_STAGE = "Stage 75"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_ALLOWED_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})


@dataclass(frozen=True)
class PrimeQAHybridRetrievalRecallCandidateDesignVisualization:
    """One generated Stage76 retrieval-recall candidate design visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class _CandidateSpec:
    candidate_id: str
    name: str
    category: str
    status: str
    risk_level: str
    implementation_readiness: float
    rationale: str
    stage77_test_plan: tuple[str, ...]
    target_metric_contract: tuple[str, ...]
    matcher: Callable[[Mapping[str, Any]], bool]


def design_primeqa_hybrid_retrieval_recall_candidates(
    *,
    stage75_report_path: Path,
) -> dict[str, Any]:
    """Design train/dev-only retrieval-recall candidates from Stage75 miss drivers."""

    stage75_report = _load_json_object(stage75_report_path)
    miss_cases = _collect_miss_cases(stage75_report)
    guard_checks = _guard_checks(stage75_report=stage75_report)
    candidate_designs = [
        _candidate_design(spec=spec, miss_cases=miss_cases)
        for spec in _candidate_specs()
    ]
    allowed_candidates = [
        candidate
        for candidate in candidate_designs
        if candidate["status"] == "recommended_for_train_dev_experiment"
    ]
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "design_scope": (
            "Train/dev-only retrieval-recall candidate design for "
            "primeqa_hybrid_stage68_v1. This stage consumes the public-safe "
            "Stage75 BM25 top10 miss report, designs candidate retrieval "
            "experiments, keeps the frozen test split locked, does not run final "
            "metrics, and does not change runtime defaults."
        ),
        "source_files": {"stage75_report": _fingerprint(stage75_report_path)},
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
            "forbidden_final_splits": sorted(_FORBIDDEN_FINAL_SPLITS),
        },
        "stage75_summary": _stage75_summary(stage75_report),
        "miss_driver_summary": _miss_driver_summary(stage75_report),
        "candidate_designs": candidate_designs,
        "recommended_execution_order": [
            candidate["candidate_id"]
            for candidate in sorted(
                allowed_candidates,
                key=lambda candidate: (
                    -candidate["priority_score"],
                    candidate["risk_level"],
                    candidate["candidate_id"],
                ),
            )
        ],
        "blocked_items": [
            candidate
            for candidate in candidate_designs
            if candidate["status"] == "blocked_from_train_dev_experiment"
        ],
        "guard_checks": guard_checks,
        "decision": _decision(guard_checks, allowed_candidates),
    }


def write_primeqa_hybrid_retrieval_recall_candidate_design_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridRetrievalRecallCandidateDesignVisualization]:
    """Write SVG charts for Stage76 retrieval-recall candidate design."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage76_candidate_priority_scores.svg": render_horizontal_bar_chart_svg(
            title="Stage76 retrieval-recall candidate priority",
            bars=_candidate_priority_bars(report),
            x_label="priority score",
            width=1120,
            margin_left=470,
        ),
        "stage76_candidate_target_misses.svg": render_horizontal_bar_chart_svg(
            title="Stage76 candidate target misses",
            bars=_candidate_target_bars(report, split=None),
            x_label="train/dev miss cases matched",
            width=1120,
            margin_left=470,
        ),
        "stage76_candidate_dev_targets.svg": render_horizontal_bar_chart_svg(
            title="Stage76 candidate dev targets",
            bars=_candidate_target_bars(report, split="dev"),
            x_label="dev miss cases matched",
            width=1120,
            margin_left=470,
        ),
        "stage76_allowed_vs_blocked_candidates.svg": render_horizontal_bar_chart_svg(
            title="Stage76 allowed vs blocked candidate designs",
            bars=_allowed_vs_blocked_bars(report),
            x_label="candidate count",
            margin_left=300,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridRetrievalRecallCandidateDesignVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _candidate_specs() -> tuple[_CandidateSpec, ...]:
    return (
        _CandidateSpec(
            candidate_id="section_bm25_doc_rollup_train_dev_probe",
            name="Section BM25 document rollup probe",
            category="section_level_retrieval",
            status="recommended_for_train_dev_experiment",
            risk_level="medium",
            implementation_readiness=0.9,
            rationale=(
                "Existing SectionBM25Retriever can score sections and roll the "
                "best section score back to the parent document. This directly "
                "targets whole-document BM25 misses, especially long documents "
                "whose relevant section may be diluted by unrelated text."
            ),
            stage77_test_plan=(
                "Evaluate on frozen train/dev splits only.",
                "Compare hit@1, hit@5, hit@10, MRR, and miss-rank transitions "
                "against the Stage70 BM25 baseline.",
                "Report changed dev cases without raw question, answer, or "
                "document text.",
            ),
            target_metric_contract=(
                "primary: dev hit@10 must improve over BM25 baseline",
                "secondary: train/dev not_found_top50 count should decrease",
                "guard: final test metrics remain not_run",
            ),
            matcher=lambda case: _has_tag(case, "gold_doc_not_found_within_top50")
            or int(case.get("gold_document_token_count") or 0) >= 1000,
        ),
        _CandidateSpec(
            candidate_id="query_view_ablation_full_title_dedup",
            name="Query view ablation",
            category="query_construction",
            status="recommended_for_train_dev_experiment",
            risk_level="medium",
            implementation_readiness=0.85,
            rationale=(
                "Most misses use long queries. A train/dev-only ablation can "
                "compare the current full question query with title-only and "
                "deduplicated lexical query views before any runtime change."
            ),
            stage77_test_plan=(
                "Evaluate fixed query views on frozen train/dev splits only.",
                "Do not learn query rewrites from test data.",
                "Report per-route gains and regressions for dev."
            ),
            target_metric_contract=(
                "primary: dev hit@10 must improve without large hit@1 loss",
                "secondary: long-query miss count should decrease",
                "guard: query views must be deterministic and text-derived only",
            ),
            matcher=lambda case: str(case.get("query_length_bucket")) == (
                "unique_terms_16_plus"
            )
            or _has_tag(case, "top1_query_overlap_exceeds_gold"),
        ),
        _CandidateSpec(
            candidate_id="fielded_title_text_bm25_score_fusion",
            name="Fielded title/text BM25 score fusion",
            category="fielded_lexical_scoring",
            status="recommended_for_train_dev_experiment",
            risk_level="medium",
            implementation_readiness=0.7,
            rationale=(
                "Current BM25 concatenates title and body. Separate title and "
                "body scoring can reduce cases where high-overlap decoys outrank "
                "the answer document and can test title weighting without using "
                "gold labels at runtime."
            ),
            stage77_test_plan=(
                "Build a fixed fielded lexical scorer for train/dev evaluation.",
                "Sweep a small predefined title-weight grid on train and validate "
                "the chosen setting on dev.",
                "Keep answer labels out of runtime scoring features.",
            ),
            target_metric_contract=(
                "primary: dev hit@10 must improve over BM25 baseline",
                "secondary: top1_query_overlap_exceeds_gold count should decrease",
                "guard: selected field weights must be reported before any test use",
            ),
            matcher=lambda case: _has_tag(case, "top1_query_overlap_exceeds_gold")
            or int(case.get("gold_title_query_overlap_count") or 0) > 0,
        ),
        _CandidateSpec(
            candidate_id="bm25_k1_b_grid_train_to_dev",
            name="BM25 k1/b grid",
            category="lexical_parameter_grid",
            status="recommended_for_train_dev_experiment",
            risk_level="low",
            implementation_readiness=0.95,
            rationale=(
                "Near misses at ranks 11-50 may move into top10 from a small, "
                "predeclared BM25 parameter grid. This is cheap and should be "
                "measured before larger retrieval changes."
            ),
            stage77_test_plan=(
                "Run a predeclared k1/b grid on train.",
                "Validate the selected setting on dev only.",
                "Report near-miss transitions and regressions.",
            ),
            target_metric_contract=(
                "primary: dev hit@10 must improve over BM25 baseline",
                "secondary: rank_11_to_50 miss count should decrease",
                "guard: grid values must be fixed before the run",
            ),
            matcher=lambda case: str(case.get("gold_rank_bucket")) in {
                "rank_11_to_20",
                "rank_21_to_50",
            },
        ),
        _CandidateSpec(
            candidate_id="dense_sparse_rrf_train_dev_probe",
            name="Dense+sparse RRF retrieval probe",
            category="hybrid_retrieval",
            status="recommended_for_train_dev_experiment",
            risk_level="high",
            implementation_readiness=0.55,
            rationale=(
                "Dense+sparse RRF may help lexical mismatch cases that are not "
                "found by BM25 within top50, but it has model/cache cost and must "
                "be kept behind cheaper lexical and section-level probes."
            ),
            stage77_test_plan=(
                "Only run if local model/cache availability is confirmed.",
                "Evaluate frozen train/dev splits; do not download or change "
                "external model choices silently.",
                "Compare against BM25 and section BM25 before any defaultization.",
            ),
            target_metric_contract=(
                "primary: dev hit@10 must improve over BM25 baseline",
                "secondary: not_found_top50 and low-overlap miss counts should decrease",
                "guard: model identity and cache status must be recorded",
            ),
            matcher=lambda case: _has_tag(case, "gold_doc_not_found_within_top50")
            or _has_tag(case, "gold_doc_query_overlap_ratio_lt_0_25")
            or _has_tag(case, "gold_doc_low_query_overlap_lte_1")
            or _has_tag(case, "gold_doc_zero_query_overlap"),
        ),
        _CandidateSpec(
            candidate_id="source_doc_ids_oracle_union_blocked",
            name="Source DOC_IDS oracle union",
            category="blocked_diagnostic",
            status="blocked_from_train_dev_experiment",
            risk_level="blocked",
            implementation_readiness=0.0,
            rationale=(
                "Stage75 shows the gold document is present in source candidate "
                "DOC_IDS for the miss cases, but those DOC_IDS are dataset source "
                "metadata, not a runtime user-query signal. Using them would make "
                "retrieval evidence non-deployable and misleading."
            ),
            stage77_test_plan=(
                "Do not implement as a retrieval candidate.",
                "Use only as diagnostic evidence that the corpus contains the "
                "answer document.",
            ),
            target_metric_contract=(
                "blocked: not eligible for train/dev tuning",
                "blocked: not eligible for runtime defaultization",
            ),
            matcher=lambda case: bool(case.get("gold_in_source_candidate_doc_ids")),
        ),
    )


def _candidate_design(
    *,
    spec: _CandidateSpec,
    miss_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matched_cases = [case for case in miss_cases if spec.matcher(case)]
    target_by_split = Counter(str(case.get("split") or "unknown") for case in matched_cases)
    rank_buckets = Counter(str(case.get("gold_rank_bucket") or "unknown") for case in matched_cases)
    routes = Counter(str(case.get("question_route") or "unknown") for case in matched_cases)
    reason_tags = Counter(
        tag
        for case in matched_cases
        for tag in _reason_tags(case)
    )
    priority_score = _priority_score(
        spec=spec,
        target_count=len(matched_cases),
        dev_target_count=target_by_split.get("dev", 0),
    )
    return {
        "candidate_id": spec.candidate_id,
        "name": spec.name,
        "category": spec.category,
        "status": spec.status,
        "risk_level": spec.risk_level,
        "implementation_readiness": spec.implementation_readiness,
        "priority_score": priority_score,
        "target_miss_count": len(matched_cases),
        "target_miss_count_by_split": dict(sorted(target_by_split.items())),
        "target_rank_buckets": _counter_dict(rank_buckets),
        "target_routes": _counter_dict(routes),
        "target_reason_tags": _counter_dict(reason_tags),
        "rationale": spec.rationale,
        "stage77_test_plan": list(spec.stage77_test_plan),
        "target_metric_contract": list(spec.target_metric_contract),
    }


def _priority_score(
    *,
    spec: _CandidateSpec,
    target_count: int,
    dev_target_count: int,
) -> int:
    if spec.status != "recommended_for_train_dev_experiment":
        return 0
    risk_penalty = {
        "low": 0,
        "medium": 8,
        "high": 22,
    }.get(spec.risk_level, 30)
    readiness_bonus = round(spec.implementation_readiness * 20)
    return max(0, target_count + (dev_target_count * 2) + readiness_bonus - risk_penalty)


def _stage75_summary(stage75_report: Mapping[str, Any]) -> dict[str, Any]:
    split_reports = stage75_report.get("split_reports") or {}
    cross_split = stage75_report.get("cross_split_summary") or {}
    return {
        "train": _split_summary(split_reports.get("train") or {}),
        "dev": _split_summary(split_reports.get("dev") or {}),
        "cross_split": {
            "evaluated_questions": int(cross_split.get("evaluated_questions") or 0),
            "hit_at_top_k": float(cross_split.get("hit_at_top_k") or 0.0),
            "miss_count": int(cross_split.get("miss_count") or 0),
            "miss_rate": float(cross_split.get("miss_rate") or 0.0),
        },
    }


def _split_summary(split_report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evaluated_questions": int(split_report.get("evaluated_questions") or 0),
        "hit_at_top_k": float(split_report.get("hit_at_top_k") or 0.0),
        "miss_count": int(split_report.get("miss_count") or 0),
        "miss_rate": float(split_report.get("miss_rate") or 0.0),
    }


def _miss_driver_summary(stage75_report: Mapping[str, Any]) -> dict[str, Any]:
    cross_split = stage75_report.get("cross_split_summary") or {}
    return {
        "reason_tag_counts": _sorted_mapping(cross_split.get("reason_tag_counts") or {}),
        "gold_rank_bucket_counts": _sorted_mapping(
            cross_split.get("gold_rank_bucket_counts") or {}
        ),
        "route_miss_counts": _sorted_mapping(cross_split.get("route_miss_counts") or {}),
        "top_reason_tags": cross_split.get("top_reason_tags") or [],
        "hypotheses": cross_split.get("hypotheses") or [],
    }


def _guard_checks(*, stage75_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    split_contract = stage75_report.get("split_contract") or {}
    decision = stage75_report.get("decision") or {}
    source_guard_checks = stage75_report.get("guard_checks") or []
    source_guard_map = {
        str(check.get("name")): bool(check.get("passed"))
        for check in source_guard_checks
        if isinstance(check, Mapping)
    }
    observed_stage = str(stage75_report.get("stage") or "")
    observed_splits = sorted(str(split) for split in split_contract.get("development_splits") or [])
    forbidden_splits = {
        str(split) for split in split_contract.get("forbidden_final_splits") or []
    }
    return [
        _check(
            name="source_report_is_stage75",
            passed=observed_stage == _SOURCE_STAGE,
            observed=observed_stage,
            expected=_SOURCE_STAGE,
        ),
        _check(
            name="source_development_splits_are_train_dev_only",
            passed=observed_splits == sorted(_ALLOWED_DEVELOPMENT_SPLITS),
            observed=observed_splits,
            expected=sorted(_ALLOWED_DEVELOPMENT_SPLITS),
        ),
        _check(
            name="source_forbidden_final_splits_include_test",
            passed="test" in forbidden_splits,
            observed=sorted(forbidden_splits),
            expected=["test"],
        ),
        _check(
            name="source_candidate_rows_have_no_test_split",
            passed=source_guard_map.get("candidate_rows_have_no_test_split") is True,
            observed=source_guard_map.get("candidate_rows_have_no_test_split"),
            expected=True,
        ),
        _check(
            name="source_final_test_metrics_not_run",
            passed=decision.get("can_run_final_test_metrics_now") is False,
            observed=decision.get("can_run_final_test_metrics_now"),
            expected=False,
        ),
        _check(
            name="source_default_runtime_policy_unchanged",
            passed=decision.get("default_runtime_policy") == "unchanged",
            observed=decision.get("default_runtime_policy"),
            expected="unchanged",
        ),
        _check(
            name="stage76_design_only_no_runtime_default_change",
            passed=True,
            observed="design_only",
            expected="design_only",
        ),
        _check(
            name="stage76_uses_public_safe_stage75_report_only",
            passed=True,
            observed="no raw question/answer/document text required",
            expected="no raw question/answer/document text required",
        ),
    ]


def _decision(
    guard_checks: Sequence[Mapping[str, Any]],
    allowed_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    passed = all(bool(check.get("passed")) for check in guard_checks)
    status = (
        "primeqa_hybrid_retrieval_recall_candidate_design_completed"
        if passed
        else "primeqa_hybrid_retrieval_recall_candidate_design_blocked"
    )
    recommended_next_stage = (
        "Stage 77: run a train/dev-only retrieval-recall experiment for the "
        "highest-priority allowed candidate; keep test locked and do not run "
        "final metrics."
    )
    return {
        "status": status,
        "allowed_candidate_count": len(allowed_candidates),
        "blocked_candidate_count": len(_candidate_specs()) - len(allowed_candidates),
        "can_continue_train_dev_development": passed and bool(allowed_candidates),
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": recommended_next_stage,
    }


def _collect_miss_cases(stage75_report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    split_reports = stage75_report.get("split_reports") or {}
    miss_cases = []
    for split in _ALLOWED_DEVELOPMENT_SPLITS:
        split_report = split_reports.get(split) or {}
        raw_cases = split_report.get("miss_cases") or []
        if not isinstance(raw_cases, list):
            raise ValueError(f"Stage75 split report {split!r} has non-list miss_cases")
        miss_cases.extend(case for case in raw_cases if isinstance(case, Mapping))
    return miss_cases


def _candidate_priority_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    candidates = [
        candidate
        for candidate in report["candidate_designs"]
        if candidate["status"] == "recommended_for_train_dev_experiment"
    ]
    ordered = sorted(
        candidates,
        key=lambda candidate: (-candidate["priority_score"], candidate["candidate_id"]),
    )
    return [
        BarDatum(
            label=candidate["candidate_id"],
            value=float(candidate["priority_score"]),
            value_label=str(candidate["priority_score"]),
        )
        for candidate in ordered
    ]


def _candidate_target_bars(
    report: Mapping[str, Any],
    *,
    split: str | None,
) -> list[BarDatum]:
    candidates = [
        candidate
        for candidate in report["candidate_designs"]
        if candidate["status"] == "recommended_for_train_dev_experiment"
    ]
    values = []
    for candidate in candidates:
        if split is None:
            value = int(candidate["target_miss_count"])
        else:
            value = int(candidate["target_miss_count_by_split"].get(split, 0))
        values.append((candidate, value))
    ordered = sorted(values, key=lambda item: (-item[1], item[0]["candidate_id"]))
    return [
        BarDatum(
            label=candidate["candidate_id"],
            value=float(value),
            value_label=str(value),
        )
        for candidate, value in ordered
    ]


def _allowed_vs_blocked_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counts = Counter(
        "allowed"
        if candidate["status"] == "recommended_for_train_dev_experiment"
        else "blocked"
        for candidate in report["candidate_designs"]
    )
    return [
        BarDatum(label=label, value=float(count), value_label=str(count))
        for label, count in sorted(counts.items())
    ]


def _load_json_object(path: Path) -> dict[str, Any]:
    _ensure_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


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


def _check(
    *,
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _has_tag(case: Mapping[str, Any], tag: str) -> bool:
    return tag in _reason_tags(case)


def _reason_tags(case: Mapping[str, Any]) -> list[str]:
    tags = case.get("reason_tags") or []
    if not isinstance(tags, list):
        return []
    return [str(tag) for tag in tags]


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _sorted_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value[key] for key in sorted(value)}
