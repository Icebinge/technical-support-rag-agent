from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 84"
_CREATED_AT = "2026-07-15"
_SPLIT_NAME = "primeqa_hybrid_stage68_v1"
_PROTOCOL_VERSION = "primeqa_hybrid_split_v1"
_ALLOWED_DEVELOPMENT_SPLITS = ("train", "dev")
_FORBIDDEN_FINAL_SPLITS = frozenset({"test"})
_RECOMMENDED_ROUTE = "second_wave_retrieval_candidate_design"
_BLOCKED_CANDIDATE_ID = "source_doc_ids_oracle_union_blocked"


@dataclass(frozen=True)
class PrimeQAHybridSecondWaveRetrievalCandidateDesignVisualization:
    """One generated Stage84 second-wave retrieval candidate design visualization."""

    name: str
    path: str


@dataclass(frozen=True)
class _SecondWaveCandidateSpec:
    candidate_id: str
    name: str
    category: str
    status: str
    risk_level: str
    implementation_readiness: float
    prior_signal_key: str
    rationale: str
    stage85_protocol_outline: tuple[str, ...]
    target_metric_contract: tuple[str, ...]
    runtime_evidence_policy: tuple[str, ...]
    matcher: Callable[[Mapping[str, Any]], bool]


def design_primeqa_hybrid_second_wave_retrieval_candidates(
    *,
    stage75_report_path: Path,
    stage76_report_path: Path,
    stage77_report_path: Path,
    stage78_report_path: Path,
    stage79_report_path: Path,
    stage80_report_path: Path,
    stage81_report_path: Path,
    stage82_report_path: Path,
    stage83_report_path: Path,
    user_confirmed_route: bool,
    confirmation_note: str,
) -> dict[str, Any]:
    """Design second-wave train/dev-only retrieval candidates from saved reports."""

    started_at = time.perf_counter()
    reports = {
        "stage75": _load_json_object(stage75_report_path),
        "stage76": _load_json_object(stage76_report_path),
        "stage77": _load_json_object(stage77_report_path),
        "stage78": _load_json_object(stage78_report_path),
        "stage79": _load_json_object(stage79_report_path),
        "stage80": _load_json_object(stage80_report_path),
        "stage81": _load_json_object(stage81_report_path),
        "stage82": _load_json_object(stage82_report_path),
        "stage83": _load_json_object(stage83_report_path),
    }
    loaded_at = time.perf_counter()
    miss_cases = _collect_miss_cases(reports["stage75"])
    first_wave_summary = _first_wave_summary(reports)
    prior_route_evidence = _prior_route_evidence(reports)
    candidate_designs = [
        _candidate_design(
            spec=spec,
            miss_cases=miss_cases,
            prior_route_evidence=prior_route_evidence,
        )
        for spec in _candidate_specs()
    ]
    recommended_candidates = [
        candidate
        for candidate in candidate_designs
        if candidate["status"] == "recommended_for_train_dev_protocol_design"
    ]
    execution_order = [
        candidate["candidate_id"]
        for candidate in sorted(
            recommended_candidates,
            key=lambda candidate: (
                -candidate["priority_score"],
                candidate["risk_level"],
                candidate["candidate_id"],
            ),
        )
    ]
    guard_checks = _guard_checks(
        reports=reports,
        candidate_designs=candidate_designs,
        user_confirmed_route=user_confirmed_route,
    )
    checked_at = time.perf_counter()
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "design_scope": (
            "Train/dev-only second-wave retrieval candidate design for "
            "primeqa_hybrid_stage68_v1. This stage consumes saved public-safe "
            "Stage75-Stage83 reports, does not load the frozen test split, does "
            "not run new retrieval metrics, does not run final metrics, does not "
            "use source DOC_IDS as runtime retrieval evidence, and does not "
            "change runtime defaults."
        ),
        "user_confirmation": {
            "route_id": _RECOMMENDED_ROUTE,
            "confirmed": bool(user_confirmed_route),
            "confirmation_note": confirmation_note,
        },
        "split_contract": {
            "split_name": _SPLIT_NAME,
            "protocol_version": _PROTOCOL_VERSION,
            "development_splits": list(_ALLOWED_DEVELOPMENT_SPLITS),
            "forbidden_final_splits": sorted(_FORBIDDEN_FINAL_SPLITS),
        },
        "source_files": {
            "stage75_report": _fingerprint(stage75_report_path),
            "stage76_report": _fingerprint(stage76_report_path),
            "stage77_report": _fingerprint(stage77_report_path),
            "stage78_report": _fingerprint(stage78_report_path),
            "stage79_report": _fingerprint(stage79_report_path),
            "stage80_report": _fingerprint(stage80_report_path),
            "stage81_report": _fingerprint(stage81_report_path),
            "stage82_report": _fingerprint(stage82_report_path),
            "stage83_report": _fingerprint(stage83_report_path),
        },
        "stage75_miss_summary": _stage75_miss_summary(reports["stage75"], miss_cases),
        "first_wave_summary": first_wave_summary,
        "prior_route_evidence": prior_route_evidence,
        "candidate_designs": candidate_designs,
        "recommended_execution_order": execution_order,
        "blocked_items": [
            candidate
            for candidate in candidate_designs
            if candidate["status"] == "blocked_from_train_dev_experiment"
        ],
        "guard_checks": guard_checks,
        "decision": _decision(
            guard_checks=guard_checks,
            recommended_execution_order=execution_order,
        ),
        "timing_seconds": {
            "load_reports": round(loaded_at - started_at, 3),
            "design_and_guard": round(checked_at - loaded_at, 3),
            "total": round(checked_at - started_at, 3),
        },
    }


def write_primeqa_hybrid_second_wave_retrieval_candidate_design_visualizations(
    report: Mapping[str, Any],
    output_dir: Path,
) -> list[PrimeQAHybridSecondWaveRetrievalCandidateDesignVisualization]:
    """Write SVG charts for Stage84 second-wave retrieval candidate design."""

    output_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "stage84_second_wave_candidate_priority_scores.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage84 second-wave candidate priority",
                bars=_candidate_priority_bars(report),
                x_label="priority score",
                width=1240,
                margin_left=520,
            )
        ),
        "stage84_second_wave_candidate_target_misses.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage84 second-wave candidate target misses",
                bars=_candidate_target_bars(report, split=None),
                x_label="train/dev miss cases matched",
                width=1240,
                margin_left=520,
            )
        ),
        "stage84_second_wave_candidate_dev_targets.svg": render_horizontal_bar_chart_svg(
            title="Stage84 second-wave candidate dev targets",
            bars=_candidate_target_bars(report, split="dev"),
            x_label="dev miss cases matched",
            width=1240,
            margin_left=520,
        ),
        "stage84_second_wave_candidate_prior_signal_scores.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage84 second-wave prior signal scores",
                bars=_candidate_prior_signal_bars(report),
                x_label="prior signal score",
                width=1240,
                margin_left=520,
            )
        ),
        "stage84_second_wave_allowed_vs_blocked_candidates.svg": (
            render_horizontal_bar_chart_svg(
                title="Stage84 second-wave allowed vs blocked",
                bars=_allowed_vs_blocked_bars(report),
                x_label="candidate count",
                margin_left=320,
            )
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(
            PrimeQAHybridSecondWaveRetrievalCandidateDesignVisualization(
                name=filename,
                path=str(path),
            )
        )
    return artifacts


def _candidate_specs() -> tuple[_SecondWaveCandidateSpec, ...]:
    return (
        _SecondWaveCandidateSpec(
            candidate_id="structured_query_keyphrase_compaction_design",
            name="Structured query keyphrase compaction design",
            category="query_construction",
            status="recommended_for_train_dev_protocol_design",
            risk_level="medium",
            implementation_readiness=0.72,
            prior_signal_key="query_view_ablation",
            rationale=(
                "Stage75 misses are dominated by long queries, while Stage77 shows "
                "that simple title-only or de-duplicated query replacement is not "
                "stable. The second wave should design a structured query "
                "compaction protocol that preserves high-signal product, error, "
                "version, and action terms instead of deleting terms wholesale."
            ),
            stage85_protocol_outline=(
                "Define deterministic keyphrase buckets before any metric run.",
                "Select candidate query views on train only.",
                "Validate the train-selected view on dev without using test.",
                "Report changed sample IDs and rank transitions without raw text.",
            ),
            target_metric_contract=(
                "primary: train-selected dev hit@10 must improve over BM25 baseline",
                "secondary: top10 regression count must be lower than improvement count",
                "guard: no query view may be selected by dev-only performance",
            ),
            runtime_evidence_policy=(
                "May use runtime question text and deterministic token features.",
                "Must not use answer document IDs, gold labels, or source DOC_IDS.",
            ),
            matcher=lambda case: _is_long_query(case)
            or _has_tag(case, "top1_query_overlap_exceeds_gold"),
        ),
        _SecondWaveCandidateSpec(
            candidate_id="selective_dense_sparse_low_overlap_gate_design",
            name="Selective dense+sparse low-overlap gate design",
            category="hybrid_retrieval_gate",
            status="recommended_for_train_dev_protocol_design",
            risk_level="high",
            implementation_readiness=0.58,
            prior_signal_key="dense_sparse_rrf",
            rationale=(
                "Stage81 dense+sparse RRF improved train hit@10 and reduced dev "
                "not-found@50, but the train-selected challenger still regressed "
                "dev hit@10. A second-wave protocol should design a train-only "
                "runtime-observable gate for lexical low-overlap or not-found-like "
                "cases before any dense route can be retried."
            ),
            stage85_protocol_outline=(
                "Reuse only already-confirmed local dense caches.",
                "Define runtime-observable gate features on train before dev validation.",
                "Compare gated dense+sparse retrieval against BM25 on train/dev only.",
                "Block the route if top10 regressions dominate on dev.",
            ),
            target_metric_contract=(
                "primary: train-selected gated policy must improve dev hit@10",
                "secondary: dev not-found@50 should decrease without hit@1 collapse",
                "guard: no downloads and no dev-selected gate thresholds",
            ),
            runtime_evidence_policy=(
                "May use query tokens, candidate scores, overlap counts, and local dense scores.",
                "Must not use source DOC_IDS, answer document IDs, or test labels.",
            ),
            matcher=lambda case: _has_tag(case, "gold_doc_not_found_within_top50")
            or _has_tag(case, "gold_doc_query_overlap_ratio_lt_0_25")
            or _has_tag(case, "gold_doc_low_query_overlap_lte_1")
            or _has_tag(case, "gold_doc_zero_query_overlap"),
        ),
        _SecondWaveCandidateSpec(
            candidate_id="section_signal_guarded_expansion_design",
            name="Section signal guarded expansion design",
            category="section_signal_gate",
            status="recommended_for_train_dev_protocol_design",
            risk_level="medium",
            implementation_readiness=0.64,
            prior_signal_key="section_bm25",
            rationale=(
                "Stage79 section rollup regressed overall but rescued a small set "
                "of deep-rank misses. A second-wave protocol should preserve the "
                "lesson as a gated section signal design rather than repeating an "
                "ungated section replacement."
            ),
            stage85_protocol_outline=(
                "Profile Stage79 section improvements and regressions from train only.",
                "Define a fixed section-signal promotion contract before dev validation.",
                "Validate only the train-selected contract on dev.",
                "Report top10 and search-depth transitions separately.",
            ),
            target_metric_contract=(
                "primary: dev hit@10 must improve over BM25 baseline",
                "secondary: search-depth improvements must exceed regressions",
                "guard: section signal must not demote existing BM25 top10 hits by default",
            ),
            runtime_evidence_policy=(
                "May use runtime section BM25 scores and document BM25 scores.",
                "Must not use gold answer rank, source DOC_IDS, or test labels.",
            ),
            matcher=lambda case: _has_tag(case, "gold_doc_not_found_within_top50")
            or int(case.get("gold_document_token_count") or 0) >= 1000,
        ),
        _SecondWaveCandidateSpec(
            candidate_id="score_margin_bm25_normalization_gate_design",
            name="Score-margin BM25 normalization gate design",
            category="lexical_parameter_gate",
            status="recommended_for_train_dev_protocol_design",
            risk_level="medium",
            implementation_readiness=0.7,
            prior_signal_key="bm25_k1_b_grid",
            rationale=(
                "Stage82 showed b=0.95 configs with better dev hit@10, but those "
                "configs were not train-selected and cannot be chosen from dev. "
                "The only safe next use of that evidence is a train-only design "
                "for score-margin or length-normalization gates."
            ),
            stage85_protocol_outline=(
                "Define score-margin and document-length proxy features on train.",
                "Select any adaptive BM25 normalization rule on train only.",
                "Validate the selected rule on dev without using dev to choose b.",
                "Keep the fixed Stage82 grid values as historical evidence only.",
            ),
            target_metric_contract=(
                "primary: train-selected rule must improve dev hit@10",
                "secondary: rank 11-50 near misses should decrease",
                "guard: dev-only b=0.95 observations cannot select a runtime rule",
            ),
            runtime_evidence_policy=(
                "May use BM25 scores, document length, and candidate rank features.",
                "Must not use source DOC_IDS, answer document IDs, or dev-only selection.",
            ),
            matcher=lambda case: str(case.get("gold_rank_bucket")) in {
                "rank_11_to_20",
                "rank_21_to_50",
            }
            or _has_tight_score_margin(case),
        ),
        _SecondWaveCandidateSpec(
            candidate_id="lexical_cluster_diversity_rerank_design",
            name="Lexical cluster diversity rerank design",
            category="candidate_diversity",
            status="recommended_for_train_dev_protocol_design",
            risk_level="medium",
            implementation_readiness=0.62,
            prior_signal_key="fielded_title_bm25",
            rationale=(
                "Stage75 shows many misses where top-ranked decoys have equal or "
                "higher lexical overlap than the answer document. Stage78 also "
                "suggests title signal can improve hit@1/MRR without improving "
                "hit@10. A second-wave candidate should design a diversity-aware "
                "rerank that separates near-duplicate lexical clusters before "
                "choosing top10 documents."
            ),
            stage85_protocol_outline=(
                "Define public-safe lexical cluster features from candidate scores.",
                "Choose any diversity penalty or cluster cap on train only.",
                "Validate the selected design on dev.",
                "Report regressions caused by cluster suppression explicitly.",
            ),
            target_metric_contract=(
                "primary: dev hit@10 must improve over BM25 baseline",
                "secondary: top1-overlap decoy misses should decrease",
                "guard: no title/body text should be written to reports",
            ),
            runtime_evidence_policy=(
                "May use runtime candidate scores, overlap counts, "
                "title-overlap counts, and ranks.",
                "Must not use source DOC_IDS, answer document IDs, or raw "
                "document text in reports.",
            ),
            matcher=lambda case: _has_tag(case, "top1_query_overlap_exceeds_gold")
            or _has_title_plateau(case)
            or _has_tight_score_margin(case),
        ),
        _SecondWaveCandidateSpec(
            candidate_id=_BLOCKED_CANDIDATE_ID,
            name="Source DOC_IDS oracle union",
            category="blocked_diagnostic",
            status="blocked_from_train_dev_experiment",
            risk_level="blocked",
            implementation_readiness=0.0,
            prior_signal_key="source_doc_ids_blocked",
            rationale=(
                "Source DOC_IDS explain that the answer document exists in source "
                "metadata, but they are not runtime user-query evidence. The "
                "second wave must not reintroduce this oracle route."
            ),
            stage85_protocol_outline=(
                "Do not implement as a retrieval candidate.",
                "Use only as diagnostic evidence that the corpus contains the answer document.",
            ),
            target_metric_contract=(
                "blocked: not eligible for train/dev tuning",
                "blocked: not eligible for runtime defaultization",
            ),
            runtime_evidence_policy=(
                "Forbidden: source DOC_IDS are not runtime retrieval evidence.",
            ),
            matcher=lambda case: bool(case.get("gold_in_source_candidate_doc_ids")),
        ),
    )


def _candidate_design(
    *,
    spec: _SecondWaveCandidateSpec,
    miss_cases: Sequence[Mapping[str, Any]],
    prior_route_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    matched_cases = [case for case in miss_cases if spec.matcher(case)]
    split_counts = Counter(str(case.get("split") or "unknown") for case in matched_cases)
    rank_buckets = Counter(str(case.get("gold_rank_bucket") or "unknown") for case in matched_cases)
    routes = Counter(str(case.get("question_route") or "unknown") for case in matched_cases)
    reason_tags = Counter(tag for case in matched_cases for tag in _reason_tags(case))
    prior_signal = prior_route_evidence.get(spec.prior_signal_key, {})
    priority_score = _priority_score(
        spec=spec,
        target_count=len(matched_cases),
        dev_target_count=split_counts.get("dev", 0),
        prior_signal_score=float(prior_signal.get("signal_score") or 0.0),
    )
    return {
        "candidate_id": spec.candidate_id,
        "name": spec.name,
        "category": spec.category,
        "status": spec.status,
        "risk_level": spec.risk_level,
        "implementation_readiness": spec.implementation_readiness,
        "prior_signal_key": spec.prior_signal_key,
        "prior_signal_score": float(prior_signal.get("signal_score") or 0.0),
        "priority_score": priority_score,
        "target_miss_count": len(matched_cases),
        "target_miss_count_by_split": dict(sorted(split_counts.items())),
        "target_rank_buckets": _counter_dict(rank_buckets),
        "target_routes": _counter_dict(routes),
        "target_reason_tags": _counter_dict(reason_tags),
        "rationale": spec.rationale,
        "stage85_protocol_outline": list(spec.stage85_protocol_outline),
        "target_metric_contract": list(spec.target_metric_contract),
        "runtime_evidence_policy": list(spec.runtime_evidence_policy),
    }


def _priority_score(
    *,
    spec: _SecondWaveCandidateSpec,
    target_count: int,
    dev_target_count: int,
    prior_signal_score: float,
) -> int:
    if spec.status != "recommended_for_train_dev_protocol_design":
        return 0
    risk_penalty = {
        "low": 0,
        "medium": 10,
        "high": 24,
    }.get(spec.risk_level, 40)
    readiness_bonus = round(spec.implementation_readiness * 24)
    prior_signal_bonus = round(prior_signal_score * 36)
    return max(
        0,
        target_count
        + (dev_target_count * 2)
        + readiness_bonus
        + prior_signal_bonus
        - risk_penalty,
    )


def _stage75_miss_summary(
    stage75_report: Mapping[str, Any],
    miss_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cross_split = stage75_report.get("cross_split_summary") or {}
    return {
        "evaluated_questions": int(cross_split.get("evaluated_questions") or 0),
        "hit_at_top_k": float(cross_split.get("hit_at_top_k") or 0.0),
        "miss_count": int(cross_split.get("miss_count") or len(miss_cases)),
        "miss_count_by_split": _counter_dict(
            Counter(str(case.get("split") or "unknown") for case in miss_cases)
        ),
        "rank_bucket_counts": _counter_dict(
            Counter(str(case.get("gold_rank_bucket") or "unknown") for case in miss_cases)
        ),
        "route_miss_counts": _counter_dict(
            Counter(str(case.get("question_route") or "unknown") for case in miss_cases)
        ),
        "query_length_bucket_counts": _counter_dict(
            Counter(str(case.get("query_length_bucket") or "unknown") for case in miss_cases)
        ),
        "reason_tag_counts": _counter_dict(
            Counter(tag for case in miss_cases for tag in _reason_tags(case))
        ),
        "source_doc_ids_oracle_presence_count": sum(
            bool(case.get("gold_in_source_candidate_doc_ids")) for case in miss_cases
        ),
    }


def _first_wave_summary(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    stage83 = reports["stage83"]
    return {
        "stage76_allowed_candidates_exhausted": bool(
            (stage83.get("decision") or {}).get("stage76_allowed_candidates_exhausted")
        ),
        "runtime_advancing_candidate_count": int(
            (stage83.get("decision") or {}).get("runtime_advancing_candidate_count") or 0
        ),
        "candidate_outcomes": stage83.get("candidate_outcomes") or [],
        "dev_only_observations": stage83.get("dev_only_observations") or [],
        "blocked_candidate": stage83.get("blocked_candidate") or {},
    }


def _prior_route_evidence(
    reports: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    stage77_decision = reports["stage77"].get("decision") or {}
    stage78_decision = reports["stage78"].get("decision") or {}
    stage79_decision = reports["stage79"].get("decision") or {}
    stage81_decision = reports["stage81"].get("decision") or {}
    stage82_decision = reports["stage82"].get("decision") or {}
    stage81_selected = _selected_comparison(
        reports["stage81"],
        split="dev",
        selected_id=stage81_decision.get("selected_config_id"),
    )
    stage82_best_dev = _best_dev_comparison(reports["stage82"])
    return {
        "query_view_ablation": {
            "source_stage": "Stage 77",
            "signal_score": 0.35,
            "observed": {
                "selected_id": stage77_decision.get("train_selected_view_id"),
                "selected_dev_hit10_delta": stage77_decision.get(
                    "train_selected_dev_hit10_delta"
                ),
                "selected_dev_top10_net": _net(
                    stage77_decision.get("train_selected_dev_top10_improvements"),
                    stage77_decision.get("train_selected_dev_top10_regressions"),
                ),
            },
            "lesson": (
                "Simple title-only or de-duplicated query replacement regressed; "
                "a second-wave query route must be structured and train-selected."
            ),
        },
        "fielded_title_bm25": {
            "source_stage": "Stage 78",
            "signal_score": 0.5,
            "observed": {
                "selected_id": stage78_decision.get("train_selected_config_id"),
                "selected_dev_hit10_delta": stage78_decision.get(
                    "train_selected_dev_hit10_delta"
                ),
                "selected_dev_top10_net": _net(
                    stage78_decision.get("train_selected_dev_top10_improvements"),
                    stage78_decision.get("train_selected_dev_top10_regressions"),
                ),
                "selected_dev_mrr_delta": (
                    (
                        reports["stage78"]
                        .get("train_selection", {})
                        .get("selected_dev_comparison_to_baseline", {})
                    ).get("mrr_delta")
                ),
            },
            "lesson": (
                "Small title weight did not improve dev hit@10, but title signal "
                "still improved dev hit@1/MRR; use it only in a more constrained "
                "candidate-diversity design."
            ),
        },
        "section_bm25": {
            "source_stage": "Stage 79",
            "signal_score": 0.45,
            "observed": {
                "selected_id": stage79_decision.get("candidate_config_id"),
                "selected_dev_hit10_delta": stage79_decision.get(
                    "candidate_dev_hit10_delta"
                ),
                "selected_dev_top10_net": _net(
                    stage79_decision.get("candidate_dev_top10_improvements"),
                    stage79_decision.get("candidate_dev_top10_regressions"),
                ),
                "dev_search_depth_net": (
                    (reports["stage79"].get("comparisons_to_baseline") or {})
                    .get("dev", {})
                    .get("search_depth_net_improvement_count")
                ),
            },
            "lesson": (
                "Ungated section rollup regressed, but it produced a few deep-rank "
                "rescues; the signal should be redesigned as a guarded feature."
            ),
        },
        "dense_sparse_rrf": {
            "source_stage": "Stage 81",
            "signal_score": 0.68,
            "observed": {
                "selected_id": stage81_decision.get("selected_config_id"),
                "selected_dev_hit10_delta": stage81_decision.get(
                    "selected_dev_hit10_delta"
                ),
                "selected_dev_top10_net": _net(
                    stage81_decision.get("selected_dev_top10_improvements"),
                    stage81_decision.get("selected_dev_top10_regressions"),
                ),
                "selected_dev_not_found_delta": stage81_decision.get(
                    "selected_dev_not_found_at_search_depth_delta"
                ),
                "selected_dev_search_depth_net": stage81_selected.get(
                    "search_depth_net_improvement_count"
                ),
            },
            "lesson": (
                "Dense+sparse retrieval reduced not-found@50 but regressed top10; "
                "the next dense route must be selective and train-gated."
            ),
        },
        "bm25_k1_b_grid": {
            "source_stage": "Stage 82",
            "signal_score": 0.48,
            "observed": {
                "selected_id": stage82_decision.get("selected_config_id"),
                "selected_dev_hit10_delta": stage82_decision.get(
                    "selected_dev_hit10_delta"
                ),
                "best_dev_non_selected_config_id": stage82_best_dev.get("config_id"),
                "best_dev_non_selected_hit10_delta": stage82_best_dev.get(
                    "hit10_delta"
                ),
                "not_selectable_reason": stage82_best_dev.get("not_selectable_reason"),
            },
            "lesson": (
                "BM25 normalization has a dev-only signal, but any second-wave use "
                "must be train-selected and cannot pick b=0.95 from dev."
            ),
        },
        "source_doc_ids_blocked": {
            "source_stage": "Stage 83",
            "signal_score": 0.0,
            "observed": reports["stage83"].get("blocked_candidate") or {},
            "lesson": "Source DOC_IDS remain blocked runtime evidence.",
        },
    }


def _guard_checks(
    *,
    reports: Mapping[str, Mapping[str, Any]],
    candidate_designs: Sequence[Mapping[str, Any]],
    user_confirmed_route: bool,
) -> list[dict[str, Any]]:
    expected_stages = {
        "stage75": "Stage 75",
        "stage76": "Stage 76",
        "stage77": "Stage 77",
        "stage78": "Stage 78",
        "stage79": "Stage 79",
        "stage80": "Stage 80",
        "stage81": "Stage 81",
        "stage82": "Stage 82",
        "stage83": "Stage 83",
    }
    allowed_candidate_ids = [
        str(candidate.get("candidate_id"))
        for candidate in candidate_designs
        if candidate.get("status") == "recommended_for_train_dev_protocol_design"
    ]
    decisions = {
        key: report.get("decision") or {}
        for key, report in reports.items()
    }
    return [
        _check(
            name="source_reports_are_expected_stages",
            passed=all(
                str(reports[key].get("stage") or "") == expected
                for key, expected in expected_stages.items()
            ),
            observed={key: reports[key].get("stage") for key in expected_stages},
            expected=expected_stages,
        ),
        _check(
            name="user_confirmed_stage84_recommended_route",
            passed=user_confirmed_route,
            observed=user_confirmed_route,
            expected=True,
        ),
        _check(
            name="stage83_recommended_route_matches_stage84",
            passed=decisions["stage83"].get("recommended_next_route_option")
            == _RECOMMENDED_ROUTE,
            observed=decisions["stage83"].get("recommended_next_route_option"),
            expected=_RECOMMENDED_ROUTE,
        ),
        _check(
            name="stage83_required_confirmation_was_respected",
            passed=decisions["stage83"].get("requires_user_confirmation_before_next_route")
            is True
            and user_confirmed_route,
            observed={
                "stage83_requires_confirmation": decisions["stage83"].get(
                    "requires_user_confirmation_before_next_route"
                ),
                "user_confirmed_route": user_confirmed_route,
            },
            expected=True,
        ),
        _check(
            name="stage76_candidates_are_exhausted",
            passed=decisions["stage83"].get("stage76_allowed_candidates_exhausted")
            is True,
            observed=decisions["stage83"].get("stage76_allowed_candidates_exhausted"),
            expected=True,
        ),
        _check(
            name="stage83_has_no_runtime_advancing_candidate",
            passed=int(decisions["stage83"].get("runtime_advancing_candidate_count") or 0)
            == 0,
            observed=decisions["stage83"].get("runtime_advancing_candidate_count"),
            expected=0,
        ),
        _check(
            name="source_doc_ids_candidate_not_reintroduced",
            passed=_BLOCKED_CANDIDATE_ID not in allowed_candidate_ids,
            observed=allowed_candidate_ids,
            expected=f"exclude {_BLOCKED_CANDIDATE_ID}",
        ),
        _check(
            name="source_doc_ids_candidate_remains_blocked",
            passed=(
                (reports["stage83"].get("blocked_candidate") or {}).get("status")
                == "blocked_from_train_dev_experiment"
            ),
            observed=(reports["stage83"].get("blocked_candidate") or {}).get("status"),
            expected="blocked_from_train_dev_experiment",
        ),
        _check(
            name="all_source_decisions_keep_final_test_locked",
            passed=all(
                decision.get("can_run_final_test_metrics_now") is False
                for decision in decisions.values()
            ),
            observed={
                key: decision.get("can_run_final_test_metrics_now")
                for key, decision in decisions.items()
            },
            expected=False,
        ),
        _check(
            name="all_source_decisions_forbid_test_tuning",
            passed=all(
                decision.get("can_use_test_for_tuning") is False
                for decision in decisions.values()
            ),
            observed={
                key: decision.get("can_use_test_for_tuning")
                for key, decision in decisions.items()
            },
            expected=False,
        ),
        _check(
            name="all_source_decisions_keep_runtime_defaults_unchanged",
            passed=all(
                decision.get("default_runtime_policy") == "unchanged"
                for decision in decisions.values()
            ),
            observed={
                key: decision.get("default_runtime_policy")
                for key, decision in decisions.items()
            },
            expected="unchanged",
        ),
        _check(
            name="stage84_design_only_no_new_retrieval_metrics",
            passed=True,
            observed="design_only",
            expected="design_only",
        ),
        _check(
            name="stage84_final_test_metrics_not_run",
            passed=True,
            observed="not_run",
            expected="not_run",
        ),
        _check(
            name="stage84_default_runtime_policy_unchanged",
            passed=True,
            observed="unchanged",
            expected="unchanged",
        ),
    ]


def _decision(
    *,
    guard_checks: Sequence[Mapping[str, Any]],
    recommended_execution_order: Sequence[str],
) -> dict[str, Any]:
    failed_checks = [str(check["name"]) for check in guard_checks if not check["passed"]]
    if failed_checks:
        return {
            "status": "primeqa_hybrid_second_wave_retrieval_candidate_design_blocked",
            "failed_checks": failed_checks,
            "can_continue_train_dev_development": False,
            "requires_user_confirmation_before_train_dev_run": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        }
    recommended_next_candidate_id = (
        recommended_execution_order[0] if recommended_execution_order else None
    )
    return {
        "status": "primeqa_hybrid_second_wave_retrieval_candidate_design_completed",
        "recommended_next_candidate_id": recommended_next_candidate_id,
        "recommended_execution_order": list(recommended_execution_order),
        "requires_user_confirmation_before_train_dev_run": True,
        "can_continue_train_dev_development": bool(recommended_execution_order),
        "can_open_final_test_gate_now": False,
        "can_run_final_test_metrics_now": False,
        "can_use_test_for_tuning": False,
        "default_runtime_policy": "unchanged",
        "recommended_next_stage": (
            "Stage 85: confirm and freeze the train/dev-only protocol for the "
            f"recommended second-wave candidate {recommended_next_candidate_id}; "
            "keep test locked and do not run final metrics."
        ),
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


def _selected_comparison(
    report: Mapping[str, Any],
    *,
    split: str,
    selected_id: Any,
) -> Mapping[str, Any]:
    comparisons = (report.get("comparisons_to_baseline") or {}).get(split) or {}
    if not isinstance(comparisons, Mapping):
        return {}
    value = comparisons.get(str(selected_id))
    return value if isinstance(value, Mapping) else {}


def _best_dev_comparison(report: Mapping[str, Any]) -> dict[str, Any]:
    selected_id = (report.get("decision") or {}).get("selected_config_id")
    comparisons = (report.get("comparisons_to_baseline") or {}).get("dev") or {}
    if not isinstance(comparisons, Mapping):
        return {}
    baseline_metrics = (
        (report.get("metrics_by_split") or {})
        .get("dev", {})
        .get("full_document_bm25_baseline", {})
    )
    baseline_hit10 = (baseline_metrics.get("hit_at_k") or {}).get("hit@10")
    best: dict[str, Any] = {}
    for config_id, comparison in comparisons.items():
        if config_id == selected_id or not isinstance(comparison, Mapping):
            continue
        hit10_delta = comparison.get("hit@10_delta")
        if hit10_delta is None:
            continue
        if not best or float(hit10_delta) > float(best.get("hit10_delta") or -999):
            best = {
                "config_id": config_id,
                "hit10_delta": round(float(hit10_delta), 4),
                "baseline_dev_hit10": baseline_hit10,
                "not_selectable_reason": (
                    "Not selected by the train-only rule; using it would be dev-set selection."
                ),
            }
    return best


def _candidate_priority_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=candidate["candidate_id"],
            value=float(candidate["priority_score"]),
            value_label=str(candidate["priority_score"]),
        )
        for candidate in _ordered_recommended_candidates(report)
    ]


def _candidate_target_bars(
    report: Mapping[str, Any],
    *,
    split: str | None,
) -> list[BarDatum]:
    values = []
    for candidate in _recommended_candidates(report):
        if split is None:
            value = int(candidate["target_miss_count"])
        else:
            value = int(candidate["target_miss_count_by_split"].get(split, 0))
        values.append((candidate, value))
    return [
        BarDatum(
            label=candidate["candidate_id"],
            value=float(value),
            value_label=str(value),
        )
        for candidate, value in sorted(
            values,
            key=lambda item: (-item[1], item[0]["candidate_id"]),
        )
    ]


def _candidate_prior_signal_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    return [
        BarDatum(
            label=candidate["candidate_id"],
            value=float(candidate["prior_signal_score"]),
            value_label=f"{candidate['prior_signal_score']:.2f}",
        )
        for candidate in _ordered_recommended_candidates(report)
    ]


def _allowed_vs_blocked_bars(report: Mapping[str, Any]) -> list[BarDatum]:
    counts = Counter(
        "allowed"
        if candidate["status"] == "recommended_for_train_dev_protocol_design"
        else "blocked"
        for candidate in report["candidate_designs"]
    )
    return [
        BarDatum(label=label, value=float(count), value_label=str(count))
        for label, count in sorted(counts.items())
    ]


def _ordered_recommended_candidates(
    report: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    return sorted(
        _recommended_candidates(report),
        key=lambda candidate: (-candidate["priority_score"], candidate["candidate_id"]),
    )


def _recommended_candidates(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        candidate
        for candidate in report["candidate_designs"]
        if candidate["status"] == "recommended_for_train_dev_protocol_design"
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


def _is_long_query(case: Mapping[str, Any]) -> bool:
    return (
        str(case.get("query_length_bucket")) == "unique_terms_16_plus"
        or int(case.get("query_unique_token_count") or 0) >= 16
    )


def _has_tight_score_margin(case: Mapping[str, Any]) -> bool:
    top_results = case.get("top_results") or []
    if not isinstance(top_results, list) or len(top_results) < 10:
        return False
    first = top_results[0] if isinstance(top_results[0], Mapping) else {}
    tenth = top_results[9] if isinstance(top_results[9], Mapping) else {}
    return float(first.get("score") or 0.0) - float(tenth.get("score") or 0.0) <= 25.0


def _has_title_plateau(case: Mapping[str, Any]) -> bool:
    gold_title_overlap = int(case.get("gold_title_query_overlap_count") or 0)
    if gold_title_overlap < 3:
        return False
    top_results = case.get("top_results") or []
    if not isinstance(top_results, list):
        return False
    strong_title_count = sum(
        1
        for result in top_results
        if isinstance(result, Mapping)
        and int(result.get("title_query_overlap_count") or 0) >= gold_title_overlap
    )
    return strong_title_count >= 5


def _net(improvements: Any, regressions: Any) -> int:
    return int(improvements or 0) - int(regressions or 0)


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))
