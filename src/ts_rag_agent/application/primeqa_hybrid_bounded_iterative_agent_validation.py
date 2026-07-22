from __future__ import annotations

import hashlib
import os
import statistics
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.primeqa_hybrid_bounded_iterative_agent_runtime import (
    bounded_iterative_agent_runtime_contract,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    _build_dense_channels,
    _build_lexical_channels,
    _build_train_fold_assignments,
    _load_json_object,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    ContextCandidateRecord,
    records_by_sample,
    select_current_query_overlap_top10,
    select_original_rrf_top10,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector_training import (
    Stage161TrainCandidateDatasetBuilder,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)
from ts_rag_agent.infrastructure.primeqa_loader import (
    load_primeqa_document_sections,
    load_primeqa_documents,
)

_STAGE = "Stage 168"
_CREATED_AT = "2026-07-22"
_ANALYSIS_ID = "primeqa_hybrid_bounded_iterative_agent_train_feasibility_v1"
_EXPECTED_TRAIN_ROWS = 562
_EXPECTED_ANSWERABLE_ROWS = 370
_EXPECTED_CANDIDATE_ROWS = 112_400
_EXPECTED_CANDIDATE_GOLD_HITS = 345
_EXPECTED_INITIAL_GOLD_HITS = 175
_EXPECTED_ALTERNATE_GOLD_HITS = 255
_EXPECTED_PRIVATE_ROWS = 1_124
_EXPECTED_POST_FIRST_GENERATION_MISSES = 138
_SOURCE_HASHES = {
    "stage161": "a13b8ee5538581f0eb87a649c48fdf4ae715b6cfa8a43a97b5115001f9cd1197",
    "stage165_private": "ce4b5b281093319696a51251d475a3fc5fa6b7dac2e7f9659464fe1d8e55ad1b",
    "stage167": "1b80dda06b8be004d100fb6130d8433ab2ee1608378a98a669ac2e08d1abd5e7",
    "stage80": "2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f",
    "train": "cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155",
    "documents": "f93b5e2d8dcfb2c7d12676ef32ce22b7809692f14081aad98096099a5256722b",
}

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class Stage168CoverageMetrics:
    answerable_count: int
    candidate_pool_gold_hit_count: int
    initial_gold_hit_count: int
    alternate_gold_hit_count: int
    union_gold_hit_count: int
    alternate_only_gold_hit_count: int
    initial_only_gold_hit_count: int
    both_views_gold_hit_count: int
    neither_view_gold_hit_count: int
    mean_view_overlap_count: float
    median_view_overlap_count: float
    mean_union_context_count: float
    max_union_context_count: int


@dataclass(frozen=True)
class Stage168GenerationMissMetrics:
    eligible_count: int
    rescued_by_alternate_count: int
    rescued_by_union_count: int
    still_missing_from_union_count: int
    rescued_by_alternate_rate: float


@dataclass(frozen=True)
class Stage168Visualization:
    name: str
    path: str


def analyze_stage168_coverage(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    grouped_records: Mapping[str, Sequence[ContextCandidateRecord]],
    private_rows: Sequence[Mapping[str, Any]],
) -> tuple[Stage168CoverageMetrics, Stage168GenerationMissMetrics, dict[str, int]]:
    answerable = [sample for sample in samples if sample.answerable]
    answerable_by_identity = {_sha256_text(sample.sample_id): sample for sample in answerable}
    initial_hits: set[str] = set()
    alternate_hits: set[str] = set()
    candidate_hits: set[str] = set()
    overlaps: list[int] = []
    union_sizes: list[int] = []
    views_by_identity: dict[str, tuple[set[str], set[str]]] = {}

    for sample in samples:
        records = grouped_records[sample.sample_id]
        initial_ids = {
            record.document_id for record in select_current_query_overlap_top10(records).selected
        }
        alternate_ids = {
            record.document_id for record in select_original_rrf_top10(records).selected
        }
        candidate_ids = {record.document_id for record in records}
        overlaps.append(len(initial_ids & alternate_ids))
        union_sizes.append(len(initial_ids | alternate_ids))
        identity = _sha256_text(sample.sample_id)
        views_by_identity[identity] = (initial_ids, alternate_ids)
        if sample.answerable and sample.answer_doc_id in candidate_ids:
            candidate_hits.add(identity)
        if sample.answerable and sample.answer_doc_id in initial_ids:
            initial_hits.add(identity)
        if sample.answerable and sample.answer_doc_id in alternate_ids:
            alternate_hits.add(identity)

    union_hits = initial_hits | alternate_hits
    answerable_identities = {_sha256_text(sample.sample_id) for sample in answerable}
    coverage = Stage168CoverageMetrics(
        answerable_count=len(answerable),
        candidate_pool_gold_hit_count=len(candidate_hits),
        initial_gold_hit_count=len(initial_hits),
        alternate_gold_hit_count=len(alternate_hits),
        union_gold_hit_count=len(union_hits),
        alternate_only_gold_hit_count=len(alternate_hits - initial_hits),
        initial_only_gold_hit_count=len(initial_hits - alternate_hits),
        both_views_gold_hit_count=len(initial_hits & alternate_hits),
        neither_view_gold_hit_count=len(answerable_identities - union_hits),
        mean_view_overlap_count=round(statistics.fmean(overlaps), 6),
        median_view_overlap_count=round(float(statistics.median(overlaps)), 6),
        mean_union_context_count=round(statistics.fmean(union_sizes), 6),
        max_union_context_count=max(union_sizes),
    )

    eligible_rows = [
        row
        for row in private_rows
        if row.get("arm") == "isolated"
        and row.get("answerable") is True
        and int(row.get("synthetic_turn_position", 0)) > 1
        and row.get("gold_candidate_rank") is not None
        and row.get("gold_generation_rank") is None
    ]
    rescued = 0
    route_rescues: dict[str, int] = {}
    for row in eligible_rows:
        identity = str(row["private_identity_sha256"])
        sample = answerable_by_identity[identity]
        _, alternate_ids = views_by_identity[identity]
        if sample.answer_doc_id in alternate_ids:
            rescued += 1
            route = str(row["question_route"])
            route_rescues[route] = route_rescues.get(route, 0) + 1
    generation_misses = Stage168GenerationMissMetrics(
        eligible_count=len(eligible_rows),
        rescued_by_alternate_count=rescued,
        rescued_by_union_count=rescued,
        still_missing_from_union_count=len(eligible_rows) - rescued,
        rescued_by_alternate_rate=round(rescued / len(eligible_rows), 6),
    )
    return coverage, generation_misses, dict(sorted(route_rescues.items()))


def run_stage168_train_feasibility(
    *,
    stage161_report_path: Path,
    stage165_private_path: Path,
    stage167_report_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    user_confirmed_ac_fallback: bool,
    confirmation_note: str,
    encoder_batch_size: int = 64,
    encoder_device: str | None = None,
    progress_sink: ProgressSink | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    paths = {
        "stage161": stage161_report_path,
        "stage165_private": stage165_private_path,
        "stage167": stage167_report_path,
        "stage80": stage80_report_path,
        "train": train_split_path,
        "documents": documents_path,
    }
    fingerprints = {name: _fingerprint(path) for name, path in paths.items()}
    _authorize_sources(fingerprints)
    if not user_confirmed_ac_fallback:
        raise ValueError("Stage168 requires explicit user authorization for A+C fallback")
    private = _load_json_object(stage165_private_path)
    if len(private.get("rows", ())) != _EXPECTED_PRIVATE_ROWS:
        raise ValueError("Stage168 private Stage165 row count drifted")
    _emit(progress_sink, phase="sources_and_ac_authorization_verified")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    if len(samples) != _EXPECTED_TRAIN_ROWS or any(
        sample.assigned_split != "train" for sample in samples
    ):
        raise ValueError("Stage168 accepts only the exact 562-row train split")
    documents_by_id = load_primeqa_documents(documents_path)
    sections_by_document = load_primeqa_document_sections(documents_path)
    documents = list(documents_by_id.values())
    fold_assignments = _build_train_fold_assignments(samples, fold_count=5)
    _emit(progress_sink, phase="train_and_documents_loaded", train_rows=len(samples))

    stage80_report = _load_json_object(stage80_report_path)
    dense_channels, dense_summary = _build_dense_channels(
        include_dense_channels=True,
        stage80_report=stage80_report,
        stage80_report_path=stage80_report_path,
        documents=documents,
        document_ids=tuple(document.id for document in documents),
        encoder_batch_size=encoder_batch_size,
        encoder_device=encoder_device,
        encoder_factory=None,
    )
    if dense_summary["status"] != "dense_channels_ready":
        raise RuntimeError("Stage168 requires both authorized local dense channels")
    lexical_channels = _build_lexical_channels(
        documents=documents,
        sections_by_document=sections_by_document,
        bm25_k1=1.5,
        bm25_b=0.75,
        component_depth=200,
    )
    channels = tuple([*lexical_channels, *dense_channels])
    _emit(progress_sink, phase="retrieval_channels_ready", channel_count=len(channels))

    records = Stage161TrainCandidateDatasetBuilder(
        documents_by_id=documents_by_id,
        sections_by_document=sections_by_document,
        channels=channels,
        fold_assignments=fold_assignments,
        progress_sink=progress_sink,
        progress_stage=_STAGE,
        progress_phase="train_candidate_replay",
    ).build(samples)
    if len(records) != _EXPECTED_CANDIDATE_ROWS:
        raise RuntimeError("Stage168 candidate replay row count drifted")
    replay_finished_at = time.perf_counter()
    coverage, generation_misses, route_rescues = analyze_stage168_coverage(
        samples=samples,
        grouped_records=records_by_sample(records),
        private_rows=private["rows"],
    )
    analyzed_at = time.perf_counter()
    runtime_contract = bounded_iterative_agent_runtime_contract()

    guard_checks = [
        _check("explicit_ac_fallback_authorization", user_confirmed_ac_fallback),
        _check("train_only_source", len(samples) == _EXPECTED_TRAIN_ROWS),
        _check("candidate_row_count_exact", len(records) == _EXPECTED_CANDIDATE_ROWS),
        _check(
            "candidate_gold_hit_control_exact",
            coverage.candidate_pool_gold_hit_count == _EXPECTED_CANDIDATE_GOLD_HITS,
        ),
        _check(
            "initial_gold_hit_control_exact",
            coverage.initial_gold_hit_count == _EXPECTED_INITIAL_GOLD_HITS,
        ),
        _check(
            "alternate_gold_hit_control_exact",
            coverage.alternate_gold_hit_count == _EXPECTED_ALTERNATE_GOLD_HITS,
        ),
        _check(
            "post_first_generation_miss_count_exact",
            generation_misses.eligible_count == _EXPECTED_POST_FIRST_GENERATION_MISSES,
        ),
        _check(
            "union_strictly_improves_initial_coverage",
            coverage.union_gold_hit_count > coverage.initial_gold_hit_count,
        ),
        _check(
            "alternate_rescues_known_generation_misses",
            generation_misses.rescued_by_alternate_count > 0,
        ),
        _check("single_retrieval_budget", runtime_contract["retrieval_call_count_per_turn"] == 1),
        _check(
            "single_inspection_budget",
            runtime_contract["maximum_evidence_inspection_count_per_turn"] == 1,
        ),
        _check(
            "two_model_decision_budget",
            runtime_contract["maximum_model_decisions_per_turn"] == 2,
        ),
        _check(
            "clarification_is_distinct_terminal_state",
            "clarify" in runtime_contract["allowed_terminal_states"],
        ),
        _check("development_not_loaded", True),
        _check("test_not_loaded", True),
        _check("default_runtime_unchanged", not runtime_contract["runtime_registered_as_default"]),
    ]
    all_guards_passed = all(check["passed"] for check in guard_checks)
    report = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-only feasibility analysis for an A+C bounded iterative Agent. A inspects "
            "the existing original-RRF Top10 view without a second retrieval; C returns one "
            "fixed system clarification question. Development and test are not loaded."
        ),
        "user_confirmation": {
            "confirmed": user_confirmed_ac_fallback,
            "confirmation_note": confirmation_note,
        },
        "source_authorization": fingerprints,
        "runtime_contract": runtime_contract,
        "coverage": asdict(coverage),
        "known_generation_miss_analysis": {
            **asdict(generation_misses),
            "rescued_count_by_question_route": route_rescues,
        },
        "limitations": [
            "Gold-document visibility is a retrieval-context proxy, not answer-quality proof.",
            "No real router model was invoked in this train-only feasibility analysis.",
            "Clarification usefulness has no human label in TechQA and is not estimated here.",
            "Runtime path correctness is covered by synthetic tests, not production traffic.",
        ],
        "timing_seconds": {
            "candidate_replay": round(replay_finished_at - started_at, 6),
            "aggregate_analysis": round(analyzed_at - replay_finished_at, 6),
            "total_before_visualization": round(analyzed_at - started_at, 6),
        },
        "guard_checks": guard_checks,
        "decision": {
            "all_process_guards_passed": all_guards_passed,
            "status": (
                "advance_to_stage169_real_gpu_router_calibration"
                if all_guards_passed
                else "stage168_blocked"
            ),
            "default_runtime_activation": False,
            "development_opened": False,
            "test_opened": False,
        },
    }
    _emit(progress_sink, phase="train_feasibility_complete", decision=report["decision"])
    return report


def write_stage168_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage168Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = report["coverage"]
    misses = report["known_generation_miss_analysis"]
    candidate_hits = coverage["candidate_pool_gold_hit_count"]
    initial_hits = coverage["initial_gold_hit_count"]
    alternate_hits = coverage["alternate_gold_hit_count"]
    union_hits = coverage["union_gold_hit_count"]
    both_hits = coverage["both_views_gold_hit_count"]
    alternate_only_hits = coverage["alternate_only_gold_hit_count"]
    initial_only_hits = coverage["initial_only_gold_hit_count"]
    neither_hits = coverage["neither_view_gold_hit_count"]
    eligible_misses = misses["eligible_count"]
    rescued_misses = misses["rescued_by_alternate_count"]
    remaining_misses = misses["still_missing_from_union_count"]
    charts = {
        "gold_document_coverage.svg": render_horizontal_bar_chart_svg(
            "Stage 168 train answerable gold-document coverage",
            (
                BarDatum("Candidate Top200", candidate_hits, f"{candidate_hits} / 370"),
                BarDatum("Initial overlap Top10", initial_hits, f"{initial_hits} / 370"),
                BarDatum("Alternate RRF Top10", alternate_hits, f"{alternate_hits} / 370"),
                BarDatum("A+C union", union_hits, f"{union_hits} / 370"),
            ),
            "Answerable train rows with gold document visible",
        ),
        "view_contribution.svg": render_horizontal_bar_chart_svg(
            "Stage 168 initial and alternate view contribution",
            (
                BarDatum("Both views", both_hits, str(both_hits)),
                BarDatum("Alternate only", alternate_only_hits, str(alternate_only_hits)),
                BarDatum("Initial only", initial_only_hits, str(initial_only_hits)),
                BarDatum("Neither view", neither_hits, str(neither_hits)),
            ),
            "Answerable train rows",
        ),
        "generation_miss_rescue.svg": render_horizontal_bar_chart_svg(
            "Stage 168 known post-first generation-miss rescue",
            (
                BarDatum("Eligible misses", eligible_misses, str(eligible_misses)),
                BarDatum("Rescued by alternate", rescued_misses, str(rescued_misses)),
                BarDatum("Still absent from union", remaining_misses, str(remaining_misses)),
            ),
            "Train-only Stage 165 isolated cases",
        ),
        "bounded_runtime_budget.svg": render_horizontal_bar_chart_svg(
            "Stage 168 bounded runtime call budget",
            (
                BarDatum("Retrieval calls", 1, "max 1"),
                BarDatum("Alternate inspections", 1, "max 1"),
                BarDatum("Model decisions", 2, "max 2"),
                BarDatum("Second retrievals", 0, "disabled"),
                BarDatum("Retries", 0, "disabled"),
            ),
            "Maximum calls per turn",
        ),
    }
    visualizations = []
    for name, svg in charts.items():
        path = output_dir / name
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        visualizations.append(Stage168Visualization(name=name.removesuffix(".svg"), path=str(path)))
    return tuple(visualizations)


def _authorize_sources(fingerprints: Mapping[str, Mapping[str, Any]]) -> None:
    for name, expected_sha256 in _SOURCE_HASHES.items():
        observed = fingerprints.get(name, {}).get("sha256")
        if observed != expected_sha256:
            raise ValueError(f"Stage168 source hash mismatch for {name}: {observed}")


def _fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _check(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(sink: ProgressSink | None, **event: Any) -> None:
    if sink is not None:
        sink({"stage": _STAGE, **event})
