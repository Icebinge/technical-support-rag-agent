from __future__ import annotations

import hashlib
import json
import statistics
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application import primeqa_hybrid_listwise_agent_e2e as stage178
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application.listwise_runtime_reranker import (
    ListwiseUnionPrimaryContextSelectionPolicy,
    PrecomputedListwiseScoreProvider,
)
from ts_rag_agent.application.primeqa_hybrid_agent_runtime_observability import (
    AgentWorkflowObservationSink,
    PublicSafeAgentWorkflowObservationEvent,
)
from ts_rag_agent.application.primeqa_hybrid_high_recall_union_comparison import (
    _build_train_fold_assignments,
)
from ts_rag_agent.application.primeqa_hybrid_optional_sidecar_agent_runtime import (
    PrimeQAHybridProcessRuntimeResourceFactory,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
    QueryOverlapCandidateScoringPolicy,
)
from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg
from ts_rag_agent.domain.dataset import PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
    load_primeqa_hybrid_split_samples,
)

_STAGE = "Stage 179"
_CREATED_AT = "2026-07-22"
_ANALYSIS_ID = "primeqa_hybrid_listwise_agent_failure_attribution_v1"
_EXPECTED_ROWS = 562
_EXPECTED_ANSWERABLE = 370
_EXPECTED_PAIR_SCORES = 9_714
_EXPECTED_FOLDS = 5
_SOURCE_HASHES = {
    "stage178_public": "e57e3f09bcc65657a3f8783e97e6767b690095e2cffd5d252d51e181eaf533c9",
    "stage178_private": "6fffa820773dea8892dc1d441aff1c3ef3df54ff368b82bf1c9a09b961f0857a",
    "stage178_alignment": "e2398024edf128ad0628900d25eb1ccc9c83c437fb474921fe136e2603e47272",
    "stage128": "012ca36c0559f3533ea2e89160fcb3cee7fb12daa89fb68c69dcf27d9d2ce63e",
    "stage125": "dfaee9eb5688a2a91e3f3d3695def5e32ad87494cdfdd31a00a0434df53ccd65",
    "stage80": "2441bb1cb1e7888299d3f57962b18cd59df84e2086ac281105abcacfc144880f",
    "train": "cabd93e0b972c47384c4bf5cc2cd215a7fc519b2df4f81fba61db73c931aa155",
}
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answer_doc_id",
        "candidate_doc_ids",
        "document_id",
        "gold_answer",
        "pair_identity",
        "question_text",
        "sample_id",
        "scores",
    }
)

ProgressSink = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class Stage179DiagnosticRow:
    fold_id: str
    union_gold_hit: bool
    prefix_gold_hit: bool
    baseline_context_hit: bool
    candidate_context_hit: bool
    baseline_gold_cited: bool
    candidate_gold_cited: bool
    baseline_f1: float
    candidate_f1: float
    candidate_gold_context_rank: int | None

    @property
    def f1_delta(self) -> float:
        return self.candidate_f1 - self.baseline_f1


@dataclass(frozen=True)
class Stage179Visualization:
    name: str
    path: str


class _ValidatingObservationSink(AgentWorkflowObservationSink):
    def __init__(self) -> None:
        self.event_count = 0

    def emit(self, event: PublicSafeAgentWorkflowObservationEvent) -> None:
        event.to_public_dict()
        self.event_count += 1


def run_stage179_failure_attribution(
    *,
    stage178_public_path: Path,
    stage178_private_path: Path,
    stage178_alignment_path: Path,
    stage128_protocol_path: Path,
    stage125_protocol_path: Path,
    stage80_report_path: Path,
    train_split_path: Path,
    documents_path: Path,
    encoder_batch_size: int = 64,
    progress_sink: ProgressSink | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    source_paths = {
        "stage178_public": stage178_public_path,
        "stage178_private": stage178_private_path,
        "stage178_alignment": stage178_alignment_path,
        "stage128": stage128_protocol_path,
        "stage125": stage125_protocol_path,
        "stage80": stage80_report_path,
        "train": train_split_path,
    }
    fingerprints = {
        name: stage173._resolved_fingerprint(path) for name, path in source_paths.items()
    }
    mismatches = [
        name
        for name, fingerprint in fingerprints.items()
        if fingerprint["sha256"] != _SOURCE_HASHES[name]
    ]
    if mismatches:
        raise ValueError(f"Stage179 source authorization failed: {mismatches}")
    public = _load_json(stage178_public_path)
    private = _load_json(stage178_private_path)
    alignment = _load_json(stage178_alignment_path)
    _authorize_stage178(public=public, private=private, alignment=alignment)
    authorized_at = time.perf_counter()
    _emit(progress_sink, phase="sources_authorized")

    samples = load_primeqa_hybrid_split_samples(train_split_path)
    if len(samples) != _EXPECTED_ROWS or any(row.assigned_split != "train" for row in samples):
        raise ValueError("Stage179 accepts only the exact train split")
    if sum(row.answerable for row in samples) != _EXPECTED_ANSWERABLE:
        raise ValueError("Stage179 answerable row count drifted")
    fold_assignments = _build_train_fold_assignments(samples, fold_count=_EXPECTED_FOLDS)
    loaded_at = time.perf_counter()

    resource_factory = PrimeQAHybridProcessRuntimeResourceFactory(
        stage128_protocol_path=stage128_protocol_path,
        stage125_protocol_path=stage125_protocol_path,
        stage80_report_path=stage80_report_path,
        documents_path=documents_path,
        encoder_batch_size=encoder_batch_size,
        encoder_device="cpu",
    )
    resources = resource_factory.build_shared()
    resources_at = time.perf_counter()
    provider = PrecomputedListwiseScoreProvider(private["scores"])
    baseline_sink = _ValidatingObservationSink()
    candidate_sink = _ValidatingObservationSink()
    baseline_workflow = stage178._workflow(
        candidate_pool_retriever=resources.candidate_pool_retriever,
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(),
        observation_sink=baseline_sink,
    )
    candidate_workflow = stage178._workflow(
        candidate_pool_retriever=resources.candidate_pool_retriever,
        observation_adapter=PrimeQAHybridSidecarObservationAdapter(
            primary_context_selection_policy=ListwiseUnionPrimaryContextSelectionPolicy(
                score_provider=provider
            )
        ),
        observation_sink=candidate_sink,
    )
    traces, diagnostic_rows = _replay_paired_agent(
        samples=samples,
        fold_assignments=fold_assignments,
        baseline_workflow=baseline_workflow,
        candidate_workflow=candidate_workflow,
        progress_sink=progress_sink,
    )
    replayed_at = time.perf_counter()
    attribution = analyze_diagnostic_rows(diagnostic_rows)
    reproduced = stage178.evaluate_agent_e2e(traces)
    finished_at = time.perf_counter()
    report: dict[str, Any] = {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "analysis_id": _ANALYSIS_ID,
        "analysis_scope": (
            "Train-only deterministic replay attributing the Stage178A listwise Agent "
            "result across prefix, two-view union, selected context, citation, and answer F1."
        ),
        "source_authorization": fingerprints,
        "frozen_protocol": {
            "selection_or_training_run": False,
            "quality_source": "stage178a_five_fold_oof_pair_logits",
            "fold_count": _EXPECTED_FOLDS,
            "development_and_test_closed": True,
            "context_to_citation_bottleneck_rule": (
                "context_gain_uncited_count >= context_gain_candidate_cited_count"
            ),
            "answer_content_bottleneck_rule": ("candidate_gold_cited_and_f1_worsened_count > 0"),
            "reranker_instability_rule": "context_hit_to_miss_count > 0",
            "fallback_strategy_enabled": False,
            "runtime_registered_as_default": False,
        },
        "attribution": attribution,
        "stage178_metric_reproduction": {
            "profiles": reproduced["profiles"],
            "deltas": reproduced["deltas"],
            "folds": reproduced["folds"],
            "changed_verified_answer_count": reproduced["changed_verified_answer_count"],
        },
        "runtime": {
            "resource_factory_build_count": resource_factory.build_count,
            "baseline_observation_event_count": baseline_sink.event_count,
            "candidate_observation_event_count": candidate_sink.event_count,
            "precomputed_score_provider": asdict(provider.counters()),
            "baseline_workflow_counters": asdict(baseline_workflow.counters()),
            "candidate_workflow_counters": asdict(candidate_workflow.counters()),
        },
        "timing_seconds": {
            "source_authorization": round(authorized_at - started_at, 6),
            "load_train_and_folds": round(loaded_at - authorized_at, 6),
            "runtime_resource_build": round(resources_at - loaded_at, 6),
            "paired_agent_replay": round(replayed_at - resources_at, 6),
            "attribution_and_validation": round(finished_at - replayed_at, 6),
            "wall": round(finished_at - started_at, 6),
        },
        "execution_boundaries": {
            "train_loaded": True,
            "development_loaded": False,
            "test_loaded": False,
            "model_fit_count": 0,
            "checkpoint_loaded": False,
            "agent_turn_count": sum(len(rows) for rows in traces.values()),
            "retry_action_count": sum(
                row.retry_action_count for rows in traces.values() for row in rows
            ),
            "fallback_action_count": sum(
                row.fallback_action_count for rows in traces.values() for row in rows
            ),
            "runtime_registered_as_default": False,
        },
    }
    forbidden = sorted(_forbidden_keys_found(report))
    report["public_safe_contract"] = {
        "forbidden_keys": sorted(_FORBIDDEN_PUBLIC_KEYS),
        "forbidden_keys_found": forbidden,
    }
    report["process_guards"] = _process_guards(
        report=report,
        public=public,
        private=private,
        diagnostic_rows=diagnostic_rows,
        forbidden=forbidden,
    )
    all_guards = all(row["passed"] for row in report["process_guards"])
    report["decision"] = {
        "status": (
            "stage179_failure_attribution_completed"
            if all_guards
            else "stage179_failure_attribution_invalid"
        ),
        "primary_bottleneck": attribution["diagnostic_decision"]["primary_bottleneck"],
        "recommended_next_direction": attribution["diagnostic_decision"][
            "recommended_next_direction"
        ],
        "stage178b_authorized": False,
        "development_opened": False,
        "test_opened": False,
        "default_runtime_activation": False,
    }
    _emit(progress_sink, phase="analysis_complete", decision=report["decision"])
    return report


def _replay_paired_agent(
    *,
    samples: Sequence[PrimeQAHybridSplitSample],
    fold_assignments: Mapping[str, str],
    baseline_workflow: Any,
    candidate_workflow: Any,
    progress_sink: ProgressSink | None,
) -> tuple[
    dict[str, tuple[stage178.AgentE2ETrace, ...]],
    tuple[Stage179DiagnosticRow, ...],
]:
    traces: dict[str, list[stage178.AgentE2ETrace]] = {"baseline": [], "candidate": []}
    rows = []
    scorer = QueryOverlapCandidateScoringPolicy()
    for index, sample in enumerate(samples, start=1):
        runtime_query = PrimeQARuntimeQuery(
            id=sample.sample_id,
            title=sample.question_title,
            text=sample.question_text,
        )
        question = sample.to_primeqa_question()
        order = (
            ("baseline", "candidate")
            if int(hashlib.sha256(sample.sample_id.encode()).hexdigest(), 16) % 2 == 0
            else ("candidate", "baseline")
        )
        runs = {}
        for arm in order:
            workflow = baseline_workflow if arm == "baseline" else candidate_workflow
            started_at = time.perf_counter()
            run = workflow.run(runtime_query)
            latency = time.perf_counter() - started_at
            public_trace = run.public_safe_trace
            traces[arm].append(
                stage178.AgentE2ETrace(
                    sample=sample,
                    fold_id=fold_assignments[sample.sample_id],
                    question=question,
                    candidate_pool_results=tuple(run.candidate_pool_results),
                    generation_context_results=tuple(run.generation_context_results),
                    verified_answer=run.verified_answer,
                    latency_seconds=latency,
                    tool_call_count=public_trace.tool_call_count,
                    retry_action_count=public_trace.retry_action_count,
                    fallback_action_count=public_trace.fallback_action_count,
                )
            )
            runs[arm] = run
        if sample.answerable and sample.answer_doc_id is not None:
            baseline_trace = traces["baseline"][-1]
            candidate_trace = traces["candidate"][-1]
            prefix = tuple(runs["candidate"].candidate_pool_results[:200])
            query_terms = scorer.query_terms(question)
            overlap = scorer.rank(query_terms=query_terms, candidates=prefix)[:10]
            original = sorted(prefix, key=lambda result: (result.rank, result.document.id))[:10]
            union = tuple({result.document.id: result for result in (*overlap, *original)}.values())
            gold_id = sample.answer_doc_id
            candidate_rank = _document_rank(
                candidate_trace.generation_context_results,
                gold_id,
            )
            rows.append(
                Stage179DiagnosticRow(
                    fold_id=fold_assignments[sample.sample_id],
                    prefix_gold_hit=any(row.document.id == gold_id for row in prefix),
                    union_gold_hit=any(row.document.id == gold_id for row in union),
                    baseline_context_hit=baseline_trace.context_gold_hit,
                    candidate_context_hit=candidate_trace.context_gold_hit,
                    baseline_gold_cited=baseline_trace.gold_cited,
                    candidate_gold_cited=candidate_trace.gold_cited,
                    baseline_f1=baseline_trace.answer_f1,
                    candidate_f1=candidate_trace.answer_f1,
                    candidate_gold_context_rank=candidate_rank,
                )
            )
        if index % 25 == 0 or index == len(samples):
            _emit(
                progress_sink, phase="paired_replay_progress", completed=index, total=len(samples)
            )
    return {name: tuple(values) for name, values in traces.items()}, tuple(rows)


def analyze_diagnostic_rows(rows: Sequence[Stage179DiagnosticRow]) -> dict[str, Any]:
    if len(rows) != _EXPECTED_ANSWERABLE:
        raise ValueError("Stage179 diagnostics require 370 answerable rows")
    context_transitions = Counter(
        _transition(row.baseline_context_hit, row.candidate_context_hit) for row in rows
    )
    citation_transitions = Counter(
        _transition(row.baseline_gold_cited, row.candidate_gold_cited) for row in rows
    )
    context_gain = [
        row for row in rows if not row.baseline_context_hit and row.candidate_context_hit
    ]
    context_loss = [
        row for row in rows if row.baseline_context_hit and not row.candidate_context_hit
    ]
    cited_candidate = [row for row in rows if row.candidate_gold_cited]
    gain_cited = [row for row in context_gain if row.candidate_gold_cited]
    gain_uncited = [row for row in context_gain if not row.candidate_gold_cited]
    cited_f1_worse = [row for row in cited_candidate if row.f1_delta < 0]
    fold_reports = {
        fold_id: _stratum_summary([row for row in rows if row.fold_id == fold_id])
        for fold_id in sorted({row.fold_id for row in rows})
    }
    rank_reports = {
        label: _rank_summary(
            [
                row
                for row in rows
                if row.candidate_gold_context_rank is not None
                and lower <= row.candidate_gold_context_rank <= upper
            ]
        )
        for label, lower, upper in (
            ("rank_1", 1, 1),
            ("rank_2_3", 2, 3),
            ("rank_4_5", 4, 5),
            ("rank_6_10", 6, 10),
        )
    }
    conversion_bottleneck = len(gain_uncited) >= len(gain_cited)
    content_bottleneck = bool(cited_f1_worse)
    reranker_instability = bool(context_loss)
    if conversion_bottleneck:
        primary = "context_to_citation_conversion"
        direction = "design_runtime_visible_citation_aware_composition_oof"
    elif content_bottleneck:
        primary = "cited_evidence_to_answer_content_fidelity"
        direction = "design_runtime_visible_answer_content_fidelity_oof"
    elif reranker_instability:
        primary = "reranker_context_stability"
        direction = "design_listwise_context_stability_constraint_oof"
    else:
        primary = "no_dominant_failure_pattern"
        direction = "stop_stage178_branch"
    return {
        "pipeline_waterfall": {
            "prefix_gold_hit_count": sum(row.prefix_gold_hit for row in rows),
            "union_gold_hit_count": sum(row.union_gold_hit for row in rows),
            "candidate_context_gold_hit_count": sum(row.candidate_context_hit for row in rows),
            "candidate_gold_citation_count": sum(row.candidate_gold_cited for row in rows),
            "candidate_positive_f1_count": sum(row.candidate_f1 > 0 for row in rows),
            "union_to_context_drop_count": sum(row.union_gold_hit for row in rows)
            - sum(row.candidate_context_hit for row in rows),
            "context_to_citation_drop_count": sum(row.candidate_context_hit for row in rows)
            - sum(row.candidate_gold_cited for row in rows),
        },
        "context_transitions": dict(sorted(context_transitions.items())),
        "citation_transitions": dict(sorted(citation_transitions.items())),
        "context_gain_attribution": {
            **_stratum_summary(context_gain),
            "candidate_gold_cited_count": len(gain_cited),
            "candidate_gold_uncited_count": len(gain_uncited),
            "citation_conversion_rate": _ratio(len(gain_cited), len(context_gain)),
        },
        "context_loss_attribution": _stratum_summary(context_loss),
        "candidate_gold_cited_attribution": {
            **_stratum_summary(cited_candidate),
            "f1_worsened_count": len(cited_f1_worse),
        },
        "candidate_gold_rank_attribution": rank_reports,
        "all_answerable_outcomes": _stratum_summary(rows),
        "fold_reports": fold_reports,
        "diagnostic_decision": {
            "context_to_citation_bottleneck": conversion_bottleneck,
            "answer_content_bottleneck": content_bottleneck,
            "reranker_instability": reranker_instability,
            "primary_bottleneck": primary,
            "recommended_next_direction": direction,
            "stage178b_authorized": False,
            "runtime_change_authorized": False,
        },
    }


def _stratum_summary(rows: Sequence[Stage179DiagnosticRow]) -> dict[str, Any]:
    deltas = [row.f1_delta for row in rows]
    return {
        "count": len(rows),
        "f1_improved_count": sum(delta > 0 for delta in deltas),
        "f1_tied_count": sum(delta == 0 for delta in deltas),
        "f1_worsened_count": sum(delta < 0 for delta in deltas),
        "mean_f1_delta": round(statistics.fmean(deltas), 6) if deltas else 0.0,
        "f1_delta_sum": round(sum(deltas), 6),
        "context_gain_count": sum(
            not row.baseline_context_hit and row.candidate_context_hit for row in rows
        ),
        "citation_gain_count": sum(
            not row.baseline_gold_cited and row.candidate_gold_cited for row in rows
        ),
        "citation_loss_count": sum(
            row.baseline_gold_cited and not row.candidate_gold_cited for row in rows
        ),
    }


def _rank_summary(rows: Sequence[Stage179DiagnosticRow]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "gold_cited_count": sum(row.candidate_gold_cited for row in rows),
        "gold_citation_rate": _ratio(sum(row.candidate_gold_cited for row in rows), len(rows)),
        "mean_f1": round(statistics.fmean(row.candidate_f1 for row in rows), 6) if rows else 0.0,
        "f1_worsened_count": sum(row.f1_delta < 0 for row in rows),
    }


def write_stage179_visualizations(
    *, report: Mapping[str, Any], output_dir: Path
) -> tuple[Stage179Visualization, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    attribution = report["attribution"]
    waterfall = attribution["pipeline_waterfall"]
    charts = {
        "pipeline_waterfall.svg": _chart(
            "Stage 179 candidate evidence pipeline",
            tuple(
                BarDatum(name, float(value), str(value))
                for name, value in waterfall.items()
                if not name.endswith("drop_count")
            ),
            "answerable train questions",
        ),
        "context_transitions.svg": _chart(
            "Stage 179 context gold-hit transitions",
            tuple(
                BarDatum(name, float(value), str(value))
                for name, value in attribution["context_transitions"].items()
            ),
            "answerable train questions",
        ),
        "citation_transitions.svg": _chart(
            "Stage 179 gold-citation transitions",
            tuple(
                BarDatum(name, float(value), str(value))
                for name, value in attribution["citation_transitions"].items()
            ),
            "answerable train questions",
        ),
        "context_gain_outcomes.svg": _chart(
            "Stage 179 context-gain F1 outcomes",
            tuple(
                BarDatum(
                    name,
                    float(attribution["context_gain_attribution"][name]),
                    str(attribution["context_gain_attribution"][name]),
                )
                for name in (
                    "f1_improved_count",
                    "f1_tied_count",
                    "f1_worsened_count",
                    "candidate_gold_cited_count",
                    "candidate_gold_uncited_count",
                )
            ),
            "context-gain questions",
        ),
        "fold_f1_delta.svg": _chart(
            "Stage 179 held-out fold F1 deltas",
            tuple(
                BarDatum(fold, values["mean_f1_delta"], f"{values['mean_f1_delta']:+.6f}")
                for fold, values in attribution["fold_reports"].items()
            ),
            "candidate minus baseline F1",
        ),
        "rank_citation_rate.svg": _chart(
            "Stage 179 candidate gold citation rate by context rank",
            tuple(
                BarDatum(rank, values["gold_citation_rate"], f"{values['gold_citation_rate']:.4f}")
                for rank, values in attribution["candidate_gold_rank_attribution"].items()
            ),
            "gold citation rate",
        ),
    }
    written = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        ET.parse(path)
        written.append(Stage179Visualization(filename.removesuffix(".svg"), str(path)))
    return tuple(written)


def _process_guards(
    *,
    report: Mapping[str, Any],
    public: Mapping[str, Any],
    private: Mapping[str, Any],
    diagnostic_rows: Sequence[Stage179DiagnosticRow],
    forbidden: Sequence[str],
) -> list[dict[str, Any]]:
    boundaries = report["execution_boundaries"]
    reproduced = report["stage178_metric_reproduction"]
    expected = public["agent_e2e"]
    return [
        _gate(
            "stage178_status_is_insufficient",
            public["decision"]["status"] == "stage178a_listwise_tool_agent_e2e_insufficient",
        ),
        _gate("exact_private_pair_scores", len(private["scores"]) == _EXPECTED_PAIR_SCORES),
        _gate("exact_answerable_rows", len(diagnostic_rows) == _EXPECTED_ANSWERABLE),
        _gate("five_folds", len(report["attribution"]["fold_reports"]) == _EXPECTED_FOLDS),
        _gate(
            "stage178_profiles_reproduced",
            _core_profiles(reproduced["profiles"]) == _core_profiles(expected["profiles"]),
        ),
        _gate(
            "stage178_deltas_reproduced",
            _core_deltas(reproduced["deltas"]) == _core_deltas(expected["deltas"]),
        ),
        _gate("one_resource_build", report["runtime"]["resource_factory_build_count"] == 1),
        _gate("exact_agent_turns", boundaries["agent_turn_count"] == 1_124),
        _gate("retry_count_zero", boundaries["retry_action_count"] == 0),
        _gate("fallback_count_zero", boundaries["fallback_action_count"] == 0),
        _gate("no_model_fit", boundaries["model_fit_count"] == 0),
        _gate("checkpoint_not_loaded", boundaries["checkpoint_loaded"] is False),
        _gate("development_not_loaded", boundaries["development_loaded"] is False),
        _gate("test_not_loaded", boundaries["test_loaded"] is False),
        _gate("default_runtime_unchanged", boundaries["runtime_registered_as_default"] is False),
        _gate("public_report_safe", not forbidden),
    ]


def _core_profiles(profiles: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "context_gold_hit_count",
        "gold_citation_count",
        "answerable_refusal_count",
        "unanswerable_false_answer_count",
    )
    return {
        arm: {
            **{key: values[key] for key in keys},
            "verified_metrics": values["verified_metrics"],
        }
        for arm, values in profiles.items()
    }


def _core_deltas(deltas: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in deltas.items() if key != "latency_p95_seconds"}


def _authorize_stage178(
    *, public: Mapping[str, Any], private: Mapping[str, Any], alignment: Mapping[str, Any]
) -> None:
    if public.get("decision", {}).get("status") != "stage178a_listwise_tool_agent_e2e_insufficient":
        raise ValueError("Stage178 public report does not authorize failure attribution")
    if (
        private.get("format_id") != "stage178a_oof_pair_logits_v1"
        or private.get("pair_count") != _EXPECTED_PAIR_SCORES
    ):
        raise ValueError("Stage178 private OOF artifact contract drifted")
    if stage178._canonical_sha256(private) != public["private_oof_artifact"]["canonical_sha256"]:
        raise ValueError("Stage178 private OOF canonical hash mismatch")
    if not all(
        alignment.get("decision", {}).get(key) is True
        for key in (
            "full_prefix_contract_exact",
            "selection_surface_exact",
            "live_union_fully_covered_by_stage177_pairs",
        )
    ):
        raise ValueError("corrected runtime alignment is not authorized")


def _forbidden_keys_found(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        found = {str(key) for key in value if str(key) in _FORBIDDEN_PUBLIC_KEYS}
        for child in value.values():
            found.update(_forbidden_keys_found(child))
        return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        found: set[str] = set()
        for child in value:
            found.update(_forbidden_keys_found(child))
        return found
    return set()


def _document_rank(results: Sequence[RetrievalResult], document_id: str) -> int | None:
    return next(
        (index for index, row in enumerate(results, start=1) if row.document.id == document_id),
        None,
    )


def _transition(before: bool, after: bool) -> str:
    return f"{'hit' if before else 'miss'}_to_{'hit' if after else 'miss'}"


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Stage179 source must be a JSON object: {path}")
    return value


def _gate(name: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed)}


def _emit(sink: ProgressSink | None, **event: Any) -> None:
    if sink is not None:
        sink({"stage": _STAGE, **event})


def _chart(title: str, bars: Sequence[BarDatum], x_label: str) -> str:
    return render_horizontal_bar_chart_svg(
        title=title, bars=bars, x_label=x_label, width=1480, margin_left=680, margin_right=220
    )
