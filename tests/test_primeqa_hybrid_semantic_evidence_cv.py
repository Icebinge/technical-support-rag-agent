from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path

from typer.testing import CliRunner

from scripts.cross_validate_primeqa_hybrid_semantic_evidence import app, main
from ts_rag_agent.application import primeqa_hybrid_evidence_entailment_cv as stage172
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as analysis
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    ContextCandidateRecord,
)
from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import PrimeQAHybridSplitSample


class FakeSemanticScorer:
    def score(self, pairs, *, progress_sink):
        _ = progress_sink
        scores = {pair.private_identity: (8.0 if pair.positive_label else -4.0) for pair in pairs}
        return scores, analysis.SemanticScoringSummary(
            pair_count=len(pairs),
            event_batch_count=1,
            scoring_seconds=0.1,
            pairs_per_second=len(pairs) * 10.0,
        )


def test_spec_grid_is_frozen_cross_product() -> None:
    specs = analysis.build_stage173_specs()

    assert len(specs) == 68
    assert len({spec.spec_id for spec in specs}) == 68
    assert {spec.feature_profile for spec in specs} == {"semantic_only", "hybrid"}


def test_query_aware_policy_selects_relevant_tail_with_bounded_text() -> None:
    policy = analysis.QueryAwareCrossEncoderTextPolicy(excerpt_chars=120)
    document = PrimeQADocument(
        id="doc",
        title="Adapter guide",
        text=("unrelated material " * 40) + "configure adapter using acmectl apply",
    )

    passage = policy.passage(
        question="How do I configure the adapter with acmectl?",
        document=document,
    )

    assert "configure adapter using acmectl" in passage
    assert len(passage) <= len(document.title) + 2 + 120


def test_semantic_summary_has_exact_feature_contract_and_final_gain() -> None:
    records = _records("sample", initial_ranks=range(11, 21))
    visible = (*records[10:20], *records[:10])
    scores = {record.document_id: float(record.baseline_rank) for record in visible}
    initial_scores = {record.document_id: scores[record.document_id] for record in records[10:20]}

    summary = analysis.summarize_semantic_view(
        visible_records=visible,
        scores=scores,
        initial_scores=initial_scores,
        initial_document_ids=frozenset(initial_scores),
        phase="final",
    )

    assert tuple(summary) == analysis._SEMANTIC_FEATURE_NAMES
    assert len(summary) == 12
    assert summary["semantic_score_max"] == 20.0
    assert summary["semantic_gain_over_initial_max"] == 0.0
    assert summary["semantic_top1_new_alternate_indicator"] == 0.0


def test_case_builder_scores_each_final_union_pair_once_and_reuses_initial_subset() -> None:
    samples = (
        _sample("sample-a", answerable=True, answer_doc_id="doc-005"),
        _sample("sample-b", answerable=False, answer_doc_id=None),
    )
    grouped = {
        sample.sample_id: _records(sample.sample_id, initial_ranks=range(11, 21))
        for sample in samples
    }
    documents = {
        f"doc-{rank:03d}": PrimeQADocument(
            id=f"doc-{rank:03d}",
            title=f"Document {rank}",
            text="Adapter configuration procedure.",
        )
        for rank in range(1, 201)
    }

    cases, pairs, scores, summary = analysis.build_semantic_evidence_cases(
        samples=samples,
        grouped_records=grouped,
        documents_by_id=documents,
        scorer=FakeSemanticScorer(),
        text_policy=analysis.QueryAwareCrossEncoderTextPolicy(),
        progress_sink=None,
    )

    assert len(cases) == 4
    assert len(pairs) == len(scores) == 40
    assert summary.pair_count == 40
    assert all(len(case.features) == 41 for case in cases)
    alternate_final = next(
        case
        for case in cases
        if case.stratum == "alternate_only_gold_visible" and case.phase == "final"
    )
    assert alternate_final.features["semantic_top1_new_alternate_indicator"] == 1.0
    assert alternate_final.sufficient_label is True


def test_feature_profiles_have_frozen_dimensions() -> None:
    cases = _evaluation_cases()

    semantic = analysis._feature_matrix(
        cases,
        analysis._FEATURE_NAMES_BY_PROFILE["semantic_only"],
    )
    hybrid = analysis._feature_matrix(
        cases,
        analysis._FEATURE_NAMES_BY_PROFILE["hybrid"],
    )

    assert semantic.shape == (len(cases), 14)
    assert hybrid.shape == (len(cases), 41)


def test_full_train_selection_uses_five_fold_oof_and_finds_learnable_signal() -> None:
    cases = _evaluation_cases(fold_count=5)

    selected = analysis._select_full_train_spec(
        cases=cases,
        specs=analysis.build_stage173_specs(),
    )

    assert selected["fold_count"] == 5
    assert selected["metrics"].case_count == len(cases)
    assert selected["eligible"] is True


def test_public_key_scan_rejects_raw_pair_fields() -> None:
    assert analysis._forbidden_keys_found({"nested": {"passage_text": "private"}}) == {
        "passage_text"
    }
    assert not analysis._forbidden_keys_found({"pair_level_roc_auc": 0.8})


def test_visualizations_write_eight_parseable_svgs(tmp_path: Path) -> None:
    visualizations = analysis.write_stage173_visualizations(
        report=_visual_report(),
        output_dir=tmp_path,
    )

    assert len(visualizations) == 8
    for visualization in visualizations:
        ET.parse(visualization.path)


def test_cli_exposes_only_train_protocol_inputs() -> None:
    result = CliRunner().invoke(app, ["--help"])
    parameters = set(inspect.signature(main).parameters)

    assert result.exit_code == 0
    assert parameters == {
        "model_snapshot",
        "output",
        "visualization_dir",
        "encoder_batch_size",
    }


def _sample(
    sample_id: str,
    *,
    answerable: bool,
    answer_doc_id: str | None,
) -> PrimeQAHybridSplitSample:
    return PrimeQAHybridSplitSample(
        split_name="fixture",
        protocol_version="fixture-v1",
        assigned_split="train",
        split_subtype="group_random_train",
        source_split="train",
        sample_id=sample_id,
        question_id=sample_id,
        question_title="Adapter",
        question_text="How do I configure it?",
        answerable=answerable,
        answer="answer" if answerable else "",
        answer_doc_id=answer_doc_id,
        candidate_doc_ids=(),
        start_offset=None,
        end_offset=None,
    )


def _records(
    sample_id: str,
    *,
    initial_ranks: range,
) -> tuple[ContextCandidateRecord, ...]:
    initial_rank_set = set(initial_ranks)
    return tuple(
        ContextCandidateRecord(
            sample_id=sample_id,
            fold_id="fold-0",
            document_id=f"doc-{rank:03d}",
            baseline_rank=rank,
            answerable=True,
            is_gold=False,
            features={
                "stage116_rrf_score": 1.0 / rank,
                "current_query_overlap_combined_score": float(rank in initial_rank_set),
                "current_query_overlap_count": float(rank in initial_rank_set),
                "current_query_overlap_ratio": 0.5,
                "route_hit_count": 3.0,
                "lexical_route_hit_count": 2.0,
                "dense_route_hit_count": 1.0,
                "best_route_inverse_rank": 1.0 / rank,
                "query_token_coverage": 0.5,
                "query_body_token_coverage": 0.4,
                "query_title_token_overlap": 1.0,
                "query_section_heading_overlap": 1.0,
                "query_special_token_match_count": 0.0,
                "bm25_top10_indicator": float(rank <= 10),
            },
        )
        for rank in range(1, 201)
    )


def _evaluation_cases(fold_count: int = 1) -> tuple[stage172.EvidenceViewCase, ...]:
    cases = []
    strata = (
        "initial_gold_visible",
        "alternate_only_gold_visible",
        "union_gold_missing_candidate_hit",
        "candidate_pool_gold_missing",
        "unanswerable",
    )
    for fold in range(fold_count):
        for repetition in range(2):
            for index, stratum in enumerate(strata):
                sufficient_initial = stratum == "initial_gold_visible"
                sufficient_final = stratum in {
                    "initial_gold_visible",
                    "alternate_only_gold_visible",
                }
                for phase, sufficient in (
                    ("initial", sufficient_initial),
                    ("final", sufficient_final),
                ):
                    base_features = {
                        name: (
                            float(phase == "final")
                            if name == "phase_final"
                            else 10.0
                            if name == "visible_document_count"
                            else 0.0
                        )
                        for name in stage172._MODEL_FEATURE_NAMES
                    }
                    semantic = {
                        name: (8.0 if sufficient else -8.0)
                        for name in analysis._SEMANTIC_FEATURE_NAMES
                    }
                    cases.append(
                        stage172.EvidenceViewCase(
                            private_identity=(
                                f"fold-{fold}-rep-{repetition}-group-{index}-{phase}"
                            ),
                            group_identity=f"fold-{fold}-rep-{repetition}-group-{index}",
                            fold_id=f"fold-{fold}",
                            phase=phase,
                            stratum=stratum,
                            features={**base_features, **semantic},
                            sufficient_label=sufficient,
                        )
                    )
    return tuple(cases)


def _visual_report() -> dict:
    rates = {
        "initial_visible_compose_rate": 0.8,
        "alternate_only_inspect_rate": 0.7,
        "alternate_only_final_compose_rate": 0.75,
        "alternate_only_path_success_rate": 0.5,
        "insufficient_final_compose_rate": 0.1,
    }
    return {
        "nested_cv": {
            "oof_metrics": rates,
            "oof_quality_gates": [
                {"name": name, "passed": True, "observed": value, "threshold": value}
                for name, value in rates.items()
            ],
            "outer_fold_metrics": {f"fold-{index}": rates for index in range(5)},
            "outer_folds": [
                {
                    "heldout_fold": f"fold-{index}",
                    "selected_spec": {
                        "feature_profile": ("semantic_only" if index < 3 else "hybrid")
                    },
                }
                for index in range(5)
            ],
        },
        "stage172_comparison": {"stage172_oof_metrics": rates},
        "semantic_diagnostics": {
            "pair_level_roc_auc": 0.8,
            "view_max_roc_auc": 0.75,
            "positive_pair_top1_rate": 0.7,
        },
        "timing_seconds": {
            "candidate_replay": 90.0,
            "semantic_pair_build_and_score": 20.0,
            "nested_cv": 5.0,
        },
        "resource_consumption": {
            "process_peak_working_set_bytes": 4 * 1024**3,
            "process_peak_private_usage_bytes": 5 * 1024**3,
            "gpu_peak_allocated_bytes": 1 * 1024**3,
            "gpu_peak_reserved_bytes": 2 * 1024**3,
            "minimum_system_available_memory_bytes": 3 * 1024**3,
        },
    }
