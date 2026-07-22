from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path

from typer.testing import CliRunner

from scripts.cross_validate_primeqa_hybrid_evidence_entailment import app, main
from ts_rag_agent.application import primeqa_hybrid_evidence_entailment_cv as analysis
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    ContextCandidateRecord,
)
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import (
    PrimeQAHybridSplitSample,
)


def test_model_spec_grid_is_frozen_cross_product() -> None:
    specs = analysis.build_stage172_specs()

    assert len(specs) == 34
    assert len({spec.spec_id for spec in specs}) == 34
    assert {spec.model_family for spec in specs} == {"logistic", "histogram_gbdt"}


def test_evidence_view_summary_has_exact_runtime_only_contract() -> None:
    records = _records("sample-a")
    visible = tuple(record for record in records if 11 <= record.baseline_rank <= 20)

    summary = analysis.summarize_evidence_view(
        records=records,
        visible_records=visible,
        phase="final",
    )

    assert tuple(summary) == analysis._MODEL_FEATURE_NAMES
    assert len(summary) == 29
    assert summary["phase_final"] == 1.0
    assert summary["visible_document_count"] == 10.0
    assert not ({"answer", "answerable", "is_gold", "document_id"} & set(summary))


def test_case_builder_keeps_two_views_of_question_in_one_fold() -> None:
    answerable = _sample("sample-a", answerable=True, answer_doc_id="doc-015")
    unanswerable = _sample("sample-b", answerable=False, answer_doc_id=None)
    grouped = {
        "sample-a": _records("sample-a", initial_ranks=range(11, 21)),
        "sample-b": _records("sample-b", initial_ranks=range(1, 11)),
    }

    cases = analysis.build_evidence_view_cases(
        samples=(answerable, unanswerable),
        grouped_records=grouped,
    )

    assert len(cases) == 4
    answerable_cases = [case for case in cases if case.stratum == "initial_gold_visible"]
    assert {case.phase for case in answerable_cases} == {"initial", "final"}
    assert {case.group_identity for case in answerable_cases} == {
        answerable_cases[0].group_identity
    }
    assert {case.fold_id for case in answerable_cases} == {"fold-0"}
    assert all(case.sufficient_label for case in answerable_cases)
    assert not any(case.sufficient_label for case in cases if case.stratum == "unanswerable")


def test_prediction_metrics_capture_required_hierarchical_path() -> None:
    cases = _evaluation_cases()
    predictions = {case.private_identity: _known_score(case) for case in cases}
    specs = {
        fold_id: analysis.EvidenceModelSpec("logistic", 0.5)
        for fold_id in {case.fold_id for case in cases}
    }

    metrics = analysis.evaluate_predictions(cases, predictions, specs)

    assert metrics.initial_visible_compose_rate == 1.0
    assert metrics.alternate_only_inspect_rate == 1.0
    assert metrics.alternate_only_final_compose_rate == 1.0
    assert metrics.alternate_only_path_success_rate == 1.0
    assert metrics.insufficient_final_compose_rate == 0.0


def test_selection_key_prioritizes_all_fold_safety() -> None:
    safe = _selection_row(safe_fold_count=5, exact_path=0.4, false_compose=0.2)
    unsafe = _selection_row(safe_fold_count=4, exact_path=1.0, false_compose=0.0)

    assert analysis._spec_selection_key(safe) > analysis._spec_selection_key(unsafe)


def test_full_train_selection_uses_grouped_oof_predictions() -> None:
    cases = _evaluation_cases(fold_count=5)

    selected = analysis._select_full_train_spec(
        cases=cases,
        specs=analysis.build_stage172_specs(),
    )

    assert selected["spec"] in analysis.build_stage172_specs()
    assert selected["fold_count"] == 5
    assert selected["metrics"].case_count == len(cases)


def test_public_key_scan_finds_private_runtime_inputs() -> None:
    assert analysis._forbidden_keys_found({"nested": {"sample_id": "private"}}) == {"sample_id"}
    assert not analysis._forbidden_keys_found({"feature_names": ["rrf_top1"]})


def test_visualizations_write_six_parseable_svgs(tmp_path: Path) -> None:
    visualizations = analysis.write_stage172_visualizations(
        report=_visual_report(),
        output_dir=tmp_path,
    )

    assert len(visualizations) == 6
    for visualization in visualizations:
        ET.parse(visualization.path)


def test_cli_exposes_no_dev_test_generation_retry_or_fallback_inputs() -> None:
    result = CliRunner().invoke(app, ["--help"])
    parameters = set(inspect.signature(main).parameters)

    assert result.exit_code == 0
    assert parameters == {"output", "visualization_dir", "encoder_batch_size"}


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
        question_title="title",
        question_text="question",
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
    initial_ranks: range = range(11, 21),
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


def _evaluation_cases(fold_count: int = 1) -> tuple[analysis.EvidenceViewCase, ...]:
    cases = []
    strata = (
        "initial_gold_visible",
        "alternate_only_gold_visible",
        "union_gold_missing_candidate_hit",
        "candidate_pool_gold_missing",
        "unanswerable",
    )
    for fold in range(fold_count):
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
                cases.append(
                    analysis.EvidenceViewCase(
                        private_identity=f"fold-{fold}-group-{index}-{phase}",
                        group_identity=f"fold-{fold}-group-{index}",
                        fold_id=f"fold-{fold}",
                        phase=phase,
                        stratum=stratum,
                        features={
                            name: (
                                float(sufficient)
                                if name == "rrf_top1"
                                else float(phase == "final")
                                if name == "phase_final"
                                else 10.0
                                if name == "visible_document_count"
                                else float(index + fold / 10)
                            )
                            for name in analysis._MODEL_FEATURE_NAMES
                        },
                        sufficient_label=sufficient,
                    )
                )
    return tuple(cases)


def _known_score(case: analysis.EvidenceViewCase) -> float:
    if case.stratum == "initial_gold_visible":
        return 0.9
    if case.stratum == "alternate_only_gold_visible":
        return 0.9 if case.phase == "final" else 0.1
    return 0.1


def _selection_row(
    *,
    safe_fold_count: int,
    exact_path: float,
    false_compose: float,
) -> dict:
    metrics = analysis.EvidenceProxyMetrics(
        case_count=10,
        positive_count=2,
        negative_count=8,
        predicted_sufficient_count=2,
        true_positive_count=2,
        false_positive_count=0,
        true_negative_count=8,
        false_negative_count=0,
        balanced_accuracy=1.0,
        roc_auc=1.0,
        initial_visible_compose_rate=1.0,
        alternate_only_inspect_rate=1.0,
        alternate_only_final_compose_rate=1.0,
        alternate_only_path_success_rate=exact_path,
        insufficient_final_compose_rate=false_compose,
    )
    return {
        "spec": analysis.EvidenceModelSpec("logistic", 0.5),
        "metrics": metrics,
        "gates": analysis._quality_gates(metrics),
        "safe_fold_count": safe_fold_count,
        "fold_count": 5,
    }


def _visual_report() -> dict:
    metrics = {
        "initial_visible_compose_rate": 0.8,
        "alternate_only_inspect_rate": 0.6,
        "alternate_only_final_compose_rate": 0.75,
        "alternate_only_path_success_rate": 0.5,
        "insufficient_final_compose_rate": 0.1,
    }
    return {
        "nested_cv": {
            "oof_metrics": metrics,
            "oof_quality_gates": [
                {"name": name, "passed": True, "observed": value, "threshold": value}
                for name, value in metrics.items()
            ],
            "outer_fold_metrics": {f"fold-{index}": metrics for index in range(5)},
            "outer_folds": [
                {"heldout_fold": f"fold-{index}", "inner_eligible_spec_count": index + 1}
                for index in range(5)
            ],
        },
        "case_summary": {"positive_view_count": 400, "negative_view_count": 724},
    }
