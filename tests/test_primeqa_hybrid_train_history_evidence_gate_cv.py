from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.analyze_primeqa_hybrid_train_history_evidence_gate_cv import app, main
from ts_rag_agent.application import primeqa_hybrid_train_history_evidence_gate_cv as analysis
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    ContextCandidateRecord,
)


def test_gate_family_is_frozen_cross_product() -> None:
    specs = analysis.build_stage167_gate_specs()

    assert len(specs) == 84
    assert len({spec.spec_id for spec in specs}) == 84


def test_pair_labels_separate_benefit_harm_and_neutral() -> None:
    beneficial = _case(1, answerable=True, isolated_refused=False, synthetic_refused=True)
    harmful = _case(2, answerable=False, isolated_refused=False, synthetic_refused=True)
    neutral = _case(3, answerable=True, isolated_refused=True, synthetic_refused=True)

    assert beneficial.beneficial_label is True
    assert beneficial.harmful_label is False
    assert harmful.beneficial_label is False
    assert harmful.harmful_label is True
    assert neutral.beneficial_label is False
    assert neutral.harmful_label is False


def test_evidence_summary_has_exact_non_gold_feature_contract() -> None:
    records = tuple(_record(rank) for rank in range(1, 201))

    summary = analysis.summarize_candidate_evidence(sample_id="sample-a", records=records)

    assert tuple(summary.values) == analysis._EVIDENCE_FEATURE_NAMES
    assert summary.values["rrf_top1_top2_margin"] > 0
    assert summary.values["selection_prefix_overlap_count"] == 10.0
    assert all("gold" not in name for name in summary.values)


def test_policy_requires_benefit_and_harm_thresholds() -> None:
    case = _case(1, answerable=True, isolated_refused=False, synthetic_refused=True)
    spec = analysis.Stage167GateSpec("logistic", 0.6, 0.2)

    selected = analysis.evaluate_policy(
        (case,),
        {case.private_identity_sha256: analysis.Stage167Prediction(0.8, 0.1)},
        spec,
    )
    blocked = analysis.evaluate_policy(
        (case,),
        {case.private_identity_sha256: analysis.Stage167Prediction(0.8, 0.3)},
        spec,
    )

    assert selected.isolated_selection_count == 1
    assert blocked.isolated_selection_count == 0


def test_selected_spec_rejects_missing_predictions_instead_of_falling_back() -> None:
    case = _case(1, answerable=True, isolated_refused=False, synthetic_refused=True)
    spec = analysis.Stage167GateSpec("logistic", 0.6, 0.2)

    with pytest.raises(ValueError, match="prediction is missing"):
        analysis.evaluate_policy((case,), {}, spec)


def test_feature_matrix_contains_only_frozen_runtime_inputs() -> None:
    cases = (
        _case(1, answerable=True, isolated_refused=False, synthetic_refused=True),
        _case(2, answerable=False, isolated_refused=False, synthetic_refused=True),
    )

    matrix = analysis._feature_matrix(cases)

    assert matrix.shape == (2, len(analysis._EVIDENCE_FEATURE_NAMES) + 8 + 3)
    assert matrix[0, : len(analysis._EVIDENCE_FEATURE_NAMES)].tolist() == [1.0] * len(
        analysis._EVIDENCE_FEATURE_NAMES
    )


def test_strict_gain_ignores_isolated_selection_count() -> None:
    neutral_selection_delta = {
        "isolated_selection_count": 4,
        "answerable_refusal_count": 0,
        "answerable_f1_sum": 0.0,
        "answerable_gold_citation_count": 0,
        "unanswerable_false_answer_count": 0,
    }

    assert analysis._strict_nonregression_values(neutral_selection_delta) is True
    assert analysis._has_strict_gain(neutral_selection_delta) is False


def test_visualizations_write_six_parseable_svgs(tmp_path: Path) -> None:
    report = _visual_report()

    visualizations = analysis.write_stage167_visualizations(report=report, output_dir=tmp_path)

    assert len(visualizations) == 6
    for visualization in visualizations:
        ET.parse(visualization.path)


def test_cli_exposes_no_development_or_test_inputs() -> None:
    result = CliRunner().invoke(app, ["--help"])
    parameters = inspect.signature(main).parameters

    assert result.exit_code == 0
    assert "user_confirmed_stage167" in parameters
    assert not ({"dev", "development", "dev_split"} & set(parameters))
    assert not ({"test", "test_split"} & set(parameters))


def _case(
    index: int,
    *,
    answerable: bool,
    isolated_refused: bool,
    synthetic_refused: bool,
) -> analysis.Stage167PairCase:
    return analysis.Stage167PairCase(
        private_identity_sha256=f"{index:064x}",
        diagnostic_group_sha256=f"{index + 100:064x}",
        fold_id=index % 5,
        question_route="other",
        turn_position=2 + index % 3,
        answerable=answerable,
        evidence={name: float(index) for name in analysis._EVIDENCE_FEATURE_NAMES},
        isolated_refused=isolated_refused,
        synthetic_refused=synthetic_refused,
        isolated_f1=0.8 if answerable and not isolated_refused else 0.0,
        synthetic_f1=0.8 if answerable and not synthetic_refused else 0.0,
        isolated_gold_cited=answerable and not isolated_refused,
        synthetic_gold_cited=answerable and not synthetic_refused,
    )


def _record(rank: int) -> ContextCandidateRecord:
    return ContextCandidateRecord(
        sample_id="sample-a",
        fold_id="fold_1",
        document_id=f"doc-{rank:03d}",
        baseline_rank=rank,
        answerable=False,
        is_gold=False,
        features={
            "stage116_rrf_score": 1.0 / rank,
            "current_query_overlap_combined_score": 201.0 - rank,
            "current_query_overlap_count": float(10 - min(rank, 10)),
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


def _visual_report() -> dict:
    folds = [
        {
            "heldout_fold": fold,
            "inner_eligible_spec_count": fold,
            "heldout_metrics": {"isolated_selection_count": fold + 1},
            "heldout_delta": {"unanswerable_false_answer_count": fold % 2},
        }
        for fold in range(5)
    ]
    return {
        "nested_cv": {
            "outer_folds": folds,
            "oof_delta": {
                "answerable_refusal_count": -1,
                "answerable_f1_sum": 0.2,
                "answerable_gold_citation_count": 1,
                "unanswerable_false_answer_count": 0,
            },
        },
        "case_summary": {
            "beneficial_label_count": 10,
            "harmful_label_count": 5,
            "neutral_label_count": 20,
        },
        "guard_checks": [{"name": "train_only", "passed": True}],
    }
