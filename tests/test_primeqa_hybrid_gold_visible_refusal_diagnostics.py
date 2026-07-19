from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path

from typer.testing import CliRunner

from scripts.analyze_primeqa_hybrid_gold_visible_refusals import app
from scripts.audit_primeqa_hybrid_gold_visible_refusal_contract import (
    app as correction_app,
)
from ts_rag_agent.application import (
    primeqa_hybrid_bounded_dynamic_agent_failure_diagnostics_protocol as stage160,
)
from ts_rag_agent.application import (
    primeqa_hybrid_gold_visible_refusal_contract_correction as correction,
)
from ts_rag_agent.application import (
    primeqa_hybrid_gold_visible_refusal_diagnostics as stage164,
)
from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    StructuredRouterPromptPolicy,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQARuntimeQuery


def test_analyzer_distinguishes_gold_document_from_answer_evidence_visibility() -> None:
    sample = _diagnostic_sample(answer="target fix")
    row = _stage160_row(refused=True)
    document = PrimeQADocument(
        id="gold-doc",
        title="Troubleshooting",
        text=("x" * 650) + " target fix appears here",
    )

    profiles = stage164.GoldVisibleRefusalAnalyzer(
        prompt_policy=StructuredRouterPromptPolicy()
    ).build_profiles(
        diagnostic_samples=[sample],
        stage160_rows=[row],
        documents_by_id={document.id: document},
    )

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.gold_generation_rank == 4
    assert profile.answer_found_in_full_document is True
    assert profile.answer_exact_span_visible is False
    assert profile.answer_all_tokens_visible is False
    assert profile.answer_token_recall_visible == 0.0
    assert profile.answer_visibility_class == "no_answer_tokens"


def test_analyzer_detects_exact_answer_inside_prompt_excerpt() -> None:
    sample = _diagnostic_sample(answer="target fix")
    row = _stage160_row(refused=False)
    document = PrimeQADocument(
        id="gold-doc",
        title="Target fix procedure",
        text="Apply the target fix and restart the service.",
    )

    profile = stage164.GoldVisibleRefusalAnalyzer(
        prompt_policy=StructuredRouterPromptPolicy()
    ).build_profiles(
        diagnostic_samples=[sample],
        stage160_rows=[row],
        documents_by_id={document.id: document},
    )[0]

    assert profile.answer_exact_span_visible is True
    assert profile.answer_all_tokens_visible is True
    assert profile.answer_token_recall_visible == 1.0
    assert profile.answer_visibility_class == "exact_span_visible"


def test_risk_aligned_auc_respects_declared_direction() -> None:
    assert stage164._risk_aligned_auc([3.0, 4.0], [1.0, 2.0], risk_direction="higher") == 1.0
    assert stage164._risk_aligned_auc([1.0, 2.0], [3.0, 4.0], risk_direction="lower") == 1.0
    assert stage164._risk_aligned_auc([1.0], [1.0], risk_direction="higher") == 0.5


def test_binary_association_reports_refusal_risk_difference() -> None:
    profiles = [
        _profile(refused=True, exact_visible=False),
        _profile(refused=True, exact_visible=False, identity="b"),
        _profile(refused=False, exact_visible=True, identity="c"),
        _profile(refused=True, exact_visible=True, identity="d"),
    ]

    result = stage164._binary_association(
        profiles,
        attribute="answer_exact_span_visible",
        risk_value=False,
    )

    assert result["risk_refusal_rate"] == 1.0
    assert result["reference_refusal_rate"] == 0.5
    assert result["refusal_rate_difference_risk_minus_reference"] == 0.5


def test_fold_stability_counts_only_comparable_folds() -> None:
    profiles = [
        _profile(refused=True, exact_visible=False, fold_id=0, identity="a"),
        _profile(refused=False, exact_visible=True, fold_id=0, identity="b"),
        _profile(refused=True, exact_visible=False, fold_id=1, identity="c"),
    ]

    result = stage164._binary_fold_stability(
        profiles,
        attribute="answer_exact_span_visible",
        risk_value=False,
    )

    assert result["comparable_fold_count"] == 1
    assert result["risk_direction_fold_count"] == 1
    assert result["folds"]["fold_2"]["comparable"] is False


def test_stage164_protocol_blocks_agent_fit_test_runtime_and_fallback() -> None:
    protocol = stage164._frozen_protocol()

    assert protocol["cohort_expected_count"] == 36
    assert protocol["grouped_fold_role"] == "direction_stability_only_no_fit_or_selection"
    assert protocol["causal_claim_allowed"] is False
    assert all(protocol["blocked"].values())


def test_stage164_private_report_contains_only_hashed_numeric_features() -> None:
    report = stage164._private_report([_profile(refused=True, exact_visible=False)])

    assert report["row_count"] == 1
    assert report["contains_raw_question"] is False
    assert report["contains_raw_answer"] is False
    assert report["contains_raw_document_id"] is False
    assert report["contains_raw_document_text"] is False
    assert "private_identity_sha256" in report["rows"][0]


def test_stage164_visualizations_write_ten_parseable_svgs(tmp_path: Path) -> None:
    profiles = []
    for fold_id in range(5):
        profiles.extend(
            [
                _profile(
                    refused=True,
                    exact_visible=False,
                    fold_id=fold_id,
                    identity=f"{fold_id}-risk",
                ),
                _profile(
                    refused=False,
                    exact_visible=True,
                    fold_id=fold_id,
                    identity=f"{fold_id}-reference",
                ),
            ]
        )
    analysis = stage164._analyze_profiles(profiles)
    report = {
        **analysis,
        "guard_checks": [{"name": "diagnostic_only", "passed": True}],
    }

    visualizations = stage164.write_stage164_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 10
    for visualization in visualizations:
        assert ET.parse(visualization.path).getroot().tag.endswith("svg")


def test_stage164_cli_has_dev_join_but_no_train_test_or_model_options() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--dev-split" in result.stdout
    assert "--train-split" not in result.stdout
    assert "--test-split" not in result.stdout
    assert "--model-snapshot" not in result.stdout


def test_stage164_run_source_has_no_agent_or_retrieval_execution() -> None:
    source = inspect.getsource(stage164.run_primeqa_hybrid_gold_visible_refusal_diagnostics)
    authorization_source = inspect.getsource(stage164._authorize_sources)

    assert "validate_primeqa_hybrid_bounded_dynamic_agent" not in source
    assert "candidate_pool_retriever" not in source
    assert "Qwen" not in source
    assert "load_stage160_dev_diagnostic_samples" in source
    assert "stage160_report = _load_json_object" in authorization_source
    assert "stage160 = _load_json_object" not in authorization_source


def test_stage164_decision_stays_diagnostic_and_keeps_test_closed() -> None:
    report = {
        "primary_hypothesis_assessment": {
            "aggregate_visibility_gap_observed": True,
            "fold_stable_visibility_gap_observed": True,
        },
        "guard_checks": [{"name": "diagnostic_only", "passed": True}],
    }

    decision = stage164._decision(report=report, all_guards_passed=True)

    assert decision["status"] == "primeqa_hybrid_gold_visible_refusal_diagnostics_completed"
    assert decision["policy_selected"] is False
    assert decision["agent_rerun"] is False
    assert decision["test_gate_opened"] is False
    assert decision["next_direction"] == (
        "design_train_only_prompt_evidence_visibility_intervention"
    )


def test_stage164_primary_hypothesis_requires_cross_fold_direction_majority() -> None:
    binary = {
        "answer_exact_span_visible": {"refusal_rate_difference_risk_minus_reference": 0.1},
        "answer_all_tokens_visible": {"refusal_rate_difference_risk_minus_reference": 0.1},
    }
    folds = {
        "answer_exact_span_visible": {
            "risk_direction_fold_count": 2,
            "opposite_direction_fold_count": 3,
            "comparable_fold_count": 5,
        }
    }

    assessment = stage164._primary_hypothesis_assessment(
        binary_associations=binary,
        fold_stability=folds,
    )

    assert assessment["aggregate_visibility_gap_observed"] is True
    assert assessment["fold_stable_visibility_gap_observed"] is False

    decision = stage164._decision(
        report={
            "primary_hypothesis_assessment": assessment,
            "guard_checks": [{"name": "diagnostic_only", "passed": True}],
        },
        all_guards_passed=True,
    )
    assert decision["next_direction"] == (
        "design_train_only_router_history_and_question_alignment_diagnostics"
    )


def test_stage164_corrected_rank_guard_uses_membership_not_dense_position() -> None:
    source = inspect.getsource(stage164._guard_checks)

    assert "gold_generation_context_membership_exact" in source
    assert "generation_context_count" in source
    assert "profile.gold_generation_rank <= 10" not in source


def test_stage164_correction_cli_has_no_split_document_or_runtime_options() -> None:
    result = CliRunner().invoke(correction_app, ["--help"])
    source = inspect.getsource(correction.run_stage164_contract_correction)

    assert result.exit_code == 0
    assert "--dev-split" not in result.stdout
    assert "--test-split" not in result.stdout
    assert "--documents" not in result.stdout
    assert "--model-snapshot" not in result.stdout
    assert "load_stage160_dev_diagnostic_samples" not in source
    assert "load_primeqa_documents" not in source
    assert "candidate_pool_retriever" not in source


def test_stage164_correction_visualizations_write_two_parseable_svgs(
    tmp_path: Path,
) -> None:
    report = {
        "rank_semantics_correction": {
            "generation_context_count_maximum": 10,
            "gold_generation_rank_maximum": 14,
            "rank_above_ten_count": 1,
        },
        "hypothesis_interpretation_correction": {
            "corrected_assessment": {
                "exact_span_fold_direction_count": 2,
                "exact_span_fold_opposite_direction_count": 3,
            }
        },
        "stable_observed_patterns": {
            "post_first_turn_risk_direction_fold_count": 5,
        },
    }

    visualizations = correction.write_stage164_contract_correction_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 2
    for visualization in visualizations:
        assert ET.parse(visualization.path).getroot().tag.endswith("svg")


def test_stage164_correction_metric_snapshot_excludes_interpretation_fields() -> None:
    original = {
        "cohort_summary": {},
        "answer_visibility_summary": {},
        "fixed_binary_associations": {},
        "fixed_numeric_associations": {},
        "question_route_summary": {},
        "fold_stability": {},
        "exploratory_feature_ranking": [],
        "primary_hypothesis_assessment": {"old": True},
        "decision": {"status": "invalid"},
    }

    snapshot = correction._metric_snapshot(original)

    assert "primary_hypothesis_assessment" not in snapshot
    assert "decision" not in snapshot


def _diagnostic_sample(*, answer: str) -> stage160.Stage160DiagnosticSample:
    return stage160.Stage160DiagnosticSample(
        private_identity_sha256="a" * 64,
        query_digest_sha256="b" * 64,
        diagnostic_group_sha256="c" * 64,
        answerable=True,
        gold_answer=answer,
        gold_document_id="gold-doc",
        gold_document_sha256="d" * 64,
        question_route="how_to_or_lookup",
        split_subtype="group_random_dev",
        runtime_query=PrimeQARuntimeQuery(
            id="query",
            title="How do I apply the fix?",
            text="Need target repair instructions.",
        ),
    )


def _stage160_row(*, refused: bool) -> dict:
    return {
        "private_identity_sha256": "a" * 64,
        "fold_id": 0,
        "refused": refused,
        "question_route": "how_to_or_lookup",
        "split_subtype": "group_random_dev",
        "gold_generation_rank": 4,
        "gold_candidate_rank": 8,
        "gold_verification_rank": 8,
        "top_candidate_score": 1.0,
        "gold_candidate_score": 0.5,
        "router_input_token_count": 2200,
        "retained_state_bytes": 1000,
        "turn_position": 1,
        "completed_turn_count": 0,
        "router_generation_latency_ms": 1200.0,
    }


def _profile(
    *,
    refused: bool,
    exact_visible: bool,
    fold_id: int = 0,
    identity: str = "a",
) -> stage164.GoldVisibleRefusalCaseProfile:
    return stage164.GoldVisibleRefusalCaseProfile(
        private_identity_sha256=identity,
        diagnostic_group_sha256=f"group-{identity}",
        fold_id=fold_id,
        refused=refused,
        question_route="how_to_or_lookup",
        split_subtype="group_random_dev",
        gold_generation_rank=7 if refused else 2,
        gold_candidate_rank=20 if refused else 5,
        gold_verification_rank=20 if refused else 5,
        top_candidate_score=1.0,
        gold_candidate_score=0.4 if refused else 0.8,
        gold_score_ratio_to_top=0.4 if refused else 0.8,
        router_input_token_count=3000 if refused else 2000,
        retained_state_bytes=1500 if refused else 900,
        turn_position=3 if refused else 1,
        completed_turn_count=2 if refused else 0,
        router_generation_latency_ms=3000.0 if refused else 1000.0,
        answer_token_count=8 if refused else 3,
        gold_document_length_chars=5000 if refused else 500,
        gold_excerpt_truncated=refused,
        answer_found_in_full_document=True,
        answer_exact_span_visible=exact_visible,
        answer_all_tokens_visible=exact_visible,
        answer_token_recall_visible=0.2 if refused else 1.0,
        question_token_recall_in_gold_prompt=0.3 if refused else 0.8,
        answer_character_start=1000 if refused else 10,
        answer_visibility_class=(
            "exact_span_visible" if exact_visible else "partial_answer_tokens"
        ),
    )
