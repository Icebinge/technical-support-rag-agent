from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from scripts.audit_primeqa_hybrid_untouched_rrf_dev_contract import (
    app as correction_app,
)
from scripts.validate_primeqa_hybrid_untouched_rrf_dev import app
from ts_rag_agent.application import (
    primeqa_hybrid_untouched_rrf_dev_contract_correction as correction,
)
from ts_rag_agent.application import primeqa_hybrid_untouched_rrf_dev_validation as validation
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector_training import (
    PrimeQAHybridCandidateDatasetBuilder,
    Stage161TrainCandidateDatasetBuilder,
)


def test_stage163_comparison_reports_aggregate_fold_and_case_deltas() -> None:
    current = _evaluation(hit=3, f1=0.2, citations=2, signature="current")
    candidate = _evaluation(hit=4, f1=0.3, citations=3, signature="candidate")

    comparison = validation._comparison(candidate=candidate, current=current)

    delta = comparison["aggregate_delta_vs_current"]
    assert delta["context_gold_hit_count"] == 1
    assert delta["average_token_f1_all_answerable"] == 0.1
    assert delta["gold_citation_count"] == 1
    assert delta["answerable_f1_improved_count"] == 1
    assert delta["changed_verified_answer_count"] == 1
    assert comparison["minimum_fold_hit_rate_delta"] == 0.1
    assert comparison["minimum_fold_f1_delta"] == 0.1


def test_stage163_strict_policy_guards_require_every_fold_non_regression() -> None:
    current = _evaluation(hit=3, f1=0.2, citations=2, signature="current")
    candidate = _evaluation(hit=4, f1=0.3, citations=3, signature="candidate")
    comparison = validation._comparison(candidate=candidate, current=current)

    guards = validation._policy_guard_results(
        candidate=candidate,
        current=current,
        comparison=comparison,
        policy_structure={"policy_is_exact_untouched_rrf_top10": True},
    )

    assert all(guards.values())
    comparison["minimum_fold_f1_delta"] = -0.000001
    guards = validation._policy_guard_results(
        candidate=candidate,
        current=current,
        comparison=comparison,
        policy_structure={"policy_is_exact_untouched_rrf_top10": True},
    )
    assert guards["every_fold_f1_not_below_current"] is False


def test_stage163_protocol_is_one_shot_dev_only_and_blocks_tuning() -> None:
    protocol = validation._frozen_protocol()

    assert protocol["validation_split"] == "dev"
    assert protocol["candidate_policy"] == "untouched_original_rrf_ranks_1_through_10"
    assert protocol["policy_search_or_tuning_on_dev"] is False
    assert protocol["grouped_fold_role"] == "stability_report_and_strict_non_regression_only"
    assert protocol["blocked"] == {
        "train_load": True,
        "test_load": True,
        "dev_fit_selection_or_tuning": True,
        "agent_runtime": True,
        "runtime_defaultization": True,
        "fallback": True,
        "query_rewrite": True,
        "second_retrieval": True,
    }


def test_stage163_visualizations_write_ten_parseable_svgs(tmp_path: Path) -> None:
    current = _evaluation(hit=3, f1=0.2, citations=2, signature="current")
    candidate = _evaluation(hit=4, f1=0.3, citations=3, signature="candidate")
    comparison = validation._comparison(candidate=candidate, current=current)
    report = {
        "dev_results": {
            "current": {"aggregate": current["aggregate"], "folds": current["folds"]},
            "rrf": {"aggregate": candidate["aggregate"], "folds": candidate["folds"]},
        },
        "policy_comparison": comparison,
        "policy_guard_results": {"strict": True},
        "guard_checks": [{"name": "dev_only", "passed": True}],
    }

    visualizations = validation.write_stage163_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 10
    for visualization in visualizations:
        path = Path(visualization.path)
        assert path.is_file()
        assert ET.parse(path).getroot().tag.endswith("svg")


def test_stage163_cli_exposes_dev_but_no_train_or_test_split_options() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--dev-split" in result.stdout
    assert "--train-split" not in result.stdout
    assert "--test-split" not in result.stdout


def test_stage163_relabels_shared_candidate_builder_stage_and_phase() -> None:
    constructor = inspect.signature(PrimeQAHybridCandidateDatasetBuilder.__init__)
    source = inspect.getsource(validation.run_primeqa_hybrid_untouched_rrf_dev_validation)

    assert Stage161TrainCandidateDatasetBuilder is PrimeQAHybridCandidateDatasetBuilder
    assert constructor.parameters["progress_stage"].default == "Stage 161"
    assert constructor.parameters["progress_phase"].default == "train_candidate_pool_build"
    assert "progress_stage=_STAGE" in source
    assert 'progress_phase="dev_candidate_pool_build"' in source


def test_shared_candidate_builder_rejects_empty_progress_phase() -> None:
    try:
        PrimeQAHybridCandidateDatasetBuilder(
            documents_by_id={},
            sections_by_document={},
            channels=(),
            fold_assignments={},
            progress_phase="",
        )
    except ValueError as error:
        assert str(error) == "candidate builder progress phase must not be empty"
    else:
        raise AssertionError("empty progress phase must be rejected")


def test_stage163_decision_opens_only_optional_agent_integration_after_pass() -> None:
    report = {
        "policy_adoption": {
            "all_strict_policy_guards_passed": True,
            "failed_policy_guards": [],
        },
        "guard_checks": [{"name": "dev_only", "passed": True}],
    }

    decision = validation._decision(report=report, process_guards_passed=True)

    assert decision["status"] == "primeqa_hybrid_untouched_rrf_dev_validated"
    assert decision["next_direction"] == (
        "integrate_untouched_rrf_as_optional_nondefault_agent_context_policy"
    )
    assert decision["test_gate_opened"] is False
    assert decision["runtime_registered_as_default"] is False


def test_stage163_correction_summarizes_runtime_top400_and_prefix_depths() -> None:
    rows = [
        {
            "candidate_pool_count": 400,
            "verification_context_count": 200,
            "generation_context_count": 10,
        }
        for _ in range(121)
    ]

    summary = correction._summarize_runtime_contract(rows)

    assert summary == {
        "row_count": 121,
        "candidate_pool_depth_minimum": 400,
        "candidate_pool_depth_maximum": 400,
        "verification_context_depth_minimum": 200,
        "verification_context_depth_maximum": 200,
        "generation_context_depth_minimum": 10,
        "generation_context_depth_maximum": 10,
        "contains_case_rows_in_public_correction": False,
    }


def test_stage163_correction_cli_has_no_split_or_retrieval_options() -> None:
    result = CliRunner().invoke(correction_app, ["--help"])
    source = inspect.getsource(correction.run_stage163_contract_correction)

    assert result.exit_code == 0
    assert "--train-split" not in result.stdout
    assert "--dev-split" not in result.stdout
    assert "--test-split" not in result.stdout
    assert "load_primeqa_hybrid_split_samples" not in source
    assert "PrimeQAHybridCandidateDatasetBuilder" not in source
    assert "_evaluate_selection_run" not in source


def test_stage163_correction_visualizations_write_two_parseable_svgs(
    tmp_path: Path,
) -> None:
    report = {
        "contract_evidence": {
            "stage160_runtime": {
                "candidate_pool_depth_minimum": 400,
                "verification_context_depth_minimum": 200,
                "generation_context_depth_minimum": 10,
            },
            "stage163_offline": {"candidate_pool_depth_minimum": 200},
        },
        "correction": {
            "original_process_guard_pass_count": 16,
            "corrected_process_guard_pass_count": 17,
        },
        "policy_result": {"strict_policy_guard_pass_count": 6},
    }

    visualizations = correction.write_stage163_contract_correction_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 2
    for visualization in visualizations:
        assert ET.parse(visualization.path).getroot().tag.endswith("svg")


def _evaluation(
    *,
    hit: int,
    f1: float,
    citations: int,
    signature: str,
) -> dict:
    aggregate = {
        "context_gold_hit_count": hit,
        "context_gold_hit_rate": hit / 10,
        "average_token_f1_all_answerable": f1,
        "average_token_f1_completed_answerable": f1,
        "gold_citation_count": citations,
        "answerable_refusal_count": 1,
        "unanswerable_false_answer_count": 0,
    }
    folds = {
        f"fold_{index}": {
            **aggregate,
            "context_gold_hit_rate": hit / 10,
        }
        for index in range(5)
    }
    private_case = SimpleNamespace(
        answerable=True,
        token_f1_all=f1,
        answer_signature=(signature,),
    )
    return {
        "aggregate": aggregate,
        "folds": folds,
        "private_cases": {"sample": private_case},
    }
