from __future__ import annotations

import inspect
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from typer.testing import CliRunner

from scripts.cross_validate_primeqa_hybrid_view_calibration import app, main
from tests.test_primeqa_hybrid_grouped_ranking_cv import (
    FakeRankingTrainer,
    _nested_fixture,
    _records,
    _sample,
)
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application import primeqa_hybrid_view_calibration_cv as analysis


def test_frozen_policy_and_spec_contract() -> None:
    assert [policy.name for policy in analysis._POLICIES] == [
        "absolute_top1",
        "top1_top2_none_margin",
        "candidate_mass_vs_none",
        "bounded_absolute_relative",
    ]
    assert len(analysis._LOGIT_THRESHOLDS) == 21
    assert len(analysis._BOUNDED_THRESHOLDS) == 21
    assert analysis._BOUNDED_THRESHOLDS[0] == -1.0
    assert analysis._BOUNDED_THRESHOLDS[-1] == 1.0
    assert analysis._spec_count() == 84
    assert analysis._EXPECTED_NESTED_FITS == 25


def test_calibration_policy_scores_match_frozen_definitions() -> None:
    logits = (3.0, 1.0, -2.0)
    scores = {policy.name: policy.score(logits) for policy in analysis._POLICIES}

    assert scores["absolute_top1"] == 3.0
    assert scores["top1_top2_none_margin"] == 2.0
    assert math.isclose(
        scores["candidate_mass_vs_none"],
        3.0 + math.log(1.0 + math.exp(-2.0) + math.exp(-5.0)),
    )
    assert math.isclose(
        scores["bounded_absolute_relative"],
        0.5 * math.tanh(3.0 / 4.0) + 0.5 * math.tanh(2.0 / 4.0),
    )


def test_candidate_mass_increases_when_visible_evidence_is_added() -> None:
    policy = analysis.CandidateMassVsNonePolicy()

    initial = policy.score((2.0, 1.0, -1.0))
    final = policy.score((3.0, 2.0, 1.0, -1.0))

    assert final > initial


def test_view_cases_compute_all_policy_scores_from_same_logits() -> None:
    sample = _sample("sample", "alternate_only_gold_visible")
    records = _records(sample.sample_id, "fold-0", initial_ranks=range(11, 21))
    logits = {
        stage173._pair_identity(sample.sample_id, record.document_id): (
            5.0
            if record.document_id == sample.answer_doc_id
            else (-1.0 if record.baseline_rank % 2 else -2.0)
        )
        for record in (*records[10:20], *records[:10])
    }

    cases, scores = analysis.build_calibration_view_cases(
        samples=(sample,),
        grouped_records={sample.sample_id: records},
        pair_logits=logits,
    )

    initial = next(case for case in cases if case.phase == "initial")
    final = next(case for case in cases if case.phase == "final")
    assert scores["absolute_top1"][initial.private_identity] == -1.0
    assert scores["absolute_top1"][final.private_identity] == 5.0
    assert scores["top1_top2_none_margin"][initial.private_identity] == -1.0
    assert scores["top1_top2_none_margin"][final.private_identity] == 5.0
    assert initial.sufficient_label is False
    assert final.sufficient_label is True


def test_nested_protocol_runs_twenty_five_listwise_fits() -> None:
    samples, grouped_records, pair_rows = _nested_fixture()
    trainer = FakeRankingTrainer()

    result = analysis.run_grouped_nested_calibration(
        samples=samples,
        grouped_records=grouped_records,
        pair_rows=pair_rows,
        trainer=trainer,
        progress_sink=None,
    )

    assert len(trainer.fit_ids) == 25
    assert len(set(trainer.fit_ids)) == 25
    assert {family for family, _ in trainer.fit_ids} == {"listwise_none"}
    assert len(result["fit_summaries"]) == 25
    assert len(result["outer_folds"]) == 5
    assert len(result["oof_cases"]) == len(samples) * 2
    assert all(row["selected_inner_eligible"] for row in result["outer_folds"])
    assert all(
        row["heldout_metrics"]["alternate_only_path_success_rate"] == 1.0
        for row in result["outer_folds"]
    )


def test_policy_selection_requires_quality_and_fold_safety() -> None:
    samples, grouped_records, pair_rows = _nested_fixture()
    trainer = FakeRankingTrainer()
    result = analysis.run_grouped_nested_calibration(
        samples=samples,
        grouped_records=grouped_records,
        pair_rows=pair_rows,
        trainer=trainer,
        progress_sink=None,
    )

    selected = analysis._select_policy_threshold(
        cases=result["policy_oof_cases"],
        scores_by_policy=result["policy_oof_scores"],
    )

    assert selected["eligible"] is True
    assert selected["safe_fold_count"] == 5
    assert selected["metrics"].alternate_only_path_success_rate == 1.0
    assert len(selected["policy_diagnostics"]) == 4


def test_public_key_scan_rejects_raw_pair_logits() -> None:
    assert analysis._forbidden_keys_found({"nested": {"pair_logit": 3.0}}) == {"pair_logit"}
    assert not analysis._forbidden_keys_found({"selected_threshold": 1.0})


def test_visualizations_write_nine_parseable_svgs(tmp_path: Path) -> None:
    visualizations = analysis.write_stage176_visualizations(
        report=_visual_report(),
        output_dir=tmp_path,
    )

    assert len(visualizations) == 9
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


def _visual_report() -> dict:
    rates = {
        "balanced_accuracy": 0.8,
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
                if name != "balanced_accuracy"
            ],
            "outer_fold_metrics": {f"fold-{index}": rates for index in range(5)},
            "selected_policy_counts": {
                "absolute_top1": 1,
                "top1_top2_none_margin": 1,
                "candidate_mass_vs_none": 2,
                "bounded_absolute_relative": 1,
            },
            "policy_full_oof_diagnostics": {
                policy.name: {"metrics": rates} for policy in analysis._POLICIES
            },
        },
        "stage175_comparison": {"stage175_oof_metrics": rates},
        "training_diagnostics": {
            "first_epoch_loss_mean": 0.5,
            "final_epoch_loss_mean": 0.2,
        },
        "timing_seconds": {"candidate_replay": 90.0, "nested_calibration": 300.0},
        "resource_consumption": {
            "process_peak_working_set_bytes": 4 * 1024**3,
            "process_peak_private_usage_bytes": 6 * 1024**3,
            "gpu_peak_allocated_bytes": 2 * 1024**3,
            "gpu_peak_reserved_bytes": 3 * 1024**3,
            "minimum_system_available_memory_bytes": 3 * 1024**3,
        },
    }
