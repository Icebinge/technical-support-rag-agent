from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path

from typer.testing import CliRunner

from scripts.evaluate_primeqa_hybrid_listwise_reranker import app, main
from tests.test_primeqa_hybrid_grouped_ranking_cv import (
    FakeRankingTrainer,
    _nested_fixture,
)
from ts_rag_agent.application import primeqa_hybrid_listwise_reranker_cv as analysis


def test_frozen_oof_metric_and_gate_contract() -> None:
    assert analysis._EXPECTED_OOF_FITS == 5
    assert analysis._METHODS == (
        "original_rrf",
        "frozen_cross_encoder",
        "listwise_oof",
    )
    assert analysis._RECALL_DEPTHS == (1, 3, 5, 10)
    assert analysis._BOOTSTRAP_REPLICATES == 2_000
    assert analysis._MRR_FOLD_WIN_MINIMUM == 4
    assert analysis._RECALL10_NONINFERIORITY_MARGIN == -0.02


def test_grouped_oof_runs_exactly_five_listwise_fits() -> None:
    samples, grouped_records, pair_rows = _nested_fixture()
    trainer = FakeRankingTrainer()

    result = analysis.run_grouped_oof_listwise_reranking(
        samples=samples,
        grouped_records=grouped_records,
        pair_rows=pair_rows,
        trainer=trainer,
        progress_sink=None,
    )

    assert len(trainer.fit_ids) == 5
    assert len(set(trainer.fit_ids)) == 5
    assert {family for family, _ in trainer.fit_ids} == {"listwise_none"}
    assert len(result["fit_summaries"]) == 5
    assert len(result["pair_logits"]) == len(pair_rows)


def test_query_ranks_keep_missing_gold_out_of_conditional_metrics() -> None:
    samples, grouped_records, pair_rows = _nested_fixture()
    frozen_scores = {row.pair.private_identity: row.frozen_score for row in pair_rows}
    listwise_logits = {
        row.pair.private_identity: (8.0 if row.pair.positive_label else -1.0) for row in pair_rows
    }

    rows = analysis.build_query_rank_rows(
        samples=samples,
        grouped_records=grouped_records,
        frozen_scores=frozen_scores,
        listwise_logits=listwise_logits,
    )
    metrics = analysis.evaluate_reranking(rows)

    assert metrics["answerable_query_count"] == 20
    assert metrics["gold_present_query_count"] == 10
    assert metrics["candidate_pool_gold_coverage"] == 0.5
    assert metrics["method_metrics"]["listwise_oof"]["conditional_recall_at"]["1"] == 1.0
    assert metrics["method_metrics"]["listwise_oof"]["all_answerable_recall_at"]["1"] == 0.5


def test_paired_bootstrap_and_fold_gates_pass_for_consistent_improvement() -> None:
    rows = tuple(
        analysis.QueryRankRow(
            fold_id=f"fold-{index % 5}",
            answerable=True,
            gold_present=True,
            original_rrf_rank=3,
            frozen_cross_encoder_rank=2,
            listwise_oof_rank=1,
        )
        for index in range(100)
    )

    evaluation = analysis.evaluate_reranking(rows)

    assert evaluation["all_quality_gates_passed"] is True
    assert all(gate["passed"] for gate in evaluation["quality_gates"])
    assert (
        evaluation["paired_bootstrap"]["original_rrf"]["metrics"]["mean_reciprocal_rank"][
            "ci95_lower"
        ]
        > 0
    )


def test_recall10_gate_allows_only_frozen_two_point_noninferiority() -> None:
    gate = analysis._gate("noninferior", -0.02, -0.02, "ge")
    failed = analysis._gate("noninferior", -0.021, -0.02, "ge")

    assert gate["passed"] is True
    assert failed["passed"] is False


def test_public_key_scan_rejects_private_gold_document() -> None:
    assert analysis._forbidden_keys_found({"nested": {"gold_document_id": "private"}}) == {
        "gold_document_id"
    }
    assert not analysis._forbidden_keys_found({"gold_present_query_count": 267})


def test_visualizations_write_eight_parseable_svgs(tmp_path: Path) -> None:
    visualizations = analysis.write_stage177_visualizations(
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


def _visual_report() -> dict:
    method_row = {
        "mean_reciprocal_rank": 0.7,
        "conditional_recall_at": {str(depth): 0.8 for depth in analysis._RECALL_DEPTHS},
        "all_answerable_recall_at": {str(depth): 0.5 for depth in analysis._RECALL_DEPTHS},
    }
    methods = {method: method_row for method in analysis._METHODS}
    return {
        "reranking_evaluation": {
            "method_metrics": methods,
            "fold_metrics": {f"fold-{index}": methods for index in range(5)},
            "quality_gates": [
                {"name": "mrr", "observed": 0.1, "passed": True},
                {"name": "recall", "observed": 0.1, "passed": True},
            ],
        },
        "training_diagnostics": {
            "first_epoch_loss_mean": 0.5,
            "final_epoch_loss_mean": 0.2,
        },
        "timing_seconds": {"candidate_replay": 90.0, "listwise_grouped_oof": 150.0},
        "resource_consumption": {
            "process_peak_working_set_bytes": 4 * 1024**3,
            "process_peak_private_usage_bytes": 6 * 1024**3,
            "gpu_peak_allocated_bytes": 2 * 1024**3,
            "gpu_peak_reserved_bytes": 3 * 1024**3,
            "minimum_system_available_memory_bytes": 3 * 1024**3,
        },
    }
