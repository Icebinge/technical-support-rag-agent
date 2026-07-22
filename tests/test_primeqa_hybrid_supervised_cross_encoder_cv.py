from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path

from typer.testing import CliRunner

from scripts.cross_validate_primeqa_hybrid_supervised_cross_encoder import app, main
from ts_rag_agent.application import primeqa_hybrid_evidence_entailment_cv as stage172
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application import primeqa_hybrid_supervised_cross_encoder_cv as analysis
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    ContextCandidateRecord,
)
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import PrimeQAHybridSplitSample


class FakeFoldTrainer:
    def __init__(self) -> None:
        self.fit_ids = []

    def fit_predict(
        self,
        *,
        training_rows,
        evaluation_rows,
        fit_id,
        training_fold_count,
        evaluation_fold_count,
        progress_sink,
    ):
        _ = progress_sink
        self.fit_ids.append(fit_id)
        selected = analysis.select_hard_negative_training_rows(training_rows)
        positives = sum(row.pair.positive_label for row in selected)
        predictions = {
            row.pair.private_identity: (0.9 if row.pair.positive_label else 0.1)
            for row in evaluation_rows
        }
        return predictions, analysis.FitSummary(
            fit_id=fit_id,
            training_fold_count=training_fold_count,
            evaluation_fold_count=evaluation_fold_count,
            training_pair_count=len(selected),
            positive_training_pair_count=positives,
            negative_training_pair_count=len(selected) - positives,
            evaluation_pair_count=len(evaluation_rows),
            optimizer_step_count=10,
            first_epoch_mean_loss=0.5,
            final_epoch_mean_loss=0.2,
            fit_seconds=1.0,
            inference_seconds=0.5,
        )


def test_frozen_threshold_and_training_contract() -> None:
    assert len(analysis._THRESHOLDS) == 21
    assert analysis._THRESHOLDS[-3:] == (0.95, 0.975, 0.99)
    assert analysis._TRAIN_EPOCHS == 2
    assert analysis._TRAIN_BATCH_SIZE == 32
    assert analysis._EXPECTED_NESTED_FITS == 25


def test_hard_negative_selection_uses_frozen_score_and_group_limits() -> None:
    rows = (
        *_pair_group("positive", positive_rank=3, count=8),
        *_pair_group("negative", positive_rank=None, count=8),
    )

    selected = analysis.select_hard_negative_training_rows(rows)
    positive_group = [row for row in selected if row.group_identity == "positive"]
    negative_group = [row for row in selected if row.group_identity == "negative"]

    assert len(positive_group) == 5
    assert sum(row.pair.positive_label for row in positive_group) == 1
    assert [row.frozen_score for row in positive_group if not row.pair.positive_label] == [
        8.0,
        7.0,
        5.0,
        4.0,
    ]
    assert len(negative_group) == 2
    assert [row.frozen_score for row in negative_group] == [8.0, 7.0]


def test_pair_fold_mapping_uses_question_group_fold() -> None:
    pairs = tuple(row.pair for row in _pair_group("group-a", positive_rank=1, count=3))
    frozen_scores = {pair.private_identity: float(index) for index, pair in enumerate(pairs)}
    base_cases = (
        _base_case("group-a", "fold-3", "initial", True),
        _base_case("group-a", "fold-3", "final", True),
    )

    rows = analysis.build_pair_fold_rows(
        pairs=pairs,
        base_cases=base_cases,
        frozen_scores=frozen_scores,
    )

    assert {row.fold_id for row in rows} == {"fold-3"}
    assert {row.group_identity for row in rows} == {"group-a"}


def test_probability_view_cases_use_max_visible_pair_probability() -> None:
    sample = _sample("sample", "fold-0", "alternate_only_gold_visible")
    records = _records(sample.sample_id, "fold-0", initial_ranks=range(11, 21))
    probabilities = {
        stage173._pair_identity(sample.sample_id, record.document_id): (
            0.9 if record.document_id == sample.answer_doc_id else 0.1
        )
        for record in (*records[10:20], *records[:10])
    }

    cases, scores = analysis.build_probability_view_cases(
        samples=(sample,),
        grouped_records={sample.sample_id: records},
        pair_probabilities=probabilities,
    )

    initial = next(case for case in cases if case.phase == "initial")
    final = next(case for case in cases if case.phase == "final")
    assert scores[initial.private_identity] == 0.1
    assert scores[final.private_identity] == 0.9
    assert initial.sufficient_label is False
    assert final.sufficient_label is True


def test_nested_protocol_runs_exactly_twenty_five_grouped_fits() -> None:
    samples, grouped_records, pair_rows = _nested_fixture()
    trainer = FakeFoldTrainer()

    result = analysis.run_grouped_nested_fine_tuning(
        samples=samples,
        grouped_records=grouped_records,
        pair_rows=pair_rows,
        trainer=trainer,
        progress_sink=None,
    )

    assert len(trainer.fit_ids) == 25
    assert len(set(trainer.fit_ids)) == 25
    assert len(result["fit_summaries"]) == 25
    assert len(result["outer_folds"]) == 5
    assert len(result["oof_cases"]) == len(samples) * 2
    assert len(result["oof_view_scores"]) == len(samples) * 2
    assert all(row["selected_threshold_inner_eligible"] for row in result["outer_folds"])
    assert all(
        row["heldout_metrics"]["alternate_only_path_success_rate"] == 1.0
        for row in result["outer_folds"]
    )


def test_threshold_selection_requires_every_fold_safety() -> None:
    samples, grouped_records, pair_rows = _nested_fixture()
    trainer = FakeFoldTrainer()
    result = analysis.run_grouped_nested_fine_tuning(
        samples=samples,
        grouped_records=grouped_records,
        pair_rows=pair_rows,
        trainer=trainer,
        progress_sink=None,
    )

    selected = analysis._select_threshold(
        cases=result["oof_cases"],
        scores=result["oof_view_scores"],
        thresholds=analysis._THRESHOLDS,
    )

    assert selected["eligible"] is True
    assert selected["safe_fold_count"] == 5
    assert selected["metrics"].alternate_only_path_success_rate == 1.0


def test_public_key_scan_rejects_training_pair_identity() -> None:
    assert analysis._forbidden_keys_found({"nested": {"training_pair_identity": "private"}}) == {
        "training_pair_identity"
    }
    assert not analysis._forbidden_keys_found({"optimizer_step_count": 100})


def test_visualizations_write_eight_parseable_svgs(tmp_path: Path) -> None:
    visualizations = analysis.write_stage174_visualizations(
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


def _pair_group(
    group_identity: str,
    *,
    positive_rank: int | None,
    count: int,
) -> tuple[analysis.PairFoldRow, ...]:
    return tuple(
        analysis.PairFoldRow(
            pair=stage173.SemanticPairInput(
                private_identity=f"{group_identity}:doc-{rank}",
                question_text="question",
                passage_text=f"document {rank}",
                positive_label=rank == positive_rank,
            ),
            fold_id="fold-0",
            group_identity=group_identity,
            frozen_score=float(count - rank + 1),
        )
        for rank in range(1, count + 1)
    )


def _base_case(
    group_identity: str,
    fold_id: str,
    phase: str,
    sufficient: bool,
) -> stage172.EvidenceViewCase:
    return stage172.EvidenceViewCase(
        private_identity=f"{group_identity}-{phase}",
        group_identity=group_identity,
        fold_id=fold_id,
        phase=phase,
        stratum="initial_gold_visible",
        features={},
        sufficient_label=sufficient,
    )


def _nested_fixture():
    samples = []
    grouped_records = {}
    pair_rows = []
    strata = (
        "initial_gold_visible",
        "alternate_only_gold_visible",
        "union_gold_missing_candidate_hit",
        "candidate_pool_gold_missing",
        "unanswerable",
    )
    for fold_index in range(5):
        fold_id = f"fold-{fold_index}"
        for stratum in strata:
            sample_id = f"{fold_id}-{stratum}"
            sample = _sample(sample_id, fold_id, stratum)
            records = _records(sample_id, fold_id, initial_ranks=range(11, 21))
            samples.append(sample)
            grouped_records[sample_id] = records
            final = (*records[10:20], *records[:10])
            group_identity = stage172._sha256_text(sample_id)
            for record in final:
                positive = bool(sample.answerable and sample.answer_doc_id == record.document_id)
                pair_rows.append(
                    analysis.PairFoldRow(
                        pair=stage173.SemanticPairInput(
                            private_identity=stage173._pair_identity(
                                sample_id,
                                record.document_id,
                            ),
                            question_text="question",
                            passage_text="document",
                            positive_label=positive,
                        ),
                        fold_id=fold_id,
                        group_identity=group_identity,
                        frozen_score=(10.0 if positive else 1.0 / record.baseline_rank),
                    )
                )
    return tuple(samples), grouped_records, tuple(pair_rows)


def _sample(
    sample_id: str,
    fold_id: str,
    stratum: str,
) -> PrimeQAHybridSplitSample:
    _ = fold_id
    answerable = stratum != "unanswerable"
    answer_doc_id = {
        "initial_gold_visible": f"{sample_id}-doc-015",
        "alternate_only_gold_visible": f"{sample_id}-doc-005",
        "union_gold_missing_candidate_hit": f"{sample_id}-doc-050",
        "candidate_pool_gold_missing": f"{sample_id}-doc-300",
        "unanswerable": None,
    }[stratum]
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
    fold_id: str,
    *,
    initial_ranks: range,
) -> tuple[ContextCandidateRecord, ...]:
    initial_rank_set = set(initial_ranks)
    return tuple(
        ContextCandidateRecord(
            sample_id=sample_id,
            fold_id=fold_id,
            document_id=f"{sample_id}-doc-{rank:03d}",
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
                {"heldout_fold": f"fold-{index}", "selected_threshold": 0.5} for index in range(5)
            ],
        },
        "stage173_comparison": {"stage173_oof_metrics": rates},
        "training_diagnostics": {
            "first_epoch_loss_mean": 0.5,
            "final_epoch_loss_mean": 0.2,
        },
        "timing_seconds": {
            "candidate_replay": 90.0,
            "nested_fine_tuning": 200.0,
        },
        "resource_consumption": {
            "process_peak_working_set_bytes": 4 * 1024**3,
            "process_peak_private_usage_bytes": 6 * 1024**3,
            "gpu_peak_allocated_bytes": 2 * 1024**3,
            "gpu_peak_reserved_bytes": 3 * 1024**3,
            "minimum_system_available_memory_bytes": 3 * 1024**3,
        },
    }
