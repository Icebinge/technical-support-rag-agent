from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from typer.testing import CliRunner

from scripts.cross_validate_primeqa_hybrid_grouped_ranking import app, main
from ts_rag_agent.application import primeqa_hybrid_evidence_entailment_cv as stage172
from ts_rag_agent.application import primeqa_hybrid_grouped_ranking_cv as analysis
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application import primeqa_hybrid_supervised_cross_encoder_cv as stage174
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    ContextCandidateRecord,
)
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import PrimeQAHybridSplitSample


class FakeRankingTrainer:
    def __init__(self) -> None:
        self.fit_ids = []

    def fit_predict(
        self,
        *,
        family,
        training_rows,
        evaluation_rows,
        fit_id,
        training_fold_count,
        evaluation_fold_count,
        progress_sink,
    ):
        _ = progress_sink
        self.fit_ids.append((family, fit_id))
        groups = analysis.build_sampled_training_groups(training_rows)
        positive_groups = sum(analysis._positive_index(group) is not None for group in groups)
        positive_logit = 6.0 if family == "pairwise_anchor" else 5.0
        negative_logit = -2.0 if family == "pairwise_anchor" else -1.0
        predictions = {
            row.pair.private_identity: (
                positive_logit if row.pair.positive_label else negative_logit
            )
            for row in evaluation_rows
        }
        return predictions, analysis.RankingFitSummary(
            family=family,
            fit_id=fit_id,
            training_fold_count=training_fold_count,
            evaluation_fold_count=evaluation_fold_count,
            training_group_count=len(groups),
            positive_training_group_count=positive_groups,
            negative_training_group_count=len(groups) - positive_groups,
            training_pair_count=sum(len(group) for group in groups),
            evaluation_pair_count=len(evaluation_rows),
            optimizer_step_count=10,
            first_epoch_mean_loss=0.5,
            final_epoch_mean_loss=0.2,
            fit_seconds=1.0,
            inference_seconds=0.5,
        )


def test_frozen_family_margin_and_fit_contract() -> None:
    assert analysis._FAMILIES == ("pairwise_anchor", "listwise_none")
    assert len(analysis._MARGIN_THRESHOLDS) == 21
    assert analysis._MARGIN_THRESHOLDS[0] == -4.0
    assert analysis._MARGIN_THRESHOLDS[-1] == 8.0
    assert analysis._EXPECTED_NESTED_FITS == 50
    assert analysis._TRAIN_EPOCHS == 2
    assert analysis._TRAIN_PAIR_BUDGET == 32


def test_pairwise_objective_rewards_correct_order_and_none_boundary() -> None:
    objective = analysis.PairwiseAnchorObjective()
    good = objective.loss(
        logits=torch.tensor([3.0, -2.0, -1.0]),
        positive_index=0,
        torch_module=torch,
    )
    bad = objective.loss(
        logits=torch.tensor([-2.0, 3.0, 2.0]),
        positive_index=0,
        torch_module=torch,
    )
    negative_only_good = objective.loss(
        logits=torch.tensor([-3.0, -2.0]),
        positive_index=None,
        torch_module=torch,
    )
    negative_only_bad = objective.loss(
        logits=torch.tensor([2.0, 3.0]),
        positive_index=None,
        torch_module=torch,
    )

    assert good < bad
    assert negative_only_good < negative_only_bad


def test_listwise_objective_uses_explicit_none_choice() -> None:
    objective = analysis.ListwiseNoneObjective()
    positive_good = objective.loss(
        logits=torch.tensor([4.0, -1.0]),
        positive_index=0,
        torch_module=torch,
    )
    positive_bad = objective.loss(
        logits=torch.tensor([-1.0, 4.0]),
        positive_index=0,
        torch_module=torch,
    )
    none_good = objective.loss(
        logits=torch.tensor([-3.0, -2.0]),
        positive_index=None,
        torch_module=torch,
    )
    none_bad = objective.loss(
        logits=torch.tensor([3.0, 2.0]),
        positive_index=None,
        torch_module=torch,
    )

    assert positive_good < positive_bad
    assert none_good < none_bad


def test_sampled_groups_preserve_questions_and_pair_budget() -> None:
    rows = (
        *_pair_group("positive", positive_rank=3, count=8),
        *_pair_group("negative", positive_rank=None, count=8),
    )

    groups = analysis.build_sampled_training_groups(rows)
    batches = analysis._pack_group_batches(groups * 8)

    assert [len(group) for group in groups] == [2, 5]
    assert analysis._positive_index(groups[0]) is None
    assert analysis._positive_index(groups[1]) is not None
    assert all(sum(len(group) for group in batch) <= 32 for batch in batches)


def test_margin_view_cases_use_top1_against_top2_or_none() -> None:
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

    cases, scores = analysis.build_margin_view_cases(
        samples=(sample,),
        grouped_records={sample.sample_id: records},
        pair_logits=logits,
    )

    initial = next(case for case in cases if case.phase == "initial")
    final = next(case for case in cases if case.phase == "final")
    assert scores[initial.private_identity] == -1.0
    assert scores[final.private_identity] == 5.0
    assert initial.sufficient_label is False
    assert final.sufficient_label is True


def test_nested_protocol_runs_fifty_grouped_family_fits() -> None:
    samples, grouped_records, pair_rows = _nested_fixture()
    trainer = FakeRankingTrainer()

    result = analysis.run_grouped_nested_ranking(
        samples=samples,
        grouped_records=grouped_records,
        pair_rows=pair_rows,
        trainer=trainer,
        progress_sink=None,
    )

    assert len(trainer.fit_ids) == 50
    assert len(set(trainer.fit_ids)) == 50
    assert sum(family == "pairwise_anchor" for family, _ in trainer.fit_ids) == 25
    assert sum(family == "listwise_none" for family, _ in trainer.fit_ids) == 25
    assert len(result["fit_summaries"]) == 50
    assert len(result["outer_folds"]) == 5
    assert len(result["oof_cases"]) == len(samples) * 2
    assert all(row["selected_inner_eligible"] for row in result["outer_folds"])
    assert all(
        row["heldout_metrics"]["alternate_only_path_success_rate"] == 1.0
        for row in result["outer_folds"]
    )


def test_family_selection_requires_fold_safety_and_quality() -> None:
    samples, grouped_records, pair_rows = _nested_fixture()
    trainer = FakeRankingTrainer()
    result = analysis.run_grouped_nested_ranking(
        samples=samples,
        grouped_records=grouped_records,
        pair_rows=pair_rows,
        trainer=trainer,
        progress_sink=None,
    )

    rows = [
        analysis._select_threshold(
            family=family,
            cases=result["family_oof_cases"][family],
            scores=result["family_oof_scores"][family],
        )
        for family in analysis._FAMILIES
    ]
    selected = max(rows, key=analysis._family_selection_key)

    assert selected["eligible"] is True
    assert selected["safe_fold_count"] == 5
    assert selected["metrics"].alternate_only_path_success_rate == 1.0


def test_public_key_scan_rejects_group_training_labels() -> None:
    assert analysis._forbidden_keys_found({"nested": {"positive_index": 0}}) == {"positive_index"}
    assert not analysis._forbidden_keys_found({"optimizer_step_count": 100})


def test_visualizations_write_nine_parseable_svgs(tmp_path: Path) -> None:
    visualizations = analysis.write_stage175_visualizations(
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


def _pair_group(
    group_identity: str,
    *,
    positive_rank: int | None,
    count: int,
) -> tuple[stage174.PairFoldRow, ...]:
    return tuple(
        stage174.PairFoldRow(
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
            sample = _sample(sample_id, stratum)
            records = _records(sample_id, fold_id, initial_ranks=range(11, 21))
            samples.append(sample)
            grouped_records[sample_id] = records
            final = (*records[10:20], *records[:10])
            group_identity = stage172._sha256_text(sample_id)
            for record in final:
                positive = bool(sample.answerable and sample.answer_doc_id == record.document_id)
                pair_rows.append(
                    stage174.PairFoldRow(
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


def _sample(sample_id: str, stratum: str) -> PrimeQAHybridSplitSample:
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
            "selected_family_counts": {"pairwise_anchor": 3, "listwise_none": 2},
            "family_full_oof_diagnostics": {
                family: {"metrics": rates} for family in analysis._FAMILIES
            },
        },
        "stage174_comparison": {"stage174_oof_metrics": rates},
        "training_diagnostics": {
            "by_family": {family: {"final_epoch_loss_mean": 0.2} for family in analysis._FAMILIES}
        },
        "timing_seconds": {"candidate_replay": 90.0, "nested_ranking": 400.0},
        "resource_consumption": {
            "process_peak_working_set_bytes": 4 * 1024**3,
            "process_peak_private_usage_bytes": 6 * 1024**3,
            "gpu_peak_allocated_bytes": 2 * 1024**3,
            "gpu_peak_reserved_bytes": 3 * 1024**3,
            "minimum_system_available_memory_bytes": 3 * 1024**3,
        },
    }
