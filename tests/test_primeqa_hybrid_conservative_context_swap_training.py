from __future__ import annotations

import inspect
import xml.etree.ElementTree as ET
from pathlib import Path

from typer.testing import CliRunner

from scripts.train_primeqa_hybrid_conservative_context_swap import app
from ts_rag_agent.application import (
    primeqa_hybrid_conservative_context_swap_training as training,
)
from ts_rag_agent.application.primeqa_hybrid_conservative_context_swap_selector import (
    ConservativeSwapSelectorConfig,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    ContextCandidateRecord,
    ScorerFitSummary,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector_training import (
    Stage161TrainCandidateDatasetBuilder,
)
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import PrimeQAHybridSplitSample


class _FakeScorer:
    def fit(self, records, *, protected_prefix_depth: int) -> ScorerFitSummary:
        return ScorerFitSummary(
            model_family="fake",
            training_group_count=len({record.sample_id for record in records}),
            positive_candidate_count=1,
            negative_candidate_count=1,
            training_example_count=2,
            feature_count=1,
        )

    def score(self, records) -> list[float]:
        return [float(record.features["utility"]) for record in records]


def test_nested_outer_run_uses_twenty_inner_fits_and_five_outer_refits(
    monkeypatch,
) -> None:
    samples = [_sample(index) for index in range(5)]
    records = tuple(
        record for index in range(5) for record in _candidate_pool(index, f"fold_{index}")
    )
    monkeypatch.setattr(training, "create_candidate_scorer", lambda family: _FakeScorer())
    monkeypatch.setattr(
        training,
        "_evaluate_threshold_candidates",
        lambda **kwargs: [
            {
                "threshold": 0.0,
                "metrics": _aggregate(hit=3, f1=0.2, citations=2),
            }
        ],
    )
    config = ConservativeSwapSelectorConfig(
        config_id="test_nested",
        model_family="pairwise_logistic",
        protected_prefix_depth=9,
        promotion_budget=1,
    )

    result = training._nested_outer_oof_selection_run(
        config=config,
        records=records,
        samples=samples,
        documents_by_id={},
    )

    assert len(result.nested_fit_summaries) == 25
    assert len(result.outer_fold_summaries) == 5
    assert set(result.selection_run.selections) == {sample.sample_id for sample in samples}
    assert all(summary["inner_fold_count"] == 4 for summary in result.outer_fold_summaries)
    assert all(
        summary["outer_train_row_count"] == 4
        and summary["outer_validation_row_count"] == 1
        and summary["outer_validation_used_for_threshold_selection"] is False
        for summary in result.outer_fold_summaries
    )


def test_inner_threshold_selection_uses_frozen_lexicographic_order() -> None:
    lower_hit = {"threshold": 0.9, "metrics": _aggregate(hit=4, f1=0.9, citations=9)}
    slower_swap = {
        "threshold": 0.2,
        "metrics": _aggregate(hit=5, f1=0.3, citations=4, swaps=0.8),
    }
    conservative = {
        "threshold": 0.8,
        "metrics": _aggregate(hit=5, f1=0.3, citations=4, swaps=0.2),
    }

    selected = training._select_inner_threshold([lower_hit, slower_swap, conservative])

    assert selected["threshold"] == 0.8


def test_stage162_strict_guards_require_improvement_over_rrf() -> None:
    rrf = {"aggregate": _aggregate(hit=5, f1=0.2, citations=3)}
    candidate = {"aggregate": _aggregate(hit=6, f1=0.2, citations=3)}
    comparison = {
        "vs_untouched_rrf": {"context_gold_hit_count_delta": 1},
        "minimum_fold_hit_rate_delta_vs_rrf": 0.0,
        "minimum_fold_f1_delta_vs_rrf": 0.0,
    }
    swap_audit = {
        "protected_prefix_violation_count": 0,
        "promotion_budget_violation_count": 0,
    }
    outer = [
        {
            "inner_fold_count": 4,
            "outer_validation_used_for_threshold_selection": False,
            "dev_used": False,
            "test_used": False,
            "selected_threshold": 0.1,
        }
        for _ in range(5)
    ]

    guards = training._config_guard_results(
        evaluation=candidate,
        rrf=rrf,
        comparison=comparison,
        swap_audit=swap_audit,
        outer_fold_summaries=outer,
    )

    assert all(guards.values())
    comparison["vs_untouched_rrf"]["context_gold_hit_count_delta"] = 0
    guards = training._config_guard_results(
        evaluation=candidate,
        rrf=rrf,
        comparison=comparison,
        swap_audit=swap_audit,
        outer_fold_summaries=outer,
    )
    assert guards["context_hit_strictly_improves_untouched_rrf"] is False


def test_stage162_protocol_closes_dev_test_runtime_and_fallback() -> None:
    protocol = training._frozen_protocol()

    assert protocol["selection_split"] == "train"
    assert protocol["outer_cv"] == "grouped_five_fold_out_of_fold_family_evaluation"
    assert protocol["threshold_grid"]["positive_margin_quantiles"] == [
        0.5,
        0.7,
        0.8,
        0.9,
        0.95,
    ]
    assert protocol["blocked"] == {
        "dev_load": True,
        "test_load": True,
        "fallback": True,
        "runtime_defaultization": True,
        "query_rewrite": True,
        "second_retrieval": True,
    }


def test_stage162_visualizations_write_ten_parseable_svgs(tmp_path: Path) -> None:
    aggregate = _aggregate(hit=5, f1=0.2, citations=3)
    aggregate.update(
        {
            "context_gold_hit_rate": 0.5,
            "selection_latency_average_ms": 0.4,
        }
    )
    report = {
        "control_results": {
            "current": {"aggregate": aggregate},
            "rrf": {"aggregate": aggregate},
        },
        "config_results": [
            {
                "config": {"config_id": "candidate"},
                "train_nested_oof_metrics": {"aggregate": aggregate},
                "nested_fit_summary": {"selected_threshold_average": 0.2},
                "comparison": {
                    "minimum_fold_hit_rate_delta_vs_rrf": 0.01,
                    "minimum_fold_f1_delta_vs_rrf": 0.02,
                },
                "train_nested_cv_selectable": True,
            }
        ],
        "guard_checks": [{"name": "train_only", "passed": True}],
    }

    visualizations = training.write_stage162_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 10
    for visualization in visualizations:
        path = Path(visualization.path)
        assert path.is_file()
        assert ET.parse(path).getroot().tag.endswith("svg")


def test_stage162_cli_has_no_dev_or_test_split_options() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--train-split" in result.stdout
    assert "--dev" not in result.stdout
    assert "--test" not in result.stdout


def test_stage162_explicitly_relabels_reused_candidate_builder_progress() -> None:
    constructor = inspect.signature(Stage161TrainCandidateDatasetBuilder.__init__)
    stage162_source = inspect.getsource(
        training.run_primeqa_hybrid_conservative_context_swap_training
    )

    assert constructor.parameters["progress_stage"].default == "Stage 161"
    assert "progress_stage=_STAGE" in stage162_source


def test_candidate_builder_rejects_empty_progress_stage() -> None:
    try:
        Stage161TrainCandidateDatasetBuilder(
            documents_by_id={},
            sections_by_document={},
            channels=(),
            fold_assignments={},
            progress_stage="",
        )
    except ValueError as error:
        assert str(error) == "candidate builder progress stage must not be empty"
    else:
        raise AssertionError("empty progress stage must be rejected")


def _aggregate(
    *,
    hit: int,
    f1: float,
    citations: int,
    swaps: float = 0.5,
) -> dict:
    return {
        "context_gold_hit_count": hit,
        "average_token_f1_all_answerable": f1,
        "gold_citation_count": citations,
        "answerable_refusal_count": 1,
        "unanswerable_false_answer_count": 1,
        "average_tail_promotion_count": swaps,
        "selection_latency_average_ms": 0.5,
    }


def _sample(index: int) -> PrimeQAHybridSplitSample:
    return PrimeQAHybridSplitSample(
        split_name="primeqa_hybrid_stage68_v1",
        protocol_version="primeqa_hybrid_split_v1",
        assigned_split="train",
        split_subtype="random_grouped",
        source_split="train",
        sample_id=f"sample_{index}",
        question_id=f"question_{index}",
        question_title=f"Question {index}",
        question_text="text",
        answerable=True,
        answer="answer",
        answer_doc_id=f"sample_{index}_doc_050",
        candidate_doc_ids=(f"sample_{index}_doc_050",),
        start_offset=None,
        end_offset=None,
    )


def _candidate_pool(index: int, fold_id: str) -> tuple[ContextCandidateRecord, ...]:
    return tuple(
        ContextCandidateRecord(
            sample_id=f"sample_{index}",
            fold_id=fold_id,
            document_id=f"sample_{index}_doc_{rank:03d}",
            baseline_rank=rank,
            answerable=True,
            is_gold=rank == 50,
            features={"utility": 1.0 if rank == 50 else -float(rank)},
        )
        for rank in range(1, 201)
    )
