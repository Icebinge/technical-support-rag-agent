from __future__ import annotations

import hashlib
import inspect
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.analyze_primeqa_hybrid_bounded_iterative_agent import app, main
from ts_rag_agent.application import primeqa_hybrid_bounded_iterative_agent_validation as validation
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    ContextCandidateRecord,
)
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import PrimeQAHybridSplitSample


def test_coverage_measures_complementary_views_and_generation_miss_rescue() -> None:
    samples = (
        _sample("alternate-only", "doc-001"),
        _sample("initial-only", "doc-011"),
        _sample("both", "doc-001"),
    )
    grouped = {
        "alternate-only": _records("alternate-only", initial_from_rank=11),
        "initial-only": _records("initial-only", initial_from_rank=11),
        "both": _records("both", initial_from_rank=1),
    }
    private_rows = (
        _generation_miss_row("alternate-only", route="other"),
        _generation_miss_row("initial-only", route="error_or_log"),
    )

    coverage, misses, route_rescues = validation.analyze_stage168_coverage(
        samples=samples,
        grouped_records=grouped,
        private_rows=private_rows,
    )

    assert coverage.answerable_count == 3
    assert coverage.candidate_pool_gold_hit_count == 3
    assert coverage.initial_gold_hit_count == 2
    assert coverage.alternate_gold_hit_count == 2
    assert coverage.union_gold_hit_count == 3
    assert coverage.alternate_only_gold_hit_count == 1
    assert coverage.initial_only_gold_hit_count == 1
    assert coverage.both_views_gold_hit_count == 1
    assert coverage.neither_view_gold_hit_count == 0
    assert coverage.mean_view_overlap_count == pytest.approx(10 / 3, abs=1e-6)
    assert coverage.mean_union_context_count == pytest.approx(50 / 3, abs=1e-6)
    assert coverage.max_union_context_count == 20
    assert misses.eligible_count == 2
    assert misses.rescued_by_alternate_count == 1
    assert misses.still_missing_from_union_count == 1
    assert route_rescues == {"other": 1}


def test_visualizations_write_four_parseable_svgs(tmp_path: Path) -> None:
    report = {
        "coverage": {
            "candidate_pool_gold_hit_count": 345,
            "initial_gold_hit_count": 175,
            "alternate_gold_hit_count": 255,
            "union_gold_hit_count": 280,
            "both_views_gold_hit_count": 150,
            "alternate_only_gold_hit_count": 105,
            "initial_only_gold_hit_count": 25,
            "neither_view_gold_hit_count": 90,
        },
        "known_generation_miss_analysis": {
            "eligible_count": 138,
            "rescued_by_alternate_count": 40,
            "still_missing_from_union_count": 98,
        },
    }

    visualizations = validation.write_stage168_visualizations(report=report, output_dir=tmp_path)

    assert len(visualizations) == 4
    for visualization in visualizations:
        ET.parse(visualization.path)


def test_cli_exposes_no_development_or_test_inputs() -> None:
    result = CliRunner().invoke(app, ["--help"])
    parameters = inspect.signature(main).parameters

    assert result.exit_code == 0
    assert "user_confirmed_ac_fallback" in parameters
    assert not ({"dev", "development", "dev_split"} & set(parameters))
    assert not ({"test", "test_split"} & set(parameters))


def test_source_authorization_rejects_unknown_hash() -> None:
    fingerprints = {name: {"sha256": sha256} for name, sha256 in validation._SOURCE_HASHES.items()}
    fingerprints["train"] = {"sha256": "0" * 64}

    with pytest.raises(ValueError, match="source hash mismatch for train"):
        validation._authorize_sources(fingerprints)


def _sample(sample_id: str, gold_document_id: str) -> PrimeQAHybridSplitSample:
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
        answerable=True,
        answer="answer",
        answer_doc_id=gold_document_id,
        candidate_doc_ids=(),
        start_offset=None,
        end_offset=None,
    )


def _records(sample_id: str, *, initial_from_rank: int) -> tuple[ContextCandidateRecord, ...]:
    initial_ranks = set(range(initial_from_rank, initial_from_rank + 10))
    return tuple(
        ContextCandidateRecord(
            sample_id=sample_id,
            fold_id="fold-0",
            document_id=f"doc-{rank:03d}",
            baseline_rank=rank,
            answerable=True,
            is_gold=False,
            features={
                "current_query_overlap_combined_score": float(rank in initial_ranks),
            },
        )
        for rank in range(1, 201)
    )


def _generation_miss_row(sample_id: str, *, route: str) -> dict[str, object]:
    return {
        "private_identity_sha256": hashlib.sha256(sample_id.encode("utf-8")).hexdigest(),
        "arm": "isolated",
        "answerable": True,
        "synthetic_turn_position": 2,
        "gold_candidate_rank": 20,
        "gold_generation_rank": None,
        "question_route": route,
    }
