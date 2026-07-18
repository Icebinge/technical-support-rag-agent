from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from typer.testing import CliRunner

from scripts.train_primeqa_hybrid_protected_context_selector import app
from ts_rag_agent.application import (
    primeqa_hybrid_protected_context_selector_training as training,
)
from ts_rag_agent.application.primeqa_hybrid_protected_context_selector import (
    RUNTIME_FEATURE_NAMES,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQADocumentSection
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import PrimeQAHybridSplitSample


def test_runtime_feature_extractor_reproduces_query_overlap_formula() -> None:
    document = PrimeQADocument(
        id="doc-1",
        title="GPU firmware update",
        text="Install package ABC-123 and restart the service.",
    )
    extractor = training.RuntimeVisibleCandidateFeatureExtractor(
        documents_by_id={document.id: document},
        sections_by_document={
            document.id: [
                PrimeQADocumentSection(
                    document_id=document.id,
                    section_id="Firmware installation",
                    text=document.text,
                )
            ]
        },
    )

    features = extractor.extract(
        query="GPU ABC-123 restart",
        document_id=document.id,
        baseline_rank=8,
        rrf_score=0.25,
        route_rank_maps={
            "full_document_bm25": {document.id: 3},
            "dense_cache__encoder": {document.id: 7},
        },
        route_score_maps={
            "full_document_bm25": {document.id: 2.0},
            "dense_cache__encoder": {document.id: 0.8},
        },
    )

    overlap_count = features["current_query_overlap_count"]
    expected = overlap_count + features["current_query_overlap_ratio"] + 0.35 / math.log2(9)
    assert features["current_query_overlap_combined_score"] == expected
    assert features["route_hit_count"] == 2.0
    assert features["lexical_route_hit_count"] == 1.0
    assert features["dense_route_hit_count"] == 1.0
    assert features["bm25_top10_indicator"] == 1.0
    assert set(RUNTIME_FEATURE_NAMES) <= set(features)


def test_strict_guards_require_improvement_without_quality_regression() -> None:
    current = _evaluation(hit=5, f1=0.2, citations=3, refusals=2, false_answers=1)
    rrf = _evaluation(hit=4, f1=0.18, citations=2, refusals=3, false_answers=1)
    candidate = _evaluation(hit=6, f1=0.2, citations=3, refusals=2, false_answers=1)
    comparison = {
        "vs_current_query_overlap": {"context_gold_hit_count_delta": 1},
        "minimum_fold_hit_rate_delta": 0.0,
        "minimum_fold_f1_delta": 0.0,
    }

    guards = training._config_guard_results(
        evaluation=candidate,
        current=current,
        rrf=rrf,
        comparison=comparison,
    )

    assert all(guards.values())
    candidate["aggregate"]["gold_citation_count"] = 2
    guards = training._config_guard_results(
        evaluation=candidate,
        current=current,
        rrf=rrf,
        comparison=comparison,
    )
    assert guards["gold_citations_not_below_current"] is False


def test_config_selection_uses_frozen_lexicographic_order() -> None:
    slower = _config_result("slower", hit=7, f1=0.21, citations=4, latency=2.0)
    faster = _config_result("faster", hit=7, f1=0.21, citations=4, latency=1.0)
    lower_hit = _config_result("lower_hit", hit=6, f1=0.9, citations=9, latency=0.1)

    selection = training._select_config([slower, faster, lower_hit])

    assert selection["status"] == "train_cv_safe_config_selected"
    assert selection["selected_config_id"] == "faster"
    assert selection["dev_used"] is False
    assert selection["test_used"] is False


def test_grouped_fold_summary_detects_cross_fold_duplicate_group() -> None:
    samples = [
        _sample("sample-a", title="Same Question", document_id="doc-1"),
        _sample("sample-b", title="same question", document_id="doc-1"),
    ]

    summary = training._fold_assignment_summary(
        samples,
        {"sample-a": "fold_0", "sample-b": "fold_1"},
    )

    assert summary["group_count"] == 1
    assert summary["cross_fold_group_violation_count"] == 1
    assert summary["raw_group_values_written"] is False


def test_protocol_and_public_contract_close_dev_test_and_gold_features() -> None:
    protocol = training._frozen_protocol()
    safe = training._public_safe_contract(
        {
            "frozen_protocol": protocol,
            "aggregate": {"count": 1},
        }
    )

    assert protocol["selection_split"] == "train"
    assert protocol["blocked"]["dev_load"] is True
    assert protocol["blocked"]["test_load"] is True
    assert protocol["blocked"]["fallback"] is True
    assert all("gold" not in feature.lower() for feature in protocol["runtime_features"])
    assert safe["public_safe"] is True
    assert training._public_safe_contract({"sample_id": "private"})["public_safe"] is False


def test_stage161_visualizations_write_ten_parseable_svgs(tmp_path: Path) -> None:
    aggregate = {
        "context_gold_hit_rate": 0.7,
        "average_token_f1_all_answerable": 0.2,
        "gold_citation_count": 3,
        "answerable_refusal_count": 2,
        "unanswerable_false_answer_count": 1,
        "average_tail_promotion_count": 2.5,
        "selection_latency_average_ms": 0.4,
    }
    report = {
        "control_results": {
            "current": {"aggregate": aggregate},
            "rrf": {"aggregate": aggregate},
        },
        "config_results": [
            {
                "config": {"config_id": "candidate"},
                "train_oof_metrics": {"aggregate": aggregate},
                "comparison": {"minimum_fold_hit_rate_delta": 0.01},
                "train_cv_selectable": True,
            }
        ],
        "guard_checks": [{"name": "train_only", "passed": True}],
    }

    visualizations = training.write_stage161_visualizations(
        report=report,
        output_dir=tmp_path,
    )

    assert len(visualizations) == 10
    for visualization in visualizations:
        path = Path(visualization.path)
        assert path.is_file()
        assert ET.parse(path).getroot().tag.endswith("svg")


def test_stage161_cli_has_no_dev_or_test_input_options() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--train-split" in result.stdout
    assert "--dev" not in result.stdout
    assert "--test" not in result.stdout


def _evaluation(
    *,
    hit: int,
    f1: float,
    citations: int,
    refusals: int,
    false_answers: int,
) -> dict:
    return {
        "aggregate": {
            "context_gold_hit_count": hit,
            "average_token_f1_all_answerable": f1,
            "gold_citation_count": citations,
            "answerable_refusal_count": refusals,
            "unanswerable_false_answer_count": false_answers,
            "protected_prefix_violation_count": 0,
        }
    }


def _config_result(
    config_id: str,
    *,
    hit: int,
    f1: float,
    citations: int,
    latency: float,
) -> dict:
    return {
        "config": {"config_id": config_id},
        "train_cv_selectable": True,
        "train_oof_metrics": {
            "aggregate": {
                "context_gold_hit_count": hit,
                "average_token_f1_all_answerable": f1,
                "gold_citation_count": citations,
                "unanswerable_false_answer_count": 0,
                "average_tail_promotion_count": 1.0,
                "selection_latency_average_ms": latency,
            }
        },
    }


def _sample(sample_id: str, *, title: str, document_id: str) -> PrimeQAHybridSplitSample:
    return PrimeQAHybridSplitSample(
        split_name="primeqa_hybrid_stage68_v1",
        protocol_version="primeqa_hybrid_split_v1",
        assigned_split="train",
        split_subtype="random_grouped",
        source_split="train",
        sample_id=sample_id,
        question_id=sample_id,
        question_title=title,
        question_text="",
        answerable=True,
        answer="answer",
        answer_doc_id=document_id,
        candidate_doc_ids=(document_id,),
        start_offset=None,
        end_offset=None,
    )
