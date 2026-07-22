from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.evaluate_primeqa_hybrid_listwise_agent_e2e import app, main
from ts_rag_agent.application import primeqa_hybrid_grouped_ranking_cv as stage175
from ts_rag_agent.application import primeqa_hybrid_listwise_agent_e2e as analysis
from ts_rag_agent.application import primeqa_hybrid_semantic_evidence_cv as stage173
from ts_rag_agent.application.listwise_runtime_reranker import (
    LISTWISE_CONTEXT_DEPTH,
    ListwiseUnionPrimaryContextSelectionPolicy,
    PrecomputedListwiseScoreProvider,
    checkpoint_public_summary,
    load_and_validate_listwise_checkpoint_manifest,
    write_listwise_checkpoint_manifest,
)
from ts_rag_agent.application.primeqa_hybrid_sidecar_observation_validation import (
    PrimeQAHybridSidecarObservationAdapter,
)
from ts_rag_agent.domain.answer import AnswerCitation, GeneratedAnswer
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult
from ts_rag_agent.infrastructure.primeqa_hybrid_split_loader import PrimeQAHybridSplitSample


def test_default_adapter_remains_query_overlap_top10() -> None:
    query = PrimeQARuntimeQuery(id="q", text="target")
    candidates = _candidates()

    bundle = PrimeQAHybridSidecarObservationAdapter().observe(
        question=query,
        candidate_pool_results=candidates,
    )

    assert len(bundle.answer_context_results) == LISTWISE_CONTEXT_DEPTH
    assert bundle.answer_context_results[0].document.id == "doc-11"


def test_listwise_policy_reranks_only_union_of_two_top10_views() -> None:
    query = PrimeQARuntimeQuery(id="q", text="target")
    candidates = _candidates()
    scores = {
        stage173._pair_identity("q", result.document.id): (
            100.0 if result.document.id == "doc-10" else float(-result.rank)
        )
        for result in candidates
    }
    provider = PrecomputedListwiseScoreProvider(scores)
    adapter = PrimeQAHybridSidecarObservationAdapter(
        primary_context_selection_policy=ListwiseUnionPrimaryContextSelectionPolicy(
            score_provider=provider
        )
    )

    bundle = adapter.observe(question=query, candidate_pool_results=candidates)

    assert len(bundle.answer_context_results) == LISTWISE_CONTEXT_DEPTH
    assert bundle.answer_context_results[0].document.id == "doc-10"
    assert "doc-21" not in {result.document.id for result in bundle.answer_context_results}
    assert provider.counters().call_count == 1
    assert provider.counters().pair_count == 20


def test_checkpoint_manifest_authenticates_every_exported_file(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    manifest = write_listwise_checkpoint_manifest(
        checkpoint_path=tmp_path,
        stage177_report_sha256="a" * 64,
        train_source_sha256="b" * 64,
        training_row_count=562,
        training_pair_count=1_925,
        optimizer_step_count=122,
        first_epoch_mean_loss=1.2,
        final_epoch_mean_loss=0.8,
    )

    loaded = load_and_validate_listwise_checkpoint_manifest(tmp_path)
    summary = checkpoint_public_summary(tmp_path)

    assert loaded == manifest
    assert summary["all_file_hashes_valid"] is True
    assert summary["file_count"] == 3
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_and_validate_listwise_checkpoint_manifest(tmp_path)


def test_agent_e2e_metrics_keep_context_and_answer_quality_separate() -> None:
    baseline = []
    candidate = []
    for index in range(10):
        sample = _sample(index)
        baseline.append(_trace(sample=sample, fold_id=f"fold_{index % 5}", improved=False))
        candidate.append(_trace(sample=sample, fold_id=f"fold_{index % 5}", improved=True))

    result = analysis.evaluate_agent_e2e(
        {"baseline": tuple(baseline), "candidate": tuple(candidate)}
    )

    assert result["paired_question_count"] == 10
    assert result["deltas"]["context_gold_hit_count"] == 10
    assert result["deltas"]["gold_citation_count"] == 10
    assert result["deltas"]["verified_average_token_f1"] > 0
    assert result["paired_bootstrap"]["metrics"]["answer_f1"]["ci95_lower"] > 0


def test_stage178a_protocol_is_strict_and_train_only() -> None:
    assert analysis._BOOTSTRAP_REPLICATES == 2_000
    assert analysis._MINIMUM_STABLE_FOLDS == 4
    assert analysis._EXPECTED_OOF_FITS == 5
    assert analysis._EXPECTED_FULL_FITS == 1
    assert analysis._CPU_RERANK_P95_LIMIT_SECONDS == 1.0
    assert (
        analysis._SOURCE_HASHES["stage178_alignment"]
        == "e2398024edf128ad0628900d25eb1ccc9c83c437fb474921fe136e2603e47272"
    )
    assert (
        "checkpoint_path"
        in inspect.signature(stage175.LocalGroupedRankingTrainer.fit_predict).parameters
    )


def test_cli_exposes_no_development_or_test_input() -> None:
    result = CliRunner().invoke(app, ["--help"])
    parameters = set(inspect.signature(main).parameters)

    assert result.exit_code == 0
    assert parameters == {
        "model_snapshot",
        "checkpoint",
        "output",
        "private_output",
        "visualization_dir",
        "encoder_batch_size",
    }


def _candidates() -> tuple[RetrievalResult, ...]:
    return tuple(
        RetrievalResult(
            document=PrimeQADocument(
                id=f"doc-{rank}",
                title=f"Document {rank}",
                text=("target " if rank >= 11 else "unrelated ") + f"body {rank}",
            ),
            score=1.0 / rank,
            rank=rank,
        )
        for rank in range(1, 401)
    )


def _sample(index: int) -> PrimeQAHybridSplitSample:
    return PrimeQAHybridSplitSample(
        split_name="fixture",
        protocol_version="v1",
        assigned_split="train",
        split_subtype="random",
        source_split="train",
        sample_id=f"sample-{index}",
        question_id=f"question-{index}",
        question_title="Install",
        question_text="How do I install it?",
        answerable=True,
        answer="run installer",
        answer_doc_id=f"gold-{index}",
        candidate_doc_ids=(),
        start_offset=None,
        end_offset=None,
    )


def _trace(
    *,
    sample: PrimeQAHybridSplitSample,
    fold_id: str,
    improved: bool,
) -> analysis.AgentE2ETrace:
    gold = PrimeQADocument(
        id=sample.answer_doc_id or "",
        title="Gold",
        text="run installer",
    )
    distractor = PrimeQADocument(id="distractor", title="Other", text="other")
    context_document = gold if improved else distractor
    answer = GeneratedAnswer(
        question_id=sample.sample_id,
        answer="run installer" if improved else "wrong",
        citations=(
            [
                AnswerCitation(
                    document_id=gold.id,
                    title=gold.title,
                    retrieval_rank=1,
                    evidence_score=10.0,
                )
            ]
            if improved
            else []
        ),
        refused=False,
    )
    result = RetrievalResult(document=context_document, score=1.0, rank=1)
    return analysis.AgentE2ETrace(
        sample=sample,
        fold_id=fold_id,
        question=sample.to_primeqa_question(),
        candidate_pool_results=(result,),
        generation_context_results=(result,),
        verified_answer=answer,
        latency_seconds=0.1,
        tool_call_count=3,
        retry_action_count=0,
        fallback_action_count=0,
    )
