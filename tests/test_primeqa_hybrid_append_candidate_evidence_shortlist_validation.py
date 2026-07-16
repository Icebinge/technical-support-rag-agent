from __future__ import annotations

from ts_rag_agent.application.primeqa_hybrid_append_candidate_evidence_shortlist_validation import (
    _AppendCandidateEvidenceShortlister,
    _check,
    _decision,
    _public_safe_contract,
    _select_config_on_train,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQAQuestion
from ts_rag_agent.domain.retrieval import RetrievalResult


def test_append_shortlister_keeps_sidecar_config_prefix_only() -> None:
    question = _question()
    candidates = _candidate_results()
    shortlister = _AppendCandidateEvidenceShortlister(
        protected_prefix_slots=10,
        replacement_append_slots=0,
    )

    selected = shortlister.shortlist(question=question, candidates=candidates, top_k=10)

    assert len(selected) == 10
    assert all(result.rank <= 200 for result in selected)


def test_append_shortlister_allows_bounded_high_signal_append_replacement() -> None:
    question = _question()
    candidates = _candidate_results()
    shortlister = _AppendCandidateEvidenceShortlister(
        protected_prefix_slots=9,
        replacement_append_slots=1,
    )

    selected = shortlister.shortlist(question=question, candidates=candidates, top_k=10)

    assert len(selected) == 10
    assert sum(result.rank > 200 for result in selected) == 1
    assert any(result.document.id == "append-gold-like" for result in selected)


def test_append_shortlister_caps_append_replacements_at_config_budget() -> None:
    question = _question()
    candidates = _candidate_results(extra_high_append=True)
    shortlister = _AppendCandidateEvidenceShortlister(
        protected_prefix_slots=8,
        replacement_append_slots=2,
    )

    selected = shortlister.shortlist(question=question, candidates=candidates, top_k=10)

    assert len(selected) == 10
    assert sum(result.rank > 200 for result in selected) == 2


def test_train_selection_uses_guard_then_train_metrics_only() -> None:
    reviews = [
        _review("blocked", guard=False, citation_delta=4, f1_delta=0.3, churn=0.1),
        _review("safe-weak", guard=True, citation_delta=0, f1_delta=0.01, churn=0.1),
        _review("safe-strong", guard=True, citation_delta=1, f1_delta=0.0, churn=0.3),
    ]

    selection = _select_config_on_train(reviews)

    assert selection["selected_config_id"] == "safe-strong"
    assert selection["eligible_config_count"] == 2
    assert selection["dev_used_for_selection"] is False
    assert selection["dev_used_for_retuning"] is False
    assert selection["selection_ranking"][0]["config_id"] == "safe-strong"


def test_decision_records_no_selection_as_completed_not_guard_blocked() -> None:
    decision = _decision(
        guard_checks=[_check(name="guard", passed=True, observed=True, expected=True)],
        train_selection={
            "selected_config_id": None,
            "selected_profile_id": None,
            "eligible_config_count": 0,
            "selection_ranking": [
                {
                    "train_gold_citation_count_delta_vs_stage116": -1,
                    "train_verified_f1_delta_vs_stage116": 0.0,
                }
            ],
        },
    )

    assert decision["status"].endswith("completed_no_selection")
    assert decision["can_continue_train_dev_development"] is True
    assert decision["selected_config_id"] is None
    assert decision["can_run_final_test_metrics_now"] is False


def test_public_safe_contract_detects_forbidden_keys() -> None:
    clean = _public_safe_contract({"metric": {"gold_citation_count": 3}})
    dirty = _public_safe_contract({"question_text": "private"})

    assert clean["forbidden_keys_found"] == []
    assert dirty["forbidden_keys_found"] == ["question_text"]


def _question() -> PrimeQAQuestion:
    return PrimeQAQuestion(
        id="q1",
        title="install adapter",
        text="How do I fix adapter install error special token?",
        answer="Use the adapter installation technote.",
        answerable=True,
        answer_doc_id="append-gold-like",
    )


def _candidate_results(*, extra_high_append: bool = False) -> list[RetrievalResult]:
    results = [
        RetrievalResult(
            document=PrimeQADocument(
                id=f"prefix-{index:02d}",
                title="ordinary prefix doc",
                text="adapter install reference",
            ),
            score=1.0,
            rank=index,
        )
        for index in range(1, 13)
    ]
    results.append(
        RetrievalResult(
            document=PrimeQADocument(
                id="append-gold-like",
                title="special adapter install error",
                text=(
                    "adapter install error special token adapter install error "
                    "special token"
                ),
            ),
            score=1.0,
            rank=201,
        )
    )
    if extra_high_append:
        results.append(
            RetrievalResult(
                document=PrimeQADocument(
                    id="append-second",
                    title="special token adapter install",
                    text="adapter install error special token workaround",
                ),
                score=1.0,
                rank=202,
            )
        )
    return results


def _review(
    config_id: str,
    *,
    guard: bool,
    citation_delta: int,
    f1_delta: float,
    churn: float,
) -> dict:
    return {
        "config_id": config_id,
        "profile_id": f"profile-{config_id}",
        "family_id": "family",
        "protected_prefix_slots": 9,
        "replacement_append_slots": 1,
        "train_changed_verified_answer_rate_vs_stage116": churn,
        "deltas_vs_stage116_control": {
            "verified_gold_citation_count_delta": citation_delta,
            "verified_average_token_f1_delta": f1_delta,
            "gold_hit_count_at_profile_depth_delta": 9,
        },
        "train_cv_guard": {
            "passed": guard,
            "failed_checks": [] if guard else ["gold_citation"],
        },
    }
