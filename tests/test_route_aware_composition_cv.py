from ts_rag_agent.application.route_aware_composition_cv import (
    cross_validate_route_aware_composition_policy,
    route_aware_cv_result_to_dict,
)


def test_cross_validation_selects_threshold_that_preserves_train_citations():
    report = {
        "cases": [
            _install_risk_case("q1"),
            _install_risk_case("q2"),
            _install_risk_case("q3"),
            _install_risk_case("q4"),
        ]
    }

    result = cross_validate_route_aware_composition_policy(
        answer_gap_report=report,
        install_upgrade_score_margin_grid=[20.0, 45.0],
        fold_count=2,
    )
    result_dict = route_aware_cv_result_to_dict(result)

    assert result.aggregate_validation.fold_count == 2
    assert result.aggregate_validation.selected_margin_counts == {"45.0": 2}
    assert result.aggregate_validation.citation_delta == 0
    assert result.aggregate_validation.citation_lost_count == 0
    assert result_dict["aggregate_validation"]["total_validation_cases"] == 4


def test_cross_validation_rejects_invalid_fold_count():
    try:
        cross_validate_route_aware_composition_policy(
            answer_gap_report={"cases": [_install_risk_case("q1")]},
            fold_count=2,
        )
    except ValueError as exc:
        assert "fold_count must be no larger" in str(exc)
    else:
        raise AssertionError("fold_count larger than cases should fail")


def _install_risk_case(question_id: str) -> dict:
    return {
        "question_id": question_id,
        "question_title": f"How do I install product {question_id}?",
        "question_route": "install_upgrade_config",
        "gold_answer_doc_id": "gold",
        "gold_answer": "Use the setup package from the gold document.",
        "selected_candidates": [
            _candidate(
                document_id="other",
                score=120.0,
                rank=1,
                sentence="Install the unrelated package from another document.",
            ),
            _candidate(
                document_id="gold",
                score=90.0,
                rank=2,
                sentence="Use the setup package from the gold document.",
            ),
        ],
    }


def _candidate(document_id: str, score: float, rank: int, sentence: str) -> dict:
    return {
        "document_id": document_id,
        "title": f"title {document_id}",
        "retrieval_rank": rank,
        "candidate_score": score,
        "sentence": sentence,
    }
