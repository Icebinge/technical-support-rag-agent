from ts_rag_agent.application.route_aware_composition_policy import (
    RouteAwareCompositionPolicy,
    analyze_route_aware_composition_policy,
    route_aware_composition_result_to_dict,
)


def test_route_aware_policy_uses_top1_only_for_strong_direct_routes():
    report = {
        "cases": [
            {
                "question_id": "q1",
                "question_title": "How do I install product A?",
                "question_route": "install_upgrade_config",
                "gold_answer_doc_id": "gold",
                "gold_answer": "Install product A with the setup command.",
                "selected_answer_token_f1": 0.4,
                "selected_candidates": [
                    _candidate(
                        document_id="gold",
                        score=140.0,
                        rank=1,
                        sentence="Install product A with the setup command.",
                    ),
                    _candidate(
                        document_id="other",
                        score=80.0,
                        rank=2,
                        sentence="General product information unrelated to setup.",
                    ),
                ],
            },
            {
                "question_id": "q2",
                "question_title": "Security Bulletin CVE-2020-1234",
                "question_route": "security_bulletin_vulnerability_detail",
                "gold_answer_doc_id": "gold",
                "gold_answer": "CVE-2020-1234 has important impact.",
                "selected_answer_token_f1": 0.6,
                "selected_candidates": [
                    _candidate(
                        document_id="other",
                        score=150.0,
                        rank=1,
                        sentence="CVE-2020-1234 summary text.",
                    ),
                    _candidate(
                        document_id="gold",
                        score=120.0,
                        rank=2,
                        sentence="CVE-2020-1234 has important impact.",
                    ),
                ],
            },
        ]
    }

    result = analyze_route_aware_composition_policy(
        answer_gap_report=report,
        policy=RouteAwareCompositionPolicy(
            strong_first_score_min=100.0,
            strong_first_score_ratio_min=1.15,
            strong_first_score_margin_min=20.0,
            install_upgrade_score_margin_min=20.0,
            enable_how_to_top1=True,
        ),
    )

    assert result.summary.total_cases == 2
    assert result.summary.strategy_counts["top1_direct_strong_signal"] == 1
    assert result.summary.strategy_counts["keep_top3_citation_sensitive"] == 1
    assert result.summary.baseline_gold_citation_count == 2
    assert result.summary.policy_gold_citation_count == 2
    assert result.changed_cases[0].question_id == "q1"


def test_route_aware_policy_uses_stricter_install_upgrade_margin_and_disables_how_to():
    report = {
        "cases": [
            {
                "question_id": "q1",
                "question_title": "How do I install product A?",
                "question_route": "install_upgrade_config",
                "gold_answer_doc_id": "gold",
                "gold_answer": "Install product A with the setup command.",
                "selected_candidates": [
                    _candidate(
                        document_id="gold",
                        score=120.0,
                        rank=1,
                        sentence="Install product A with the setup command.",
                    ),
                    _candidate(
                        document_id="other",
                        score=90.0,
                        rank=2,
                        sentence="General product information unrelated to setup.",
                    ),
                ],
            },
            {
                "question_id": "q2",
                "question_title": "How can I export a private key?",
                "question_route": "how_to_or_lookup",
                "gold_answer_doc_id": "gold",
                "gold_answer": "Use the export key action.",
                "selected_candidates": [
                    _candidate(
                        document_id="gold",
                        score=120.0,
                        rank=1,
                        sentence="Use the export key action.",
                    ),
                    _candidate(
                        document_id="other",
                        score=90.0,
                        rank=2,
                        sentence="General product information unrelated to keys.",
                    ),
                ],
            },
        ]
    }

    result = analyze_route_aware_composition_policy(
        answer_gap_report=report,
        policy=RouteAwareCompositionPolicy(
            strong_first_score_min=100.0,
            strong_first_score_ratio_min=1.15,
            strong_first_score_margin_min=20.0,
            install_upgrade_score_margin_min=45.0,
        ),
    )

    strategy_by_question_id = {
        case.question_id: case.strategy
        for case in result.changed_cases
    }
    assert result.summary.strategy_counts["keep_top3_default"] == 2
    assert "top1_direct_strong_signal" not in result.summary.strategy_counts
    assert "q1" not in strategy_by_question_id
    assert "q2" not in strategy_by_question_id


def test_route_aware_policy_can_enable_how_to_top1_explicitly():
    report = {
        "cases": [
            {
                "question_id": "q1",
                "question_title": "How can I export a private key?",
                "question_route": "how_to_or_lookup",
                "gold_answer_doc_id": "gold",
                "gold_answer": "Use the export key action.",
                "selected_candidates": [
                    _candidate(
                        document_id="gold",
                        score=120.0,
                        rank=1,
                        sentence="Use the export key action.",
                    ),
                    _candidate(
                        document_id="other",
                        score=90.0,
                        rank=2,
                        sentence="General product information unrelated to keys.",
                    ),
                ],
            },
        ]
    }

    result = analyze_route_aware_composition_policy(
        answer_gap_report=report,
        policy=RouteAwareCompositionPolicy(enable_how_to_top1=True),
    )

    assert result.summary.strategy_counts["top1_direct_strong_signal"] == 1
    assert result.changed_cases[0].question_id == "q1"


def test_route_aware_policy_rejects_weak_top1_and_deduplicates_same_doc():
    report = {
        "cases": [
            {
                "question_id": "q1",
                "question_route": "how_to_or_lookup",
                "gold_answer_doc_id": "gold",
                "gold_answer": "download the attached guide",
                "selected_answer_token_f1": 0.5,
                "selected_candidates": [
                    _candidate(
                        document_id="gold",
                        score=70.0,
                        rank=1,
                        sentence="Download the attached guide.",
                    ),
                    _candidate(
                        document_id="same",
                        score=60.0,
                        rank=2,
                        sentence="Download the attached guide.",
                    ),
                ],
            },
            {
                "question_id": "q2",
                "question_route": "other",
                "gold_answer_doc_id": "doc-a",
                "gold_answer": "restart service",
                "selected_answer_token_f1": 0.3,
                "selected_candidates": [
                    _candidate(
                        document_id="doc-a",
                        score=90.0,
                        rank=1,
                        sentence="Restart service from the panel.",
                    ),
                    _candidate(
                        document_id="doc-a",
                        score=80.0,
                        rank=1,
                        sentence="Restart service from the panel.",
                    ),
                    _candidate(
                        document_id="doc-b",
                        score=70.0,
                        rank=2,
                        sentence="Use another panel.",
                    ),
                ],
            },
        ]
    }

    result = analyze_route_aware_composition_policy(answer_gap_report=report)
    result_dict = route_aware_composition_result_to_dict(result)

    assert result.summary.strategy_counts["keep_top3_default"] == 1
    assert result.summary.strategy_counts["dedup_same_document"] == 1
    assert result_dict["summary"]["total_cases"] == 2


def _candidate(document_id: str, score: float, rank: int, sentence: str) -> dict:
    return {
        "document_id": document_id,
        "title": f"title {document_id}",
        "retrieval_rank": rank,
        "candidate_score": score,
        "sentence": sentence,
    }
