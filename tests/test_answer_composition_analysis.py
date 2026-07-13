from ts_rag_agent.application.answer_composition_analysis import (
    analyze_answer_composition_report,
    composition_result_to_dict,
)


def test_answer_composition_analysis_counts_top1_and_oracle_gains():
    report = {
        "cases": [
            {
                "question_id": "q1",
                "question_title": "Security Bulletin CVE-2020-1234",
                "question_route": "security_bulletin_vulnerability_detail",
                "selected_selector_name": "section_span_bm25_sentence",
                "gold_answer": "CVEID CVE-2020-1234 important fix",
                "selected_answer_token_f1": 0.5,
                "selected_candidates": [
                    _candidate(
                        document_id="gold",
                        sentence="CVEID CVE-2020-1234 important fix",
                        candidate_f1=1.0,
                    ),
                    _candidate(
                        document_id="other",
                        sentence="unrelated noisy paragraph",
                        candidate_f1=0.0,
                    ),
                ],
            },
            {
                "question_id": "q2",
                "question_title": "How do I configure product?",
                "question_route": "install_upgrade_config",
                "selected_selector_name": "answer_aware_bm25_sentence",
                "gold_answer": "set value restart service",
                "selected_answer_token_f1": 0.8,
                "selected_candidates": [
                    _candidate(
                        document_id="gold",
                        sentence="set value restart service",
                        candidate_f1=1.0,
                    ),
                    _candidate(
                        document_id="gold",
                        sentence="set value restart service",
                        candidate_f1=1.0,
                    ),
                ],
            },
        ]
    }

    result = analyze_answer_composition_report(
        answer_gap_report=report,
        f1_gain_margin=0.03,
    )

    assert result.summary.total_cases == 2
    assert result.summary.top1_beats_current == 2
    assert result.summary.best_prefix_oracle_beats_current == 2
    assert result.summary.multi_document_answer_count == 1
    assert result.summary.duplicate_answer_count == 1
    assert result.summary.question_route_counts["security_bulletin_vulnerability_detail"] == 1
    assert result.top1_gain_cases[0].question_id == "q1"


def test_composition_result_to_dict_is_json_safe():
    result = analyze_answer_composition_report(
        answer_gap_report={
            "cases": [
                {
                    "question_id": "q1",
                    "gold_answer": "restart service",
                    "selected_candidates": [
                        _candidate(
                            document_id="gold",
                            sentence="restart service",
                            candidate_f1=1.0,
                        )
                    ],
                }
            ]
        }
    )

    result_dict = composition_result_to_dict(result)

    assert result_dict["summary"]["total_cases"] == 1
    assert result_dict["oracle_gap_cases"] == []


def _candidate(document_id: str, sentence: str, candidate_f1: float) -> dict:
    return {
        "document_id": document_id,
        "retrieval_rank": 1,
        "sentence": sentence,
        "candidate_token_f1": candidate_f1,
    }
