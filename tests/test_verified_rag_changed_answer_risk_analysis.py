from ts_rag_agent.application.verified_rag_changed_answer_risk_analysis import (
    analyze_verified_rag_changed_answer_risk,
    write_changed_answer_risk_visualizations,
)
from ts_rag_agent.domain.dataset import PrimeQAQuestion


def test_analyze_verified_rag_changed_answer_risk_counts_routes_and_rank_risk():
    baseline = _report(
        samples=[
            _sample(
                "q1",
                answerable=True,
                gold_doc_id="doc-a",
                verified_answer="baseline unrelated",
                verified_refused=False,
                verified_citations=[_citation("doc-x", rank=2, score=50.0)],
                reasons=["verified"],
                gold_answer="candidate exact answer",
            ),
            _sample(
                "q2",
                answerable=False,
                gold_doc_id="-",
                verified_answer="I do not have enough verified evidence to answer this question.",
                verified_refused=True,
                verified_citations=[],
                reasons=["citation_rank_too_low"],
                gold_answer="-",
            ),
        ],
        policy="top_k",
    )
    candidate = _report(
        samples=[
            _sample(
                "q1",
                answerable=True,
                gold_doc_id="doc-a",
                verified_answer="candidate exact answer",
                verified_refused=False,
                verified_citations=[_citation("doc-a", rank=1, score=90.0)],
                reasons=["verified"],
                gold_answer="candidate exact answer",
            ),
            _sample(
                "q2",
                answerable=False,
                gold_doc_id="-",
                verified_answer="candidate answers with mixed rank citations",
                verified_refused=False,
                verified_citations=[
                    _citation("doc-y", rank=1, score=70.0),
                    _citation("doc-z", rank=4, score=110.0),
                ],
                reasons=["verified"],
                gold_answer="-",
            ),
        ],
        policy="candidate_score_gte_60_guarded_reranker",
    )

    analysis = analyze_verified_rag_changed_answer_risk(
        baseline,
        candidate,
        questions=[
            _question("q1", "Why does the server show an exception?", answerable=True),
            _question("q2", "How do you configure this server?", answerable=False),
        ],
    )

    assert analysis["summary"]["changed_verified_answers"] == 2
    assert analysis["summary"]["changed_answerable"] == 1
    assert analysis["summary"]["changed_unanswerable"] == 1
    assert analysis["summary"]["answerable_improved"] == 1
    assert analysis["summary"]["unanswerable_refusal_regressions"] == 1
    assert analysis["summary"]["candidate_has_out_of_rank_citation"] == 1
    assert analysis["route_distribution"]["all_changed"] == {
        "error_or_log": 1,
        "install_upgrade_config": 1,
    }
    assert analysis["outcome_distribution"] == {
        "answerable_f1_improved": 1,
        "unanswerable_refusal_regression": 1,
    }
    assert analysis["risk_observations"][
        "would_block_unanswerable_regressions_if_all_citations_rank_lte_max"
    ] == 1
    assert analysis["unanswerable_refusal_regression_cases"][0]["question_id"] == "q2"


def test_write_changed_answer_risk_visualizations(tmp_path):
    analysis = analyze_verified_rag_changed_answer_risk(
        _report(
            samples=[
                _sample(
                    "q1",
                    answerable=True,
                    gold_doc_id="doc-a",
                    verified_answer="old",
                    verified_refused=False,
                    verified_citations=[_citation("doc-a", rank=1, score=30.0)],
                    reasons=["verified"],
                    gold_answer="new",
                )
            ],
            policy="top_k",
        ),
        _report(
            samples=[
                _sample(
                    "q1",
                    answerable=True,
                    gold_doc_id="doc-a",
                    verified_answer="new",
                    verified_refused=False,
                    verified_citations=[_citation("doc-a", rank=1, score=30.0)],
                    reasons=["verified"],
                    gold_answer="new",
                )
            ],
            policy="candidate",
        ),
        questions=[_question("q1", "Why does the server show an exception?", answerable=True)],
    )

    artifacts = write_changed_answer_risk_visualizations(analysis, tmp_path)

    assert [artifact.name for artifact in artifacts] == [
        "stage47_changed_by_route.svg",
        "stage47_changed_by_outcome.svg",
        "stage47_out_of_rank_by_outcome.svg",
        "stage47_candidate_worst_rank.svg",
    ]
    assert all((tmp_path / artifact.name).exists() for artifact in artifacts)


def _report(samples: list[dict], policy: str) -> dict:
    return {
        "split": "dev",
        "rag": {
            "evidence_selector": "selector",
            "composition_policy": policy,
            "candidate_reranker": None,
            "retrieval_top_k": 5,
            "max_sentences": 3,
            "min_sentence_score": 2.0,
            "max_candidates_per_document": 3,
            "min_evidence_score": 15.0,
            "max_citation_rank": 3,
            "min_citations": 1,
        },
        "metrics": {
            "verified": {"total_questions": len(samples)},
        },
        "samples": samples,
    }


def _sample(
    question_id: str,
    answerable: bool,
    gold_doc_id: str,
    verified_answer: str,
    verified_refused: bool,
    verified_citations: list[dict],
    reasons: list[str],
    gold_answer: str,
) -> dict:
    return {
        "question_id": question_id,
        "question_title": f"Title {question_id}",
        "answerable": answerable,
        "gold_answer_doc_id": gold_doc_id,
        "verified_answer": {
            "answer": verified_answer,
            "refused": verified_refused,
            "citations": verified_citations,
        },
        "verification": {"reasons": reasons},
        "gold_answer": gold_answer,
    }


def _citation(document_id: str, rank: int, score: float) -> dict:
    return {
        "document_id": document_id,
        "title": document_id,
        "retrieval_rank": rank,
        "evidence_score": score,
    }


def _question(question_id: str, title: str, answerable: bool) -> PrimeQAQuestion:
    return PrimeQAQuestion(
        id=question_id,
        title=title,
        text="",
        answer="candidate exact answer" if answerable else "-",
        answerable=answerable,
        answer_doc_id="doc-a" if answerable else None,
    )
