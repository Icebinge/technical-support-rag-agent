from ts_rag_agent.application.verified_rag_report_comparison import (
    compare_verified_rag_reports,
    write_verified_rag_comparison_visualizations,
)


def test_compare_verified_rag_reports_counts_gold_citations_and_changes():
    baseline = _report(
        policy="top_k",
        samples=[
            _sample(
                "q1",
                answerable=True,
                gold_doc_id="doc-a",
                original_answer="old answer [doc-a]",
                verified_answer="old answer [doc-a]",
                original_citations=["doc-a"],
                verified_citations=["doc-a"],
                gold_answer="old better answer",
            ),
            _sample(
                "q2",
                answerable=True,
                gold_doc_id="doc-b",
                original_answer="baseline unrelated",
                verified_answer="baseline unrelated",
                original_citations=["doc-x"],
                verified_citations=["doc-x"],
                gold_answer="candidate exact answer",
            ),
            _sample(
                "q3",
                answerable=False,
                gold_doc_id="-",
                original_answer="baseline unanswerable",
                verified_answer="baseline unanswerable",
                original_citations=["doc-x"],
                verified_citations=["doc-x"],
                gold_answer="-",
            ),
        ],
    )
    candidate = _report(
        policy="candidate_score_gte_60_guarded_reranker",
        samples=[
            _sample(
                "q1",
                answerable=True,
                gold_doc_id="doc-a",
                original_answer="old answer [doc-a]",
                verified_answer="old answer [doc-a]",
                original_citations=["doc-a"],
                verified_citations=["doc-a"],
                gold_answer="old better answer",
            ),
            _sample(
                "q2",
                answerable=True,
                gold_doc_id="doc-b",
                original_answer="candidate exact answer [doc-b]",
                verified_answer="candidate exact answer [doc-b]",
                original_citations=["doc-b"],
                verified_citations=["doc-b"],
                gold_answer="candidate exact answer",
            ),
            _sample(
                "q3",
                answerable=False,
                gold_doc_id="-",
                original_answer="candidate unanswerable",
                verified_answer="candidate unanswerable",
                original_citations=["doc-y"],
                verified_citations=["doc-y"],
                gold_answer="-",
            ),
        ],
    )

    comparison = compare_verified_rag_reports(baseline, candidate)

    assert comparison["sample_completeness"]["complete"] is True
    assert comparison["exact_gold_citations"]["baseline"]["verified"] == {
        "answerable_count": 2,
        "generated_answerable_count": 2,
        "gold_cited_count": 1,
    }
    assert comparison["exact_gold_citations"]["candidate"]["verified"] == {
        "answerable_count": 2,
        "generated_answerable_count": 2,
        "gold_cited_count": 2,
    }
    assert comparison["exact_gold_citations"]["verified_citation_gained_question_ids"] == [
        "q2"
    ]
    assert comparison["changed_answers"]["verified"]["all_count"] == 2
    assert comparison["changed_answers"]["verified"]["answerable_question_ids"] == ["q2"]
    assert comparison["changed_answers"]["verified"]["unanswerable_question_ids"] == ["q3"]
    assert comparison["verified_answerable_f1_outcomes"]["improved_count"] == 1
    assert comparison["verified_answerable_f1_outcomes"]["regressed_count"] == 0


def test_write_verified_rag_comparison_visualizations(tmp_path):
    comparison = compare_verified_rag_reports(
        _report(
            policy="top_k",
            samples=[
                _sample(
                    "q1",
                    answerable=True,
                    gold_doc_id="doc-a",
                    original_answer="old answer",
                    verified_answer="old answer",
                    original_citations=["doc-a"],
                    verified_citations=["doc-a"],
                    gold_answer="old answer",
                )
            ],
        ),
        _report(
            policy="candidate",
            samples=[
                _sample(
                    "q1",
                    answerable=True,
                    gold_doc_id="doc-a",
                    original_answer="new answer",
                    verified_answer="new answer",
                    original_citations=["doc-a"],
                    verified_citations=["doc-a"],
                    gold_answer="new answer",
                )
            ],
        ),
    )

    artifacts = write_verified_rag_comparison_visualizations(comparison, tmp_path)

    assert [artifact.name for artifact in artifacts] == [
        "verified_rag_metric_deltas.svg",
        "verified_rag_changed_answers.svg",
        "verified_rag_answerable_f1_outcomes.svg",
    ]
    assert all((tmp_path / artifact.name).exists() for artifact in artifacts)


def _report(policy: str, samples: list[dict]) -> dict:
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
            "original": _metrics(total_questions=len(samples)),
            "verified": _metrics(total_questions=len(samples)),
        },
        "verification": {"newly_refused": 0},
        "samples": samples,
    }


def _metrics(total_questions: int) -> dict:
    return {
        "total_questions": total_questions,
        "generated_answerable_questions": 1,
        "refused_answerable_questions": 0,
        "refused_unanswerable_questions": 0,
        "gold_doc_citation_rate": 1.0,
        "average_token_f1": 0.5,
    }


def _sample(
    question_id: str,
    answerable: bool,
    gold_doc_id: str,
    original_answer: str,
    verified_answer: str,
    original_citations: list[str],
    verified_citations: list[str],
    gold_answer: str,
) -> dict:
    return {
        "question_id": question_id,
        "question_title": f"Title {question_id}",
        "answerable": answerable,
        "gold_answer_doc_id": gold_doc_id,
        "original_answer": {
            "answer": original_answer,
            "refused": False,
            "citations": [_citation(document_id) for document_id in original_citations],
        },
        "verified_answer": {
            "answer": verified_answer,
            "refused": False,
            "citations": [_citation(document_id) for document_id in verified_citations],
        },
        "gold_answer": gold_answer,
    }


def _citation(document_id: str) -> dict:
    return {
        "document_id": document_id,
        "title": document_id,
        "retrieval_rank": 1,
        "evidence_score": 10.0,
    }
