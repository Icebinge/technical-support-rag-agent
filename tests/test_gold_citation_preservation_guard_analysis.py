from ts_rag_agent.application.gold_citation_preservation_guard_analysis import (
    PreservationGuardScenario,
    analyze_gold_citation_preservation_guards,
    write_preservation_guard_visualizations,
)


def test_preserve_baseline_out_of_rank_docs_restores_lost_gold_citation():
    baseline = _report(
        samples=[
            _sample(
                question_id="q1",
                answerable=True,
                gold_doc_id="gold-doc",
                original_citations=[
                    _citation("rank-one-doc", rank=1, score=120.0),
                    _citation("gold-doc", rank=4, score=82.0),
                ],
                verified_answer="baseline cites gold",
                verified_citations=[
                    _citation("rank-one-doc", rank=1, score=120.0),
                    _citation("gold-doc", rank=4, score=82.0),
                ],
                gold_answer="baseline cites gold",
            ),
            _sample(
                question_id="q2",
                answerable=True,
                gold_doc_id="doc-b",
                original_citations=[_citation("doc-b", rank=1, score=90.0)],
                verified_answer="same answer",
                verified_citations=[_citation("doc-b", rank=1, score=90.0)],
                gold_answer="same answer",
            ),
        ],
        policy="top_k",
    )
    candidate = _report(
        samples=[
            _sample(
                question_id="q1",
                answerable=True,
                gold_doc_id="gold-doc",
                original_citations=[
                    _citation("similar-doc", rank=2, score=90.0),
                    _citation("rank-one-doc", rank=1, score=120.0),
                ],
                verified_answer="candidate drops gold",
                verified_citations=[
                    _citation("similar-doc", rank=2, score=90.0),
                    _citation("rank-one-doc", rank=1, score=120.0),
                ],
                gold_answer="baseline cites gold",
            ),
            _sample(
                question_id="q2",
                answerable=True,
                gold_doc_id="doc-b",
                original_citations=[_citation("doc-b", rank=1, score=90.0)],
                verified_answer="same answer",
                verified_citations=[_citation("doc-b", rank=1, score=90.0)],
                gold_answer="same answer",
            ),
        ],
        policy="candidate",
    )

    analysis = analyze_gold_citation_preservation_guards(
        [
            PreservationGuardScenario(
                label="unit",
                baseline_report=baseline,
                candidate_report=candidate,
            )
        ]
    )

    scenario = analysis["scenarios"][0]
    candidate_as_is = _guard_result(scenario, "candidate_as_is")
    preserve_out_of_rank = _guard_result(
        scenario,
        "preserve_baseline_out_of_rank_docs",
    )

    assert candidate_as_is["metric_deltas_vs_baseline"]["gold_cited_count"] == -1
    assert preserve_out_of_rank["metric_deltas_vs_baseline"]["gold_cited_count"] == 0
    assert preserve_out_of_rank["blocked_changed_answer_count"] == 1
    assert preserve_out_of_rank["blocked_question_ids"] == ["q1"]
    assert preserve_out_of_rank["blocked_gold_citation_loss_count"] == 1
    assert preserve_out_of_rank["blocked_cases"][0]["dropped_protected_document_ids"] == [
        "gold-doc"
    ]


def test_write_preservation_guard_visualizations(tmp_path):
    analysis = analyze_gold_citation_preservation_guards(
        [
            PreservationGuardScenario(
                label="unit",
                baseline_report=_report(
                    samples=[
                        _sample(
                            question_id="q1",
                            answerable=True,
                            gold_doc_id="doc-a",
                            original_citations=[_citation("doc-a", rank=1, score=90.0)],
                            verified_answer="old answer",
                            verified_citations=[_citation("doc-a", rank=1, score=90.0)],
                            gold_answer="new answer",
                        )
                    ],
                    policy="top_k",
                ),
                candidate_report=_report(
                    samples=[
                        _sample(
                            question_id="q1",
                            answerable=True,
                            gold_doc_id="doc-a",
                            original_citations=[_citation("doc-a", rank=1, score=90.0)],
                            verified_answer="new answer",
                            verified_citations=[_citation("doc-a", rank=1, score=90.0)],
                            gold_answer="new answer",
                        )
                    ],
                    policy="candidate",
                ),
            )
        ]
    )

    artifacts = write_preservation_guard_visualizations(analysis, tmp_path)

    assert [artifact.name for artifact in artifacts] == [
        "stage50_unit_f1_delta.svg",
        "stage50_unit_gold_citation_delta.svg",
        "stage50_unit_blocked_changed_count.svg",
    ]
    assert all((tmp_path / artifact.name).exists() for artifact in artifacts)


def _guard_result(scenario: dict, label: str) -> dict:
    for result in scenario["guard_results"]:
        if result["guard_label"] == label:
            return result
    raise AssertionError(f"Missing guard result: {label}")


def _report(samples: list[dict], policy: str) -> dict:
    return {
        "split": "unit",
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
    original_citations: list[dict],
    verified_answer: str,
    verified_citations: list[dict],
    gold_answer: str,
) -> dict:
    return {
        "question_id": question_id,
        "question_title": f"Title {question_id}",
        "answerable": answerable,
        "gold_answer_doc_id": gold_doc_id,
        "original_answer": {
            "answer": verified_answer,
            "refused": False,
            "citations": original_citations,
        },
        "verified_answer": {
            "answer": verified_answer,
            "refused": False,
            "citations": verified_citations,
        },
        "gold_answer": gold_answer,
    }


def _citation(document_id: str, rank: int, score: float) -> dict:
    return {
        "document_id": document_id,
        "title": document_id,
        "retrieval_rank": rank,
        "evidence_score": score,
    }
