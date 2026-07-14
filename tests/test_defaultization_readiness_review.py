import json

from ts_rag_agent.application.defaultization_readiness_review import (
    SplitReviewSources,
    review_defaultization_readiness,
    write_defaultization_review_visualizations,
)


def test_defaultization_review_selects_stage51_when_rank_contained_loses_citation():
    review = review_defaultization_readiness(
        sources=[
            SplitReviewSources(
                split="dev",
                topk_report=_report("dev", "top_k", f1=0.2755, gold_count=1),
                rank_contained_report=_report(
                    "dev",
                    "candidate_score_gte_60_rank_contained_guarded_reranker",
                    f1=0.2763,
                    gold_count=1,
                ),
                candidate_report=_report(
                    "dev",
                    _stage51_policy(),
                    f1=0.2760,
                    gold_count=1,
                ),
                candidate_risk_report=_risk_report(),
            ),
            SplitReviewSources(
                split="train",
                topk_report=_report("train", "top_k", f1=0.2557, gold_count=2),
                rank_contained_report=_report(
                    "train",
                    "candidate_score_gte_60_rank_contained_guarded_reranker",
                    f1=0.2565,
                    gold_count=1,
                ),
                candidate_report=_report(
                    "train",
                    _stage51_policy(),
                    f1=0.2565,
                    gold_count=2,
                ),
                candidate_risk_report=_risk_report(),
            ),
        ]
    )

    assert review["overall_decision"] == {
        "candidate_passes_dev_train_readiness": True,
        "rank_contained_passes_dev_train_safety": False,
        "unique_heldout_candidate": _stage51_policy(),
        "default_runtime_change": "not_allowed_before_heldout_evaluation",
        "status": "stage51_is_unique_candidate_for_single_heldout_evaluation",
    }
    train_review = review["split_reviews"][1]
    assert train_review["rank_contained_vs_topk"]["verified_gold_citation_delta"] == -1
    assert all(check["passed"] for check in train_review["candidate_readiness_checks"])
    assert review["heldout_test_protocol"]["heldout_dataset"] == "nvidia/TechQA-RAG-Eval"


def test_defaultization_review_blocks_candidate_with_negative_f1_delta():
    review = review_defaultization_readiness(
        sources=[
            SplitReviewSources(
                split="dev",
                topk_report=_report("dev", "top_k", f1=0.30, gold_count=1),
                rank_contained_report=_report("dev", "rank", f1=0.31, gold_count=1),
                candidate_report=_report("dev", _stage51_policy(), f1=0.29, gold_count=1),
                candidate_risk_report=_risk_report(),
            )
        ]
    )

    assert not review["overall_decision"]["candidate_passes_dev_train_readiness"]
    checks = review["split_reviews"][0]["candidate_readiness_checks"]
    failed = [check["name"] for check in checks if not check["passed"]]
    assert failed == ["verified_f1_delta_vs_topk_non_negative"]


def test_defaultization_review_writes_visualizations(tmp_path):
    review = review_defaultization_readiness(
        sources=[
            SplitReviewSources(
                split="dev",
                topk_report=_report("dev", "top_k", f1=0.2755, gold_count=1),
                rank_contained_report=_report("dev", "rank", f1=0.2763, gold_count=1),
                candidate_report=_report("dev", _stage51_policy(), f1=0.2760, gold_count=1),
                candidate_risk_report=_risk_report(),
            )
        ]
    )

    artifacts = write_defaultization_review_visualizations(review, tmp_path)

    assert {artifact.name for artifact in artifacts} == {
        "stage52_verified_f1_by_policy.svg",
        "stage52_gold_citation_delta_vs_topk.svg",
        "stage52_changed_answer_risk.svg",
        "stage52_readiness_pass_count.svg",
    }
    for artifact in artifacts:
        assert (tmp_path / artifact.name).read_text(encoding="utf-8").startswith("<svg")


def _report(
    split: str,
    policy: str,
    f1: float,
    gold_count: int,
) -> dict:
    samples = [
        _answerable_sample("q1", "doc-1", gold_count >= 1),
        _answerable_sample("q2", "doc-2", gold_count >= 2),
        _unanswerable_sample("q3"),
    ]
    return {
        "split": split,
        "rag": {
            "composition_policy": policy,
            "candidate_reranker": None,
        },
        "metrics": {
            "verified": {
                "total_questions": len(samples),
                "generated_answerable_questions": 2,
                "gold_doc_citation_rate": round(gold_count / 2, 4),
                "average_token_f1": f1,
                "refused_answerable_questions": 0,
                "refused_unanswerable_questions": 0,
            }
        },
        "verification": {
            "newly_refused": 0,
        },
        "samples": samples,
    }


def _answerable_sample(question_id: str, gold_doc_id: str, cites_gold: bool) -> dict:
    cited_doc_id = gold_doc_id if cites_gold else f"not-{gold_doc_id}"
    return {
        "question_id": question_id,
        "answerable": True,
        "gold_answer_doc_id": gold_doc_id,
        "verified_answer": {
            "refused": False,
            "citations": [{"document_id": cited_doc_id}],
        },
    }


def _unanswerable_sample(question_id: str) -> dict:
    return {
        "question_id": question_id,
        "answerable": False,
        "gold_answer_doc_id": None,
        "verified_answer": {
            "refused": False,
            "citations": [],
        },
    }


def _risk_report() -> dict:
    return {
        "summary": {
            "changed_verified_answers": 1,
            "changed_answerable": 1,
            "changed_unanswerable": 0,
            "unanswerable_refusal_regressions": 0,
            "answerable_improved": 1,
            "answerable_regressed": 0,
            "answerable_tied_changed": 0,
            "candidate_has_out_of_rank_citation": 0,
            "unanswerable_regression_has_out_of_rank_citation": 0,
        }
    }


def _stage51_policy() -> str:
    return (
        "candidate_score_gte_60_rank_contained_"
        "preserve_baseline_out_of_rank_guarded_reranker"
    )


def test_fake_reports_are_json_serializable():
    json.dumps(_report("dev", "top_k", f1=0.2, gold_count=1))
