import json
from pathlib import Path

from ts_rag_agent.application.primeqa_hybrid_candidate_reranker_development import (
    run_primeqa_hybrid_candidate_reranker_development,
    write_primeqa_hybrid_candidate_reranker_development_visualizations,
)


def test_primeqa_hybrid_candidate_reranker_development_runs_train_dev_only(tmp_path):
    paths = _write_fixture(tmp_path, include_test_candidate=False)

    run = run_primeqa_hybrid_candidate_reranker_development(
        candidate_dataset_path=paths["candidate_dataset"],
        candidate_summary_path=paths["candidate_summary"],
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        fold_count=3,
        model_names=("logistic_best_candidate", "ridge_candidate_token_f1"),
        policy_model_names=("logistic_best_candidate", "ridge_candidate_token_f1"),
        max_answer_candidates=1,
    )
    visualizations = write_primeqa_hybrid_candidate_reranker_development_visualizations(
        run=run,
        output_dir=tmp_path / "visuals",
    )

    assert run.report["stage"] == "Stage 71"
    assert run.report["loaded_candidate_summary"]["candidate_splits"] == ["dev", "train"]
    assert run.report["loaded_gold_answer_summary"]["answer_count_by_split"] == {
        "dev": 3,
        "train": 3,
    }
    assert run.report["train_only_model_cv"]["fold_count"] == 3
    assert [
        result["model_name"]
        for result in run.report["train_to_dev_policy_validations"]
    ] == ["logistic_best_candidate", "ridge_candidate_token_f1"]
    assert all(
        result["train_split"] == "train"
        for result in run.report["train_to_dev_policy_validations"]
    )
    assert all(
        result["evaluation_split"] == "dev"
        for result in run.report["train_to_dev_policy_validations"]
    )
    assert all(check["passed"] for check in run.report["guard_checks"])
    assert run.report["decision"]["can_run_final_test_metrics_now"] is False
    assert len(visualizations) == 20
    assert {artifact.group for artifact in visualizations} == {
        "train_only_model_cv",
        "logistic_best_candidate_train_cv_policy",
        "logistic_best_candidate_train_to_dev_policy",
        "ridge_candidate_token_f1_train_cv_policy",
        "ridge_candidate_token_f1_train_to_dev_policy",
    }


def test_primeqa_hybrid_candidate_reranker_development_blocks_test_rows(tmp_path):
    paths = _write_fixture(tmp_path, include_test_candidate=True)

    run = run_primeqa_hybrid_candidate_reranker_development(
        candidate_dataset_path=paths["candidate_dataset"],
        candidate_summary_path=paths["candidate_summary"],
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        fold_count=3,
        model_names=("logistic_best_candidate",),
        policy_model_names=("logistic_best_candidate",),
        max_answer_candidates=1,
    )

    checks = {check["name"]: check for check in run.report["guard_checks"]}
    assert checks["candidate_artifact_splits_are_train_dev_only"]["passed"] is False
    assert checks["candidate_rows_have_no_test_split"]["passed"] is False
    assert run.report["decision"]["status"] == (
        "primeqa_hybrid_candidate_reranker_development_blocked"
    )


def _write_fixture(tmp_path: Path, *, include_test_candidate: bool) -> dict[str, Path]:
    train_split = tmp_path / "train.jsonl"
    dev_split = tmp_path / "dev.jsonl"
    train_samples = [
        _split_sample("train", f"primeqa_train:TRAIN_Q{index:03d}")
        for index in range(1, 4)
    ]
    dev_samples = [
        _split_sample("dev", f"primeqa_dev:DEV_Q{index:03d}")
        for index in range(1, 4)
    ]
    _write_jsonl(train_split, train_samples)
    _write_jsonl(dev_split, dev_samples)

    rows = [
        candidate
        for sample in [*train_samples, *dev_samples]
        for candidate in _candidate_rows(
            split=sample["assigned_split"],
            question_id=sample["sample_id"],
        )
    ]
    if include_test_candidate:
        rows.extend(_candidate_rows(split="test", question_id="primeqa_test:TEST_Q001"))

    candidate_dataset = tmp_path / "candidates.jsonl"
    _write_jsonl(candidate_dataset, rows)
    candidate_summary = tmp_path / "candidate_summary.json"
    candidate_summary.write_text(
        json.dumps(_candidate_summary(rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "train_split": train_split,
        "dev_split": dev_split,
        "candidate_dataset": candidate_dataset,
        "candidate_summary": candidate_summary,
    }


def _split_sample(assigned_split: str, sample_id: str) -> dict:
    return {
        "dataset": "primeqa_techqa",
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "assigned_split": assigned_split,
        "split_subtype": f"group_random_{assigned_split}",
        "assignment_reason": "group_random_remainder_split",
        "source_split": sample_id.split(":", maxsplit=1)[0],
        "source_row_index": 1,
        "sample_id": sample_id,
        "question_id": sample_id.split(":", maxsplit=1)[1],
        "question_title": "How to restart the service?",
        "question_text": "",
        "question": "How to restart the service?",
        "answerable": True,
        "answer": "Restart the service from the admin console.",
        "answer_doc_id": "doc-gold",
        "candidate_doc_ids": ["doc-gold", "doc-noise"],
        "answer_span": {"start_offset": 0, "end_offset": 43},
        "metadata": {
            "group_hash": f"{sample_id}:group",
            "candidate_doc_count": 2,
            "candidate_doc_hash": f"{sample_id}:docs",
        },
    }


def _candidate_rows(split: str, question_id: str) -> list[dict]:
    return [
        _candidate_row(
            split=split,
            question_id=question_id,
            candidate_rank=1,
            score=95.0,
            token_f1=0.1,
            is_best=False,
            document_id="doc-noise",
            sentence="Original weak answer.",
        ),
        _candidate_row(
            split=split,
            question_id=question_id,
            candidate_rank=2,
            score=70.0,
            token_f1=0.8,
            is_best=True,
            document_id="doc-gold",
            sentence="Restart the service from the admin console.",
        ),
    ]


def _candidate_row(
    *,
    split: str,
    question_id: str,
    candidate_rank: int,
    score: float,
    token_f1: float,
    is_best: bool,
    document_id: str,
    sentence: str,
) -> dict:
    return {
        "split": split,
        "question_id": question_id,
        "candidate_id": f"{question_id}::candidate_{candidate_rank:03d}",
        "candidate_rank": candidate_rank,
        "runtime_features": {
            "selector_name": "fixture_selector",
            "question_route": "other",
            "retrieval_rank": candidate_rank,
            "retrieval_score": 10.0 - candidate_rank,
            "candidate_score": score,
            "candidate_token_count": 8,
            "candidate_sentence_count": 1,
            "question_token_count": 6,
            "query_term_count": 4,
            "query_overlap_count": 3 if is_best else 1,
            "query_overlap_ratio": 0.75 if is_best else 0.25,
            "candidate_query_coverage_ratio": 0.6 if is_best else 0.2,
            "title_query_overlap_count": 1,
            "title_query_overlap_ratio": 0.25,
            "answer_signal_score": 3.0 if is_best else 0.1,
            "problem_noise_score": 0.0 if is_best else 1.4,
            "has_answer_heading": is_best,
            "has_problem_heading": not is_best,
            "has_question_heading": False,
            "has_url": False,
            "has_trace_noise": not is_best,
            "symbol_ratio": 0.02,
        },
        "gold_labels": {
            "candidate_token_f1": token_f1,
            "is_gold_document": is_best,
            "is_best_candidate_for_question": is_best,
            "best_candidate_token_f1_for_question": 0.8,
            "f1_gap_to_best_candidate": 0.0 if is_best else 0.7,
        },
        "metadata": {
            "question_title": "How to restart the service?",
            "question_route": "other",
            "document_id": document_id,
            "document_title": "Fixture document",
            "candidate_sentence": sentence,
        },
    }


def _candidate_summary(rows: list[dict]) -> dict:
    rows_by_split = {}
    question_ids_by_split = {}
    rows_by_route = {}
    for row in rows:
        rows_by_split[row["split"]] = rows_by_split.get(row["split"], 0) + 1
        question_ids_by_split.setdefault(row["split"], set()).add(row["question_id"])
        route = row["runtime_features"]["question_route"]
        rows_by_route[route] = rows_by_route.get(route, 0) + 1
    return {
        "stage": "Stage 69",
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "candidate_artifact_summary": {
            "summary": {
                "splits": sorted(rows_by_split),
                "selector_name": "fixture_selector",
                "total_questions": sum(
                    len(question_ids) for question_ids in question_ids_by_split.values()
                ),
                "total_rows": len(rows),
                "questions_by_split": {
                    split: len(question_ids)
                    for split, question_ids in question_ids_by_split.items()
                },
                "rows_by_split": rows_by_split,
                "rows_by_route": rows_by_route,
                "average_rows_per_question": 2.0,
                "average_top_candidate_token_f1": 0.1,
                "average_best_candidate_token_f1": 0.8,
                "average_oracle_gain_vs_top_candidate": 0.7,
                "questions_with_gold_document_candidate": sum(
                    len(question_ids) for question_ids in question_ids_by_split.values()
                ),
                "gold_document_candidate_rows": sum(
                    bool(row["gold_labels"]["is_gold_document"]) for row in rows
                ),
            }
        },
        "guard_checks": [],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
