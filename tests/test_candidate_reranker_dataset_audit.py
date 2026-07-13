from pathlib import Path

from ts_rag_agent.application.candidate_reranker_dataset_audit import (
    audit_candidate_reranker_dataset,
    candidate_reranker_dataset_audit_to_dict,
    write_audit_visualizations,
)


def test_audit_candidate_reranker_dataset_summarizes_distribution_and_gain(
    tmp_path: Path,
):
    rows = [
        _row("dev", "q1", 1, "other", 0.1),
        _row("dev", "q1", 2, "other", 0.5, is_gold_document=True),
        _row("train", "q2", 1, "error_or_log", 0.6),
        _row("train", "q2", 2, "error_or_log", 0.2),
    ]
    summary_report = _summary_report(rows)

    audit = audit_candidate_reranker_dataset(rows=rows, summary_report=summary_report)
    audit_dict = candidate_reranker_dataset_audit_to_dict(audit)
    visualizations = write_audit_visualizations(audit=audit, output_dir=tmp_path)

    assert audit.total_rows == 4
    assert audit.total_questions == 2
    assert audit.questions_by_split == {"dev": 1, "train": 1}
    assert _rank_bucket(audit, "rank_1").count == 1
    assert _rank_bucket(audit, "rank_2").count == 1
    assert _f1_bucket(audit, "0.40-0.60").count == 1
    assert _f1_bucket(audit, "0.60-0.80").count == 1
    assert audit.route_oracle_gain[0].question_route == "other"
    assert audit.route_oracle_gain[0].average_oracle_gain_vs_top_candidate == 0.4
    assert audit.split_summaries[0].split == "dev"
    assert audit.split_summaries[0].best_rank_1_rate == 0.0
    assert audit.split_summaries[1].split == "train"
    assert audit.split_summaries[1].best_rank_1_rate == 1.0
    assert audit.consistency_audit.total_rows_match is True
    assert audit.consistency_audit.rows_by_route_match is True
    assert audit.feature_leakage_audit.label_leakage_detected_from_keys is False
    assert audit_dict["total_rows"] == 4
    assert {visualization.name for visualization in visualizations} == {
        "candidate_label_f1_distribution.svg",
        "best_candidate_rank_distribution.svg",
        "route_oracle_gain.svg",
        "split_oracle_gain.svg",
    }
    assert (tmp_path / "route_oracle_gain.svg").read_text(encoding="utf-8").startswith(
        "<svg"
    )


def test_audit_detects_obvious_runtime_feature_label_leakage():
    rows = [
        _row(
            "dev",
            "q1",
            1,
            "other",
            0.1,
            runtime_extra={"candidate_token_f1": 0.1},
        ),
        _row("dev", "q1", 2, "other", 0.5, is_gold_document=True),
    ]

    audit = audit_candidate_reranker_dataset(
        rows=rows,
        summary_report=_summary_report(rows, question_count=1),
    )

    assert audit.feature_leakage_audit.label_leakage_detected_from_keys is True
    assert "candidate_token_f1" in (
        audit.feature_leakage_audit.suspicious_runtime_feature_keys
    )


def test_audit_reports_consistency_mismatches_without_hiding_them():
    rows = [
        _row("dev", "q1", 1, "other", 0.1),
        _row("dev", "q1", 2, "other", 0.5),
    ]
    summary_report = _summary_report(rows, question_count=1)
    summary_report["summary"]["total_rows"] = 3

    audit = audit_candidate_reranker_dataset(rows=rows, summary_report=summary_report)

    assert audit.consistency_audit.total_rows_match is False
    assert audit.consistency_audit.actual_total_rows == 2
    assert audit.consistency_audit.summary_total_rows == 3


def _row(
    split: str,
    question_id: str,
    candidate_rank: int,
    route: str,
    candidate_token_f1: float,
    is_gold_document: bool = False,
    runtime_extra: dict | None = None,
) -> dict:
    runtime_features = {
        "selector_name": "test_selector",
        "question_route": route,
        "retrieval_rank": candidate_rank,
        "retrieval_score": 10.0 - candidate_rank,
        "candidate_score": 20.0 - candidate_rank,
        "candidate_token_count": 8,
        "candidate_sentence_count": 1,
        "title_query_overlap_count": 2,
        "title_query_overlap_ratio": 0.5,
    }
    runtime_features.update(runtime_extra or {})
    return {
        "split": split,
        "question_id": question_id,
        "candidate_id": f"{question_id}::candidate_{candidate_rank:03d}",
        "candidate_rank": candidate_rank,
        "runtime_features": runtime_features,
        "gold_labels": {
            "candidate_token_f1": candidate_token_f1,
            "is_gold_document": is_gold_document,
            "is_best_candidate_for_question": False,
            "best_candidate_token_f1_for_question": 0.0,
            "f1_gap_to_best_candidate": 0.0,
        },
        "metadata": {
            "question_title": "Test question",
            "document_id": "doc",
            "document_title": "Test document",
            "candidate_sentence": "Candidate text.",
        },
    }


def _summary_report(rows: list[dict], question_count: int = 2) -> dict:
    question_summaries = [
        {
            "split": "dev",
            "question_id": "q1",
            "question_route": "other",
            "candidate_count": 2,
            "gold_document_candidate_count": 1,
            "top_candidate_token_f1": 0.1,
            "best_candidate_token_f1": 0.5,
            "best_candidate_rank": 2,
            "oracle_gain_vs_top_candidate": 0.4,
        }
    ]
    if question_count == 2:
        question_summaries.append(
            {
                "split": "train",
                "question_id": "q2",
                "question_route": "error_or_log",
                "candidate_count": 2,
                "gold_document_candidate_count": 0,
                "top_candidate_token_f1": 0.6,
                "best_candidate_token_f1": 0.6,
                "best_candidate_rank": 1,
                "oracle_gain_vs_top_candidate": 0.0,
            }
        )

    rows_by_split = {}
    rows_by_route = {}
    for row in rows:
        rows_by_split[row["split"]] = rows_by_split.get(row["split"], 0) + 1
        route = row["runtime_features"]["question_route"]
        rows_by_route[route] = rows_by_route.get(route, 0) + 1

    return {
        "summary": {
            "total_rows": len(rows),
            "total_questions": question_count,
            "rows_by_split": rows_by_split,
            "rows_by_route": rows_by_route,
        },
        "question_summaries": question_summaries,
    }


def _rank_bucket(audit, label: str):
    return next(
        bucket for bucket in audit.best_candidate_rank_distribution if bucket.label == label
    )


def _f1_bucket(audit, label: str):
    return next(
        bucket for bucket in audit.candidate_token_f1_distribution if bucket.label == label
    )
