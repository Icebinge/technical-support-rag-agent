from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.msqa_stage51_changed_case_review import (
    review_msqa_stage51_changed_cases,
    write_msqa_stage51_changed_case_review_visualizations,
)

app = typer.Typer(help="Review Stage64 MSQA Stage51 changed cases.")


@app.command()
def main(
    stage64_report: Annotated[
        Path,
        typer.Option("--stage64-report", help="Stage64 adapter comparison JSON."),
    ],
    split_jsonl: Annotated[
        Path,
        typer.Option("--split-jsonl", help="Stage57 frozen MSQA JSONL split."),
    ],
    candidate_jsonl: Annotated[
        Path,
        typer.Option("--candidate-jsonl", help="Stage63 capped candidate JSONL."),
    ],
    adapter_report: Annotated[
        Path,
        typer.Option("--adapter-report", help="Stage63 capped adapter report JSON."),
    ],
    distribution_report: Annotated[
        Path,
        typer.Option("--distribution-report", help="Stage63 distribution report JSON."),
    ],
    candidate_reranker_dataset: Annotated[
        Path,
        typer.Option("--candidate-reranker-dataset", help="Stage31 reranker JSONL."),
    ],
    stage31_summary: Annotated[
        Path,
        typer.Option("--stage31-summary", help="Stage31 reranker summary JSON."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output Stage65 changed-case review JSON."),
    ],
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", help="Candidate reranker model name."),
    ] = "logistic_best_candidate",
    train_split: Annotated[
        str,
        typer.Option("--train-split", help="Candidate reranker training split."),
    ] = "train",
    max_answer_candidates: Annotated[
        int,
        typer.Option("--max-answer-candidates", help="Stage51 answer candidate count."),
    ] = 3,
    max_citation_rank: Annotated[
        int,
        typer.Option("--max-citation-rank", help="Stage51 rank-contained guard rank."),
    ] = 3,
    sample_limit: Annotated[
        int,
        typer.Option("--sample-limit", help="Sample cases saved per bucket."),
    ] = 20,
) -> None:
    """Write the Stage65 changed-case and citation-tradeoff review."""

    for path in [
        stage64_report,
        split_jsonl,
        candidate_jsonl,
        adapter_report,
        distribution_report,
        candidate_reranker_dataset,
        stage31_summary,
    ]:
        _ensure_file_exists(path)

    report = review_msqa_stage51_changed_cases(
        stage64_report_path=stage64_report,
        split_jsonl_path=split_jsonl,
        candidate_jsonl_path=candidate_jsonl,
        adapter_report_path=adapter_report,
        distribution_report_path=distribution_report,
        candidate_reranker_dataset_path=candidate_reranker_dataset,
        stage31_summary_path=stage31_summary,
        model_name=model,
        train_split=train_split,
        max_answer_candidates=max_answer_candidates,
        max_citation_rank=max_citation_rank,
        sample_limit=sample_limit,
    )
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_msqa_stage51_changed_case_review_visualizations(
            report=report,
            output_dir=visualization_dir,
        )
    report = {
        **report,
        "visualizations": [
            {"name": artifact.name, "path": artifact.path} for artifact in visualizations
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved MSQA Stage51 changed-case review: {output}")


def _console_summary(report: MappingForSummary) -> dict[str, Any]:
    changed = report["changed_case_summary"]
    cohorts = report["cohort_summaries"]
    return {
        "stage": report["stage"],
        "rebuild_contract": report["rebuild_contract"],
        "consistency_checks_passed": report["decision"]["consistency_checks_passed"],
        "changed_case_summary": {
            "question_count": changed["question_count"],
            "changed_answer_count": changed["changed_answer_count"],
            "changed_answer_rate": changed["changed_answer_rate"],
            "top3_regression_count": changed["top3_regression_count"],
            "top3_improvement_count": changed["top3_improvement_count"],
            "net_top3_delta_sum": changed["net_top3_delta_sum"],
            "citation_gained_count": changed["citation_gained_count"],
            "citation_lost_count": changed["citation_lost_count"],
            "citation_delta": changed["citation_delta"],
        },
        "changed_route_counts": cohorts["changed"]["route_counts"],
        "regression_route_counts": cohorts["top3_regressions"]["route_counts"],
        "source_transition_counts": cohorts["changed"]["source_transition_counts"],
        "decision": report["decision"],
        "visualizations": report["visualizations"],
    }


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


MappingForSummary = dict[str, Any]


if __name__ == "__main__":
    app()
