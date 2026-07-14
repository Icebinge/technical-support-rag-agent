from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.msqa_stage51_adapter_comparison import (
    compare_msqa_stage51_capped_adapter,
    write_msqa_stage51_adapter_comparison_visualizations,
)

app = typer.Typer(help="Compare Stage51 on the capped Stage63 MSQA candidate pool.")


@app.command()
def main(
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
        typer.Option("--output", help="Output Stage64 comparison report JSON."),
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
    """Run the Stage64 capped MSQA Stage51 adapter comparison."""

    for path in [
        split_jsonl,
        candidate_jsonl,
        adapter_report,
        distribution_report,
        candidate_reranker_dataset,
        stage31_summary,
    ]:
        _ensure_file_exists(path)

    report = compare_msqa_stage51_capped_adapter(
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
        visualizations = write_msqa_stage51_adapter_comparison_visualizations(
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
    typer.echo(f"Saved MSQA Stage51 capped adapter comparison: {output}")


def _console_summary(report: MappingForSummary) -> dict[str, Any]:
    metrics = report["metrics"]
    return {
        "stage": report["stage"],
        "comparison_contract": {
            "candidate_pool_rebuilt": report["comparison_contract"][
                "candidate_pool_rebuilt"
            ],
            "candidate_pool_rows": report["comparison_contract"][
                "candidate_pool_rows"
            ],
            "model_name": report["comparison_contract"]["model_name"],
            "train_split": report["comparison_contract"]["train_split"],
            "max_answer_candidates": report["comparison_contract"][
                "max_answer_candidates"
            ],
            "rank_contained_max_retrieval_rank": report["comparison_contract"][
                "rank_contained_max_retrieval_rank"
            ],
        },
        "metrics": {
            "question_count": metrics["question_count"],
            "baseline_top3_average_answer_token_f1": metrics[
                "baseline_top3_average_answer_token_f1"
            ],
            "stage51_top3_average_answer_token_f1": metrics[
                "stage51_top3_average_answer_token_f1"
            ],
            "top3_average_delta_vs_baseline": metrics[
                "top3_average_delta_vs_baseline"
            ],
            "changed_answer_count": metrics["changed_answer_count"],
            "replacement_count": metrics["replacement_count"],
            "gold_source_citation_delta": metrics["gold_source_citation_delta"],
            "citation_lost_count": metrics["citation_lost_count"],
            "citation_gained_count": metrics["citation_gained_count"],
        },
        "decision_reason_counts": report["decision_reason_counts"],
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
