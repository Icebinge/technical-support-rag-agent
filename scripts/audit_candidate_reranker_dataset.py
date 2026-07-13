from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.candidate_reranker_dataset_audit import (
    audit_candidate_reranker_dataset,
    candidate_reranker_dataset_audit_to_dict,
    load_candidate_reranker_rows,
    load_candidate_reranker_summary,
    write_audit_visualizations,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Audit a candidate-reranker dataset and write SVG charts.")


@app.command()
def main(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", help="Input candidate-reranker JSONL dataset."),
    ],
    summary: Annotated[
        Path,
        typer.Option("--summary", help="Input Stage 31 candidate-reranker summary JSON."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output audit JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
) -> None:
    """Audit labels, ranks, route gains, split differences, and leakage risks."""

    _ensure_file_exists(dataset)
    _ensure_file_exists(summary)
    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir / f"candidate_reranker_dataset_audit_{dataset.stem}.json"
    )
    visualization_output_dir = visualization_dir or output_path.with_suffix("")

    rows = load_candidate_reranker_rows(dataset)
    summary_report = load_candidate_reranker_summary(summary)
    audit = audit_candidate_reranker_dataset(rows=rows, summary_report=summary_report)
    visualizations = write_audit_visualizations(
        audit=audit,
        output_dir=visualization_output_dir,
    )

    audit_dict = candidate_reranker_dataset_audit_to_dict(audit)
    audit_dict["source_paths"] = {
        "dataset": str(dataset),
        "summary": str(summary),
    }
    audit_dict["visualizations"] = [
        {"name": visualization.name, "path": visualization.path}
        for visualization in visualizations
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    typer.echo(
        json.dumps(
            {
                "total_rows": audit.total_rows,
                "total_questions": audit.total_questions,
                "best_candidate_rank_distribution": [
                    {
                        "label": bucket.label,
                        "count": bucket.count,
                        "rate": bucket.rate,
                    }
                    for bucket in audit.best_candidate_rank_distribution
                ],
                "split_summaries": [
                    {
                        "split": split_summary.split,
                        "question_count": split_summary.question_count,
                        "average_oracle_gain_vs_top_candidate": (
                            split_summary.average_oracle_gain_vs_top_candidate
                        ),
                        "best_rank_1_rate": split_summary.best_rank_1_rate,
                        "gold_document_candidate_rate": (
                            split_summary.gold_document_candidate_rate
                        ),
                    }
                    for split_summary in audit.split_summaries
                ],
                "label_leakage_detected_from_keys": (
                    audit.feature_leakage_audit.label_leakage_detected_from_keys
                ),
                "output": str(output_path),
                "visualization_dir": str(visualization_output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Missing file: {path}")


if __name__ == "__main__":
    app()
