from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.candidate_reranker_dataset_audit import (
    load_candidate_reranker_rows,
)
from ts_rag_agent.application.candidate_reranker_policy_stability import (
    analyze_candidate_reranker_policy_stability,
    candidate_reranker_policy_stability_to_dict,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Analyze fixed constrained candidate-reranker policy stability.")


@app.command()
def main(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", help="Input candidate-reranker JSONL dataset."),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="Candidate reranker model name."),
    ] = "logistic_best_candidate",
    fold_count: Annotated[
        int,
        typer.Option("--fold-count", help="Number of deterministic question folds."),
    ] = 5,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output policy-stability JSON path."),
    ] = None,
) -> None:
    """Compare Stage 35 best policy against a simpler block-how-to-only policy."""

    _ensure_file_exists(dataset)
    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir / f"candidate_reranker_policy_stability_{dataset.stem}.json"
    )

    rows = load_candidate_reranker_rows(dataset)
    result = analyze_candidate_reranker_policy_stability(
        rows=rows,
        model_name=model,
        fold_count=fold_count,
    )
    result_dict = candidate_reranker_policy_stability_to_dict(result)
    result_dict["source_paths"] = {"dataset": str(dataset)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    typer.echo(
        json.dumps(
            {
                "model_name": result.model_name,
                "fold_count": result.fold_count,
                "primary_policy": {
                    "name": result.primary_policy.config.name,
                    **_metrics_summary(result.primary_policy.metrics),
                },
                "challenger_policy": {
                    "name": result.challenger_policy.config.name,
                    **_metrics_summary(result.challenger_policy.metrics),
                },
                "primary_vs_challenger": {
                    "average_delta_difference": (
                        result.primary_vs_challenger.average_delta_difference
                    ),
                    "regressed_count_difference": (
                        result.primary_vs_challenger.regressed_count_difference
                    ),
                    "replacement_count_difference": (
                        result.primary_vs_challenger.replacement_count_difference
                    ),
                },
                "findings": result.findings,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _metrics_summary(metrics) -> dict:
    return {
        "policy_average_token_f1": metrics.policy_average_token_f1,
        "average_delta_vs_top_candidate": metrics.average_delta_vs_top_candidate,
        "oracle_gap_closed_rate": metrics.oracle_gap_closed_rate,
        "replacement_count": metrics.replacement_count,
        "regressed_count": metrics.regressed_count,
        "final_missed_gold_document_count": metrics.final_missed_gold_document_count,
        "final_deep_rank_count": metrics.final_deep_rank_count,
    }


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Missing file: {path}")


if __name__ == "__main__":
    app()
