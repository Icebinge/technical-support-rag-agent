from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.candidate_reranker_dataset_audit import (
    load_candidate_reranker_rows,
)
from ts_rag_agent.application.guarded_candidate_reranker_answer_experiment import (
    guarded_candidate_answer_experiment_to_dict,
    run_guarded_candidate_reranker_answer_experiment,
    write_guarded_candidate_answer_visualizations,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_questions

app = typer.Typer(
    help="Evaluate guarded candidate-reranker decisions as answer-level offline proxies."
)


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
    splits: Annotated[
        str,
        typer.Option(
            "--splits",
            help="Comma-separated PrimeQA splits used to load gold answers.",
        ),
    ] = "dev,train",
    max_answer_candidates: Annotated[
        int,
        typer.Option(
            "--max-answer-candidates",
            help="Top-k size for the leading-candidate rewrite proxy mode.",
        ),
    ] = 3,
    sample_limit: Annotated[
        int,
        typer.Option("--sample-limit", help="Maximum sample cases kept per bucket."),
    ] = 20,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output answer experiment JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
) -> None:
    """Run the Stage 37 guarded answer-level offline proxy experiment."""

    _ensure_file_exists(dataset)
    _validate_options(
        fold_count=fold_count,
        max_answer_candidates=max_answer_candidates,
        sample_limit=sample_limit,
    )
    split_names = _parse_splits(splits)
    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir / f"candidate_reranker_answer_experiment_{dataset.stem}.json"
    )
    visualization_output_dir = visualization_dir or output_path.with_suffix("")

    rows = load_candidate_reranker_rows(dataset)
    gold_answers = _load_gold_answers(settings=settings, splits=split_names)
    result = run_guarded_candidate_reranker_answer_experiment(
        rows=rows,
        gold_answers_by_question_key=gold_answers,
        model_name=model,
        fold_count=fold_count,
        max_answer_candidates=max_answer_candidates,
        sample_limit=sample_limit,
    )
    visualizations = write_guarded_candidate_answer_visualizations(
        result=result,
        output_dir=visualization_output_dir,
    )
    result_dict = guarded_candidate_answer_experiment_to_dict(result)
    result_dict["source_paths"] = {
        "dataset": str(dataset),
        "gold_answer_splits": split_names,
    }
    result_dict["visualizations"] = [
        {"name": visualization.name, "path": visualization.path}
        for visualization in visualizations
    ]

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
                "max_answer_candidates": result.max_answer_candidates,
                "main_policy": _policy_summary(result.main_policy),
                "sensitivity_policy": _policy_summary(result.sensitivity_policy),
                "main_vs_sensitivity": [
                    {
                        "mode_name": delta.mode_name,
                        "policy_average_f1_difference": (
                            delta.policy_average_f1_difference
                        ),
                        "average_delta_difference": delta.average_delta_difference,
                        "regressed_count_difference": delta.regressed_count_difference,
                        "gold_citation_count_difference": (
                            delta.gold_citation_count_difference
                        ),
                    }
                    for delta in result.main_vs_sensitivity
                ],
                "findings": result.findings,
                "output": str(output_path),
                "visualization_dir": str(visualization_output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _policy_summary(policy) -> dict:
    return {
        "name": policy.config.name,
        "modes": [
            {
                "mode_name": evaluation.mode_name,
                "baseline_average_answer_token_f1": (
                    evaluation.metrics.baseline_average_answer_token_f1
                ),
                "policy_average_answer_token_f1": (
                    evaluation.metrics.policy_average_answer_token_f1
                ),
                "average_delta_vs_baseline": (
                    evaluation.metrics.average_delta_vs_baseline
                ),
                "oracle_gap_closed_rate": evaluation.metrics.oracle_gap_closed_rate,
                "replacement_count": evaluation.metrics.replacement_count,
                "regressed_count": evaluation.metrics.regressed_count,
                "baseline_gold_citation_count": (
                    evaluation.metrics.baseline_gold_citation_count
                ),
                "policy_gold_citation_count": (
                    evaluation.metrics.policy_gold_citation_count
                ),
                "gold_citation_delta": evaluation.metrics.gold_citation_delta,
            }
            for evaluation in policy.mode_evaluations
        ],
    }


def _load_gold_answers(settings: ProjectSettings, splits: list[str]) -> dict[str, str]:
    training_dir = settings.primeqa_raw_dir / "TechQA" / "training_and_dev"
    gold_answers = {}
    for split in splits:
        questions_path = _resolve_questions_path(training_dir, split)
        _ensure_file_exists(questions_path)
        for question in load_primeqa_questions(questions_path):
            if question.answerable:
                gold_answers[f"{split}::{question.id}"] = question.answer
    return gold_answers


def _parse_splits(raw_splits: str) -> list[str]:
    split_names = [split.strip().lower() for split in raw_splits.split(",") if split.strip()]
    if not split_names:
        raise typer.BadParameter("--splits must not be empty.")
    allowed = {"dev", "train"}
    invalid = sorted(set(split_names) - allowed)
    if invalid:
        raise typer.BadParameter(f"Unsupported split(s): {', '.join(invalid)}")
    return split_names


def _validate_options(
    fold_count: int,
    max_answer_candidates: int,
    sample_limit: int,
) -> None:
    if fold_count < 2:
        raise typer.BadParameter("--fold-count must be at least 2.")
    if max_answer_candidates <= 0:
        raise typer.BadParameter("--max-answer-candidates must be positive.")
    if sample_limit < 0:
        raise typer.BadParameter("--sample-limit must be non-negative.")


def _resolve_questions_path(training_dir: Path, split: str) -> Path:
    if split == "dev":
        return training_dir / "dev_Q_A.json"
    if split == "train":
        return training_dir / "training_Q_A.json"
    raise typer.BadParameter("--splits must contain only dev and train.")


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Missing file: {path}")


if __name__ == "__main__":
    app()
