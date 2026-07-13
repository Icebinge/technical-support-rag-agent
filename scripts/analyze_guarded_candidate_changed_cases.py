from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.candidate_reranker_dataset_audit import (
    load_candidate_reranker_rows,
)
from ts_rag_agent.application.guarded_candidate_reranker_changed_case_analysis import (
    analyze_guarded_candidate_changed_cases,
    guarded_candidate_changed_case_analysis_to_dict,
    write_changed_case_visualizations,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_questions

app = typer.Typer(
    help="Analyze changed cases from the guarded candidate-reranker top-k answer proxy."
)


@app.command()
def main(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", help="Input candidate-reranker JSONL dataset."),
    ],
    stage37_report: Annotated[
        Path,
        typer.Option("--stage37-report", help="Stage 37 answer experiment JSON report."),
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
            help="Top-k size used by the leading-candidate rewrite proxy.",
        ),
    ] = 3,
    sample_limit: Annotated[
        int,
        typer.Option(
            "--sample-limit",
            help="Maximum representative cases kept per case bucket.",
        ),
    ] = 50,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output changed-case analysis JSON path."),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Output directory for SVG charts."),
    ] = None,
) -> None:
    """Run Stage 38 changed-case attribution and stricter-gate audit."""

    _ensure_file_exists(dataset)
    _ensure_file_exists(stage37_report)
    _validate_options(
        fold_count=fold_count,
        max_answer_candidates=max_answer_candidates,
        sample_limit=sample_limit,
    )
    split_names = _parse_splits(splits)
    settings = ProjectSettings()
    output_path = output or (
        settings.artifact_dir / f"guarded_candidate_changed_cases_{dataset.stem}.json"
    )
    visualization_output_dir = visualization_dir or output_path.with_suffix("")

    rows = load_candidate_reranker_rows(dataset)
    report = _load_json_object(stage37_report)
    gold_answers = _load_gold_answers(settings=settings, splits=split_names)
    analysis = analyze_guarded_candidate_changed_cases(
        stage37_report=report,
        rows=rows,
        gold_answers_by_question_key=gold_answers,
        model_name=model,
        fold_count=fold_count,
        max_answer_candidates=max_answer_candidates,
        sample_limit=sample_limit,
    )
    visualizations = write_changed_case_visualizations(
        analysis=analysis,
        output_dir=visualization_output_dir,
    )
    analysis_dict = guarded_candidate_changed_case_analysis_to_dict(analysis)
    analysis_dict["source_paths"] = {
        "dataset": str(dataset),
        "stage37_report": str(stage37_report),
        "gold_answer_splits": split_names,
    }
    analysis_dict["visualizations"] = [
        {"name": visualization.name, "path": visualization.path}
        for visualization in visualizations
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(analysis_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    typer.echo(
        json.dumps(
            {
                "model_name": analysis.model_name,
                "fold_count": analysis.fold_count,
                "mode_name": analysis.mode_name,
                "policy_name": analysis.policy_name,
                "metrics": {
                    "question_count": analysis.metrics.question_count,
                    "changed_case_count": analysis.metrics.changed_case_count,
                    "average_delta_vs_baseline": (
                        analysis.metrics.average_delta_vs_baseline
                    ),
                    "improved_count": analysis.metrics.improved_count,
                    "regressed_count": analysis.metrics.regressed_count,
                    "citation_lost_count": analysis.metrics.citation_lost_count,
                    "citation_gained_count": analysis.metrics.citation_gained_count,
                    "gold_citation_delta": analysis.metrics.gold_citation_delta,
                },
                "best_gate_by_delta": _best_gate_by_delta(analysis),
                "best_gate_by_regression": _best_gate_by_regression(analysis),
                "findings": analysis.findings,
                "output": str(output_path),
                "visualization_dir": str(visualization_output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _best_gate_by_delta(analysis) -> dict:
    audit = max(
        analysis.stricter_gate_audits,
        key=lambda gate: (
            gate.average_delta_vs_baseline,
            -gate.regressed_count,
        ),
    )
    return _gate_summary(audit)


def _best_gate_by_regression(analysis) -> dict:
    audit = min(
        analysis.stricter_gate_audits,
        key=lambda gate: (
            gate.regressed_count,
            -gate.average_delta_vs_baseline,
        ),
    )
    return _gate_summary(audit)


def _gate_summary(audit) -> dict:
    return {
        "name": audit.name,
        "blocked_replacement_count": audit.blocked_replacement_count,
        "average_delta_vs_baseline": audit.average_delta_vs_baseline,
        "regressed_count": audit.regressed_count,
        "citation_lost_count": audit.citation_lost_count,
        "citation_gained_count": audit.citation_gained_count,
        "gold_citation_delta": audit.gold_citation_delta,
    }


def _load_json_object(path: Path) -> dict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise typer.BadParameter(f"JSON file must contain an object: {path}")
    return loaded


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
