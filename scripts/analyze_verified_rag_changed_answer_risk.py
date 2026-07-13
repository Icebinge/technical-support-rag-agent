from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.verified_rag_changed_answer_risk_analysis import (
    analyze_verified_rag_changed_answer_risk,
    write_changed_answer_risk_visualizations,
)
from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.primeqa_loader import load_primeqa_questions

app = typer.Typer(help="Analyze changed-answer and unanswerable risk between RAG reports.")


@app.command()
def main(
    baseline_report: Annotated[
        Path,
        typer.Option("--baseline-report", help="Baseline verified RAG report."),
    ],
    candidate_report: Annotated[
        Path,
        typer.Option("--candidate-report", help="Candidate verified RAG report."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output risk analysis JSON path."),
    ],
    questions: Annotated[
        Path | None,
        typer.Option(
            "--questions",
            help="PrimeQA questions JSON path. Defaults to the split in the baseline report.",
        ),
    ] = None,
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
) -> None:
    """Write changed-answer risk analysis for two verified RAG reports."""

    _ensure_file_exists(baseline_report)
    _ensure_file_exists(candidate_report)
    baseline = _load_json(baseline_report)
    candidate = _load_json(candidate_report)
    questions_path = questions or _resolve_questions_path(baseline)
    _ensure_file_exists(questions_path)

    analysis = analyze_verified_rag_changed_answer_risk(
        baseline_report=baseline,
        candidate_report=candidate,
        questions=load_primeqa_questions(questions_path),
    )
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_changed_answer_risk_visualizations(
            analysis=analysis,
            output_dir=visualization_dir,
        )
    report = {
        **analysis,
        "paths": {
            "baseline_report": str(baseline_report),
            "candidate_report": str(candidate_report),
            "questions": str(questions_path),
        },
        "visualizations": [
            {"name": artifact.name, "path": artifact.path} for artifact in visualizations
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(_console_summary(report), ensure_ascii=False, indent=2))
    typer.echo(f"Saved changed-answer risk analysis: {output}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_questions_path(report: MappingLike) -> Path:
    settings = ProjectSettings()
    split = str(report["split"]).strip().lower()
    training_dir = settings.primeqa_raw_dir / "TechQA" / "training_and_dev"
    if split == "dev":
        return training_dir / "dev_Q_A.json"
    if split == "train":
        return training_dir / "training_Q_A.json"
    raise typer.BadParameter(f"Unsupported report split for questions path: {split}")


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    risk = report["risk_observations"]
    return {
        "changed_verified_answers": report["summary"]["changed_verified_answers"],
        "changed_answerable": report["summary"]["changed_answerable"],
        "changed_unanswerable": report["summary"]["changed_unanswerable"],
        "unanswerable_refusal_regressions": report["summary"][
            "unanswerable_refusal_regressions"
        ],
        "route_distribution": report["route_distribution"]["all_changed"],
        "outcome_distribution": report["outcome_distribution"],
        "candidate_has_out_of_rank_citation": report["summary"][
            "candidate_has_out_of_rank_citation"
        ],
        "would_block_if_all_citations_rank_lte_max": {
            "changed_cases": risk[
                "would_block_changed_cases_if_all_citations_rank_lte_max"
            ],
            "unanswerable_regressions": risk[
                "would_block_unanswerable_regressions_if_all_citations_rank_lte_max"
            ],
        },
        "visualizations": report["visualizations"],
    }


MappingLike = dict[str, Any]


if __name__ == "__main__":
    app()
