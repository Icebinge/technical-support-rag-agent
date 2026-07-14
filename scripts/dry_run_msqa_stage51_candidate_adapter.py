from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.msqa_stage51_candidate_adapter import (
    build_msqa_stage51_candidate_adapter_dry_run,
    write_msqa_stage51_candidate_adapter_visualizations,
    write_msqa_stage51_candidate_jsonl,
)

app = typer.Typer(help="Dry-run the confirmed MSQA Stage 51 candidate adapter.")


@app.command()
def main(
    split_jsonl: Annotated[
        Path,
        typer.Option("--split-jsonl", help="Stage 57 frozen MSQA JSONL split."),
    ],
    protocol_report: Annotated[
        Path,
        typer.Option("--protocol-report", help="Stage 60 MSQA protocol report."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output Stage 61 adapter dry-run JSON path."),
    ],
    candidate_output: Annotated[
        Path,
        typer.Option("--candidate-output", help="Output full candidate JSONL path."),
    ],
    visualization_dir: Annotated[
        Path | None,
        typer.Option("--visualization-dir", help="Optional output directory for SVG charts."),
    ] = None,
    confirmed_protocol: Annotated[
        bool,
        typer.Option(
            "--confirmed-protocol",
            help="Required after user confirms Stage 60 option A.",
        ),
    ] = False,
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Answer-only source rows retrieved per query."),
    ] = 10,
    min_sentence_chars: Annotated[
        int,
        typer.Option("--min-sentence-chars", help="Minimum answer sentence length."),
    ] = 1,
    max_candidates_per_source_row: Annotated[
        int | None,
        typer.Option(
            "--max-candidates-per-source-row",
            help="Optional cap after scoring answer-sentence candidates per source row.",
        ),
    ] = None,
    sample_limit: Annotated[
        int,
        typer.Option("--sample-limit", help="Sample summaries saved in the report."),
    ] = 20,
    stage_name: Annotated[
        str,
        typer.Option("--stage-name", help="Report stage label."),
    ] = "Stage 61",
) -> None:
    """Write MSQA candidate adapter dry-run artifacts."""

    for path in [split_jsonl, protocol_report]:
        _ensure_file_exists(path)
    dry_run = build_msqa_stage51_candidate_adapter_dry_run(
        split_jsonl_path=split_jsonl,
        protocol_report_path=protocol_report,
        confirmed_protocol=confirmed_protocol,
        top_k=top_k,
        min_sentence_chars=min_sentence_chars,
        max_candidates_per_source_row=max_candidates_per_source_row,
        sample_limit=sample_limit,
        stage_name=stage_name,
    )
    write_msqa_stage51_candidate_jsonl(
        candidate_rows=dry_run.candidate_rows,
        output_path=candidate_output,
    )
    report = {
        **dry_run.report,
        "candidate_output": {
            "path": str(candidate_output),
            "rows": len(dry_run.candidate_rows),
        },
    }
    visualizations = []
    if visualization_dir is not None:
        visualizations = write_msqa_stage51_candidate_adapter_visualizations(
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
    typer.echo(f"Saved MSQA Stage 51 candidate adapter dry run: {output}")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_confirmation": report["user_confirmation"],
        "adapter_contract": {
            "source_citation_identity": report["adapter_contract"][
                "source_citation_identity"
            ],
            "candidate_construction": report["adapter_contract"][
                "candidate_construction"
            ],
            "retrieval_index_text": report["adapter_contract"]["retrieval_index_text"],
            "excluded_index_text": report["adapter_contract"]["excluded_index_text"],
            "top_k": report["adapter_contract"]["top_k"],
            "max_candidates_per_source_row": report["adapter_contract"][
                "max_candidates_per_source_row"
            ],
            "effective_candidate_pool_cap": report["adapter_contract"][
                "effective_candidate_pool_cap"
            ],
        },
        "dry_run_summary": report["dry_run_summary"],
        "source_retrieval_summary": report["source_retrieval_summary"],
        "candidate_contract_checks": report["candidate_contract_checks"],
        "decision": report["decision"],
        "candidate_output": report["candidate_output"],
        "visualizations": report["visualizations"],
    }


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Path is not a file: {path}")


if __name__ == "__main__":
    app()
