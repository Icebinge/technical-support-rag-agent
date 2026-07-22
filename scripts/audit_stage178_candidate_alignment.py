from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ts_rag_agent.application.primeqa_hybrid_stage178_candidate_alignment import (
    run_stage178_candidate_alignment_audit,
)
from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Audit Stage161 replay against the live Stage128 candidate pool.")


@app.command()
def main(
    output: Annotated[Path | None, typer.Option("--output")] = None,
    encoder_batch_size: Annotated[int, typer.Option("--encoder-batch-size")] = 64,
) -> None:
    settings = ProjectSettings()
    artifacts = settings.artifact_dir
    split_dir = artifacts / "primeqa_hybrid_split_stage68_splits"
    report = run_stage178_candidate_alignment_audit(
        stage128_protocol_path=artifacts
        / "primeqa_hybrid_agent_retrieval_integration_protocol_stage128.json",
        stage125_protocol_path=artifacts
        / "primeqa_hybrid_prefix_preserving_recall_expansion_protocol_stage125.json",
        stage80_report_path=artifacts / "primeqa_hybrid_dense_sparse_rrf_feasibility_stage80.json",
        train_split_path=split_dir / "primeqa_hybrid_split_stage68_train.jsonl",
        documents_path=settings.primeqa_raw_dir
        / "TechQA"
        / "training_and_dev"
        / "training_dev_technotes.sections.json",
        encoder_batch_size=encoder_batch_size,
    )
    output_path = output or artifacts / "stage178_candidate_alignment_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, ensure_ascii=True, indent=2))
    typer.echo(f"Saved Stage178 candidate alignment audit: {output_path}")


if __name__ == "__main__":
    app()
