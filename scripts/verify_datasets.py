from __future__ import annotations

import json

import typer

from ts_rag_agent.config import ProjectSettings
from ts_rag_agent.infrastructure.techqa_loader import (
    compute_dataset_stats,
    list_corpus_filenames,
    load_nvidia_samples,
)

app = typer.Typer(help="Verify local TechQA dataset files and write a report.")


@app.command()
def main() -> None:
    settings = ProjectSettings()
    train_json = settings.nvidia_raw_dir / "train.json"
    corpus_zip = settings.nvidia_raw_dir / "corpus.zip"
    primeqa_archive = settings.primeqa_raw_dir / "TechQA.tar.gz"

    if not train_json.exists():
        raise typer.BadParameter(f"Missing file: {train_json}")
    if not corpus_zip.exists():
        raise typer.BadParameter(f"Missing file: {corpus_zip}")

    samples = load_nvidia_samples(train_json)
    corpus_filenames = list_corpus_filenames(corpus_zip)
    stats = compute_dataset_stats(samples, corpus_filenames)

    report = {
        "nvidia_techqa_rag_eval": {
            "train_json": str(train_json),
            "corpus_zip": str(corpus_zip),
            "train_json_mb": round(train_json.stat().st_size / 1024 / 1024, 2),
            "corpus_zip_mb": round(corpus_zip.stat().st_size / 1024 / 1024, 2),
            "stats": stats.model_dump(),
        },
        "primeqa_techqa": {
            "archive": str(primeqa_archive),
            "exists": primeqa_archive.exists(),
            "archive_mb": round(primeqa_archive.stat().st_size / 1024 / 1024, 2)
            if primeqa_archive.exists()
            else None,
        },
    }

    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.artifact_dir / "dataset_verification.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    typer.echo(f"Saved verification report: {report_path}")


if __name__ == "__main__":
    app()
