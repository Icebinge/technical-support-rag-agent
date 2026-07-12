from __future__ import annotations

import shutil
from pathlib import Path

import typer
from huggingface_hub import hf_hub_download

from ts_rag_agent.config import ProjectSettings

app = typer.Typer(help="Download public datasets for the Technical Support RAG Agent.")


def _copy_from_cache(cached_path: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_path, destination)


@app.command()
def main(
    eval_only: bool = typer.Option(
        False,
        "--eval-only",
        help="Download only the lightweight NVIDIA TechQA-RAG-Eval dataset.",
    ),
    include_primeqa: bool = typer.Option(
        False,
        "--include-primeqa",
        help="Also download the original PrimeQA/TechQA archive. This file is large.",
    ),
) -> None:
    settings = ProjectSettings()

    eval_dir = settings.nvidia_raw_dir
    eval_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("train.json", "corpus.zip"):
        cached = hf_hub_download(repo_id=settings.eval_repo, repo_type="dataset", filename=filename)
        _copy_from_cache(cached, eval_dir / filename)
        typer.echo(f"Downloaded {settings.eval_repo}/{filename} -> {eval_dir / filename}")

    should_download_primeqa = include_primeqa and not eval_only
    if should_download_primeqa:
        primeqa_dir = settings.primeqa_raw_dir
        primeqa_dir.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(
            repo_id=settings.train_repo,
            repo_type="dataset",
            filename="TechQA.tar.gz",
        )
        _copy_from_cache(cached, primeqa_dir / "TechQA.tar.gz")
        typer.echo(
            f"Downloaded {settings.train_repo}/TechQA.tar.gz "
            f"-> {primeqa_dir / 'TechQA.tar.gz'}"
        )
    elif include_primeqa and eval_only:
        typer.echo("--eval-only was set, so the PrimeQA archive was not downloaded.")


if __name__ == "__main__":
    app()
