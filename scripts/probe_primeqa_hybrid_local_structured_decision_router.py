from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from ts_rag_agent.application.primeqa_hybrid_structured_decision_router import (
    Qwen3VLTransformersTextGenerationBackend,
    StrictStructuredDecisionRouter,
    structured_decision_router_contract,
)
from ts_rag_agent.domain.dataset import PrimeQADocument, PrimeQARuntimeQuery
from ts_rag_agent.domain.retrieval import RetrievalResult

app = typer.Typer(help="Probe the Stage157 local structured decision router once.")


@app.command()
def main(
    model_snapshot: Annotated[
        Path,
        typer.Option("--model-snapshot", help="Existing local Qwen snapshot directory."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional public-safe probe JSON."),
    ] = None,
) -> None:
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the selected Stage157 probe")
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    backend = Qwen3VLTransformersTextGenerationBackend.load_local(
        snapshot_path=model_snapshot,
    )
    loaded_at = time.perf_counter()
    router = StrictStructuredDecisionRouter(backend=backend)
    decision = router.decide(
        question=PrimeQARuntimeQuery(
            id="generated-stage157-probe",
            title="Generated adapter verification question",
            text="How should the adapter be verified after installation?",
        ),
        generation_context_results=_generated_evidence(),
        completed_turns=(),
    )
    finished_at = time.perf_counter()
    metrics = router.last_metrics
    if metrics is None:
        raise RuntimeError("local router completed without invocation metrics")
    report: dict[str, Any] = {
        "stage": "Stage 157",
        "probe_id": "primeqa_hybrid_local_structured_decision_router_gpu_probe_v1",
        "probe_data": "generated_synthetic_runtime_data",
        "router_contract": structured_decision_router_contract(),
        "environment": {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
        },
        "model": {
            "snapshot_revision": backend.snapshot_path.name,
            "generation_call_count": backend.generation_call_count,
            "selected_action": decision.action,
            "schema_valid": metrics.schema_valid,
        },
        "metrics": {
            **metrics.to_public_dict(),
            "model_load_latency_ms": round((loaded_at - started_at) * 1000, 3),
            "total_probe_latency_ms": round((finished_at - started_at) * 1000, 3),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "closed_boundaries": {
            "train_split_loaded": False,
            "dev_split_loaded": False,
            "test_split_loaded": False,
            "test_metrics_run": False,
            "real_documents_loaded": False,
            "retrieval_indexes_loaded": False,
            "raw_question_saved": False,
            "raw_document_saved": False,
            "raw_model_output_saved": False,
            "retry_action_count": 0,
            "fallback_action_count": 0,
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


def _generated_evidence() -> tuple[RetrievalResult, ...]:
    return tuple(
        RetrievalResult(
            document=PrimeQADocument(
                id=f"generated-document-{rank}",
                title=f"Generated adapter verification procedure {rank}",
                text=(
                    "After installation, inspect the service configuration, run the documented "
                    "health check, and confirm that the adapter reports a ready state."
                ),
            ),
            score=1.0 / rank,
            rank=rank,
        )
        for rank in range(1, 11)
    )


if __name__ == "__main__":
    app()
