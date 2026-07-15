import json
from pathlib import Path

import numpy as np

from ts_rag_agent.application.primeqa_hybrid_dense_sparse_rrf_feasibility import (
    check_primeqa_hybrid_dense_sparse_rrf_feasibility,
    write_primeqa_hybrid_dense_sparse_rrf_feasibility_visualizations,
)


def test_dense_sparse_rrf_feasibility_finds_local_cache_without_raw_text(tmp_path):
    paths = _write_fixture(tmp_path, include_cache=True)

    report = check_primeqa_hybrid_dense_sparse_rrf_feasibility(
        documents_path=paths["documents"],
        pyproject_path=paths["pyproject"],
        stage76_report_path=paths["stage76_report"],
        stage79_report_path=paths["stage79_report"],
        dense_cache_dir=paths["dense_cache_dir"],
        huggingface_hub_dir=paths["huggingface_hub_dir"],
        legacy_metric_paths=[paths["legacy_metric"]],
    )
    visualizations = write_primeqa_hybrid_dense_sparse_rrf_feasibility_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 80"
    assert report["decision"]["can_run_dense_sparse_rrf_without_download"] is True
    assert report["decision"]["requires_user_confirmation_before_train_dev_run"] is True
    assert report["dense_cache_candidates"][0]["document_ids_match_current_corpus"] is True
    assert report["dense_cache_candidates"][0]["can_run_without_model_download"] is True
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Restart the service from the admin console" not in serialized
    assert "Install the storage driver package" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage80_dense_cache_readiness.svg",
        "stage80_dependency_availability.svg",
        "stage80_candidate_options.svg",
    }


def test_dense_sparse_rrf_feasibility_blocks_without_local_cache(tmp_path):
    paths = _write_fixture(tmp_path, include_cache=False)

    report = check_primeqa_hybrid_dense_sparse_rrf_feasibility(
        documents_path=paths["documents"],
        pyproject_path=paths["pyproject"],
        stage76_report_path=paths["stage76_report"],
        stage79_report_path=paths["stage79_report"],
        dense_cache_dir=paths["dense_cache_dir"],
        huggingface_hub_dir=paths["huggingface_hub_dir"],
        legacy_metric_paths=[],
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["compatible_local_dense_cache_available"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_dense_sparse_rrf_feasibility_blocked"
    )


def _write_fixture(tmp_path: Path, *, include_cache: bool) -> dict[str, Path]:
    documents = tmp_path / "documents.json"
    documents.write_text(
        json.dumps(
            {
                "doc-restart": {
                    "id": "doc-restart",
                    "title": "Restart service",
                    "text": "Restart the service from the admin console.",
                    "sections": [],
                },
                "doc-driver": {
                    "id": "doc-driver",
                    "title": "Storage driver install",
                    "text": "Install the storage driver package.",
                    "sections": [],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "\n".join(
            [
                "[project.optional-dependencies]",
                'rag = ["rank-bm25>=0.2.2", "scikit-learn>=1.4", "sentence-transformers>=3.0"]',
            ]
        ),
        encoding="utf-8",
    )
    stage76_report = tmp_path / "stage76.json"
    stage76_report.write_text(
        json.dumps(
            {
                "stage": "Stage 76",
                "candidate_designs": [
                    {
                        "candidate_id": "dense_sparse_rrf_train_dev_probe",
                        "status": "recommended_for_train_dev_experiment",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    stage79_report = tmp_path / "stage79.json"
    stage79_report.write_text(
        json.dumps(
            {
                "stage": "Stage 79",
                "decision": {"can_open_final_test_gate_now": False},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    dense_cache_dir = tmp_path / "dense"
    dense_cache_dir.mkdir()
    huggingface_hub_dir = tmp_path / "hub"
    model_dir = huggingface_hub_dir / "models--fixture--local-model"
    (model_dir / "refs").mkdir(parents=True)
    (model_dir / "snapshots" / "abc123").mkdir(parents=True)
    (model_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    cache_path = dense_cache_dir / "fixture__local-model_64_noprefix.npz"
    if include_cache:
        np.savez_compressed(
            cache_path,
            document_ids=np.asarray(["doc-restart", "doc-driver"]),
            embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            model_name=np.asarray("fixture/local-model"),
            document_text_max_chars=np.asarray(64),
            document_prefix=np.asarray(""),
        )
    legacy_metric = tmp_path / "dense_legacy.json"
    legacy_metric.write_text(
        json.dumps(
            {
                "split": "dev",
                "paths": {"embedding_cache": str(cache_path)},
                "dense": {
                    "model_name": "fixture/local-model",
                    "query_prefix": "",
                    "document_prefix": "",
                    "cache_status": "loaded",
                },
                "metrics": {"hit_at_k": {"hit@10": 1.0}, "mrr": 1.0},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "documents": documents,
        "pyproject": pyproject,
        "stage76_report": stage76_report,
        "stage79_report": stage79_report,
        "dense_cache_dir": dense_cache_dir,
        "huggingface_hub_dir": huggingface_hub_dir,
        "legacy_metric": legacy_metric,
    }
