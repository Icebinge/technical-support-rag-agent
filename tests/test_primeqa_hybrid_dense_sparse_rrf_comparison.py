import json
from pathlib import Path

import numpy as np

from ts_rag_agent.application.primeqa_hybrid_dense_sparse_rrf_comparison import (
    run_primeqa_hybrid_dense_sparse_rrf_comparison,
    write_primeqa_hybrid_dense_sparse_rrf_comparison_visualizations,
)


class _FakeEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            normalized = text.lower()
            if "driver" in normalized or "storage" in normalized:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


def test_dense_sparse_rrf_comparison_runs_train_dev_only_and_public_safe(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_dense_sparse_rrf_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage80_report_path=paths["stage80_report"],
        top_k_values=(1, 5, 10),
        search_depth=10,
        candidate_top_k=10,
        encoder_factory=lambda _config: _FakeEncoder(),
    )
    visualizations = write_primeqa_hybrid_dense_sparse_rrf_comparison_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 81"
    assert set(report["metrics_by_split"]) == {"dev", "train"}
    assert set(report["metrics_by_split"]["dev"]) == {
        "dense_sparse_rrf__fixture_local_a__64_noprefix",
        "dense_sparse_rrf__fixture_local_b__64_query",
        "full_document_bm25_baseline",
    }
    assert report["train_selection"]["selected_config_id"].startswith("dense_sparse_rrf__")
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["loaded_data_summary"]["test_split_loaded"] is False
    assert "Restart the service from the admin console" not in serialized
    assert "Install the storage driver package" not in serialized
    assert "Driver installation troubleshooting" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage81_dense_sparse_rrf_train_hit_at_10.svg",
        "stage81_dense_sparse_rrf_dev_hit_at_10.svg",
        "stage81_dense_sparse_rrf_dev_delta_hit_at_10.svg",
        "stage81_dense_sparse_rrf_dev_not_found_at_50.svg",
        "stage81_dense_sparse_rrf_dev_top10_changes.svg",
    }


def test_dense_sparse_rrf_comparison_blocks_protocol_mismatch(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_dense_sparse_rrf_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage80_report_path=paths["stage80_report"],
        user_confirmed_protocol="single_cached_model::fixture/local-a",
        top_k_values=(1, 5, 10),
        search_depth=10,
        candidate_top_k=10,
        encoder_factory=lambda _config: _FakeEncoder(),
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_protocol_matches_stage80_option"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_dense_sparse_rrf_comparison_blocked"
    )
    assert report["metrics_by_split"] == {}


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    train_split = tmp_path / "train.jsonl"
    dev_split = tmp_path / "dev.jsonl"
    _write_jsonl(
        train_split,
        [
            _split_sample(
                sample_id="primeqa_train:TRAIN_Q001",
                assigned_split="train",
                question_title="Restart service",
                question_text="admin console restart",
                answer="Restart the service from the admin console.",
                answer_doc_id="doc-restart",
            )
        ],
    )
    _write_jsonl(
        dev_split,
        [
            _split_sample(
                sample_id="primeqa_dev:DEV_Q001",
                assigned_split="dev",
                question_title="Storage driver install",
                question_text="bundle storage driver",
                answer="Install the storage driver package.",
                answer_doc_id="doc-driver",
            )
        ],
    )
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
                    "text": "Use the storage driver bundle.",
                    "sections": [],
                },
                "doc-decoy": {
                    "id": "doc-decoy",
                    "title": "Driver installation troubleshooting",
                    "text": "general installation troubleshooting",
                    "sections": [],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    dense_cache_dir = tmp_path / "dense"
    dense_cache_dir.mkdir()
    cache_a = dense_cache_dir / "fixture__local-a_64_noprefix.npz"
    cache_b = dense_cache_dir / "fixture__local-b_64_query.npz"
    document_ids = np.asarray(["doc-restart", "doc-driver", "doc-decoy"])
    embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.2, 0.2]],
        dtype=np.float32,
    )
    np.savez_compressed(
        cache_a,
        document_ids=document_ids,
        embeddings=embeddings,
        model_name=np.asarray("fixture/local-a"),
        document_text_max_chars=np.asarray(64),
        document_prefix=np.asarray(""),
    )
    np.savez_compressed(
        cache_b,
        document_ids=document_ids,
        embeddings=embeddings,
        model_name=np.asarray("fixture/local-b"),
        document_text_max_chars=np.asarray(64),
        document_prefix=np.asarray("query: "),
    )
    hub = tmp_path / "hub"
    _write_snapshot(hub, "fixture/local-a", "aaa111")
    _write_snapshot(hub, "fixture/local-b", "bbb222")
    stage75_report = tmp_path / "stage75.json"
    stage75_report.write_text(
        json.dumps(
            {
                "stage": "Stage 75",
                "split_reports": {
                    "train": {"hit_at_top_k": 1.0},
                    "dev": {"hit_at_top_k": 1.0},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    stage80_report = tmp_path / "stage80.json"
    stage80_report.write_text(
        json.dumps(
            {
                "stage": "Stage 80",
                "decision": {
                    "can_run_dense_sparse_rrf_without_download": True,
                    "requires_user_confirmation_before_train_dev_run": True,
                },
                "candidate_options": [
                    {
                        "option_id": "compare_existing_cached_dense_models",
                        "eligible": True,
                    }
                ],
                "dense_cache_candidates": [
                    _stage80_cache_candidate(
                        cache_path=cache_a,
                        model_name="fixture/local-a",
                        document_prefix="",
                        query_prefix=None,
                        snapshot="aaa111",
                        hub=hub,
                    ),
                    _stage80_cache_candidate(
                        cache_path=cache_b,
                        model_name="fixture/local-b",
                        document_prefix="query: ",
                        query_prefix="query: ",
                        snapshot="bbb222",
                        hub=hub,
                    ),
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "train_split": train_split,
        "dev_split": dev_split,
        "documents": documents,
        "stage75_report": stage75_report,
        "stage80_report": stage80_report,
    }


def _stage80_cache_candidate(
    *,
    cache_path: Path,
    model_name: str,
    document_prefix: str,
    query_prefix: str | None,
    snapshot: str,
    hub: Path,
) -> dict:
    return {
        "cache_path": str(cache_path),
        "cache_sha256": "fixture-sha256",
        "model_name": model_name,
        "document_text_max_chars": 64,
        "document_prefix": document_prefix,
        "embedding_shape": [3, 2],
        "document_id_count": 3,
        "document_ids_match_current_corpus": True,
        "can_run_without_model_download": True,
        "huggingface_model_cache": {
            "model_cache_dir": str(hub / f"models--{model_name.replace('/', '--')}"),
            "exists": True,
            "refs_main": snapshot,
            "snapshot_count": 1,
            "snapshots": [snapshot],
        },
        "legacy_metric_matches": [
            {
                "method": "dense",
                "model_name": model_name,
                "query_prefix": query_prefix,
                "document_prefix": document_prefix,
            }
        ],
    }


def _write_snapshot(hub: Path, model_name: str, snapshot: str) -> None:
    model_dir = hub / f"models--{model_name.replace('/', '--')}"
    (model_dir / "refs").mkdir(parents=True)
    (model_dir / "snapshots" / snapshot).mkdir(parents=True)
    (model_dir / "refs" / "main").write_text(snapshot, encoding="utf-8")


def _split_sample(
    *,
    sample_id: str,
    assigned_split: str,
    question_title: str,
    question_text: str,
    answer: str,
    answer_doc_id: str,
) -> dict:
    return {
        "dataset": "primeqa_techqa",
        "split_name": "primeqa_hybrid_stage68_v1",
        "protocol_version": "primeqa_hybrid_split_v1",
        "assigned_split": assigned_split,
        "split_subtype": f"group_random_{assigned_split}",
        "assignment_reason": "group_random_remainder_split",
        "source_split": f"primeqa_{assigned_split}",
        "source_row_index": 1,
        "sample_id": sample_id,
        "question_id": sample_id.split(":", maxsplit=1)[1],
        "question_title": question_title,
        "question_text": question_text,
        "question": f"{question_title}\n\n{question_text}",
        "answerable": True,
        "answer": answer,
        "answer_doc_id": answer_doc_id,
        "candidate_doc_ids": [answer_doc_id],
        "answer_span": {"start_offset": 0, "end_offset": len(answer)},
        "metadata": {
            "group_hash": f"{sample_id}:group",
            "candidate_doc_count": 1,
            "candidate_doc_hash": f"{sample_id}:docs",
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
