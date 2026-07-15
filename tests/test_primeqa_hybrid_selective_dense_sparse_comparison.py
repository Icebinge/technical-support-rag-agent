import hashlib
import json
from pathlib import Path

import numpy as np

from ts_rag_agent.application.primeqa_hybrid_selective_dense_sparse_comparison import (
    run_primeqa_hybrid_selective_dense_sparse_comparison,
    write_primeqa_hybrid_selective_dense_sparse_comparison_visualizations,
)


class _FakeEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            normalized = text.lower()
            if "alpha" in normalized or "theta" in normalized:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


def test_selective_dense_sparse_runs_train_dev_only_and_public_safe(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_selective_dense_sparse_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage97_report_path=paths["stage97_report"],
        user_confirmed_protocol=True,
        confirmed_protocol_id="selective_dense_sparse_low_overlap_gate_train_dev_v1",
        confirmation_note="fixture confirmation",
        top_k_values=(1, 5, 10),
        search_depth=10,
        encoder_factory=lambda _config: _FakeEncoder(),
    )
    visualizations = (
        write_primeqa_hybrid_selective_dense_sparse_comparison_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    selected = report["decision"]["selected_policy_id"]
    dev_metrics = report["metrics_by_split"]["dev"][selected]
    dev_comparison = report["comparisons_to_baseline"]["dev"][selected]

    assert report["stage"] == "Stage 98"
    assert set(report["metrics_by_split"]) == {"dev", "train"}
    assert report["loaded_data_summary"]["test_split_loaded"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_selective_dense_sparse_comparison_completed"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(check["passed"] for check in report["guard_checks"])
    assert report["train_selection"]["selected_policy_id"] is not None
    assert dev_metrics["hit_at_k"]["hit@10"] == 1.0
    assert dev_comparison["hit@10_delta"] == 1.0
    assert dev_comparison["not_found_count_at_search_depth_delta"] == -1
    assert dev_comparison["promotion_count"] == 1
    assert "Use the hidden restart command" not in serialized
    assert "Restart the component with the private code" not in serialized
    assert "Private procedure" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage98_selective_dense_sparse_train_hit_at_10.svg",
        "stage98_selective_dense_sparse_dev_hit_at_10.svg",
        "stage98_selective_dense_sparse_dev_hit10_delta.svg",
        "stage98_selective_dense_sparse_dev_not_found_delta.svg",
        "stage98_selective_dense_sparse_dev_promotions.svg",
        "stage98_selective_dense_sparse_guard_check_status.svg",
    }


def test_selective_dense_sparse_blocks_without_user_confirmation(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_selective_dense_sparse_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage97_report_path=paths["stage97_report"],
        user_confirmed_protocol=False,
        confirmed_protocol_id="selective_dense_sparse_low_overlap_gate_train_dev_v1",
        confirmation_note="fixture confirmation",
        top_k_values=(1, 5, 10),
        search_depth=10,
        encoder_factory=lambda _config: _FakeEncoder(),
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage98_train_dev_run"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_selective_dense_sparse_comparison_blocked"
    )
    assert report["metrics_by_split"] == {}


def test_selective_dense_sparse_blocks_protocol_mismatch(tmp_path):
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_selective_dense_sparse_comparison(
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        stage75_report_path=paths["stage75_report"],
        stage97_report_path=paths["stage97_report"],
        user_confirmed_protocol=True,
        confirmed_protocol_id="wrong_protocol",
        confirmation_note="fixture confirmation",
        top_k_values=(1, 5, 10),
        search_depth=10,
        encoder_factory=lambda _config: _FakeEncoder(),
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["confirmed_protocol_id_matches_stage97"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_selective_dense_sparse_comparison_blocked"
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
                question_title="alpha beta gamma delta epsilon",
                question_text="zeta eta theta iota kappa lambda",
                answer="Use the hidden restart command.",
                answer_doc_id="doc-answer",
            )
        ],
    )
    _write_jsonl(
        dev_split,
        [
            _split_sample(
                sample_id="primeqa_dev:DEV_Q001",
                assigned_split="dev",
                question_title="alpha beta gamma delta epsilon",
                question_text="zeta eta theta iota kappa lambda",
                answer="Use the hidden restart command.",
                answer_doc_id="doc-answer",
            )
        ],
    )
    documents = tmp_path / "documents.json"
    documents.write_text(
        json.dumps(
            {
                "doc-answer": {
                    "id": "doc-answer",
                    "title": "Private procedure",
                    "text": "Restart the component with the private code.",
                    "sections": [],
                },
                "doc-decoy": {
                    "id": "doc-decoy",
                    "title": "Public guide",
                    "text": "General service guide with no matching secret phrase.",
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
    cache_b = dense_cache_dir / "fixture__local-b_64_noprefix.npz"
    document_ids = np.asarray(["doc-answer", "doc-decoy"])
    embeddings = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    _write_dense_cache(cache_a, document_ids, embeddings, "fixture/local-a")
    _write_dense_cache(cache_b, document_ids, embeddings, "fixture/local-b")
    snapshot_a = tmp_path / "hub" / "models--fixture--local-a" / "snapshots" / "aaa111"
    snapshot_b = tmp_path / "hub" / "models--fixture--local-b" / "snapshots" / "bbb222"
    snapshot_a.mkdir(parents=True)
    snapshot_b.mkdir(parents=True)
    stage75_report = tmp_path / "stage75.json"
    stage75_report.write_text(
        json.dumps(
            {
                "stage": "Stage 75",
                "split_reports": {
                    "train": {"hit_at_top_k": 0.0},
                    "dev": {"hit_at_top_k": 0.0},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    stage97_report = tmp_path / "stage97.json"
    stage97_report.write_text(
        json.dumps(
            _stage97_report(
                cache_a=cache_a,
                cache_b=cache_b,
                snapshot_a=snapshot_a,
                snapshot_b=snapshot_b,
            ),
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
        "stage97_report": stage97_report,
    }


def _stage97_report(
    *,
    cache_a: Path,
    cache_b: Path,
    snapshot_a: Path,
    snapshot_b: Path,
) -> dict:
    config_a = _dense_config(
        config_id="dense_sparse_rrf__fixture_local_a__64_noprefix",
        model_name="fixture/local-a",
        cache_path=cache_a,
        snapshot_path=snapshot_a,
    )
    config_b = _dense_config(
        config_id="dense_sparse_rrf__fixture_local_b__64_noprefix",
        model_name="fixture/local-b",
        cache_path=cache_b,
        snapshot_path=snapshot_b,
    )
    return {
        "stage": "Stage 97",
        "decision": {
            "status": "primeqa_hybrid_selective_dense_sparse_protocol_frozen",
            "protocol_id": "selective_dense_sparse_low_overlap_gate_train_dev_v1",
            "requires_user_confirmation_before_train_dev_run": True,
        },
        "frozen_protocol": {
            "protocol_id": "selective_dense_sparse_low_overlap_gate_train_dev_v1",
            "candidate_id": "selective_dense_sparse_low_overlap_gate_design",
            "baseline_retriever": {
                "config_id": "full_document_bm25_baseline",
                "bm25_k1": 1.5,
                "bm25_b": 0.75,
                "candidate_depth": 10,
                "primary_top_k": 10,
            },
            "dense_cache_contract": {
                "allowed_cache_source": "stage80_compatible_local_dense_caches_only",
                "allowed_dense_configs": [config_a, config_b],
                "download_required": False,
                "document_reencoding_allowed": False,
                "query_encoding_mode": "local_snapshot_path_with_local_files_only",
            },
            "candidate_policy_grid": [
                _policy(
                    policy_id="sdsl_fixture_a_v1",
                    dense_config_id=config_a["config_id"],
                    promotion_budget=2,
                    protected_count=1,
                ),
                _policy(
                    policy_id="sdsl_fixture_b_v1",
                    dense_config_id=config_b["config_id"],
                    promotion_budget=1,
                    protected_count=1,
                ),
                _policy(
                    policy_id="sdsl_fixture_a_dense_bias_v1",
                    dense_config_id=config_a["config_id"],
                    dense_weight=1.25,
                    promotion_budget=2,
                    protected_count=1,
                ),
                _policy(
                    policy_id="sdsl_fixture_b_conservative_v1",
                    dense_config_id=config_b["config_id"],
                    dense_weight=0.85,
                    promotion_budget=1,
                    protected_count=1,
                ),
            ],
            "train_selection_rule": {
                "selection_split": "train",
                "validation_split": "dev",
                "dev_selection_forbidden": True,
                "test_selection_forbidden": True,
            },
            "metrics_allowed_after_confirmation": ["hit@10", "MRR@50"],
            "public_safe_changed_case_fields": [
                "sample_id",
                "split",
                "baseline_rank",
                "challenger_rank",
                "policy_id",
                "dense_config_id",
                "baseline_rank_bucket",
                "challenger_rank_bucket",
                "gate_activation_reason_code",
                "query_length_bucket",
                "bm25_top1_overlap_bucket",
                "bm25_top10_mean_overlap_bucket",
                "dense_rank_bucket",
                "promotion_budget_used",
            ],
        },
    }


def _dense_config(
    *,
    config_id: str,
    model_name: str,
    cache_path: Path,
    snapshot_path: Path,
) -> dict:
    return {
        "config_id": config_id,
        "model_name": model_name,
        "cache_path": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "document_text_max_chars": 64,
        "document_prefix": "",
        "query_prefix": "",
        "query_prefix_source": "fixture",
        "embedding_shape": [2, 2],
        "document_id_count": 2,
        "can_run_without_model_download_in_stage80": True,
        "snapshot_path": str(snapshot_path),
        "snapshot_status": "fixture",
    }


def _policy(
    *,
    policy_id: str,
    dense_config_id: str,
    dense_weight: float = 1.0,
    promotion_budget: int,
    protected_count: int,
) -> dict:
    return {
        "policy_id": policy_id,
        "dense_config_id": dense_config_id,
        "gate_mode": "low_bm25_lexical_overlap",
        "minimum_query_token_count": 8,
        "maximum_bm25_top1_query_overlap_ratio": 0.3,
        "maximum_bm25_top10_mean_query_overlap_ratio": 0.3,
        "dense_candidate_rank_max": 10,
        "sparse_weight": 1.0,
        "dense_weight": dense_weight,
        "rrf_k": 60,
        "maximum_dense_top10_promotions_per_query": promotion_budget,
        "protected_bm25_top_rank_count": protected_count,
        "dense_candidate_must_be_outside_bm25_top10": True,
        "dense_config_present_in_stage81": True,
    }


def _write_dense_cache(
    path: Path,
    document_ids: np.ndarray,
    embeddings: np.ndarray,
    model_name: str,
) -> None:
    np.savez_compressed(
        path,
        document_ids=document_ids,
        embeddings=embeddings,
        model_name=np.asarray(model_name),
        document_text_max_chars=np.asarray(64),
        document_prefix=np.asarray(""),
    )


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
