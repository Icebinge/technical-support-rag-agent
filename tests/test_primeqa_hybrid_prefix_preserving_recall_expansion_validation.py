from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ts_rag_agent.application.primeqa_hybrid_prefix_preserving_recall_expansion_protocol import (
    _candidate_configs,
    _selection_rules,
)
from ts_rag_agent.application.primeqa_hybrid_prefix_preserving_recall_expansion_validation import (
    run_primeqa_hybrid_prefix_preserving_recall_expansion_validation,
    write_primeqa_hybrid_prefix_preserving_recall_expansion_validation_visualizations,
)


class _FakeEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            normalized = text.lower()
            if "driver" in normalized or "storage" in normalized:
                vectors.append([0.0, 1.0])
            elif "network" in normalized:
                vectors.append([0.5, 0.5])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


def test_prefix_preserving_recall_expansion_validation_runs_train_dev_only(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_prefix_preserving_recall_expansion_validation(
        stage125_protocol_path=paths["stage125_protocol"],
        stage80_report_path=paths["stage80_report"],
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        user_confirmed_validation=True,
        confirmation_note="unit test confirmed Stage126 validation",
        train_fold_count=5,
        encoder_factory=lambda _config: _FakeEncoder(),
    )
    visualizations = (
        write_primeqa_hybrid_prefix_preserving_recall_expansion_validation_visualizations(
            report=report,
            output_dir=tmp_path / "visuals",
        )
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 126"
    assert report["analysis_id"] == (
        "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_validation_v1"
    )
    assert set(report["baseline_by_split"]) == {"dev", "train"}
    assert len(report["config_reviews"]) == 6
    assert report["baseline_by_split"]["train"]["hit_at_200_count"] == 5
    assert report["baseline_by_split"]["dev"]["hit_at_200_count"] == 1
    assert report["loaded_data_summary"]["test_split_loaded"] is False
    assert report["dense_channel_preflight"]["status"] == "dense_channels_ready"
    assert report["train_selection"]["dev_used_for_selection"] is False
    assert report["dev_report_observations"]["dev_used_for_retuning"] is False
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["fallback_strategies_enabled"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert all(
        split_review["prefix_identity_violation_count"] == 0
        for review in report["config_reviews"]
        for split_review in review["split_reviews"].values()
    )
    assert all(
        split_review["hit_at_200_loss_count"] == 0
        for review in report["config_reviews"]
        for split_review in review["split_reviews"].values()
    )
    assert report["public_safe_contract"]["forbidden_keys_found"] == []
    assert all(check["passed"] for check in report["guard_checks"])
    assert "Restart the service from the admin console" not in serialized
    assert "Install the storage driver package" not in serialized
    assert "Driver installation troubleshooting" not in serialized
    assert '"question_text":' not in serialized
    assert '"answer_doc_id":' not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage126_train_target_depth_gain.svg",
        "stage126_dev_target_depth_gain.svg",
        "stage126_train_appended_gold_recovery.svg",
        "stage126_dev_appended_gold_recovery.svg",
        "stage126_train_hit200_loss.svg",
        "stage126_prefix_identity_violations.svg",
        "stage126_selected_append_count_summary.svg",
        "stage126_selection_decision_flags.svg",
        "stage126_guard_check_status.svg",
    }


def test_prefix_preserving_recall_expansion_validation_blocks_without_confirmation(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_prefix_preserving_recall_expansion_validation(
        stage125_protocol_path=paths["stage125_protocol"],
        stage80_report_path=paths["stage80_report"],
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        user_confirmed_validation=False,
        confirmation_note="not confirmed",
        train_fold_count=5,
        encoder_factory=lambda _config: _FakeEncoder(),
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_stage126_validation"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_validation_blocked"
    )
    assert report["config_reviews"] == []


def test_prefix_preserving_recall_expansion_validation_blocks_when_dense_missing(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)

    report = run_primeqa_hybrid_prefix_preserving_recall_expansion_validation(
        stage125_protocol_path=paths["stage125_protocol"],
        stage80_report_path=None,
        train_split_path=paths["train_split"],
        dev_split_path=paths["dev_split"],
        documents_path=paths["documents"],
        user_confirmed_validation=True,
        confirmation_note="unit test confirmed Stage126 validation",
        train_fold_count=5,
        encoder_factory=lambda _config: _FakeEncoder(),
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage126_dense_channels_ready"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_validation_blocked"
    )


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    train_split = tmp_path / "train.jsonl"
    dev_split = tmp_path / "dev.jsonl"
    train_rows = [
        _split_sample(
            sample_id=f"primeqa_train:TRAIN_Q{index:03d}",
            assigned_split="train",
            question_title=title,
            question_text=text,
            answer=answer,
            answer_doc_id=doc_id,
        )
        for index, (title, text, answer, doc_id) in enumerate(
            [
                (
                    "Restart service",
                    "admin console restart",
                    "Restart the service from the admin console.",
                    "doc-restart",
                ),
                (
                    "Driver install",
                    "storage driver bundle",
                    "Install the storage driver package.",
                    "doc-driver",
                ),
                (
                    "Network timeout",
                    "network timeout gateway",
                    "Check the gateway timeout settings.",
                    "doc-network",
                ),
                (
                    "CVE-2024-1234 patch",
                    "security patch CVE-2024-1234",
                    "Apply the CVE-2024-1234 security patch.",
                    "doc-cve",
                ),
                (
                    "License activation",
                    "license activation key",
                    "Activate the license key.",
                    "doc-license",
                ),
            ],
            start=1,
        )
    ]
    dev_rows = [
        _split_sample(
            sample_id="primeqa_dev:DEV_Q001",
            assigned_split="dev",
            question_title="Storage driver install",
            question_text="bundle storage driver",
            answer="Install the storage driver package.",
            answer_doc_id="doc-driver",
        )
    ]
    _write_jsonl(train_split, train_rows)
    _write_jsonl(dev_split, dev_rows)
    documents = tmp_path / "documents.json"
    document_payload = {
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
        "doc-network": {
            "id": "doc-network",
            "title": "Network timeout",
            "text": "Check the gateway timeout settings.",
            "sections": [],
        },
        "doc-cve": {
            "id": "doc-cve",
            "title": "CVE-2024-1234 patch",
            "text": "Apply the CVE-2024-1234 security patch.",
            "sections": [],
        },
        "doc-license": {
            "id": "doc-license",
            "title": "License activation",
            "text": "Activate the license key.",
            "sections": [],
        },
        "doc-decoy": {
            "id": "doc-decoy",
            "title": "Driver installation troubleshooting",
            "text": "general installation troubleshooting",
            "sections": [],
        },
    }
    documents.write_text(
        json.dumps(document_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage125_protocol = tmp_path / "stage125.json"
    stage125_protocol.write_text(
        json.dumps(_stage125_protocol(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stage80_report = _write_stage80_report(
        tmp_path=tmp_path,
        document_ids=tuple(document_payload),
    )
    return {
        "train_split": train_split,
        "dev_split": dev_split,
        "documents": documents,
        "stage125_protocol": stage125_protocol,
        "stage80_report": stage80_report,
    }


def _stage125_protocol() -> dict:
    return {
        "stage": "Stage 125",
        "protocol_id": (
            "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_protocol_v1"
        ),
        "frozen_protocol": {
            "baseline_prefix_contract": {
                "baseline_config_id": "stage116_fixed_rrf_top200_baseline",
                "prefix_depth": 200,
                "train_baseline_hit_at_200_count": 5,
                "train_baseline_hit_at_200": 1.0,
                "dev_baseline_hit_at_200_count": 1,
                "dev_baseline_hit_at_200": 1.0,
                "ranks_1_to_200_must_remain_identical": True,
                "prefix_documents_may_be_reordered": False,
                "prefix_documents_may_be_dropped": False,
                "prefix_duplicate_in_append_region_allowed": False,
                "hit_at_200_loss_count_must_be_zero_by_construction": True,
            },
            "candidate_configs": _candidate_configs(),
            "selection_rules": _selection_rules(),
        },
        "decision": {
            "status": (
                "primeqa_hybrid_stage116_prefix_preserving_recall_expansion_"
                "protocol_frozen"
            ),
            "recommended_next_direction": (
                "run_stage116_prefix_preserving_recall_expansion_train_cv_dev_"
                "validation"
            ),
            "can_continue_train_dev_development": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "fallback_strategies_enabled": False,
            "default_runtime_policy": "unchanged",
        },
        "public_safe_contract": {
            "forbidden_keys_found": [],
        },
    }


def _write_stage80_report(*, tmp_path: Path, document_ids: tuple[str, ...]) -> Path:
    dense_cache_dir = tmp_path / "dense"
    dense_cache_dir.mkdir()
    cache_a = dense_cache_dir / "fixture__local-a_64_noprefix.npz"
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.2, 0.2],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        cache_a,
        document_ids=np.asarray(document_ids),
        embeddings=embeddings,
        model_name=np.asarray("fixture/local-a"),
        document_text_max_chars=np.asarray(64),
        document_prefix=np.asarray(""),
    )
    hub = tmp_path / "hub"
    _write_snapshot(hub, "fixture/local-a", "aaa111")
    path = tmp_path / "stage80.json"
    path.write_text(
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
                        document_count=len(document_ids),
                    )
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _stage80_cache_candidate(
    *,
    cache_path: Path,
    model_name: str,
    document_prefix: str,
    query_prefix: str | None,
    snapshot: str,
    hub: Path,
    document_count: int,
) -> dict:
    return {
        "cache_path": str(cache_path),
        "cache_sha256": "fixture-sha256",
        "model_name": model_name,
        "document_text_max_chars": 64,
        "document_prefix": document_prefix,
        "embedding_shape": [document_count, 2],
        "document_id_count": document_count,
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
