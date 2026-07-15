import json

from ts_rag_agent.application.primeqa_hybrid_selective_dense_sparse_protocol import (
    freeze_primeqa_hybrid_selective_dense_sparse_protocol,
    write_primeqa_hybrid_selective_dense_sparse_protocol_visualizations,
)


def test_selective_dense_sparse_protocol_freezes_confirmed_candidate(tmp_path):
    paths = _write_fixture_reports(tmp_path)

    report = freeze_primeqa_hybrid_selective_dense_sparse_protocol(
        stage84_report_path=paths["stage84"],
        stage96_report_path=paths["stage96"],
        stage80_report_path=paths["stage80"],
        stage81_report_path=paths["stage81"],
        user_confirmed_candidate=True,
        confirmed_candidate_id="selective_dense_sparse_low_overlap_gate_design",
        confirmation_note="confirmed in test",
    )
    visualizations = write_primeqa_hybrid_selective_dense_sparse_protocol_visualizations(
        report=report,
        output_dir=tmp_path / "visuals",
    )

    serialized = json.dumps(report, ensure_ascii=False)
    assert report["stage"] == "Stage 97"
    assert report["decision"]["status"] == (
        "primeqa_hybrid_selective_dense_sparse_protocol_frozen"
    )
    assert report["decision"]["can_run_final_test_metrics_now"] is False
    assert report["decision"]["requires_user_confirmation_before_train_dev_run"] is True
    assert report["decision"]["can_use_test_for_tuning"] is False
    assert report["frozen_protocol"]["protocol_id"] == (
        "selective_dense_sparse_low_overlap_gate_train_dev_v1"
    )
    assert len(report["frozen_protocol"]["candidate_policy_grid"]) == 4
    assert all(check["passed"] for check in report["guard_checks"])
    assert (
        report["frozen_protocol"]["dense_cache_contract"]["download_required"] is False
    )
    assert (
        report["frozen_protocol"]["dense_cache_contract"][
            "document_reencoding_allowed"
        ]
        is False
    )
    assert "Restart the database service" not in serialized
    assert "Install the firmware update" not in serialized
    assert "How do I use this private dense sparse question" not in serialized
    assert {artifact.name for artifact in visualizations} == {
        "stage97_selective_dense_sparse_cache_readiness.svg",
        "stage97_selective_dense_sparse_gate_thresholds.svg",
        "stage97_selective_dense_sparse_rrf_weights.svg",
        "stage97_selective_dense_sparse_feature_group_counts.svg",
        "stage97_selective_dense_sparse_protocol_decision_flags.svg",
        "stage97_selective_dense_sparse_guard_check_status.svg",
    }


def test_selective_dense_sparse_protocol_blocks_unconfirmed_candidate(tmp_path):
    paths = _write_fixture_reports(tmp_path)

    report = freeze_primeqa_hybrid_selective_dense_sparse_protocol(
        stage84_report_path=paths["stage84"],
        stage96_report_path=paths["stage96"],
        stage80_report_path=paths["stage80"],
        stage81_report_path=paths["stage81"],
        user_confirmed_candidate=False,
        confirmed_candidate_id="selective_dense_sparse_low_overlap_gate_design",
        confirmation_note="not confirmed",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["user_confirmed_selective_dense_sparse_protocol"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_selective_dense_sparse_protocol_blocked"
    )


def test_selective_dense_sparse_protocol_blocks_candidate_mismatch(tmp_path):
    paths = _write_fixture_reports(tmp_path)

    report = freeze_primeqa_hybrid_selective_dense_sparse_protocol(
        stage84_report_path=paths["stage84"],
        stage96_report_path=paths["stage96"],
        stage80_report_path=paths["stage80"],
        stage81_report_path=paths["stage81"],
        user_confirmed_candidate=True,
        confirmed_candidate_id="source_doc_ids_oracle_union_blocked",
        confirmation_note="wrong candidate",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert (
        checks["confirmed_candidate_matches_stage96_next_candidate"]["passed"] is False
    )
    assert report["decision"]["status"] == (
        "primeqa_hybrid_selective_dense_sparse_protocol_blocked"
    )


def test_selective_dense_sparse_protocol_blocks_cache_mismatch(tmp_path):
    paths = _write_fixture_reports(tmp_path, stage81_cache_suffix="_different")

    report = freeze_primeqa_hybrid_selective_dense_sparse_protocol(
        stage84_report_path=paths["stage84"],
        stage96_report_path=paths["stage96"],
        stage80_report_path=paths["stage80"],
        stage81_report_path=paths["stage81"],
        user_confirmed_candidate=True,
        confirmed_candidate_id="selective_dense_sparse_low_overlap_gate_design",
        confirmation_note="confirmed in test",
    )

    checks = {check["name"]: check for check in report["guard_checks"]}
    assert checks["stage80_stage81_dense_cache_identities_match"]["passed"] is False
    assert report["decision"]["status"] == (
        "primeqa_hybrid_selective_dense_sparse_protocol_blocked"
    )


def _write_fixture_reports(tmp_path, *, stage81_cache_suffix: str = "") -> dict:
    reports = {
        "stage84": _stage84_report(),
        "stage96": _stage96_report(),
        "stage80": _stage80_report(),
        "stage81": _stage81_report(cache_suffix=stage81_cache_suffix),
    }
    paths = {}
    for name, report in reports.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        paths[name] = path
    return paths


def _stage84_report() -> dict:
    return {
        "stage": "Stage 84",
        "candidate_designs": [
            {
                "candidate_id": "selective_dense_sparse_low_overlap_gate_design",
                "name": "Selective dense+sparse low-overlap gate design",
                "category": "hybrid_retrieval_gate",
                "status": "recommended_for_train_dev_protocol_design",
                "risk_level": "high",
                "implementation_readiness": 0.58,
                "prior_signal_key": "dense_sparse_rrf",
                "prior_signal_score": 0.68,
                "priority_score": 159,
                "target_miss_count": 111,
                "target_miss_count_by_split": {"dev": 17, "train": 94},
                "target_rank_buckets": {
                    "not_found_top50": 110,
                    "rank_21_to_50": 1,
                },
                "rationale": (
                    "Stage81 dense+sparse RRF reduced not-found@50 but regressed "
                    "dev hit@10, so a train-only runtime gate is required."
                ),
                "stage85_protocol_outline": [
                    "Reuse only already-confirmed local dense caches.",
                    "Define runtime-observable gate features on train before dev validation.",
                    "Compare gated dense+sparse retrieval against BM25 on train/dev only.",
                    "Block the route if top10 regressions dominate on dev.",
                ],
                "target_metric_contract": [
                    "primary: train-selected gated policy must improve dev hit@10",
                    "secondary: dev not-found@50 should decrease without hit@1 collapse",
                    "guard: no downloads and no dev-selected gate thresholds",
                ],
                "runtime_evidence_policy": [
                    (
                        "May use query tokens, candidate scores, overlap counts, "
                        "and local dense scores."
                    ),
                    "Must not use source DOC_IDS, answer document IDs, or test labels.",
                ],
                "raw_question_text": "How do I use this private dense sparse question",
            }
        ],
        "decision": {
            "status": "primeqa_hybrid_second_wave_retrieval_candidate_design_completed",
            "recommended_next_candidate_id": "lexical_cluster_diversity_rerank_design",
            "requires_user_confirmation_before_train_dev_run": True,
            "can_continue_train_dev_development": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        },
    }


def _stage96_report() -> dict:
    return {
        "stage": "Stage 96",
        "candidate_queue": {
            "next_candidate_summary": {
                "candidate_id": "selective_dense_sparse_low_overlap_gate_design",
                "status": "recommended_for_train_dev_protocol_design",
            },
        },
        "decision": {
            "status": "primeqa_hybrid_score_margin_bm25_route_stopped",
            "stopped_candidate_id": "score_margin_bm25_normalization_gate_design",
            "current_route_defaultization": "blocked",
            "next_candidate_id": "selective_dense_sparse_low_overlap_gate_design",
            "can_continue_train_dev_development": True,
            "requires_user_confirmation_before_next_protocol": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        },
        "raw_question_text": "Restart the database service",
    }


def _stage80_report() -> dict:
    return {
        "stage": "Stage 80",
        "dense_cache_candidates": [
            _dense_cache_candidate(
                model_name="intfloat/e5-small-v2",
                cache_path="data\\indexes\\dense\\intfloat__e5-small-v2_512_passage.npz",
                document_text_max_chars=512,
                document_prefix="passage: ",
            ),
            _dense_cache_candidate(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_path=(
                    "data\\indexes\\dense\\sentence-transformers__all-MiniLM-L6-v2_1600.npz"
                ),
                document_text_max_chars=1600,
                document_prefix="",
            ),
        ],
        "decision": {
            "status": "primeqa_hybrid_dense_sparse_rrf_feasibility_completed",
            "compatible_local_dense_cache_count": 2,
            "can_continue_train_dev_development": True,
            "can_run_dense_sparse_rrf_without_download": True,
            "requires_user_confirmation_before_train_dev_run": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        },
    }


def _stage81_report(*, cache_suffix: str) -> dict:
    return {
        "stage": "Stage 81",
        "dense_cache_configs": [
            _dense_cache_config(
                config_id="dense_sparse_rrf__intfloat_e5_small_v2__512_passage",
                model_name="intfloat/e5-small-v2",
                cache_path=(
                    "data\\indexes\\dense\\intfloat__e5-small-v2_512_passage"
                    f"{cache_suffix}.npz"
                ),
                document_text_max_chars=512,
                document_prefix="passage: ",
                query_prefix="query: ",
            ),
            _dense_cache_config(
                config_id=(
                    "dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix"
                ),
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_path=(
                    "data\\indexes\\dense\\sentence-transformers__all-MiniLM-L6-v2_1600"
                    f"{cache_suffix}.npz"
                ),
                document_text_max_chars=1600,
                document_prefix="",
                query_prefix="",
            ),
        ],
        "decision": {
            "status": "primeqa_hybrid_dense_sparse_rrf_comparison_completed",
            "selected_config_id": (
                "dense_sparse_rrf__sentence_transformers_all_MiniLM_L6_v2__1600_noprefix"
            ),
            "selected_dev_hit10_delta": -0.0132,
            "selected_dev_not_found_at_search_depth_delta": -6,
            "can_continue_train_dev_development": True,
            "can_open_final_test_gate_now": False,
            "can_run_final_test_metrics_now": False,
            "can_use_test_for_tuning": False,
            "default_runtime_policy": "unchanged",
        },
        "raw_answer_text": "Install the firmware update",
    }


def _dense_cache_candidate(
    *,
    model_name: str,
    cache_path: str,
    document_text_max_chars: int,
    document_prefix: str,
) -> dict:
    return {
        "model_name": model_name,
        "cache_path": cache_path,
        "cache_sha256": "abc123",
        "document_text_max_chars": document_text_max_chars,
        "document_prefix": document_prefix,
        "embedding_shape": [28482, 384],
        "document_id_count": 28482,
        "document_ids_match_current_corpus": True,
        "can_run_without_reencoding_documents": True,
        "can_run_without_model_download": True,
    }


def _dense_cache_config(
    *,
    config_id: str,
    model_name: str,
    cache_path: str,
    document_text_max_chars: int,
    document_prefix: str,
    query_prefix: str,
) -> dict:
    return {
        "config_id": config_id,
        "model_name": model_name,
        "cache_path": cache_path,
        "cache_sha256": "abc123",
        "document_text_max_chars": document_text_max_chars,
        "document_prefix": document_prefix,
        "query_prefix": query_prefix,
        "query_prefix_source": "stage80_legacy_dense_metric",
        "embedding_shape": [28482, 384],
        "document_id_count": 28482,
        "can_run_without_model_download_in_stage80": True,
        "snapshot_path": "C:\\hf\\snapshot",
        "snapshot_status": "refs_main",
    }
