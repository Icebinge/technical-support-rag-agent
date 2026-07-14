from pathlib import Path

from ts_rag_agent.application.external_eval_dataset_rediscovery import (
    rediscover_external_eval_datasets,
    write_external_eval_rediscovery_visualizations,
)


def test_stage66_rediscovery_recommends_hqa_without_running_metrics():
    report = rediscover_external_eval_datasets()

    assert report["stage"] == "Stage 66"
    assert report["decision"]["recommended_candidate"] == "hqa_data_ubuntu_dialogue"
    assert report["decision"]["can_run_final_metrics_now"] is False
    assert report["decision"]["can_download_without_user_confirmation"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert report["candidate_count"] == 5
    assert report["candidates"][0]["label"] == "hqa_data_ubuntu_dialogue"
    assert report["candidates"][0]["scores"]["fit_score"] == 15
    assert report["candidates"][0]["license_name"] == "CC BY 4.0"
    assert any(
        action.startswith("Do not download HQA-Data")
        for action in report["blocked_actions"]
    )


def test_stage66_rediscovery_keeps_blocked_candidates_explicit():
    report = rediscover_external_eval_datasets()
    by_label = {candidate["label"]: candidate for candidate in report["candidates"]}

    assert by_label["hf_ubuntu_dialogue_qa"]["status"] == (
        "blocked_by_license_metadata_mismatch"
    )
    assert "MIT" in by_label["hf_ubuntu_dialogue_qa"]["license_name"]
    assert "Apache-2.0" in by_label["hf_ubuntu_dialogue_qa"]["license_name"]
    assert by_label["msdialog"]["status"] == (
        "blocked_until_access_and_license_boundary_confirmation"
    )
    assert by_label["askubuntu_stackexchange_dump"]["status"] == (
        "derivation_candidate_blocked_by_size_access_and_attribution_plan"
    )


def test_stage66_rediscovery_writes_visualizations(tmp_path):
    report = rediscover_external_eval_datasets()

    artifacts = write_external_eval_rediscovery_visualizations(
        report,
        tmp_path / "visuals",
    )

    assert {artifact.name for artifact in artifacts} == {
        "stage66_candidate_fit_score.svg",
        "stage66_candidate_domain_fit.svg",
        "stage66_candidate_citation_fit.svg",
        "stage66_candidate_effort_score.svg",
    }
    for artifact in artifacts:
        assert Path(artifact.path).read_text(encoding="utf-8").startswith("<svg")
