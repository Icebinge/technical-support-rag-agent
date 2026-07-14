from ts_rag_agent.application.external_eval_dataset_discovery import (
    discover_external_eval_datasets,
    write_external_eval_discovery_visualizations,
)


def test_external_eval_discovery_recommends_msqa_but_blocks_final_metrics():
    report = discover_external_eval_datasets()

    assert report["decision"]["recommended_candidate"] == "microsoft_msqa"
    assert report["decision"]["can_run_final_metrics_now"] is False
    assert report["decision"]["default_runtime_policy"] == "unchanged"
    assert "Do not run Stage 51 candidate metrics on MSQA in Stage 55." in report[
        "blocked_actions"
    ]


def test_msqa_candidate_requires_schema_citation_and_leakage_audits():
    report = discover_external_eval_datasets()
    candidates = {candidate["label"]: candidate for candidate in report["candidates"]}
    msqa = candidates["microsoft_msqa"]

    assert msqa["status"] == "recommended_for_stage56_schema_probe"
    assert msqa["scores"]["domain_fit_score"] == 3
    assert msqa["scores"]["citation_fit_score"] == 2
    assert msqa["scores"]["answerability_fit_score"] == 0
    assert any(
        "source-link" in required_audit
        for required_audit in msqa["required_audits_before_metrics"]
    )
    assert any(
        "leakage audit" in required_audit
        for required_audit in msqa["required_audits_before_metrics"]
    )


def test_external_eval_candidates_preserve_blocked_and_control_statuses():
    report = discover_external_eval_datasets()
    candidates = {candidate["label"]: candidate for candidate in report["candidates"]}

    assert (
        candidates["msdialog"]["status"]
        == "blocked_until_access_and_license_confirmation"
    )
    assert candidates["natural_questions"]["status"] == "control_benchmark_only"
    assert candidates["stackexchange_dumps"]["status"] == "manual_derivation_candidate_only"


def test_external_eval_discovery_visualizations_are_written(tmp_path):
    report = discover_external_eval_datasets()

    artifacts = write_external_eval_discovery_visualizations(report, tmp_path)

    assert {artifact.name for artifact in artifacts} == {
        "stage55_candidate_fit_score.svg",
        "stage55_candidate_domain_fit.svg",
        "stage55_candidate_citation_fit.svg",
        "stage55_candidate_effort_score.svg",
    }
    for artifact in artifacts:
        assert (tmp_path / artifact.name).read_text(encoding="utf-8").startswith("<svg")
