from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg

_STAGE = "Stage 66"
_CREATED_AT = "2026-07-14"


@dataclass(frozen=True)
class RediscoverySource:
    """A public source checked for the Stage66 dataset rediscovery."""

    label: str
    url: str
    observed_facts: tuple[str, ...]

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "url": self.url,
            "observed_facts": list(self.observed_facts),
        }


@dataclass(frozen=True)
class RediscoveryCandidate:
    """One external dataset candidate reviewed after MSQA Stage65."""

    label: str
    name: str
    status: str
    availability: str
    license_name: str
    record_count: int | None
    source_summary: str
    source_urls: tuple[str, ...]
    domain_fit_score: int
    schema_fit_score: int
    citation_fit_score: int
    answerability_fit_score: int
    license_fit_score: int
    independence_fit_score: int
    adapter_effort_score: int
    strengths: tuple[str, ...]
    risks: tuple[str, ...]
    required_audits_before_metrics: tuple[str, ...]
    next_action: str

    @property
    def fit_score(self) -> int:
        """Weighted discovery score, not a model-quality metric."""

        return (
            self.domain_fit_score * 2
            + self.schema_fit_score
            + self.citation_fit_score
            + self.answerability_fit_score
            + self.license_fit_score
            + self.independence_fit_score
        )

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "name": self.name,
            "status": self.status,
            "availability": self.availability,
            "license_name": self.license_name,
            "record_count": self.record_count,
            "source_summary": self.source_summary,
            "source_urls": list(self.source_urls),
            "scores": {
                "fit_score": self.fit_score,
                "domain_fit_score": self.domain_fit_score,
                "schema_fit_score": self.schema_fit_score,
                "citation_fit_score": self.citation_fit_score,
                "answerability_fit_score": self.answerability_fit_score,
                "license_fit_score": self.license_fit_score,
                "independence_fit_score": self.independence_fit_score,
                "adapter_effort_score": self.adapter_effort_score,
            },
            "strengths": list(self.strengths),
            "risks": list(self.risks),
            "required_audits_before_metrics": list(self.required_audits_before_metrics),
            "next_action": self.next_action,
        }


@dataclass(frozen=True)
class DatasetRediscoveryVisualization:
    """One generated Stage66 visualization."""

    name: str
    path: str


def rediscover_external_eval_datasets() -> dict[str, Any]:
    """Create the Stage66 source-backed external dataset rediscovery snapshot."""

    candidates = _rank_candidates(_candidate_snapshots())
    recommended = candidates[0]
    return {
        "stage": _STAGE,
        "created_at": _CREATED_AT,
        "review_scope": (
            "External dataset rediscovery after Stage65 blocked Stage51 "
            "defaultization from the MSQA adapter evidence. This report checks "
            "new or previously secondary public candidates from source pages. "
            "It does not download datasets, does not run metrics, does not tune "
            "Stage51, and does not change the default runtime."
        ),
        "source_method": {
            "method": "manual_web_discovery_from_primary_or_dataset_owner_sources",
            "source_status": "source-backed snapshot, not a live scraper output",
            "notes": [
                (
                    "Dataset facts were checked from dataset owner pages, public "
                    "hosting pages, or official access instructions."
                ),
                (
                    "No row-level leakage audit was run because Stage66 did not "
                    "download candidate datasets into local data/raw."
                ),
            ],
        },
        "selection_criteria": {
            "domain_fit": (
                "Technical-support, enterprise IT, operating-system support, or "
                "software troubleshooting domain is preferred."
            ),
            "schema_fit": (
                "Machine-readable question, answer, context/source identifiers, "
                "and split information are preferred."
            ),
            "citation_fit": (
                "Context, evidence span, source row, or accepted-answer identity "
                "must be inspectable before any RAG-style comparison."
            ),
            "answerability_fit": (
                "Native no-answer labels help refusal evaluation; answerable-only "
                "datasets can only support answer/citation risk checks."
            ),
            "license_fit": (
                "Public-safe local evaluation licensing is required before download."
            ),
            "independence_fit": (
                "The source must be outside the PrimeQA/TechQA, NVIDIA, and MSQA "
                "development loop, then still pass leakage audit before metrics."
            ),
        },
        "decision": {
            "recommended_candidate": recommended.label,
            "recommended_candidate_name": recommended.name,
            "recommended_next_stage": (
                "Stage 67: HQA-Data local schema probe, file checksum capture, "
                "context-span coverage audit, and PrimeQA/MSQA leakage protocol"
            ),
            "can_run_final_metrics_now": False,
            "can_download_without_user_confirmation": False,
            "default_runtime_policy": "unchanged",
            "reason": (
                "HQA-Data is the best source-backed next probe because it is an "
                "Ubuntu technical-support-derived QA dataset with train/test "
                "files, contexts, answer spans, and a clear CC BY 4.0 license. "
                "It is still only a schema-probe candidate because its questions "
                "and answers are generated from dialogue context rather than a "
                "natural human support-answer benchmark."
            ),
        },
        "candidate_count": len(candidates),
        "candidates": [candidate.to_report_dict() for candidate in candidates],
        "source_links": [source.to_report_dict() for source in _source_snapshots()],
        "blocked_actions": [
            "Do not download HQA-Data until the user confirms Stage67 schema probe.",
            "Do not run final or pseudo-held-out metrics in Stage66.",
            "Do not treat generated HQA QA pairs as natural user support answers.",
            (
                "Do not use the Hugging Face ubuntu_dialogue_qa mirror until its "
                "license metadata mismatch is resolved."
            ),
            "Do not use MSDialog without access approval and redistribution boundary.",
            "Do not change the default runtime policy.",
        ],
    }


def write_external_eval_rediscovery_visualizations(
    report: dict[str, Any],
    output_dir: Path,
) -> list[DatasetRediscoveryVisualization]:
    """Write compact SVG charts for the Stage66 rediscovery report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = report["candidates"]
    charts = {
        "stage66_candidate_fit_score.svg": render_horizontal_bar_chart_svg(
            title="Stage66 external candidate fit score",
            bars=_score_bars(candidates, "fit_score"),
            x_label="weighted fit score, higher is better",
            margin_left=330,
        ),
        "stage66_candidate_domain_fit.svg": render_horizontal_bar_chart_svg(
            title="Stage66 external candidate domain fit",
            bars=_score_bars(candidates, "domain_fit_score"),
            x_label="domain fit score, higher is better",
            margin_left=330,
        ),
        "stage66_candidate_citation_fit.svg": render_horizontal_bar_chart_svg(
            title="Stage66 external candidate citation fit",
            bars=_score_bars(candidates, "citation_fit_score"),
            x_label="citation/evidence fit score, higher is better",
            margin_left=330,
        ),
        "stage66_candidate_effort_score.svg": render_horizontal_bar_chart_svg(
            title="Stage66 external candidate adapter effort",
            bars=_score_bars(candidates, "adapter_effort_score"),
            x_label="adapter effort score, higher is more work",
            margin_left=330,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(DatasetRediscoveryVisualization(name=filename, path=str(path)))
    return artifacts


def _rank_candidates(
    candidates: Sequence[RediscoveryCandidate],
) -> list[RediscoveryCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.fit_score,
            -candidate.adapter_effort_score,
            candidate.domain_fit_score,
            candidate.citation_fit_score,
            candidate.license_fit_score,
        ),
        reverse=True,
    )


def _score_bars(
    candidates: Sequence[dict[str, Any]],
    score_name: str,
) -> list[BarDatum]:
    return [
        BarDatum(
            label=str(candidate["label"]),
            value=float(candidate["scores"][score_name]),
            value_label=str(candidate["scores"][score_name]),
        )
        for candidate in candidates
    ]


def _candidate_snapshots() -> tuple[RediscoveryCandidate, ...]:
    return (
        RediscoveryCandidate(
            label="hqa_data_ubuntu_dialogue",
            name="HQA-Data from Ubuntu Dialogue Corpus",
            status="recommended_for_stage67_schema_probe",
            availability="public Mendeley Data page with CSV and JSON formats",
            license_name="CC BY 4.0",
            record_count=36438,
            source_summary=(
                "Question-answer generation dataset derived from Ubuntu Dialogue "
                "Corpus conversations, with contexts and answer start/end spans."
            ),
            source_urls=(
                "https://data.mendeley.com/datasets/p85z3v45xk/1",
                "https://www.sciencedirect.com/science/article/pii/S2352340923003645",
            ),
            domain_fit_score=2,
            schema_fit_score=3,
            citation_fit_score=3,
            answerability_fit_score=0,
            license_fit_score=2,
            independence_fit_score=3,
            adapter_effort_score=2,
            strengths=(
                "Ubuntu support-dialogue origin is closer to technical support than generic QA.",
                "Public page lists train and test files in CSV and JSON formats.",
                "Context and answer start/end positions support span-grounded probing.",
                "Mendeley page lists a clear CC BY 4.0 license.",
                "Small enough for a controlled local schema probe.",
            ),
            risks=(
                "Questions and answers are generated from dialogue context, not "
                "natural user questions with human accepted answers.",
                "No native unanswerable rows for refusal evaluation.",
                "Ubuntu IRC/chat style may be noisy and different from enterprise "
                "support documentation.",
                "CC BY attribution must be preserved in any derived artifacts.",
            ),
            required_audits_before_metrics=(
                "Confirm file URLs, sizes, and checksums before parsing.",
                "Probe CSV and JSON schemas locally without committing raw files.",
                "Measure answer-span coverage and context extraction validity.",
                "Run exact and near-duplicate leakage audits against PrimeQA and MSQA questions.",
                "Freeze a project-owned HQA evaluation split before any baseline.",
            ),
            next_action=(
                "Stage67 should download HQA-Data only after confirmation, then run "
                "a local schema, checksum, span-coverage, and leakage probe."
            ),
        ),
        RediscoveryCandidate(
            label="multidoc2dial",
            name="MultiDoc2Dial",
            status="strong_document_grounding_reference_not_new_primary",
            availability="public Hugging Face dataset card and project page",
            license_name="Apache-2.0",
            record_count=None,
            source_summary=(
                "Multi-document goal-oriented dialogues grounded in documents, useful "
                "as a document-citation adapter reference."
            ),
            source_urls=(
                "https://huggingface.co/datasets/IBM/multidoc2dial",
                "https://doc2dial.github.io/multidoc2dial/",
            ),
            domain_fit_score=1,
            schema_fit_score=3,
            citation_fit_score=3,
            answerability_fit_score=1,
            license_fit_score=3,
            independence_fit_score=3,
            adapter_effort_score=3,
            strengths=(
                "Document-grounded schema is closer to citation-aware RAG.",
                "Apache-2.0 license is public-safe for local evaluation work.",
                "Can help test span/document contract logic.",
            ),
            risks=(
                "Domain is not technical support, so it is weak defaultization evidence.",
                "Dialogue turns need transformation into single-turn QA samples.",
                "Already reviewed in Stage55 as a secondary reference.",
            ),
            required_audits_before_metrics=(
                "Confirm current downloadable files and split availability.",
                "Map document spans into the local citation contract.",
                "Run leakage audit against PrimeQA, MSQA, and HQA candidates if used.",
            ),
            next_action=(
                "Keep as a fallback citation-schema reference if HQA fails schema probe."
            ),
        ),
        RediscoveryCandidate(
            label="msdialog",
            name="MSDialog",
            status="blocked_until_access_and_license_boundary_confirmation",
            availability="request access from CIIR",
            license_name="internal research only, no sharing",
            record_count=35000,
            source_summary=(
                "Microsoft Community technical-support dialogues with selected "
                "answers, user intent labels, and response-ranking variants."
            ),
            source_urls=("https://ciir.cs.umass.edu/downloads/msdialog/",),
            domain_fit_score=3,
            schema_fit_score=3,
            citation_fit_score=1,
            answerability_fit_score=1,
            license_fit_score=0,
            independence_fit_score=3,
            adapter_effort_score=4,
            strengths=(
                "Best domain match after MSQA: Microsoft product support dialogues.",
                "Complete version has 35,000 dialogs and selected-answer fields.",
                "ResponseRank has train, validation, and test partitions.",
            ),
            risks=(
                "Access requires contacting CIIR and receiving a password.",
                "Agreement limits use to internal research and forbids sharing datasets.",
                "Dialog response ranking is not a document-citation RAG task.",
                "Not suitable for committed or redistributable derived artifacts "
                "without a strict plan.",
            ),
            required_audits_before_metrics=(
                "Confirm access eligibility and license boundary with the user.",
                "Decide whether non-redistributable local artifacts are acceptable.",
                "Probe selected-answer and response-rank schemas only after access approval.",
                "Run leakage audit if local access is approved.",
            ),
            next_action=(
                "Do not use until the user explicitly approves access-request and "
                "non-redistribution constraints."
            ),
        ),
        RediscoveryCandidate(
            label="askubuntu_stackexchange_dump",
            name="Ask Ubuntu Stack Exchange Data Dump",
            status="derivation_candidate_blocked_by_size_access_and_attribution_plan",
            availability=(
                "public historical Internet Archive dump plus current account-based "
                "access instructions"
            ),
            license_name="Stack Exchange user content under CC BY-SA family",
            record_count=None,
            source_summary=(
                "Stack Exchange Q&A dump for Ask Ubuntu with post-level questions, "
                "answers, and accepted-answer IDs."
            ),
            source_urls=(
                "https://archive.org/download/stackexchange",
                "https://stackoverflow.com/help/data-dumps",
                "https://stackoverflow.blog/2014/01/23/stack-exchange-cc-data-now-hosted-by-the-internet-archive/",
            ),
            domain_fit_score=2,
            schema_fit_score=2,
            citation_fit_score=1,
            answerability_fit_score=0,
            license_fit_score=1,
            independence_fit_score=3,
            adapter_effort_score=4,
            strengths=(
                "Ask Ubuntu is a large technical troubleshooting Q&A source.",
                "Posts schema can derive question, accepted answer, and source IDs.",
                "Historical Internet Archive listing includes askubuntu.com.7z.",
            ),
            risks=(
                "Deriving a clean evaluation set is a dataset-construction project.",
                "CC BY-SA attribution and share-alike obligations complicate artifacts.",
                "Current Stack Overflow help says latest dumps require account "
                "settings and a non-LLM-training affirmation.",
                "The historical Ask Ubuntu archive is large enough to require a "
                "streaming parser plan.",
            ),
            required_audits_before_metrics=(
                "Choose exact site snapshot and access path before download.",
                "Design attribution-preserving derived artifact rules.",
                "Stream-parse Posts XML and validate accepted-answer coverage.",
                "Run leakage audit against PrimeQA, MSQA, and HQA candidates.",
            ),
            next_action=(
                "Keep parked unless HQA fails and the user approves a derived dataset plan."
            ),
        ),
        RediscoveryCandidate(
            label="hf_ubuntu_dialogue_qa",
            name="sedthh/ubuntu_dialogue_qa",
            status="blocked_by_license_metadata_mismatch",
            availability="public Hugging Face dataset page",
            license_name="Hugging Face metadata says MIT; dataset card text says Apache-2.0",
            record_count=None,
            source_summary=(
                "Hugging Face mirror filtered from Ubuntu dialogue chatlogs to Q&A "
                "pairs only."
            ),
            source_urls=("https://huggingface.co/datasets/sedthh/ubuntu_dialogue_qa",),
            domain_fit_score=2,
            schema_fit_score=2,
            citation_fit_score=1,
            answerability_fit_score=0,
            license_fit_score=0,
            independence_fit_score=3,
            adapter_effort_score=2,
            strengths=(
                "Small public hosted dataset page with Ubuntu/forum/Linux tags.",
                "Hugging Face page lists parquet format and QA task tags.",
            ),
            risks=(
                "License metadata conflicts with the dataset card text.",
                "Dataset viewer is not available on the public page.",
                "The page describes Q&A pairs only; context/span contract needs probing.",
            ),
            required_audits_before_metrics=(
                "Resolve license metadata mismatch before any download.",
                "Inspect files and schema only after the licensing ambiguity is resolved.",
                "Run leakage audit if later approved.",
            ),
            next_action=(
                "Do not use while the license metadata mismatch remains unresolved."
            ),
        ),
    )


def _source_snapshots() -> tuple[RediscoverySource, ...]:
    return (
        RediscoverySource(
            label="hqa_mendeley",
            url="https://data.mendeley.com/datasets/p85z3v45xk/1",
            observed_facts=(
                "Published 2022-12-15 with DOI 10.17632/p85z3v45xk.1.",
                "Derived from Ubuntu Dialogue Corpus conversations by dialogueID.",
                "Lists CSV and JSON formats.",
                "Lists 29,150 train QA pairs and 7,288 test QA pairs.",
                "Lists 9,364 contexts and 36,438 total QA pairs.",
                "Lists CC BY 4.0 license.",
            ),
        ),
        RediscoverySource(
            label="hqa_sciencedirect",
            url="https://www.sciencedirect.com/science/article/pii/S2352340923003645",
            observed_facts=(
                "Data article says questions and answers are contained within the context.",
                "Data article points to Mendeley Data as original data.",
                "Keywords include Ubuntu dialogue corpus and question answering generation.",
            ),
        ),
        RediscoverySource(
            label="hf_ubuntu_dialogue_qa",
            url="https://huggingface.co/datasets/sedthh/ubuntu_dialogue_qa",
            observed_facts=(
                "Hugging Face metadata lists question answering/text generation tasks.",
                "Metadata lists parquet format and English language.",
                "Metadata lists MIT license.",
                "Dataset card text says it is made available under Apache License 2.0.",
                "Dataset viewer is not available on the public page.",
            ),
        ),
        RediscoverySource(
            label="msdialog_ciir",
            url="https://ciir.cs.umass.edu/downloads/msdialog/",
            observed_facts=(
                "MSDialog is based on Microsoft Community support dialogues.",
                "Complete version lists more than 35,000 dialogs.",
                "Intent subset lists about 2,400 dialogs with selected answers.",
                "ResponseRank lists train, validation, and test partitions.",
                "Access requires contacting CIIR and agreeing to internal-research-only terms.",
            ),
        ),
        RediscoverySource(
            label="stackexchange_archive",
            url="https://archive.org/download/stackexchange",
            observed_facts=(
                "Internet Archive listing includes askubuntu.com.7z.",
                "The listed Ask Ubuntu historical archive size is about 1,022.0 MB.",
                "Listing also includes license.txt and readme.txt.",
            ),
        ),
        RediscoverySource(
            label="stackexchange_current_access",
            url="https://stackoverflow.com/help/data-dumps",
            observed_facts=(
                "Current help page says latest dumps are downloaded from profile settings.",
                "Current help page requires affirming that the file will not be used "
                "for LLM training.",
            ),
        ),
    )
