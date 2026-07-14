from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ts_rag_agent.application.svg_charts import BarDatum, render_horizontal_bar_chart_svg


@dataclass(frozen=True)
class DatasetSource:
    """A source-backed public page used during Stage 55 discovery."""

    label: str
    url: str
    observed_facts: tuple[str, ...]


@dataclass(frozen=True)
class ExternalEvalCandidate:
    """One external evaluation dataset candidate."""

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
        """Weighted fit score where domain match matters most for this project."""

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
class ExternalEvalVisualization:
    """One generated Stage 55 discovery visualization."""

    name: str
    path: str


def discover_external_eval_datasets() -> dict[str, Any]:
    """Create the Stage 55 external evaluation dataset discovery snapshot."""

    candidates = _rank_candidates(_candidate_snapshots())
    recommended = candidates[0]
    return {
        "stage": "Stage 55",
        "created_at": "2026-07-14",
        "review_scope": (
            "External independent evaluation dataset discovery after the user "
            "confirmed the Stage 54 recommended route. This report is a "
            "source-backed discovery and schema-fit audit snapshot; it does not "
            "download datasets, does not run final metrics, and does not change "
            "the default runtime policy."
        ),
        "source_method": {
            "method": "manual_web_discovery_from_primary_or_dataset_owner_sources",
            "source_status": "source-backed snapshot, not live scraper output",
            "notes": [
                (
                    "Dataset availability and licenses were checked from public "
                    "dataset owner pages, dataset cards, or official hosting pages."
                ),
                (
                    "No row-level leakage audit was run in this stage because the "
                    "candidate datasets were not downloaded into local data/raw."
                ),
            ],
        },
        "selection_criteria": {
            "domain_fit": "technical-support or enterprise IT QA is preferred.",
            "schema_fit": (
                "Question, gold answer, identifiers, and held-out split information "
                "should be machine-readable."
            ),
            "citation_fit": (
                "The dataset should support evidence-source or source-link checks "
                "instead of answer-only scoring."
            ),
            "answerability_fit": (
                "Unanswerable or no-answer labels are useful, but absence is not "
                "fatal if the dataset is otherwise the best external tech-support fit."
            ),
            "license_fit": "License must be public-safe enough for local evaluation work.",
            "independence_fit": (
                "Source must be outside the PrimeQA/TechQA and NVIDIA development loop, "
                "then still pass row-level leakage audit before metrics."
            ),
        },
        "decision": {
            "recommended_candidate": recommended.label,
            "recommended_candidate_name": recommended.name,
            "recommended_next_stage": (
                "Stage 56: MSQA local schema probe, source-link coverage audit, "
                "and PrimeQA leakage audit protocol"
            ),
            "can_run_final_metrics_now": False,
            "default_runtime_policy": "unchanged",
            "reason": (
                "MSQA has the strongest external technical-support match and public "
                "licensing profile, but it still needs a local schema probe, source-link "
                "coverage audit, and leakage audit before any held-out metrics can be "
                "truthfully reported."
            ),
        },
        "candidate_count": len(candidates),
        "candidates": [candidate.to_report_dict() for candidate in candidates],
        "source_links": [source.__dict__ for source in _source_snapshots()],
        "blocked_actions": [
            "Do not run Stage 51 candidate metrics on MSQA in Stage 55.",
            "Do not compare MSQA against top-k until a schema adapter and leakage report exist.",
            (
                "Do not treat answer-only rows as citation-ready without source-link "
                "coverage evidence."
            ),
            "Do not change the default runtime policy.",
        ],
    }


def write_external_eval_discovery_visualizations(
    report: dict[str, Any],
    output_dir: Path,
) -> list[ExternalEvalVisualization]:
    """Write compact SVG charts for Stage 55 external dataset discovery."""

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = report["candidates"]
    charts = {
        "stage55_candidate_fit_score.svg": render_horizontal_bar_chart_svg(
            title="Stage 55 external candidate fit score",
            bars=[
                BarDatum(
                    label=candidate["label"],
                    value=float(candidate["scores"]["fit_score"]),
                    value_label=str(candidate["scores"]["fit_score"]),
                )
                for candidate in candidates
            ],
            x_label="weighted fit score, higher is better",
            margin_left=330,
        ),
        "stage55_candidate_domain_fit.svg": render_horizontal_bar_chart_svg(
            title="Stage 55 external candidate domain fit",
            bars=[
                BarDatum(
                    label=candidate["label"],
                    value=float(candidate["scores"]["domain_fit_score"]),
                    value_label=str(candidate["scores"]["domain_fit_score"]),
                )
                for candidate in candidates
            ],
            x_label="domain fit score, higher is better",
            margin_left=330,
        ),
        "stage55_candidate_citation_fit.svg": render_horizontal_bar_chart_svg(
            title="Stage 55 external candidate citation fit",
            bars=[
                BarDatum(
                    label=candidate["label"],
                    value=float(candidate["scores"]["citation_fit_score"]),
                    value_label=str(candidate["scores"]["citation_fit_score"]),
                )
                for candidate in candidates
            ],
            x_label="citation/evidence fit score, higher is better",
            margin_left=330,
        ),
        "stage55_candidate_effort_score.svg": render_horizontal_bar_chart_svg(
            title="Stage 55 external candidate adapter effort",
            bars=[
                BarDatum(
                    label=candidate["label"],
                    value=float(candidate["scores"]["adapter_effort_score"]),
                    value_label=str(candidate["scores"]["adapter_effort_score"]),
                )
                for candidate in candidates
            ],
            x_label="adapter effort score, higher is more work",
            margin_left=330,
        ),
    }
    artifacts = []
    for filename, svg in charts.items():
        path = output_dir / filename
        path.write_text(svg, encoding="utf-8")
        artifacts.append(ExternalEvalVisualization(name=filename, path=str(path)))
    return artifacts


def _rank_candidates(
    candidates: Sequence[ExternalEvalCandidate],
) -> list[ExternalEvalCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.fit_score,
            -candidate.adapter_effort_score,
            candidate.domain_fit_score,
            candidate.citation_fit_score,
        ),
        reverse=True,
    )


def _candidate_snapshots() -> tuple[ExternalEvalCandidate, ...]:
    return (
        ExternalEvalCandidate(
            label="microsoft_msqa",
            name="Microsoft Q&A (MSQA)",
            status="recommended_for_stage56_schema_probe",
            availability="public GitHub archive with data files",
            license_name="Dataset: CDLA-Permissive-2.0; project code: MIT",
            record_count=32252,
            source_summary=(
                "External Microsoft product and IT technical QA benchmark collected "
                "from Microsoft Q&A, with human-generated accepted answers."
            ),
            source_urls=(
                "https://github.com/microsoft/Microsoft-Q-A-MSQA-",
                "https://aclanthology.org/2023.emnlp-industry.29/",
                "https://cdla.dev/permissive-2-0/",
            ),
            domain_fit_score=3,
            schema_fit_score=3,
            citation_fit_score=2,
            answerability_fit_score=0,
            license_fit_score=3,
            independence_fit_score=3,
            adapter_effort_score=2,
            strengths=(
                "Technical-support and enterprise IT domain match is strong.",
                "Contains question, accepted answer, tags, URL, and a provided test_id file.",
                "Dataset license is explicitly listed as CDLA-Permissive-2.0.",
                "Source README shows link standardization and Azure documentation tooling.",
            ),
            risks=(
                "Accepted-answer-only filtering means no native unanswerable rows.",
                "Gold citation coverage is not guaranteed for every row.",
                "The 99.9 MB CSV should be locally sampled before adding an adapter.",
                "The repository is archived, so future fixes from maintainers are unlikely.",
            ),
            required_audits_before_metrics=(
                "Download only after recording source URL, size, and checksum.",
                "Probe CSV headers and parse a small local sample.",
                "Measure source-link and learn.microsoft.com documentation-link coverage.",
                "Run exact and near-duplicate leakage audit against PrimeQA train/dev questions.",
                "Freeze a MSQA evaluation split before comparing top-k and Stage 51.",
            ),
            next_action=(
                "Stage 56 should perform a local MSQA schema probe and source-link "
                "coverage audit, without running answer-quality metrics yet."
            ),
        ),
        ExternalEvalCandidate(
            label="multidoc2dial",
            name="MultiDoc2Dial",
            status="secondary_document_grounded_reference",
            availability="public Hugging Face dataset card and project page",
            license_name="Apache-2.0",
            record_count=None,
            source_summary=(
                "Multi-document goal-oriented dialogues grounded in documents across "
                "several non-technical-service domains."
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
                "Grounded multi-document schema is useful as an adapter reference.",
                "Apache-2.0 license is public-safe for local evaluation work.",
                "Evidence-document structure is closer to RAG citation checks than answer-only QA.",
            ),
            risks=(
                "Domain is not technical support, so defaultization evidence would be indirect.",
                "Dialogue-oriented examples need transformation into this project's QA format.",
                "No row-level leakage audit has been run locally.",
            ),
            required_audits_before_metrics=(
                "Confirm downloadable files and exact split availability.",
                "Map document spans into the local citation contract.",
                "Run leakage audit against PrimeQA train/dev question text.",
            ),
            next_action="Use only as a fallback or schema-design reference after MSQA probe.",
        ),
        ExternalEvalCandidate(
            label="doc2dial",
            name="Doc2Dial",
            status="secondary_document_grounded_reference",
            availability="public Hugging Face dataset card and project page",
            license_name="CC-BY-3.0",
            record_count=4500,
            source_summary=(
                "Goal-oriented dialogues grounded in associated documents, with "
                "document spans and reading-comprehension style fields."
            ),
            source_urls=(
                "https://huggingface.co/datasets/IBM/doc2dial",
                "https://doc2dial.github.io/data.html",
            ),
            domain_fit_score=1,
            schema_fit_score=3,
            citation_fit_score=3,
            answerability_fit_score=1,
            license_fit_score=2,
            independence_fit_score=3,
            adapter_effort_score=3,
            strengths=(
                "Has explicit document text, spans, and grounded answer fields.",
                "Useful for testing citation-aware adapters.",
                "Dataset card lists CC-BY-3.0 license.",
            ),
            risks=(
                "Domain mismatch is material for a technical-support RAG agent.",
                "Dialogue turns need conversion into single-turn QA evaluation rows.",
                "CC-BY attribution must be preserved in any derived artifacts.",
            ),
            required_audits_before_metrics=(
                "Confirm split files and document-span coverage locally.",
                "Define attribution handling for derived evaluation artifacts.",
                "Run leakage audit against PrimeQA train/dev question text.",
            ),
            next_action="Keep as a citation-schema reference, not the primary new test set.",
        ),
        ExternalEvalCandidate(
            label="stackexchange_dumps",
            name="Stack Exchange Data Dumps",
            status="manual_derivation_candidate_only",
            availability="public Internet Archive dumps",
            license_name="CC-BY-SA with attribution requirements",
            record_count=None,
            source_summary=(
                "Network-wide community Q&A dumps, including technical sites such as "
                "Ask Ubuntu, Server Fault, Super User, and Stack Overflow."
            ),
            source_urls=(
                "https://archive.org/download/stackexchange",
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
                "Large technical QA source outside the PrimeQA/TechQA lineage.",
                "Official dumps include post-level artifacts that can be parsed reproducibly.",
            ),
            risks=(
                "Requires deriving a new test set, which is closer to dataset construction.",
                "CC-BY-SA and attribution requirements complicate redistribution.",
                "Accepted answers are not the same as source-document citations.",
                "No native no-answer rows for refusal evaluation.",
            ),
            required_audits_before_metrics=(
                "Choose a site and extraction protocol before downloading dumps.",
                "Design attribution-preserving derived artifacts.",
                "Run leakage audit against PrimeQA train/dev and any public benchmark overlap.",
            ),
            next_action="Do not use before MSQA is ruled out or a derivation protocol is approved.",
        ),
        ExternalEvalCandidate(
            label="natural_questions",
            name="Natural Questions",
            status="control_benchmark_only",
            availability="public Google Research archive",
            license_name="Apache-2.0",
            record_count=307372,
            source_summary=(
                "Open-domain Wikipedia QA with raw document HTML, long-answer "
                "annotations, short answers, yes/no labels, and null long-answer rows."
            ),
            source_urls=("https://github.com/google-research-datasets/natural-questions",),
            domain_fit_score=0,
            schema_fit_score=3,
            citation_fit_score=3,
            answerability_fit_score=3,
            license_fit_score=3,
            independence_fit_score=3,
            adapter_effort_score=3,
            strengths=(
                "Strong answerability and document-span schema.",
                "Apache-2.0 license and mature evaluation tooling.",
            ),
            risks=(
                "Domain mismatch is too large for defaultizing a technical-support policy.",
                "A good result would not prove technical-support runtime readiness.",
            ),
            required_audits_before_metrics=(
                "Use only as a control benchmark after project-specific candidates fail.",
                "Run leakage audit against PrimeQA train/dev if ever used.",
            ),
            next_action=(
                "Keep as a general QA control, not a replacement technical-support "
                "test set."
            ),
        ),
        ExternalEvalCandidate(
            label="msdialog",
            name="MSDialog",
            status="blocked_until_access_and_license_confirmation",
            availability="request access from CIIR",
            license_name="not confirmed from the public download page",
            record_count=35000,
            source_summary=(
                "Microsoft Community technical-support dialogues with answer markers "
                "and response-ranking variants."
            ),
            source_urls=("https://ciir.cs.umass.edu/downloads/msdialog/",),
            domain_fit_score=3,
            schema_fit_score=1,
            citation_fit_score=0,
            answerability_fit_score=0,
            license_fit_score=0,
            independence_fit_score=3,
            adapter_effort_score=4,
            strengths=(
                "Technical-support domain match is strong.",
                "Public page documents answer markers and response-ranking splits.",
            ),
            risks=(
                "Access requires contacting CIIR, so immediate reproducibility is blocked.",
                "Public page does not provide a clear redistributable dataset license.",
                "Forum-dialogue answers lack packaged source-document evidence.",
            ),
            required_audits_before_metrics=(
                "Confirm license and access terms with the data owner.",
                "Inspect schema locally only after legitimate access is obtained.",
                "Run leakage audit against PrimeQA train/dev if access is approved.",
            ),
            next_action="Do not use unless access and license are explicitly confirmed.",
        ),
    )


def _source_snapshots() -> tuple[DatasetSource, ...]:
    return (
        DatasetSource(
            label="microsoft_msqa_github",
            url="https://github.com/microsoft/Microsoft-Q-A-MSQA-",
            observed_facts=(
                "Repository is public and archived.",
                "README states MSQA has 32k Microsoft product and IT QA pairs.",
                "README lists 32,252 data rows and 377 tags.",
                "README states datasets are licensed under CDLA-Permissive-2.0.",
                "README shows data/test_id.txt and processed data workflow.",
            ),
        ),
        DatasetSource(
            label="microsoft_msqa_acl_paper",
            url="https://aclanthology.org/2023.emnlp-industry.29/",
            observed_facts=(
                (
                    "Paper describes MSQA as centered around Microsoft products and "
                    "IT technical problems."
                ),
                "Paper says source code and sample data are available at the Microsoft QA URL.",
            ),
        ),
        DatasetSource(
            label="cdla_permissive_2_0",
            url="https://cdla.dev/permissive-2-0/",
            observed_facts=(
                "License text permits use, modification, and sharing when terms are followed.",
                "License text imposes no restriction on computational-analysis results.",
            ),
        ),
        DatasetSource(
            label="doc2dial_huggingface",
            url="https://huggingface.co/datasets/IBM/doc2dial",
            observed_facts=(
                "Dataset card lists CC-BY-3.0 license.",
                "Dataset card describes over 4,500 annotated document-grounded conversations.",
                "Dataset fields include document text, spans, references, and QA-style answers.",
            ),
        ),
        DatasetSource(
            label="multidoc2dial_huggingface",
            url="https://huggingface.co/datasets/IBM/multidoc2dial",
            observed_facts=(
                "Dataset card lists Apache-2.0 license.",
                "Dataset is a multi-document document-grounded dialogue dataset.",
            ),
        ),
        DatasetSource(
            label="msdialog_ciir",
            url="https://ciir.cs.umass.edu/downloads/msdialog/",
            observed_facts=(
                "Public page describes Microsoft Community technical-support dialogs.",
                "Public page lists is_answer fields and response-ranking splits.",
                "Public page says researchers should email CIIR for access.",
            ),
        ),
        DatasetSource(
            label="stackexchange_archive",
            url="https://archive.org/download/stackexchange",
            observed_facts=(
                "Internet Archive listing includes individual Stack Exchange site dumps.",
                "Listing includes license.txt and technical community sites.",
            ),
        ),
        DatasetSource(
            label="stackexchange_license_blog",
            url=(
                "https://stackoverflow.blog/2014/01/23/"
                "stack-exchange-cc-data-now-hosted-by-the-internet-archive/"
            ),
            observed_facts=(
                "Stack Overflow states community-contributed content is CC BY-SA.",
                "The post lists attribution requirements for reused content.",
            ),
        ),
        DatasetSource(
            label="natural_questions_github",
            url="https://github.com/google-research-datasets/natural-questions",
            observed_facts=(
                "Repository lists Apache-2.0 license.",
                "README describes real Google-search questions answered from Wikipedia.",
                (
                    "README documents raw HTML, long-answer candidates, short answers, "
                    "and no-answer markers."
                ),
            ),
        ),
    )
