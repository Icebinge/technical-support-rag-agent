from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectSettings(BaseSettings):
    """Runtime paths and upstream dataset identifiers."""

    model_config = SettingsConfigDict(env_prefix="TS_RAG_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    eval_repo: str = "nvidia/TechQA-RAG-Eval"
    train_repo: str = "PrimeQA/TechQA"
    enable_optional_sidecar_agent: bool = False
    enable_concurrent_sidecar_agent: bool = False

    @field_validator("enable_optional_sidecar_agent", mode="before")
    @classmethod
    def validate_explicit_runtime_flag(cls, value: object) -> object:
        """Accept only explicit true/false values for the optional runtime."""

        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ValueError("TS_RAG_ENABLE_OPTIONAL_SIDECAR_AGENT must be explicit true or false")

    @field_validator("enable_concurrent_sidecar_agent", mode="before")
    @classmethod
    def validate_explicit_concurrent_runtime_flag(cls, value: object) -> object:
        """Accept only explicit true/false values for the concurrent runtime."""

        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ValueError("TS_RAG_ENABLE_CONCURRENT_SIDECAR_AGENT must be explicit true or false")

    @model_validator(mode="after")
    def validate_runtime_modes_are_mutually_exclusive(self) -> "ProjectSettings":
        """Prevent two process-owned runtime graphs from being requested together."""

        if self.enable_optional_sidecar_agent and self.enable_concurrent_sidecar_agent:
            raise ValueError("optional and concurrent sidecar runtime flags are mutually exclusive")
        return self

    @property
    def nvidia_raw_dir(self) -> Path:
        return self.data_dir / "raw" / "nvidia_techqa_rag_eval"

    @property
    def primeqa_raw_dir(self) -> Path:
        return self.data_dir / "raw" / "primeqa_techqa"
