from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectSettings(BaseSettings):
    """Runtime paths and upstream dataset identifiers."""

    model_config = SettingsConfigDict(env_prefix="TS_RAG_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    eval_repo: str = "nvidia/TechQA-RAG-Eval"
    train_repo: str = "PrimeQA/TechQA"
    enable_optional_sidecar_agent: bool = False

    @field_validator("enable_optional_sidecar_agent", mode="before")
    @classmethod
    def validate_explicit_runtime_flag(cls, value: object) -> object:
        """Accept only explicit true/false values for the optional runtime."""

        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ValueError("TS_RAG_ENABLE_OPTIONAL_SIDECAR_AGENT must be explicit true or false")

    @property
    def nvidia_raw_dir(self) -> Path:
        return self.data_dir / "raw" / "nvidia_techqa_rag_eval"

    @property
    def primeqa_raw_dir(self) -> Path:
        return self.data_dir / "raw" / "primeqa_techqa"
