from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectSettings(BaseSettings):
    """Runtime paths and upstream dataset identifiers."""

    model_config = SettingsConfigDict(env_prefix="TS_RAG_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    eval_repo: str = "nvidia/TechQA-RAG-Eval"
    train_repo: str = "PrimeQA/TechQA"

    @property
    def nvidia_raw_dir(self) -> Path:
        return self.data_dir / "raw" / "nvidia_techqa_rag_eval"

    @property
    def primeqa_raw_dir(self) -> Path:
        return self.data_dir / "raw" / "primeqa_techqa"
