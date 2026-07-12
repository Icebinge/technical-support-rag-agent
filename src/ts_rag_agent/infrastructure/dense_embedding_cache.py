from __future__ import annotations

from pathlib import Path

import numpy as np

from ts_rag_agent.domain.dataset import PrimeQADocument
from ts_rag_agent.infrastructure.dense_retriever import TextEmbeddingModel, build_document_texts


def load_or_build_document_embeddings(
    encoder: TextEmbeddingModel,
    documents: list[PrimeQADocument],
    model_name: str,
    document_text_max_chars: int,
    document_prefix: str,
    cache_path: Path,
    no_cache: bool,
) -> tuple[np.ndarray, str]:
    """读取或构建文档向量缓存。"""

    document_ids = np.asarray([document.id for document in documents])
    if not no_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_document_ids = cached["document_ids"]
        cached_model_name = str(cached["model_name"])
        cached_max_chars = int(cached["document_text_max_chars"])
        cached_document_prefix = (
            str(cached["document_prefix"]) if "document_prefix" in cached else ""
        )
        if (
            cached_model_name == model_name
            and cached_max_chars == document_text_max_chars
            and cached_document_prefix == document_prefix
            and np.array_equal(cached_document_ids, document_ids)
        ):
            return np.asarray(cached["embeddings"], dtype=np.float32), "loaded"

    texts = build_document_texts(
        documents,
        document_text_max_chars=document_text_max_chars,
        document_prefix=document_prefix,
    )
    embeddings = encoder.encode(texts)

    if not no_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            document_ids=document_ids,
            embeddings=embeddings.astype(np.float32),
            model_name=np.asarray(model_name),
            document_text_max_chars=np.asarray(document_text_max_chars),
            document_prefix=np.asarray(document_prefix),
        )

    return embeddings, "created" if not no_cache else "disabled"


def default_dense_cache_path(
    data_dir: Path,
    model_name: str,
    document_text_max_chars: int,
    document_prefix: str = "",
) -> Path:
    """生成 dense 文档向量缓存路径。"""

    safe_model_name = model_name.replace("/", "__").replace("\\", "__")
    safe_prefix = "noprefix" if not document_prefix else _safe_cache_part(document_prefix)
    return (
        data_dir
        / "indexes"
        / "dense"
        / f"{safe_model_name}_{document_text_max_chars}_{safe_prefix}.npz"
    )


def _safe_cache_part(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")[:40]
