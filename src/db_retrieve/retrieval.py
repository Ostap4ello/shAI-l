#!/usr/bin/env python3

from pathlib import Path
from typing import List

from openai import OpenAI

from .utils.faiss_utils import (
    build_faiss_index,
    load_documents,
    load_index,
    save_index,
    save_index_config,
    load_index_config,
    resolve_index_paths,
    get_default_index_path_within_db,
)

from .utils.embedding import embed_strings

import logging

logger = logging.getLogger(__name__)


def build(
    db_path: str,
    client: OpenAI,
    model: str,
    batch_size: int = 32,
    show_progress: bool = False,
    index_path_within_db: str = get_default_index_path_within_db(),
) -> None:
    doc_dir = Path(db_path)
    index_path, meta_path, config_path = resolve_index_paths(
        db_path, index_path_within_db
    )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    texts, metadata = load_documents(doc_dir)
    vectors = embed_strings(client, model, texts, batch_size)
    index = build_faiss_index(vectors)

    save_index(index, metadata, index_path, meta_path)
    save_index_config(config_path, model)


def check(
    db_path: str, index_path_within_db: str = get_default_index_path_within_db()
) -> bool:
    index_path, meta_path, config_path = resolve_index_paths(
        db_path, index_path_within_db
    )
    return index_path.exists() and meta_path.exists() and config_path.exists()


def search(
    db_path: str,
    client: OpenAI,
    query: str,
    top_k: int = 5,
    index_path_within_db: str = get_default_index_path_within_db(),
) -> List[dict]:
    if not check(db_path, index_path_within_db):
        logger.error("Index not found. Run build() first.")
        raise RuntimeError("Index not found. Run build() first.")

    index_path, meta_path, config_path = resolve_index_paths(
        db_path, index_path_within_db
    )

    config = load_index_config(config_path)
    model = config.get("model")
    if not model:
        logger.error(f"Model not found in config: {config_path}")
        raise RuntimeError(f"Model not found in config: {config_path}")

    index, metadata = load_index(index_path, meta_path)
    query_vec = embed_strings(client, model, [query], batch_size=1)
    distances, indices = index.search(query_vec, top_k)

    results: List[dict] = []
    for idx in indices[0].tolist():
        if idx < 0 or idx >= len(metadata):
            continue
        results.append(metadata[idx])

    return results
