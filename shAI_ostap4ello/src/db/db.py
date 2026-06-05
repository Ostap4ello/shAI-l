#!/usr/bin/env python3

from openai import OpenAI
from pathlib import Path
from typing import List
from datetime import datetime
import numpy as np

from .utils.documents import (
    list_db_documents,
    load_documents,
    load_documents_in_sections,
)
from .utils.config import (
    get_default_index_path_within_db,
    load_index_config,
    resolve_index_paths,
    save_index_config,
    get_empty_config as get_empty_db_config,
)
from .utils.faiss_utils import (
    build_index,
    load_index,
    save_index,
)
from ..llm import embed_strings

import logging

logger = logging.getLogger(__name__)


def build(
    db_path: str,
    client: OpenAI,
    model: str,
    batch_size: int = 32,
    index_path_within_db: str = get_default_index_path_within_db(),
    section_rows: int = 0,
) -> None:
    # Ensure index_path_within_db is a hidden folder
    if not str(index_path_within_db).startswith("."):
        raise ValueError(
            f"index_path_within_db must start with a dot ('.'): got '{index_path_within_db}'"
        )

    db_dir = Path(db_path)
    index_path, meta_path, config_path = resolve_index_paths(
        db_path, index_path_within_db
    )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading documents from: {db_dir.resolve()}")
    if section_rows == 0:
        texts, metadata = load_documents(list_db_documents(db_dir))
    elif section_rows > 0:
        texts, metadata = load_documents_in_sections(
            list_db_documents(db_dir), section_rows
        )
    else:
        raise RuntimeError("Section size cannot be less then 0")

    logger.info(f"Creating database index")
    vectors = embed_strings(client, model, texts, batch_size)
    index = build_index(vectors)
    save_index(index, metadata, index_path, meta_path)

    index_config = get_empty_db_config()
    index_config["model"] = model
    save_index_config(config_path, index_config)

    logger.info(f"Successfully built index for: {db_dir.resolve()}")


def update(
    db_path: str,
    client: OpenAI,
    batch_size: int = 32,
    index_path_within_db: str = get_default_index_path_within_db(),
    section_rows: int = 0,
) -> None:
    if not check(db_path, index_path_within_db):
        raise RuntimeError("Index not found. Run build() first.")
    db_dir = Path(db_path)

    index_path, meta_path, config_path = resolve_index_paths(
        db_path, index_path_within_db
    )

    logger.info(f"Loading database: {db_dir.resolve()}")
    index_config = load_index_config(config_path)
    model = index_config["model"]
    old_index, old_metadata_list = load_index(index_path, meta_path)
    db_mtime = get_index_info(db_path, index_path_within_db)["timestamp"]

    file_to_index = {}
    for i, meta in enumerate(old_metadata_list):
        idx = meta["path"] + " " + str(meta.get("from", ""))
        file_to_index[idx] = i

    logger.info(f"Listing documents in {db_dir.resolve()}")
    doc_paths = list_db_documents(db_dir)

    embed_queue = []
    metadata = []
    vectors = []

    for doc_path in doc_paths:
        doc_mtime = datetime.fromtimestamp(Path(doc_path).stat().st_mtime).isoformat()

        if doc_mtime > db_mtime:
            embed_queue.append(doc_path)
            continue

        found = False
        for idx in file_to_index.keys():
            if idx.startswith(str(doc_path) + " "):
                vectors.append(old_index.reconstruct(file_to_index[idx]))
                metadata.append(old_metadata_list[file_to_index[idx]])
                found = True
                break

        if not found:
            embed_queue.append(doc_path)


    if len(embed_queue) == 0:
        logger.info("No new or updated documents found. Index is up to date.")
        save_index_config(config_path, index_config)  # updates timestamp in config
        return

    logger.info(f"Loading new/updated documents ({len(embed_queue)})")
    if section_rows == 0:
        texts, updated_metadata = load_documents(embed_queue)
    elif section_rows > 0:
        texts, updated_metadata = load_documents_in_sections(embed_queue, section_rows)
    else:
        raise RuntimeError("Section size cannot be less then 0")

    logger.info(f"Updating database index")
    updated_vectors = embed_strings(client, model, texts, batch_size)
    vectors = np.vstack(vectors + updated_vectors.tolist())
    metadata = metadata + updated_metadata
    index = build_index(vectors)
    save_index(index, metadata, index_path, meta_path)

    save_index_config(config_path, index_config)  # updates timestamp in config
    logger.info(f"Successfully updated index for: {db_dir.resolve()}")


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
        raise RuntimeError("Index not found. Run build() first.")

    index_path, meta_path, config_path = resolve_index_paths(
        db_path, index_path_within_db
    )

    logger.info(f"Loading database: {Path(db_path).resolve()}")
    model = load_index_config(config_path)["model"]

    index, metadata = load_index(index_path, meta_path)
    query_vec = embed_strings(client, model, [query], batch_size=1)
    logger.info(f"Searching database")
    distances, indices = index.search(query_vec, top_k)

    results: List[dict] = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx < 0 or idx >= len(metadata):
            continue
        results.append(
            {
                "metadata": metadata[idx],
                "distance": float(distances[0][i]),
            }
        )

    return results


def search_in_files_dynamic(
    file_paths: List[str],
    client: OpenAI,
    model: str,
    query: str,
    batch_size: int = 32,
    top_k: int = 5,
    section_rows: int = 20,
) -> List[dict]:
    # TODO: dynamic function - create lazy indexing and caching instead

    file_paths = list(set(file_paths))

    texts, metadata = load_documents_in_sections(
        [Path(p) for p in file_paths], section_rows
    )

    texts.append(query)

    vectors = embed_strings(client, model, texts, batch_size)
    _ = texts.pop()

    query_vec = vectors[-1].reshape(1, -1)
    vectors = vectors[:-1]

    index = build_index(vectors)

    distances, indices = index.search(query_vec, top_k)

    results: List[dict] = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx < 0 or idx >= len(metadata):
            continue
        results.append(
            {
                "metadata": metadata[idx],
                "distance": float(distances[0][i]),
            }
        )

    return results


def get_index_info(
    db_path: str,
    index_path_within_db: str = get_default_index_path_within_db(),
) -> dict:
    if not check(db_path, index_path_within_db):
        raise RuntimeError("Index not found. Run build() first.")

    _, _, config_path = resolve_index_paths(db_path, index_path_within_db)
    return load_index_config(config_path)
