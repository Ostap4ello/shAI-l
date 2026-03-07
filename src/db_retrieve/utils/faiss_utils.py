import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np


import logging

logger = logging.getLogger(__name__)


DEFAULT_INDEX_PATH_WITHIN_DB = ".index"


def load_documents(doc_dir: Path) -> Tuple[List[str], List[dict]]:
    logger.info(f"Loading documents from: {doc_dir}")
    if not doc_dir.exists() or not doc_dir.is_dir():
        logger.error(f"Document directory not found: {doc_dir}")
        raise RuntimeError(f"Document directory not found: {doc_dir}")

    texts: List[str] = []
    metadata: List[dict] = []

    file_paths = [path for path in sorted(doc_dir.rglob("*")) if path.is_file()]
    total_files = len(file_paths)

    for idx, path in enumerate(file_paths, start=1):
        logger.debug(f"\rLoading {idx}/{total_files}: {path}")
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not content.strip():
            continue
        texts.append(content)
        metadata.append({"path": str(path)})

    if not texts:
        raise RuntimeError(f"No readable documents in: {doc_dir}")

    return texts, metadata


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def save_index(
    index: faiss.Index, metadata: List[dict], index_path: Path, meta_path: Path
) -> None:
    logger.info(f"Saving index to: {index_path}")
    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_index(index_path: Path, meta_path: Path) -> Tuple[faiss.Index, List[dict]]:
    logger.info(f"Loading index from: {index_path}")
    if not index_path.exists():
        logger.error(f"Index not found: {index_path}")
        raise RuntimeError(f"Index not found: {index_path}")
    if not meta_path.exists():
        logger.error(f"Metadata not found: {meta_path}")
        raise RuntimeError(f"Metadata not found: {meta_path}")
    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, metadata


def resolve_index_paths(
    db_path: str, index_path_within_db: str
) -> tuple[Path, Path, Path]:
    index_dir = Path(db_path) / index_path_within_db
    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "index.meta.json"
    config_path = index_dir / "config.json"
    return index_path, meta_path, config_path


def get_default_index_path_within_db() -> str:
    return DEFAULT_INDEX_PATH_WITHIN_DB


def save_index_config(config_path: Path, model: str) -> None:
    config_path.write_text(json.dumps({"model": model}, indent=2), encoding="utf-8")


def load_index_config(config_path: Path) -> dict:
    logger.info(f"Loading index config from: {config_path}")
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}. Returning empty config.")
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse config: {config_path}. Error: {exc}")
        raise RuntimeError(f"Invalid index config: {config_path}") from exc
