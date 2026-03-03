import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from openai import OpenAI


DEFAULT_INDEX_PATH_WITHIN_DB = ".index"

def load_documents(doc_dir: Path, show_progress: bool = False) -> Tuple[List[str], List[dict]]:
    if not doc_dir.exists() or not doc_dir.is_dir():
        raise RuntimeError(f"Document dir not found: {doc_dir}")

    texts: List[str] = []
    metadata: List[dict] = []

    file_paths = [path for path in sorted(doc_dir.rglob("*")) if path.is_file()]
    total_files = len(file_paths)

    for idx, path in enumerate(file_paths, start=1):
        if show_progress:
            print(f"\r" + " "*100, end="", flush=True)  # Clear line
            print(f"\rLoading {idx}/{total_files}: {path}", end="", flush=True)
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not content.strip():
            continue
        texts.append(content)
        metadata.append({"path": str(path)})

    if show_progress:
        print()

    if not texts:
        raise RuntimeError(f"No readable documents in: {doc_dir}")

    return texts, metadata


def embed_strings(
    client: OpenAI,
    model: str,
    strings: List[str],
    batch_size: int,
    show_progress: bool = False,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    total = len(strings)
    for i in range(0, total, batch_size):
        if show_progress:
            done = min(i + batch_size, total)
            print(f"\rEmbedding {done}/{total}", end="", flush=True)
        batch = strings[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        batch_vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        vectors.extend(batch_vecs)

    if show_progress:
        print()
    return np.vstack(vectors)


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def save_index(index: faiss.Index, metadata: List[dict], index_path: Path, meta_path: Path) -> None:
    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_index(index_path: Path, meta_path: Path) -> Tuple[faiss.Index, List[dict]]:
    if not index_path.exists():
        raise RuntimeError(f"Index not found: {index_path}")
    if not meta_path.exists():
        raise RuntimeError(f"Metadata not found: {meta_path}")
    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, metadata

def resolve_index_paths(db_path: str, index_path_within_db: str) -> tuple[Path, Path, Path]:
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
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid index config: {config_path}") from exc
