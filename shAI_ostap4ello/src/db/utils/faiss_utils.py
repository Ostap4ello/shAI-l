from pathlib import Path
from typing import List, Tuple
import faiss
import json
import numpy as np


import logging

logger = logging.getLogger(__name__)


DEFAULT_INDEX_PATH_WITHIN_DB = ".index"
ALLOWED_EXTENSIONS = [
    # General text/markdown
    ".txt", ".md", ".markdown", ".log", ".norg",

    # Typesetting/markup/documentation
    ".rst", ".tex", ".latex", ".asciidoc", ".adoc", ".org", ".rmd", ".qmd", ".gmi", ".gemini",

    # Tabular/notation
    ".csv", ".tsv", ".srt", ".vtt", ".bib",

    # Config/human-readable data
    ".json", ".xml", ".yaml", ".yml", ".ini", ".toml", ".cfg", ".conf", ".properties", ".plist",

    # Diagram/graph formats
    ".dot", ".plantuml",

    # Notebooks/code-literate
    ".ipynb",

    # Code (sometimes human-annotated)
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".html", ".css", ".sh", ".bat", ".env"
]


def build_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def save_index(
    index: faiss.Index, metadata: List[dict], index_path: Path, meta_path: Path
) -> None:
    logger.debug(f"Saving index to: {index_path}")
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


