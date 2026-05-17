from pathlib import Path
from typing import List, Tuple
import os

import logging

logger = logging.getLogger(__name__)


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


def list_db_documents(doc_dir: Path) -> List[Path]:
    logger.info(f"Listing documents in: {doc_dir}")
    if not doc_dir.exists() or not doc_dir.is_dir():
        logger.error(f"Document directory not found: {doc_dir}")
        raise RuntimeError(f"Document directory not found: {doc_dir}")

    doc_paths = []
    for root, _, files in os.walk(doc_dir, followlinks=True):
        for file in files:
            path = Path(root) / file
            if (
                any(part.startswith(".") for part in path.relative_to(doc_dir).parts)
                or path.suffix not in ALLOWED_EXTENSIONS
            ):
                continue
            doc_paths.append(path.absolute())

    if len(doc_paths) == 0:
        raise RuntimeError(f"No readable documents in: {doc_dir}")

    return doc_paths


def load_documents(doc_paths: List[Path]) -> Tuple[List[str], List[dict]]:
    logger.debug(f"Loading documents: {doc_paths}")

    texts: List[str] = []
    metadata: List[dict] = []

    total = len(doc_paths)
    for idx, path in enumerate(doc_paths, start=1):
        logger.debug(f"\rLoading {idx}/{total}: {path}")
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not content.strip():
            continue
        texts.append(content)
        metadata.append({"path": str(path)})

    return texts, metadata


def load_documents_in_sections( doc_paths: List[Path], section_rows: int
) -> Tuple[List[str], List[dict]]:
    logger.debug(
        f"Loading documents in sections: {doc_paths} (section={section_rows}lines)"
    )

    sections: List[str] = []
    metadata: List[dict] = []

    for doc_path in doc_paths:
        one_sections, one_metadata = _load_document_in_sections(doc_path, section_rows)
        sections = sections + one_sections
        metadata = metadata + one_metadata

    return sections, metadata


def _load_document_in_sections(
    doc_path: Path, section_rows: int
) -> Tuple[List[str], List[dict]]:
    logger.debug(
        f"Loading document in sections: {doc_path} (section={section_rows}lines)"
    )

    # TODO: handling last section - if it lacks rows. Separator?
    # Custom 

    sections: List[str] = []
    metadata: List[dict] = []

    lines = open(doc_path, "r").readlines()
    for i in range(0, len(lines), section_rows):
        sections.append("".join(lines[i : i + 20]))
        metadata.append({"path": str(doc_path), "from": i, "to": i + 20})

    return sections, metadata
