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

def load_documents(doc_dir: Path) -> Tuple[List[str], List[dict]]:
    logger.info(f"Loading documents from: {doc_dir}")
    if not doc_dir.exists() or not doc_dir.is_dir():
        logger.error(f"Document directory not found: {doc_dir}")
        raise RuntimeError(f"Document directory not found: {doc_dir}")

    texts: List[str] = []
    metadata: List[dict] = []

    file_paths = []
    for root, _, files in os.walk(doc_dir, followlinks=True):
        for file in files:
            path = Path(root) / file
            if (
                any(part.startswith(".") for part in path.relative_to(doc_dir).parts)
                or path.suffix not in ALLOWED_EXTENSIONS
            ):
                continue
            file_paths.append(path.absolute())
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

def load_documents_in_sections(doc_paths: List[Path], section_rows: int) -> Tuple[List[str], List[dict]]:
    logger.debug(f"Loading documents in sections: {doc_paths}")

    sections = []
    metadata = []

    for doc_path in doc_paths:
        one_sections, one_metadata = load_document_in_sections(doc_path, section_rows)
        sections = sections + one_sections
        metadata = metadata + one_metadata

    return sections, metadata

def load_document_in_sections(doc_path: Path, section_rows: int) -> Tuple[List[str], List[dict]]:
    logger.debug(f"Loading document in sections: {doc_path}")

    if not doc_path.exists() or not doc_path.is_file():
        logger.error(f"Document not found: {doc_path}")
        raise RuntimeError(f"Document not found: {doc_path}")

    # TODO: handling last section - if it lacks rows. Separator?
    # Custom 

    sections = []
    metadata = []

    lines = open(doc_path, "r").readlines()
    for i in range(0, len(lines), section_rows):
        sections.append("".join(lines[i : i + 20]))
        metadata.append({"path": str(doc_path), "from": i, "to": i + 20})

    return sections, metadata

