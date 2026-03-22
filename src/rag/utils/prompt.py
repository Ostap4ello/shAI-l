from re import sub
from pathlib import Path

import logging
from typing import Iterable

from ...utils.prompt import get_prompt

logger = logging.getLogger(__name__)


def get_doc_choice_prompt(
    doc_paths: list[str], query: str, preview_lines: int = 10
) -> str:
    prompt_name = "rag-choose-document"
    text = get_prompt(prompt_name)

    paths_str = "\n".join(doc_paths)
    text = sub("<<DATA>>", f"Paths:\n{paths_str}\n\n<<DATA>>", text)

    for doc_path in doc_paths:
        contents = "".join(open(doc_path).readlines()[:preview_lines])
        text = sub(
            "<<DATA>>", f"{doc_path}\n```\n{contents}\n...\n```\n\n<<DATA>>", text
        )
    text = sub("<<QUESTION>>.*\n", "", text)

    text = sub("<<QUESTION>>", f"{query}\n", text)

    return text


def get_doc_choice_answer(response: str) -> str | None:
    for line in response.strip().split("\n"):
        line = line.strip()
        if Path(line).is_file() or line == "None":
            return line
    return None


def get_single_doc_prompt(doc_path: str, query: str) -> str:
    prompt_name = "rag-scan-document"
    text = get_prompt(prompt_name)

    contents = open(doc_path).read()
    text = sub("<<DATA>>", f"```\n{contents}\n```\n", text)
    text = sub("<<QUESTION>>", f"{query}\n", text)

    return text


def get_single_doc_scanning_prompts(doc_path, query: list[str]) -> Iterable[str]:
    prompt_name = "rag-sequentially-can-document"
    template_text = get_prompt(prompt_name)
    part_length = 100

    lines = open(doc_path, "r").readlines()
    parts_count = (len(lines) - 1) // part_length + 1

    for i in range(0, parts_count):
        contents = "".join(lines[(i) * part_length : (i + 1) * part_length])
        part = sub("<<DATA>>", f"```\n{contents}\n```\n", template_text)
        part = sub("<<QUESTION>>", f"{query}\n", part)
        yield part
