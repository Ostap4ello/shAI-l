from re import sub, match
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


def _get_stripped_rag_prompt(path: str) -> str:
    text = ""
    try:
        with open(path, "r") as f:
            for line in f.readlines():
                # Comments
                if match("^[ \t]*<<#.*$", line):
                    continue
                line = sub("<<#.*$", "\n", line)
                text += line
    except Exception as e:
        logger.error(f"Error reading prompt file: {e}")
        raise RuntimeError(f"Error reading prompt file: {e}")
    return text


def get_docchoice_prompt(
    prompt_path: str, data: list[dict], query: str, preview_lines: int = 10
) -> str:
    text = _get_stripped_rag_prompt(prompt_path)

    paths_str = "\n".join([q["path"] for q in data])
    text = sub(
        "<<DATA>>", f"Paths:\n{paths_str}\n\n<<DATA>>", text
    )

    for q in data:
        contents = "".join(open(q["path"]).readlines()[:preview_lines])
        text = sub(
            "<<DATA>>", f"{q["path"]}\n```\n{contents}\n...\n```\n\n<<DATA>>", text
        )
    text = sub("<<QUESTION>>.*\n", "", text)

    text = sub("<<QUESTION>>", f"{query}\n", text)

    return text


def get_docchoice_answer(response: str) -> str | None:
    for line in response.strip().split("\n"):
        line = line.strip()
        if Path(line).is_file() or line == "None":
            return line
    return None


def get_singledoc_prompt(path: str, data: dict, query: str) -> str:
    text = _get_stripped_rag_prompt(path)

    contents = open(data["path"]).read()
    text = sub("<<DATA>>", "```\n" + contents + "\n```\n", text)
    text = sub("<<QUESTION>>", f"{query}\n", text)

    return text


def get_docscanning_prompts(
    prompt_path: str, data: dict, query: list[str]
) -> list[str]:
    text = _get_stripped_rag_prompt(prompt_path)
    part_length = 100

    lines = open(data["path"], "r").readlines()
    parts_count = (len(lines) - 1) // part_length + 1

    parts = [text for _ in range(parts_count)]

    for i in range(0, parts_count):
        contents = "".join(lines[(i) * part_length : (i + 1) * part_length])
        parts[i] = sub("<<DATA>>", f"```\n{contents}\n```\n", parts[i])
        parts[i] = sub("<<QUESTION>>", f"{query}\n", parts[i])

    return parts
