from re import sub, match
from pathlib import Path

def _parse_rag_prompt(path: str, data: list[dict], query: list[str]) -> str:
    text = ""
    with open(path, "r") as f:
        for line in f.readlines():
            # Comments
            if match("^[ \t]*<<#.*$", line):
                continue
            line = sub("<<#.*$", "\n", line)
            text += line

    for q in data:
        contents = open(q["path"]).read()
        text = sub("<<DATA>>", "```\n" + contents + "\n```\n <<DATA>>", text)
    for q in query:
        text = sub("<<QUESTION>>", q+"\n <<QUESTION>>", text)

    text = sub("<<DATA>>.*\n", "", text)
    text = sub("<<QUESTION>>.*\n", "", text)

    return text
