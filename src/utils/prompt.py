from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _strip_comments(text: str) -> str:
    """
    Remove comments using # as marker.
    Use \# to render a literal #.
    """
    stripped_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        has_newline = line.endswith("\n")
        body = line[:-1] if has_newline else line

        out: list[str] = []
        idx = 0
        while idx < len(body):
            ch = body[idx]
            if ch == "\\" and idx + 1 < len(body) and body[idx + 1] == "#":
                out.append("#")
                idx += 2
                continue
            if ch == "#":
                break
            out.append(ch)
            idx += 1

        cleaned = "".join(out)
        if has_newline:
            cleaned += "\n"
        stripped_lines.append(cleaned)

    return "".join(stripped_lines)


def get_prompt(base_name: str) -> str:
    """
    Retrieve prompt text from prompts/<base_name>.txt.
    """
    if not base_name:
        raise ValueError("Prompt base name cannot be empty.")

    prompt_path = _project_root() / "prompts" / f"{base_name}.txt"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return _strip_comments(prompt_path.read_text(encoding="utf-8"))
