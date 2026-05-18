from .adapter import *
from .prompt import get_prompt

__all__ = [
    "start_ollama",
    "stop_ollama",
    "is_ollama_running",
    "convert_man_pages_to_text",
    "pull_model",
    "rm_model",
    "ls_models",
    "get_prompt",
]
