from openai import OpenAI
from typing import List, Optional
import argparse
import logging
import os
import sys
import signal

from .db.__main__ import _cli_parser as _db_cli_parser
from .rag.__main__ import _cli_parser as _rag_cli_parser
from .llm.__main__ import _cli_parser as _llm_cli_parser
from .utils.__main__ import _cli_parser as _utils_cli_parser

from .utils import is_ollama_running, start_ollama, stop_ollama

DEFAULT_API_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen3:1.7b"

logger = logging.getLogger(__name__)


def _cli_parser():
    parser = argparse.ArgumentParser(description="ShAI-CLI")

    parser.add_argument(
        "--keep-ollama-running",
        "-K",
        type=bool,
        default=False,
        help="If true, if this script starts Ollama, it will not stop it on exit",
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )
    subparsers.add_parser(
        "db",
        parents=[_db_cli_parser()],
        add_help=False,
        help="Database indexing and retrieval",
    )
    subparsers.add_parser(
        "rag",
        parents=[_rag_cli_parser()],
        add_help=False,
        help="RAG-enabled generation",
    )
    subparsers.add_parser(
        "llm",
        parents=[_llm_cli_parser()],
        add_help=False,
        help="Direct LLM interactions",
    )
    subparsers.add_parser(
        "utils",
        parents=[_utils_cli_parser()],
        add_help=False,
        help="Miscellaneous utilities",
    )

    return parser


def get_running_client() -> OpenAI:
    client = _get_client()
    model = _get_model()
    _ollama_check_or_run()
    logger.info(f"Bringing up {model} model early")
    try:
        client.completions.create(model=model, max_tokens=1, prompt="Hello")
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        raise RuntimeError(f"Error connecting to Ollama: {e}") from e
    logger.info(f"{model} model is up and running.")
    return client


def _ollama_check_or_run() -> bool:
    if is_ollama_running():
        logger.info("Ollama is running.")
        return False
    else:
        print("Ollama is not running. Start Ollama? (y/N): ", end="")
        choice = input().strip().lower()
        if choice == "y":
            logger.info("Trying to start Ollama...")
            start_ollama()
        else:
            logger.error("Ollama is required to run this application. Exiting.")
            raise SystemExit(0)
        return True


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = DEFAULT_API_KEY
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        base_url = DEFAULT_API_BASE_URL
    client = OpenAI(api_key=api_key, base_url=base_url)
    logger.info(f"Initialized OpenAI client with base URL: {base_url}")
    return client


def cleanup(stop_ollama_on_finish: bool = False) -> None:
    logger.info("Cleaning up resources...")
    if stop_ollama_on_finish:
        stop_ollama()
    logger.info("Goodbye!")


def _get_model():
    return DEFAULT_MODEL


def handle_sigint(signum: int, frame: object) -> None:
    print()
    logging.warning("\nInterrupted. Exiting cleanly.")
    cleanup()
    raise SystemExit(0)


def loop() -> None:
    signal.signal(signal.SIGINT, handle_sigint)

    client = get_running_client()

    while True:
        user_query = ""
        try:
            user_query = input("Enter your query: ")
        except EOFError:
            logger.warning("\nNo input provided. Exiting.")
            break

        try:
            # main logic
            pass
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            break


def main(argv: Optional[List[str]] = None) -> None:
    def handle_sigint(signum: int, frame: object) -> None:
        print("\nInterrupted. Exiting cleanly.", file=sys.stderr)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    parser = _cli_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    stop_ollama_on_finish = _ollama_check_or_run()
    args.func(args)

    cleanup(stop_ollama_on_finish & (args.keep_ollama_running == False))


if __name__ == "__main__":
    main()
