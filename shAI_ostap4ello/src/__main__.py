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
from .config import DEFAULT_CONFIG_PATH, load_config

from .utils import is_ollama_running, start_ollama, stop_ollama

logger = logging.getLogger(__name__)

DEFAULT_KEEP_OLLAMA = False
DEFAULT_CREATE_CONFIG = False
DEFAULT_API_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_LOG_LEVEL = "INFO"

def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ShAI-CLI")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=DEFAULT_CONFIG_PATH,
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        default=DEFAULT_CREATE_CONFIG,
        help="If true, creates a config file with default values if it doesn't exist, then exits",
    )

    parser.add_argument(
        "--keep-ollama-running",
        "-K",
        action="store_true",
        default=DEFAULT_KEEP_OLLAMA,
        help="If true, if this script starts Ollama, it will not stop it on exit",
    )

    subparsers = parser.add_subparsers(
        dest="command", required=False, help="Available commands"
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


def _ollama_check_or_run() -> bool:
    if is_ollama_running():
        logger.info("Ollama is running.")
        return False
    else:
        print("Ollama is not running. Start Ollama?")
        print("[y]es / Yes and [k]eep Ollama running after this session / [n]o (default):")

        choice = input().strip().lower()
        if choice == "y" or choice == "k":
            logger.info("Trying to start Ollama...")
            start_ollama()
            if choice == "k":
                return False
            else:
                return True
        else:
            logger.error("Ollama is required to run this application. Exiting.")
            raise SystemExit(0)


def cleanup(stop_ollama_on_finish: bool = False) -> None:
    logger.info("Cleaning up resources...")
    if stop_ollama_on_finish:
        stop_ollama()
    logger.info("Goodbye!")

def handle_sigint(signum: int, frame: object) -> None:
    print()
    logging.warning("\nInterrupted. Exiting cleanly.")
    cleanup()
    raise SystemExit(0)

# def loop() -> None:
#     signal.signal(signal.SIGINT, handle_sigint)
#
#     client = get_running_client()
#
#     while True:
#         user_query = ""
#         try:
#             user_query = input("Enter your query: ")
#         except EOFError:
#             logger.warning("\nNo input provided. Exiting.")
#             break
#
#         try:
#             # main logic
#             pass
#         except Exception as e:
#             logger.error(f"Error processing query: {e}")
#             break


def main() -> None:
    def handle_sigint(signum: int, frame: object) -> None:
        print("\nInterrupted. Exiting cleanly.", file=sys.stderr)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    argv = sys.argv[1:]

    # Pre-parse
    pre_parser = cli_parser()
    pre_args = None
    try:
        pre_args = pre_parser.parse_args(argv)
    except Exception:
        # Ignore the error from missing required subcommand for now
        # help will be shown later when we parse the full args
        pass

    # TODO: default values in help are not updated when loading config
    if pre_args is not None:
        load_config(config_path_str=pre_args.config, create=pre_args.create_config)
        if pre_args.create_config:
            print(f"Config file created at {pre_args.config}. Exiting as requested.")
            raise SystemExit(0)

    parser = cli_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(0)

    if not hasattr(args, "log_level"):
        args.log_level = DEFAULT_LOG_LEVEL

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
