import argparse
import logging
import sys
import signal

from .config import DEFAULT_CONFIG_PATH, load_config
from .db.__main__ import _cli_parser as _db_cli_parser
from .interpreter.interpreter import main as interpreter_main
from .llm.__main__ import _cli_parser as _llm_cli_parser
from .rag.__main__ import _cli_parser as _rag_cli_parser
from .utils.__main__ import _cli_parser as _utils_cli_parser

from .utils import is_ollama_running

logger = logging.getLogger(__name__)

DEFAULT_CREATE_CONFIG = False
DEFAULT_API_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_LOG_LEVEL = "INFO"

def _interpreter_cmd(args: argparse.Namespace) -> None:
    interpreter_main(args.config, args.log_level)

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
    _interpreter_parser = subparsers.add_parser(
        "interpreter", help="ShAI shell"
    )
    _interpreter_parser.add_argument(
        "--log-level",
        type=str,
        required=False,
        default="WARNING",
        help="Set the logging level (default: WARNING)",
    )
    _interpreter_parser.set_defaults(func=_interpreter_cmd)

    return parser


def handle_sigint(signum: int, frame: object) -> None:
    print()
    logging.warning("\nInterrupted. Exiting cleanly.")
    raise SystemExit(0)


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
        logger.info(
            f"Loading config from {pre_args.config} (create={pre_args.create_config})"
        )
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

    if not is_ollama_running():
        logger.warning(
            "Ollama does not appear to be running. All LLM interactions will "
            "fail. Try using `shai utils start_ollama` to start it."
        )

    args.func(args)


if __name__ == "__main__":
    main()
