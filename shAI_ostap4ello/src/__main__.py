import argparse
import logging
from pathlib import Path
import sys
import signal


from .db.__main__ import _cli_parser as _db_cli_parser
from .llm.__main__ import _cli_parser as _llm_cli_parser
from .workflows.__main__ import _cli_parser as _w_cli_parser
from .utils.__main__ import _cli_parser as _utils_cli_parser

from .config import DEFAULT_CONFIG_PATH, load_config, get_config_value
from .utils import (
    is_ollama_running,
    pull_model,
    start_ollama,
    stop_ollama,
    wait_ollama_ready,
)

logger = logging.getLogger(__name__)

DEFAULT_CREATE_CONFIG = False
DEFAULT_API_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_LOG_LEVEL = "INFO"


def _cmd_ollama_init(_: argparse.Namespace) -> None:
    init_shai()


def init_shai():
    db_path = Path(get_config_value("db", "db_path", str)).expanduser().resolve()
    logger.info(f"Creating db folder({db_path})")
    db_path.mkdir(parents=True, exist_ok=True)

    models = [
        get_config_value("llm", "model", str),
        get_config_value("llm", "embed_model", str),
        get_config_value("rag", "model", str),
    ]
    logger.info(f"Initializing Ollama container for configured models ({models})")
    url = get_config_value("utils", "ollama_url", str)
    was_ollama_running = is_ollama_running(
        get_config_value("utils", "ollama_container_name")
    )

    if not was_ollama_running:
        try:
            start_ollama(context_length=3200, gpus="none", name="tmp_shai", create=True)
        except Exception as e:
            logger.error(f"Error: {e}")
            raise SystemExit(1)

        try:
            wait_ollama_ready(url)
        except Exception as e:
            logger.error(f"Error: {e}")
            stop_ollama(name="tmp_shai", remove=True)
            raise SystemExit(1)

    for model in models:
        try:
            pull_model(url, model)
        except Exception as e:
            logger.error(f"Error: {e}")

    if not was_ollama_running:
        try:
            stop_ollama(name="tmp_shai", remove=True)
        except Exception as e:
            logger.error(f"Error: {e}")
            raise SystemExit(1)


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ShAI-CLI")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--create-config",
        "-C",
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
        "workflows",
        aliases=["ws"],
        parents=[_w_cli_parser()],
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
        aliases=["u"],
        parents=[_utils_cli_parser()],
        add_help=False,
        help="Miscellaneous utilities",
    )
    init_parser = subparsers.add_parser(
        "init",
        help=(
            "Create default database folder, create/start Ollama container, pull"
            "models from config, then remove container and exit"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    init_parser.set_defaults(func=_cmd_ollama_init)

    return parser


def handle_sigint(signum: int, frame: object) -> None:
    print()
    logging.warning("\nInterrupted. Exiting cleanly.")
    raise SystemExit(0)


def main(argv: list | None = None) -> None:
    def handle_sigint(signum: int, frame: object) -> None:
        print("\nInterrupted. Exiting cleanly.", file=sys.stderr)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = argv[1:]

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
        if pre_args.config is None:
            pre_args.config = DEFAULT_CONFIG_PATH
        elif pre_args.create_config == True:
            logger.info(f"Creating new config ar {pre_args.config}")
        else:
            logger.info(f"Loading config from {pre_args.config}")

        load_config(config_path_str=pre_args.config, create=pre_args.create_config)

        if pre_args.create_config:
            logger.info(f"Config file created at {pre_args.config}")
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

    if not is_ollama_running(get_config_value("utils", "ollama_container_name")):
        logger.warning(
            "Ollama does not appear to be running. All LLM interactions will "
            "fail. Try using `shai utils start_ollama` to start it."
        )

    args.func(args)


if __name__ == "__main__":
    main()
