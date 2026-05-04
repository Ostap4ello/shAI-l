import argparse
import logging
import sys

from ..config import DEFAULT_CONFIG_PATH
from . import interpreter_main

logger = logging.getLogger(__name__)

DEFAULT_KEEP_OLLAMA = False
DEFAULT_LOG_LEVEL = "WARNING"


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ShAI-CLI Interpreter")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_LOG_LEVEL,
        help="Set the logging level",
    )
    return parser


def main() -> None:
    argv = sys.argv[1:]

    parser = cli_parser()
    args = parser.parse_args(argv)

    interpreter_main(config_path=args.config, log_level=args.log_level)


if __name__ == "__main__":
    main()
