from typing import List, Optional
import argparse
import logging
import signal
import sys

from .adapter import *

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def _cmd_start_ollama(args: argparse.Namespace) -> None:
    start_ollama(context_length=args.context_length, gpus=args.gpus, name=args.name)


def _cmd_stop_ollama(args: argparse.Namespace) -> None:
    stop_ollama(name=args.name)


def _cmd_is_ollama_running(args: argparse.Namespace) -> None:
    running = is_ollama_running(name=args.name)
    if running:
        print("Ollama is running.")
    else:
        print("Ollama is not running.")


def _cmd_convert(args: argparse.Namespace) -> None:
    convert_man_pages_to_text(src_dir=args.src_dir, out_dir=args.out_dir)


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI interface for ext-utils")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    parser_start = subparsers.add_parser(
        "start_ollama", help="Run the Ollama Docker container"
    )
    parser_start.add_argument(
        "--context-length", type=int, default=32000, help="Set the context length"
    )
    parser_start.add_argument("--gpus", type=str, default="all", help="Specify GPUs")
    parser_start.add_argument(
        "--name", type=str, default="ollama-node-1", help="Container name"
    )
    parser_start.set_defaults(func=_cmd_start_ollama)

    parser_stop = subparsers.add_parser(
        "stop_ollama", help="Stop the Ollama Docker container"
    )
    parser_stop.add_argument(
        "--name", type=str, default="ollama-node-1", help="Container name"
    )
    parser_stop.set_defaults(func=_cmd_stop_ollama)

    parser_status = subparsers.add_parser(
        "is_ollama_running", help="Check if the Ollama Docker container is running"
    )
    parser_status.add_argument(
        "--name", type=str, default="ollama-node-1", help="Container name"
    )
    parser_status.set_defaults(func=_cmd_is_ollama_running)

    parser_convert = subparsers.add_parser(
        "convert_man_pages", help="Convert groff files to ASCII text"
    )
    parser_convert.add_argument(
        "--src-dir", type=str, required=True, help="Source directory for groff files"
    )
    parser_convert.add_argument(
        "--out-dir", type=str, required=True, help="Output directory for text files"
    )
    parser_convert.set_defaults(func=_cmd_convert)

    return parser


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
    )
    args.func(args)


if __name__ == "__main__":
    main()
