from pathlib import Path
from typing import List, Optional
import argparse
import logging
import signal
import sys

from .adapter import *
from .fetch_man import fetch_manpages_to_db, MAN_ROOT, DEFAULT_SECTIONS, MERGE_POLICIES

DEFAULT_DOCKER_CONTAINER_NAME = "ollama-node-1"
DEFAULT_DOCKER_CONTEXT_LENGTH = 32000
DEFAULT_DOCKER_GPUS = "all"
DEFAULT_FETCH_DB_PATH = "./db"
DEFAULT_FETCH_MERGE_STRATEGY = "abort"
DEFAULT_LOG_LEVEL = "INFO"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def _cmd_start_ollama(args: argparse.Namespace) -> None:
    print(f"Starting Ollama with context length {args.context_length}, gpus {args.gpus}, name {args.name}...")
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


def _cmd_fetch_man_db(args: argparse.Namespace) -> None:
    try:
        db_path = str(Path(args.db_path).expanduser())
        fetch_manpages_to_db(
            db_path=db_path,
            sections=args.sections.split(","),
            merge_strategy=args.merge_strategy,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI interface for ext-utils",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_LOG_LEVEL,
        help="Set the logging level (default: INFO)",
    )

    parser_start = subparsers.add_parser(
        "start_ollama", help="Run the Ollama Docker container"
    )
    parser_start.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_DOCKER_CONTEXT_LENGTH,
        help="Set the context length",
    )
    parser_start.add_argument("--gpus", type=str, default=DEFAULT_DOCKER_GPUS, help="Specify GPUs")
    parser_start.add_argument(
        "--name", type=str, default=DEFAULT_DOCKER_CONTAINER_NAME, help="Container name"
    )
    parser_start.set_defaults(func=_cmd_start_ollama)

    parser_stop = subparsers.add_parser(
        "stop_ollama", help="Stop the Ollama Docker container"
    )
    parser_stop.add_argument(
        "--name", type=str, default=DEFAULT_DOCKER_CONTAINER_NAME, help="Container name"
    )
    parser_stop.set_defaults(func=_cmd_stop_ollama)

    parser_status = subparsers.add_parser(
        "is_ollama_running", help="Check if the Ollama Docker container is running"
    )
    parser_status.add_argument(
        "--name", type=str, default=DEFAULT_DOCKER_CONTAINER_NAME, help="Container name"
    )
    parser_status.set_defaults(func=_cmd_is_ollama_running)

    parser_convert = subparsers.add_parser(
        "convert_man_pages",
        help="Convert gz'd groff files to ASCII text"
        + " (recursively from src-dir to out-dir for all gz files found in src-dir)",
    )
    parser_convert.add_argument(
        "--src-dir", type=str, required=True, help="Source directory for groff files"
    )
    parser_convert.add_argument(
        "--out-dir", type=str, required=True, help="Output directory for text files"
    )
    parser_convert.set_defaults(func=_cmd_convert)

    fetch_man_cmd = subparsers.add_parser(
        "fetch_man_db",
        help="Fetch system man pages to text file database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    fetch_man_cmd.add_argument(
        "--db-path",
        default=DEFAULT_FETCH_DB_PATH,
        help="Destination folder for man text db",
    )
    fetch_man_cmd.add_argument(
        "--sections",
        default=",".join(sorted(DEFAULT_SECTIONS)),
        help="Man section numbers (comma-separated)",
    )
    fetch_man_cmd.add_argument(
        "--merge-strategy",
        default=DEFAULT_FETCH_MERGE_STRATEGY,
        choices=sorted(MERGE_POLICIES),
        help="Policy if db dir exists and not empty: abort, clean, merge-ours (keep old file on conflict), merge-theirs, skip-existing.",
    )
    fetch_man_cmd.add_argument(
        "--man-root",
        default=MAN_ROOT,
        help=f"Root directory for man pages (contains subdirs like man1, man2, etc)",
    )
    fetch_man_cmd.set_defaults(func=_cmd_fetch_man_db)

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
