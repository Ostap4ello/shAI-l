#!/usr/bin/env python3

import argparse
import signal
import sys
import os
from openai import OpenAI
from typing import List, Optional

from .db import build, search, check

import logging

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_EMBED_MODEL = "ibm/granite-embedding:125m"


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = DEFAULT_API_KEY
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        base_url = DEFAULT_API_BASE_URL
    return OpenAI(api_key=api_key, base_url=base_url)


def _cmd_build(args: argparse.Namespace) -> None:
    client = _get_client()
    model = args.model
    db_path = os.path.expanduser(args.db_path)
    build(
        db_path=db_path,
        index_path_within_db=args.index_path_within_db,
        client=client,
        model=model,
        batch_size=args.batch_size,
    )
    print("Index build complete.")


def _cmd_search(args: argparse.Namespace) -> None:
    db_path = os.path.expanduser(args.db_path)
    if not check(db_path, args.index_path_within_db):
        print("Index not found. Run 'build' command first.", file=sys.stderr)
        sys.exit(1)
    client = _get_client()
    results = search(
        db_path=db_path,
        index_path_within_db=args.index_path_within_db,
        client=client,
        query=args.query,
        top_k=args.top_k,
    )
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"[{i}]")
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Metadata:")
        for key, value in result["metadata"].items():
            print(f"    {key}: {value}")
        print()


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local retrieval indexer",
        # Note: is not actualized with config
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build_cmd = sub.add_parser(
        "build",
        help="Build the index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build_cmd.add_argument(
        "--db-path",
        default="~/.local/share/shai_db",
        help="Path to document directory",
    )
    build_cmd.add_argument(
        "--index-path-within-db",
        default=".index",
        help="Index subdirectory name (must start with a dot to be hidden)",
    )
    build_cmd.add_argument(
        "--batch-size", type=int, default=32, help="Embedding batch size"
    )
    build_cmd.add_argument(
        "--model", default=DEFAULT_EMBED_MODEL, help=f"Embedding model to use"
    )
    build_cmd.set_defaults(func=_cmd_build)

    search_cmd = sub.add_parser(
        "search",
        help="Search the index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    search_cmd.add_argument(
        "--db-path",
        default="~/.local/share/shai_db",
        help="Path to document directory",
    )
    search_cmd.add_argument(
        "--index-path-within-db",
        default=".index",
        help="Index subdirectory name (must start with a dot to be hidden)",
    )
    search_cmd.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return"
    )
    search_cmd.add_argument("query", help="Search query string")
    search_cmd.set_defaults(func=_cmd_search)

    check_cmd = sub.add_parser(
        "check",
        help="Check if index exists",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    check_cmd.add_argument(
        "--db-path",
        default="~/.local/share/shai_db",
        help="Path to document directory",
    )
    check_cmd.add_argument(
        "--index-path-within-db",
        default=".index",
        help="Index subdirectory name (must start with a dot to be hidden)",
    )
    check_cmd.set_defaults(
        func=lambda args: print(
            "Index exists."
            if check(
                db_path=os.path.expanduser(args.db_path),
                index_path_within_db=args.index_path_within_db,
            )
            else "Index not found."
        )
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

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
