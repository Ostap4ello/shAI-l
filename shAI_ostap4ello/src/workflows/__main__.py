#!/usr/bin/env python3

from openai import OpenAI
from typing import List, Optional
import argparse
import logging
import sys
import signal

from ..db.__main__ import (
    DEFAULT_DB_PATH,
    DEFAULT_INDEX_PATH_WITHIN_DB,
    DEFAULT_TOP_K_EXTENDED,
    DEFAULT_EXTENDED_SEARCH,
)

from . import rag_simple, rag_extended
from ..llm import get_client

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_EMBED_MODEL = "ibm/granite-embedding:125m"


def _cli_parser():
    parser = argparse.ArgumentParser(
        description="CLI interface for RAG pipeline."
        # Note: is not actualized with config
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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

    find_cmd = subparsers.add_parser(
        "find", help="Generate a RAG-enabled response for a query"
    )
    find_cmd.add_argument(
        "query", type=str, help="The query to process with the RAG pipeline"
    )
    find_cmd.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="Path to document directory",
    )
    find_cmd.add_argument(
        "--index-path-within-db",
        default=DEFAULT_INDEX_PATH_WITHIN_DB,
        help="Index subdirectory name (must start with a dot to be hidden)",
    )
    find_cmd.add_argument(
        "--extended-search",
        "-e",
        action="store_true",
        default=DEFAULT_EXTENDED_SEARCH,
        help="If true, section-scoped find will be applied on retieved docs, then metadata with this will be returned",
    )
    find_cmd.add_argument(
        "--top-k-extended",
        type=int,
        default=DEFAULT_TOP_K_EXTENDED,
        help="Number of results to return",
    )

    find_cmd.set_defaults(func=_cmd_rag)

    return parser


def _get_client() -> OpenAI:
    return get_client(DEFAULT_API_BASE_URL, DEFAULT_API_KEY)


def _cmd_rag(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        if args.extended_search:
            results = rag_extended(
                client=client,
                gen_model=DEFAULT_MODEL,
                query=args.query,
                db_path=args.db_path,
                index_path_within_db=args.index_path_within_db,
                top_k=5,
                top_k_extended=args.top_k_extended,
                section_size=100,  # TODO: make this configurable
            )
        else:
            results = rag_simple(
                client=client,
                gen_model=DEFAULT_MODEL,
                query=args.query,
                db_path=args.db_path,
                index_path_within_db=args.index_path_within_db,
                top_k=5,
            )
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}")
        raise SystemExit(1)

    print(results)


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
