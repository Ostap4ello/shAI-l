#!/usr/bin/env python3

from openai import OpenAI
from typing import List, Optional
import argparse
import logging
import sys
import signal

from ..db import __main__ as db_main
from ..llm import __main__ as llm_main

from . import rag_simple, rag_extended, interpreter
from ..llm import get_client

logger = logging.getLogger(__name__)

DEFAULT_LOG_LEVEL = "INFO"


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
        "-l",
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_LOG_LEVEL,
        help="Set the logging level (default: INFO)",
    )

    find_cmd = subparsers.add_parser(
        "find", help="Generate a RAG-enabled response for a query"
    )
    find_cmd.add_argument(
        "query", type=str, help="The query to process with the RAG pipeline"
    )
    find_cmd.add_argument(
        "-m",
        "--model",
        default=llm_main.DEFAULT_MODEL,
        help="The generation model to use for RAG",
    )
    find_cmd.add_argument(
        "--db-path",
        default=db_main.DEFAULT_DB_PATH,
        help="Path to document directory",
    )
    find_cmd.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=db_main.DEFAULT_TOP_K,
        help="Number of results to return for the initial search",
    )
    find_cmd.add_argument(
        "-i",
        "--index-path-within-db",
        default=db_main.DEFAULT_INDEX_PATH_WITHIN_DB,
        help="Index subdirectory name (must start with a dot to be hidden)",
    )
    find_cmd.add_argument(
        "-e",
        "--extended-search",
        action="store_true",
        default=db_main.DEFAULT_EXTS,
        help="If true, section-scoped find will be applied on retieved docs, then metadata with this will be returned",
    )
    find_cmd.add_argument(
        "-K",
        "--top-k-extended",
        type=int,
        default=db_main.DEFAULT_EXTS_TOP_K,
        help="Number of results to return",
    )
    find_cmd.add_argument(
        "-S",
        "--section-rows-extended",
        type=int,
        default=db_main.DEFAULT_EXTS_SECTION_ROWS,
        help="Number of rows per section for extended search (only applicable with --extended-search)",
    )
    find_cmd.set_defaults(func=_cmd_rag)

    interp_cmd = subparsers.add_parser(
        "interpreter", help="Interpreter for testing classifier (proof of concept)"
    )
    interp_cmd.set_defaults(func=_cmd_interpreter)

    return parser


def _get_client() -> OpenAI:
    return get_client(llm_main.DEFAULT_API_BASE_URL, llm_main.DEFAULT_API_KEY)


def _cmd_rag(args: argparse.Namespace) -> None:
    client = _get_client()
    try:
        if args.extended_search:
            results = rag_extended(
                client=client,
                gen_model=args.model,
                query=args.query,
                db_path=args.db_path,
                index_path_within_db=args.index_path_within_db,
                top_k=args.top_k,
                top_k_extended=args.top_k_extended,
                section_size=args.section_rows_extended,
            )
        else:
            results = rag_simple(
                client=client,
                gen_model=args.model,
                query=args.query,
                db_path=args.db_path,
                index_path_within_db=args.index_path_within_db,
                top_k=args.top_k,
            )
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}")
        raise SystemExit(1)

    print(results)


def _cmd_interpreter(_: argparse.Namespace) -> None:
    interpreter()


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
