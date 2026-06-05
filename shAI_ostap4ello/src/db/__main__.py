#!/usr/bin/env python3

import argparse
import signal
import sys
import os
from openai import OpenAI
from typing import List, Optional

from .db import build, get_index_info, search, check, search_in_files_dynamic
from ..llm import get_client

import logging

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_EMBED_MODEL = "ibm/granite-embedding:125m"
DEFAULT_EXTENDED_SEARCH = False
DEFAULT_TOP_K = 5
DEFAULT_TOP_K_EXTENDED = 10
DEFAULT_DB_PATH = "~/.local/share/shai_db"
DEFAULT_INDEX_PATH_WITHIN_DB = ".index"
DEFAULT_BATCH_SIZE = 32


def _get_client() -> OpenAI:
    return get_client(DEFAULT_API_BASE_URL, DEFAULT_API_KEY)

def _cmd_info(args: argparse.Namespace) -> None:
    db_path = os.path.expanduser(args.db_path)
    if not check(db_path, args.index_path_within_db):
        print("Index not found.")
        raise SystemExit(1)

    try:
        print("Index Information:")
        print(f"  Database path: {db_path}")
        info = get_index_info(db_path, args.index_path_within_db)
        for key, value in info.items():
            print(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Error retrieving index information: {e}")
        raise SystemExit(1)

def _cmd_build(args: argparse.Namespace) -> None:
    client = _get_client()
    model = args.model
    db_path = os.path.expanduser(args.db_path)
    # TODO: better error handling
    try:
        build(
            db_path=db_path,
            index_path_within_db=args.index_path_within_db,
            client=client,
            model=model,
            batch_size=args.batch_size,
            section_rows=args.section_rows,
        )
        print("Index build complete.")
    except Exception as e:
        logger.error(f"Error during index build: {e}")
        raise SystemExit(1)

    print("Index build complete.")


def _cmd_search(args: argparse.Namespace) -> None:
    read_results = args.read_results
    db_path = os.path.expanduser(args.db_path)
    expected_model = DEFAULT_EMBED_MODEL
    db_model = None
    if not check(db_path, args.index_path_within_db):
        print("Index not found. Run 'build' command first.", file=sys.stderr)
        sys.exit(1)
    client = _get_client()

    try:
        db_model = str(get_index_info(db_path, args.index_path_within_db)["model"])
        if expected_model != db_model:
            logger.warning(
                f"This database uses {db_model} embeddings (instead"
                f"of {expected_model}, specified in your config/args)"
            )
        results = search(
            db_path=db_path,
            index_path_within_db=args.index_path_within_db,
            client=client,
            query=args.query,
            top_k=args.top_k,
        )
    except Exception as e:
        logger.warning(f"Error during db search: {e}")
        raise SystemExit(1)

    if args.extended_search:
        logger.info("Performing extended search on retrieved documents")
        paths = []
        for r in results:
            paths.append(r["metadata"]["path"])

        try:
            if expected_model != db_model:
                logger.warning(
                    f"This database uses {db_model} embeddings (instead"
                    "of {expected_model}, specified in your config/args)"
                )
            results = search_in_files_dynamic(
                file_paths=paths,
                client=client,
                model=db_model,
                query=args.query,
                top_k=args.top_k_extended,
            )
        except Exception as e:
            logger.warning(f"Error during db search: {e}")
            raise SystemExit(1)

        print("Results of extended search:")
        for i, result in enumerate(results, 1):
            m = result["metadata"]
            p = m["path"]
            f = int(m["from"])
            t = int(m["to"])
            print(f"  {i}: {p}:{f}-{t}: (dist={result['distance']:.4f})")
            if read_results:
                lines = open(p, "r").readlines()[f:t]
                for line in lines:
                    print("  " + line, end="")
                print("  ---")
        print()
    else:
        print("Results:")
        for i, result in enumerate(results, 1):
            m = result["metadata"]
            p = m["path"]
            print(f"  {i}: {p}: (dist={result['distance']:.4f})")
            if read_results:
                content = open(p, "r").read()
                for line in content.splitlines():
                    print("  " + line)
                print("  ---")
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
        default=DEFAULT_DB_PATH,
        help="Path to document directory",
    )
    build_cmd.add_argument(
        "--index-path-within-db",
        default=DEFAULT_INDEX_PATH_WITHIN_DB,
        help="Index subdirectory name (must start with a dot to be hidden)",
    )
    build_cmd.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Embedding batch size",
    )
    build_cmd.add_argument(
        "--model", default=DEFAULT_EMBED_MODEL, help=f"Embedding model to use"
    )
    build_cmd.add_argument(
        "--section-rows",
        type=int,
        default=0,
        help=(
            "If set, indexes by sections of the chosen size instead of the whole document"
        ),
    )
    build_cmd.set_defaults(func=_cmd_build)

    search_cmd = sub.add_parser(
        "search",
        help="Search the index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    search_cmd.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="Path to document directory",
    )
    search_cmd.add_argument(
        "--index-path-within-db",
        default=DEFAULT_INDEX_PATH_WITHIN_DB,
        help="Index subdirectory name (must start with a dot to be hidden)",
    )
    search_cmd.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results to return"
    )
    search_cmd.add_argument(
        "--extended-search",
        "-e",
        action="store_true",
        default=DEFAULT_EXTENDED_SEARCH,
        help="If true, section-scoped search will be applied on retieved docs, then metadata with this will be returned",
    )
    search_cmd.add_argument(
        "--top-k-extended",
        type=int,
        default=DEFAULT_TOP_K_EXTENDED,
        help="Number of results to return",
    )
    search_cmd.add_argument(
        "--read-results",
        "-R",
        action="store_true",
        default=False,
        help="If true, the content of retrieved documents will be printed to stdout along with metadata",
    )
    search_cmd.add_argument("query", help="Search query string")
    search_cmd.set_defaults(func=_cmd_search)

    info_cmd = sub.add_parser(
        "info",
        help="Retrieve index information, if index exists",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    info_cmd.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="Path to document directory",
    )
    info_cmd.add_argument(
        "--index-path-within-db",
        default=DEFAULT_INDEX_PATH_WITHIN_DB,
        help="Index subdirectory name (must start with a dot to be hidden)",
    )
    info_cmd.set_defaults(func=_cmd_info)

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
