#!/usr/bin/env python3

from openai import OpenAI
from typing import List, Optional
import argparse
import logging
import os
import sys
import signal

from .rag import rag_pipeline

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_EMBED_MODEL = "ibm/granite-embedding:125m"


def _cli_parser():
    parser = argparse.ArgumentParser(description="CLI interface for RAG pipeline.")
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
    find_cmd.set_defaults(func=_cmd_find)

    return parser


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = DEFAULT_API_KEY
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        base_url = DEFAULT_API_BASE_URL
    client = OpenAI(api_key=api_key, base_url=base_url)
    logger.info(f"Initialized OpenAI client with base URL: {base_url}")
    return client


def _cmd_find(args: argparse.Namespace) -> None:
    client = _get_client()
    results = rag_pipeline(
        client=client, model=DEFAULT_MODEL, query=args.query, top_k=5
    )
    assert isinstance(results, str), "Expected RAG pipeline output to be a string"
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
