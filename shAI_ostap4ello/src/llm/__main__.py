#!/usr/bin/env python3

import argparse
import json
import signal
import sys
from openai import OpenAI
from typing import List, Optional

from ..llm import generate, embed_string, get_client

import logging

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen3:1.7b"
DEFAULT_EMBED_MODEL = "ibm/granite-embedding:125m"


def _get_client() -> OpenAI:
    return get_client(DEFAULT_API_BASE_URL, DEFAULT_API_KEY)


def _cmd_generate(args: argparse.Namespace) -> None:
    client = _get_client()
    generate(
        client=client, model=args.model, user_input=args.prompt, do_stream=args.stream
    )


def _cmd_embed_string(args: argparse.Namespace) -> None:
    client = _get_client()
    embedding = embed_string(client=client, model=args.model, string=args.input_string)
    print(json.dumps(embedding, indent=2))


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple LLM response generator with streaming support.",
        # Note: is not actualized with config
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate_cmd = sub.add_parser(
        "generate",
        help="Generate a response from the LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    generate_cmd.add_argument(
        "--model", default=DEFAULT_MODEL, help="Model to use for generation"
    )
    generate_cmd.add_argument(
        "--stream", action="store_true", default=True, help="Enable streaming output"
    )
    generate_cmd.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Disable streaming output",
    )
    generate_cmd.add_argument("prompt", help="The prompt to send to the LLM")
    generate_cmd.set_defaults(func=_cmd_generate)

    embed_cmd = sub.add_parser(
        "embed",
        help="Generate embeddings for a string",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    embed_cmd.add_argument(
        "--model",
        default=DEFAULT_EMBED_MODEL,
        help="Model to use for embedding generation",
    )
    embed_cmd.add_argument("input_string", help="The string to generate embeddings for")
    embed_cmd.set_defaults(func=_cmd_embed_string)

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
