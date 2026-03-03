#!/usr/bin/env python3

import argparse
import json
import signal
import sys
import os
from openai import OpenAI
from typing import List, Optional

from .llm import generate

from pathlib import Path

DEFAULT_API_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_EMBED_MODEL = "qwen3:1.7b"

def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = DEFAULT_API_KEY
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        base_url = DEFAULT_API_BASE_URL
    return OpenAI(api_key=api_key, base_url=base_url)


def _cmd_generate(args: argparse.Namespace) -> None:
    client = _get_client()
    generate(client=client, model=args.model, user_input=args.prompt, do_stream=args.stream)


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple LLM response generator with streaming support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate_cmd = sub.add_parser(
        "generate",
        help="Generate a response from the LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    generate_cmd.add_argument("--model", default=DEFAULT_EMBED_MODEL, help="Model to use for generation")
    generate_cmd.add_argument("--stream", action="store_true", default=True, help="Enable streaming output")
    generate_cmd.add_argument("--no-stream", action="store_false", dest="stream", help="Disable streaming output")
    generate_cmd.add_argument("prompt", help="The prompt to send to the LLM")
    generate_cmd.set_defaults(func=_cmd_generate)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    def handle_sigint(signum: int, frame: object) -> None:
        print("\nInterrupted. Exiting cleanly.", file=sys.stderr)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    parser = _cli_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
