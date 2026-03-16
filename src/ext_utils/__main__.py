import argparse
from .adapter import *

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def main():
    parser = argparse.ArgumentParser(description="CLI interface for ext-utils")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    parser_start = subparsers.add_parser("start_ollama", help="Run the Ollama Docker container")
    parser_start.add_argument("--context-length", type=int, default=32000, help="Set the context length")
    parser_start.add_argument("--gpus", type=str, default="all", help="Specify GPUs")
    parser_start.add_argument("--name", type=str, default="ollama-node-1", help="Container name")

    parser_stop = subparsers.add_parser("stop_ollama", help="Stop the Ollama Docker container")
    parser_stop.add_argument("--name", type=str, default="ollama-node-1", help="Container name")

    parser_status = subparsers.add_parser("is_ollama_running", help="Check if the Ollama Docker container is running")
    parser_status.add_argument("--name", type=str, default="ollama-node-1", help="Container name")

    parser_convert = subparsers.add_parser("convert_man_pages", help="Convert groff files to ASCII text")
    parser_convert.add_argument("--src-dir", type=str, required=True, help="Source directory for groff files")
    parser_convert.add_argument("--out-dir", type=str, required=True, help="Output directory for text files")

    args = parser.parse_args()

    if args.command == "start_ollama":
        start_ollama(context_length=args.context_length, gpus=args.gpus, name=args.name)
    elif args.command == "stop_ollama":
        stop_ollama(name=args.name)
    elif args.command == "is_ollama_running":
        running = is_ollama_running(name=args.name)
        if running:
            print("Ollama is running.")
        else:
            print("Ollama is not running.")
    elif args.command == "convert_man_pages":
        convert_man_pages_to_text(src_dir=args.src_dir, out_dir=args.out_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

