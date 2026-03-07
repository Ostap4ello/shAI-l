from src.rag import _parse_rag_prompt
import src.llm
import src.db_retrieve
import sys
import signal
from src.ext_utils import *
import argparse
import os
import logging
from openai import OpenAI


DEFAULT_API_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen3:1.7b"

ollama_was_run = 0

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Set logging levels.")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    return parser.parse_args()

def get_running_client() -> OpenAI:
    client = _get_client()
    model = _get_model()
    _ollama_check_or_run()
    logger.info(f"Bringing up {model} model early")
    try:
        client.completions.create(model=model, max_tokens=1, prompt="Hello")
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {e}")
        raise RuntimeError(f"Error connecting to Ollama: {e}") from e
    logger.info(f"{model} model is up and running.")
    return client


def _ollama_check_or_run() -> None:
    if is_ollama_running():
        logger.info("Ollama is running.")
    else:
        print("Ollama is not running. Start Ollama? (y/N): ", end="")
        choice = input().strip().lower()
        if choice == "y":
            logger.info("Trying to start Ollama...")
            start_ollama()
            ollama_was_run = 1
        else:
            logger.error("Ollama is required to run this application. Exiting.")
            raise SystemExit(0)


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


def finish() -> None:
    logger.info("Cleaning up resources...")
    if ollama_was_run:
        stop_ollama()
    logger.info("Cleanup complete. Goodbye!")


def _get_model():
    return DEFAULT_MODEL


def handle_sigint(signum: int, frame: object) -> None:
    print()
    logging.warning("\nInterrupted. Exiting cleanly.")
    finish()
    raise SystemExit(0)


def rag_pipeline(client: OpenAI, query: str) -> str:
    model = _get_model()

    # Retrieve relevant documents from the database
    retrieved_docs = src.db_retrieve.search("man-db", client, query, 1)
    logger.debug(f"Retrieved docs: {retrieved_docs}")

    # Parse and process the query
    parsed_query = _parse_rag_prompt("./prompts/ret.txt", retrieved_docs, [query])
    logger.debug(f"Parsed query for LLM:\n{parsed_query}")

    logger.info("---")
    logger.info(f"Retrieved Documents:")
    for doc in retrieved_docs:
        logger.info(f"- {doc['path']}")
    logger.info("---")

    # Generate response using LLM with retrieved context
    response = src.llm.generate(client, model, parsed_query)

    response += "\n---\nRetrieved Documents:\n"
    for doc in retrieved_docs:
        response += doc["path"] + "\n"
    response += "---\n"
    return response


def loop() -> None:
    signal.signal(signal.SIGINT, handle_sigint)

    client = get_running_client()

    while True:
        user_query = ""
        try:
            user_query = input("Enter your query: ")
        except EOFError:
            logger.warning("\nNo input provided. Exiting.")
            break
        try:
            result = rag_pipeline(client, user_query)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            break

        print(result)

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level), 
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    loop()
    finish()

if __name__ == "__main__":
    main()
