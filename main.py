from src.rag import get_docchoice_prompt, get_singledoc_prompt, get_docchoice_answer
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


def rag_pipeline(client: OpenAI, query: str, top_k:int = 5) -> str:
    model = _get_model()

    # Retrieve relevant documents from the database
    results = src.db_retrieve.search("man-db", client, query, top_k)
    logger.info("---")
    logger.info(f"Retrieved Documents:")
    for path in results:
        logger.info(f"- {path['metadata']['path']}, dist={path['distance']:.4f}")
    logger.info("---")

    # Choose the most relevant document
    doc_paths = [doc["metadata"]["path"] for doc in results]
    chosen_doc_path = None
    for i in range(5):
        parsed_query = get_docchoice_prompt("./prompts/choosing-from-docs.txt", doc_paths, query)
        logger.debug(f"Parsed query for doc choice:\n{parsed_query[:1000]}...")
        logger.info(f"Choosing the most relevant document (attempt {i+1}/5)...")
        response = src.llm.generate(client, model, parsed_query)
        response = get_docchoice_answer(response)

        if response is None:
            logger.warning(f"({i+1}/5) LLM response is not a valid document path or 'None'.")
        elif response == "None":
            logger.warning(f"({i+1}/5) Retrieved documents may not be relevant to the query.")
            raise RuntimeError("Retrieved documents may not be relevant to the query.")
        else:
            chosen_doc_path = get_docchoice_answer(response)
            logger.info(f"Chosen document: {chosen_doc_path}")
            break

    if not chosen_doc_path:
        logger.error("Failed to choose a valid document after 5 attempts. Exiting.")
        raise RuntimeError("Failed to choose a valid document after 5 attempts.")

    # Parse and process the query
    # TODO
    parsed_query = get_singledoc_prompt("./prompts/ret.txt", chosen_doc_path, query)
    logger.debug(f"Parsed query for LLM:\n{parsed_query[:1000]}...")

    # Generate response using LLM with retrieved context
    response = src.llm.generate(client, model, parsed_query)

    response += "\n---\n"
    response += f"Chosen retrieved document: {chosen_doc_path}\n"
    response += "Considered Documents:\n"
    for path in doc_paths:
        response += f"- {path}\n"
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
            result = rag_pipeline(client, user_query, 5)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            break

        print(f"\n{result}\n")

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
