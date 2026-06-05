import logging

from ..utils import is_ollama_running, start_ollama, stop_ollama
from ..llm import get_client
from . import classify_is_bash
from ..config import get_config_value

logger = logging.getLogger(__name__)

DEFAULT_KEEP_OLLAMA = False


def ollama_check_or_run() -> bool:
    if is_ollama_running(get_config_value("utils", "ollama_container_name")):
        logger.info("Ollama is running.")
        return False
    else:
        logger.info("Ollama is not running.")
        print("Ollama docker service is not running. Start Ollama?")
        print(
            "[y]es / Yes and [k]eep Ollama running after this session / [n]o (default):"
        )

        choice = input().strip().lower()
        if choice == "y" or choice == "k":
            logger.info("Trying to start Ollama...")
            gpus = get_config_value("utils", "ollama_gpus", str)
            context_length = get_config_value("utils", "ollama_context_length", int)
            create = True  # Always create a new container for this session
            name = get_config_value("utils", "ollama_container_name", str)
            start_ollama(
                context_length=context_length, gpus=gpus, name=name, create=create
            )
            print("Ollama started successfully.")
            if choice == "k":
                return False
            else:
                return True
        else:
            logger.error("Ollama is required to run this application. Exiting.")
            raise SystemExit(0)


def cleanup(stop_ollama_on_finish: bool = False) -> None:
    logger.info("Cleaning up resources...")
    name = get_config_value("utils", "ollama_container_name", str)
    remove = True  # Always create a new container for this session
    if stop_ollama_on_finish:
        stop_ollama(name=name, remove=remove)
    print("Goodbye!")


def loop() -> None:
    print(
        "NOTE: this idea was left as a playground for testing the LLM-based"
        " bash classification. It is not full-featured shell wrapper and does not"
        " execute any commands."
    )
    print("Entering main loop. Press Ctrl+C to exit.")
    model = get_config_value("llm", "model")
    base_url = get_config_value("llm", "api_base_url")
    api_key = get_config_value("llm", "api_key")
    client = get_client(base_url, api_key)
    while True:
        try:
            query = input(">> ")
        except EOFError:
            print("\nExiting...")
            break
        if query.strip().lower() in {"exit", "quit"}:
            print("Exiting...")
            break
        else:
            print(f"You entered: {query}")
            try:
                if classify_is_bash(client, model, query):
                    print("Your query was classified as bash")
                else:
                    print("Your query was classified as natural language")
            except Exception as e:
                print(f"Error happened while processing the query: {e}")


def interpreter() -> None:
    stop_ollama_on_finish = ollama_check_or_run()
    loop()
    cleanup(stop_ollama_on_finish)
