import logging
import sys
import signal

from ..utils import is_ollama_running, start_ollama, stop_ollama
from ..config import load_config

logger = logging.getLogger(__name__)

DEFAULT_KEEP_OLLAMA = False


def ollama_check_or_run() -> bool:
    if is_ollama_running():
        logger.info("Ollama is running.")
        return False
    else:
        logger.info("Ollama is not running.")
        # TODO: check rather http socket, not just docker container status,
        # don't use utils function that checks docker container.
        print("Ollama docker service is not running. Start Ollama?")
        print(
            "[y]es / Yes and [k]eep Ollama running after this session / [n]o (default):"
        )

        choice = input().strip().lower()
        if choice == "y" or choice == "k":
            logger.info("Trying to start Ollama...")
            start_ollama()
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
    if stop_ollama_on_finish:
        stop_ollama()
    print("Goodbye!")

def loop() -> None:
    print("Entering main loop. Press Ctrl+C to exit.")
    while True:
        query = input(">> ")
        if query.strip().lower() in {"exit", "quit"}:
            print("Exiting...")
            break
        else:
            print(f"You entered: {query}")


def main(config_path: str, log_level: str = "WARNING") -> None:
    def handle_sigint(signum: int, frame: object) -> None:
        print("\nInterrupted. Exiting cleanly.", file=sys.stderr)
        cleanup()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    load_config(config_path, create=False)

    stop_ollama_on_finish = ollama_check_or_run()

    loop()

    cleanup(stop_ollama_on_finish)
