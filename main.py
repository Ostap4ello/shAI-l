from src.rag import _parse_rag_prompt
import src.llm
import src.db_retrieve
import sys
import signal

import os

from openai import OpenAI

DEFAULT_API_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "qwen3:1.7b"

def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = DEFAULT_API_KEY
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        base_url = DEFAULT_API_BASE_URL
    return OpenAI(api_key=api_key, base_url=base_url)

def _get_model():
    return DEFAULT_MODEL

def handle_sigint(signum: int, frame: object) -> None:
    print("\nInterrupted. Exiting cleanly.", file=sys.stderr)
    raise SystemExit(0)

def rag_pipeline(query: str) -> str:
    client = _get_client()
    model = _get_model()

    # Retrieve relevant documents from the database
    retrieved_docs = src.db_retrieve.search("man-db", client, query, 1)

    # Parse and process the query
    parsed_query = _parse_rag_prompt("./prompts/ret.txt", retrieved_docs, [query])

    dbg = "---\nRetrieved Documents:\n"
    for doc in retrieved_docs:
        dbg += doc['path']+"\n"
    print(dbg)
    print("---")

    # Generate response using LLM with retrieved context
    response = src.llm.generate(client, model, parsed_query)

    response += "\n\nRetrieved Documents:\n"
    for doc in retrieved_docs:
        response += doc['path']+"\n"
    return response

def main() -> None:
    signal.signal(signal.SIGINT, handle_sigint)
    while True:
        user_query = ""
        try:
            user_query = input("Enter your query: ")
        except EOFError:
            print("\nNo input provided. Exiting.")
            break
        result = rag_pipeline(user_query)
        print(result)

if __name__ == "__main__":
    main()
