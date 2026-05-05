from typing import Iterable
from openai import OpenAI
import logging

from .utils.prompt import (
    get_doc_choice_prompt,
    get_single_doc_prompt,
    get_doc_choice_answer,
)

from .. import db
from .. import llm

logger = logging.getLogger(__name__)


def rag_pipeline(
    client: OpenAI, model: str, query: str, top_k: int = 5
) -> str | Iterable[str]:

    # Retrieve relevant documents from the database
    results = db.search("man-db", client, query, top_k)
    logger.info("---")
    logger.info(f"Retrieved Documents:")
    for path in results:
        logger.info(f"- {path['metadata']['path']}, dist={path['distance']:.4f}")
    logger.info("---")

    # Choose the most relevant document
    doc_paths = [doc["metadata"]["path"] for doc in results]
    chosen_doc_path = None
    for i in range(5):
        parsed_query = get_doc_choice_prompt(doc_paths, query)
        logger.debug(f"Parsed query for doc choice:\n{parsed_query[:1000]}...")
        logger.info(f"Choosing the most relevant document (attempt {i+1}/5)...")
        response = llm.generate(client, model, parsed_query)
        assert isinstance(response, str), "Expected response to be a string"
        response = get_doc_choice_answer(response)

        if response is None:
            logger.warning(
                f"({i+1}/5) LLM response is not a valid document path or 'None'."
            )
        elif response == "None":
            logger.warning(
                f"({i+1}/5) Retrieved documents may not be relevant to the query."
            )
            raise RuntimeError("Retrieved documents may not be relevant to the query.")
        else:
            chosen_doc_path = get_doc_choice_answer(response)
            logger.info(f"Chosen document: {chosen_doc_path}")
            break

    if not chosen_doc_path:
        logger.error("Failed to choose a valid document after 5 attempts. Exiting.")
        raise RuntimeError("Failed to choose a valid document after 5 attempts.")

    # Parse and process the query
    # TODO
    parsed_query = get_single_doc_prompt(chosen_doc_path, query)
    logger.debug(f"Parsed query for LLM:\n{parsed_query[:1000]}...")

    # Generate response using LLM with retrieved context
    response = llm.generate(client, model, parsed_query)
    assert isinstance(response, str), "Expected response to be a string"
    response += "\n---\n"
    response += f"Chosen retrieved document: {chosen_doc_path}\n"
    response += "Considered Documents:\n"
    for path in doc_paths:
        response += f"- {path}\n"
    response += "---\n"
    return response
