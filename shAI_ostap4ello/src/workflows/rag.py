from pathlib import Path
from openai import OpenAI
import logging

RAG_QUERY_FORMAT = """
You are efficient information retriever. Your task is to answer the questions using only the retrieved documents. Keep answers brief and include citations by quoting the exact relevant lines from the retrieved data in brackets.

Retrieved data:
%s

Question:
%s

Answer:
"""

from .. import db
from .. import llm

logger = logging.getLogger(__name__)


def load_related_documents(search_results: list[dict]) -> list[tuple[float, str, str]]:
    """
    For different outputs of db, builds a list of (dist, path, content) for the
    retrieved documents. If the retrieved document is in sections, the content
    is the sorted concatenation of the sections (with line `...` as separator)).
    """
    ranges_by_path = {}
    min_distances_by_path = {}
    for entry in search_results:
        path = entry["metadata"]["path"]
        distance = entry["distance"]
        if (
            not "path" in min_distances_by_path
            or distance < min_distances_by_path[path]
        ):
            min_distances_by_path[path] = distance

        if not path in ranges_by_path:
            ranges_by_path[path] = []
        elif ranges_by_path[path] == "full":
            continue

        if "from" in entry["metadata"] and "to" in entry["metadata"]:
            ranges_by_path[path].append(
                (entry["metadata"]["from"], entry["metadata"]["to"])
            )
        elif "from" in entry["metadata"] or "to" in entry["metadata"]:
            logger.error(
                f"Document {path} has only one of 'from' or 'to' metadata"
                " fields, skipping."
            )
        else:
            ranges_by_path[path] = "full"

    loaded_documents = []
    for path, ranges in ranges_by_path.items():
        lines = []
        try:
            lines = open(path, "r", encoding="utf-8").read()
        except Exception as e:
            logger.error(f"Error loading document {path}: {e}")
            continue

        if ranges == "full":
            loaded_documents.append((min_distances_by_path[path], path, "".join(lines)))
        else:
            ranges = sorted(ranges, key=lambda x: x[0])

            content = ""
            for from_line, to_line in ranges:
                if content != "":
                    content += "\n...\n"
                    content += "".join(lines[from_line:to_line])
            loaded_documents.append((min_distances_by_path[path], path, content))

    loaded_documents = sorted(loaded_documents, key=lambda x: x[0])
    return loaded_documents


# def choose_doc(results: list[dict], query: str, client: OpenAI, model: str) -> str:
#     # Choose the most relevant document
#     doc_paths = [doc["metadata"]["path"] for doc in results]
#     chosen_doc_path = None
#     for i in range(5):
#         parsed_query = get_doc_choice_prompt(doc_paths, query)
#         logger.debug(f"Parsed query for doc choice:\n{parsed_query[:1000]}...")
#         logger.info(f"Choosing the most relevant document (attempt {i+1}/5)...")
#         response = llm.generate(client, model, parsed_query)
#         assert isinstance(response, str), "Expected response to be a string"
#         response = get_doc_choice_answer(response)
#
#         if response is None:
#             logger.warning(
#                 f"({i+1}/5) LLM response is not a valid document path or 'None'."
#             )
#         elif response == "None":
#             logger.warning(
#                 f"({i+1}/5) Retrieved documents may not be relevant to the query."
#             )
#             raise RuntimeError("Retrieved documents may not be relevant to the query.")
#         else:
#             chosen_doc_path = get_doc_choice_answer(response)
#             logger.info(f"Chosen document: {chosen_doc_path}")
#             break
#
#     if not chosen_doc_path:
#         logger.error("Failed to choose a valid document after 5 attempts. Exiting.")
#         raise RuntimeError("Failed to choose a valid document after 5 attempts.")
#     pass
#     # Parse and process the que
#     # TODO
#     parsed_query = get_single_doc_prompt(chosen_doc_path, query)
#     logger.debug(f"Parsed query for LLM:\n{parsed_query[:1000]}...")
#
#     # Generate response using LLM with retrieved context
#     response = llm.generate(client, model, parsed_query)
#     assert isinstance(response, str), "Expected response to be a string"
#     response += "\n---\n"
#     response += f"Chosen retrieved document: {chosen_doc_path}\n"
#     response += "Considered Documents:\n"
#     for path in doc_paths:
#         response += f"- {path}\n"
#     response += "---\n"
#     return response


def answer_on_db_results(
    search_results: list[dict], query: str, client: OpenAI, gen_model: str
) -> str:
    loaded_documents = load_related_documents(search_results)
    parsed_query = RAG_QUERY_FORMAT % (
        "\n\n".join(
            [
                f"Document: {path}\n```\n{content}\n```\n"
                for _, path, content in loaded_documents
            ]
        ),
        query,
    )

    logger.info(f"Answering question using retrieved documents.")
    logger.debug(f"Parsed query for LLM:\n{parsed_query[:1000]}...")
    response = llm.generate(client, gen_model, parsed_query)

    return response


def rag_simple(
    client: OpenAI,
    gen_model: str,
    query: str,
    db_path: str,
    index_path_within_db: str,
    top_k: int,
) -> str:

    print(db_path, index_path_within_db)
    results = db.search(
        db_path, client, query, top_k, index_path_within_db=index_path_within_db
    )

    epilogue = "Retrieved documents (with distances):\n"
    for entry in results:
        epilogue += f"- {entry['metadata']['path']}, dist={entry['distance']:.4f}\n"
        if "from" in entry["metadata"] and "to" in entry["metadata"]:
            epilogue += f"  (section from line {entry['metadata']['from']} to {entry['metadata']['to']})\n"

    logger.info(epilogue)

    response = answer_on_db_results(results, query, client, gen_model)

    response += "\n---\n"
    response += epilogue

    return response
