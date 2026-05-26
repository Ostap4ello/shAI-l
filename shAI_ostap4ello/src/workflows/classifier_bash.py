from openai import OpenAI
import logging

from ..llm import generate

logger = logging.getLogger(__name__)

CLASSIFY_QUERY_FORMAT = """
Classify the following text as either "natural language" or "bash script".
Respond with ONLY one of the two options, without any additional text or
explanation.

Text:
%s

Classification:
"""

CLASSIFY_TRY_COUNT = 5

def classify_is_bash(client: OpenAI, model: str, query: str) -> bool:
    for i in range(CLASSIFY_TRY_COUNT):
        parsed_query = CLASSIFY_QUERY_FORMAT % query
        resp = generate(client, model, parsed_query)

        resp = resp.lower().strip()

        if resp == "natural language":
            return False
        elif resp == "bash script":
            return True

        logger.info(f"Could not parse answer. Retrying ({i})")

    raise RuntimeError("Could not parse LLM error when classifying")
