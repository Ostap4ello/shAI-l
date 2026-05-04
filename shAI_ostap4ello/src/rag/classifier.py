from openai import OpenAI
import logging

from .utils.prompt import get_classify_prompt

from ..llm import generate

logger = logging.getLogger(__name__)


def classify_is_bash(client: OpenAI, model: str, query: str) -> bool:
    for i in range(5):
        parsed_query = get_classify_prompt(query)
        resp = generate(client, model, parsed_query)

        resp = resp.lower().strip()

        if resp == "natural_language":
            return False
        elif resp == "bash_script":
            return True

        logger.info(f"Could not parse answer. Retrying ({i})")

    raise RuntimeError("Could not parse LLM error when classifying")


