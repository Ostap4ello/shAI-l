import numpy as np
import os

from typing import Iterable, List
from openai import OpenAI
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

import logging

logger = logging.getLogger(__name__)


def get_client(base_url: str, api_key: str) -> OpenAI:
    env_api_key = os.environ.get("OPENAI_API_KEY")
    if env_api_key:
        api_key = env_api_key
    env_base_url = os.environ.get("OPENAI_BASE_URL")
    if env_base_url:
        base_url = env_base_url
    return OpenAI(api_key=api_key, base_url=base_url)


def generate(
    client: OpenAI, model: str, user_input: str, do_stream: bool = False
) -> str | Iterable[str]:
    if not model:
        logger.error("Model is not set. Please specify a model to generate a response.")
        raise ValueError("Model is not set.")

    if not user_input:
        return ""

    response = ""

    logger.debug(f"do_stream={do_stream}")
    logger.info(f"Generating response...")
    if do_stream:
        text_chunks: list[str] = []
        try:
            for chunk in _stream_response_chunks(model, user_input, client):
                if chunk:
                    text_chunks.append(chunk)
                    print(chunk, end="", flush=True)
            print()

        except Exception as e:
            logger.error(f"Error during streaming response: {e}")
            raise e

        response = "".join(text_chunks)
    else:
        try:
            resp = client.responses.create(
                model=model,
                input=user_input,
                stream=False,
            )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise e
        for event in resp.output:
            if isinstance(event, ResponseReasoningItem):
                pass
            elif isinstance(event, ResponseOutputMessage):
                for item in event.content:
                    if isinstance(item, ResponseOutputText):
                        response += item.text
    logger.debug(f"Generated response:\n{response}")

    return response


def _stream_response_chunks(
    model: str, user_input: str, client: OpenAI
) -> Iterable[str]:
    stream = client.responses.create(
        model=model,
        input=user_input,
        stream=True,
    )

    for event in stream:
        if isinstance(event, ResponseTextDoneEvent):
            break
        elif isinstance(event, ResponseTextDeltaEvent):
            if event.delta is not None:
                yield event.delta


def embed_string(
    client: OpenAI,
    model: str,
    string: str,
) -> np.ndarray:

    batch = [string]
    logger.debug(f"Embedding string")
    resp = client.embeddings.create(model=model, input=batch)
    logger.debug(f"Received embedding.")
    vector = np.array(resp.data[0].embedding, dtype=np.float32)

    return vector


def embed_strings(
    client: OpenAI,
    model: str,
    strings: List[str],
    batch_size: int = 0,
) -> np.ndarray:
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative.")
    elif batch_size == 0:
        batch_size = len(strings)

    vectors: List[np.ndarray] = []
    total = len(strings)

    for i in range(0, total, batch_size):
        batch = strings[i : i + batch_size]
        logger.debug(f"Embedding batch - {i}-{min(i+batch_size,total)}/{total})")
        resp = client.embeddings.create(model=model, input=batch)
        logger.debug(f"Received  embedding batch.")
        batch_vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        vectors.extend(batch_vecs)

    return np.vstack(vectors)
