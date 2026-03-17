from typing import Iterable

from openai import OpenAI
from openai.types.responses import (
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

import logging

logger = logging.getLogger(__name__)


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
