from typing import Iterable

from openai import OpenAI
from openai.types.responses import ResponseOutputItem, ResponseOutputMessage, ResponseOutputText, ResponseReasoningItem, ResponseTextDeltaEvent, ResponseTextDoneEvent

def generate(client: OpenAI, model: str, user_input: str, do_stream: bool = False) -> str:
    if not model:
        raise ValueError("Model must be set to generate a response.")

    if not user_input:
        return ""

    response = ""

    if do_stream:
        text_chunks: list[str] = []
        for chunk in _stream_response_chunks(model, user_input, client):
            if chunk:
                text_chunks.append(chunk)
                print(chunk, end="", flush=True)

        print()
        response = "".join(text_chunks)
    else:
        resp = client.responses.create(
            model=model,
            input=user_input,
            stream=False,
        )
        for event in resp.output:
            if isinstance(event, ResponseReasoningItem):
                pass
            elif isinstance(event, ResponseOutputMessage):
                for item in event.content:
                    if isinstance(item, ResponseOutputText):
                        response += item.text

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
