from typing import List

import numpy as np
from openai import OpenAI

import logging

logger = logging.getLogger(__name__)


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
