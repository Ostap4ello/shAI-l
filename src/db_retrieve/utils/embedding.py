from typing import List

import numpy as np
from openai import OpenAI

import logging

logger = logging.getLogger(__name__)


def embed_strings(
    client: OpenAI,
    model: str,
    strings: List[str],
    batch_size: int,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    total = len(strings)
    for i in range(0, total, batch_size):
        batch = strings[i : i + batch_size]
        logger.debug(f"Sent batch {i}-{min(i + batch_size, total)}/{total} to embedding endpoint.")
        resp = client.embeddings.create(model=model, input=batch)
        logger.debug(f"Received batch.")
        batch_vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        vectors.extend(batch_vecs)

    return np.vstack(vectors)
