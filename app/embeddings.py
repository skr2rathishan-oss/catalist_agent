"""Retrieval text formatting and vector validation, independent of the provider."""

import math

FORMAT_VERSION = 2


def prepare_embedding_text(text, task):
    """Keep text intact; NVIDIA receives the retrieval mode as input_type."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Embedding text must not be empty")
    if task not in ("document", "query"):
        raise ValueError("Embedding task must be document or query")
    return text


def normalise_vector(values, dimensions):
    """Reject invalid vectors and return unit-length floats for cosine search."""

    if not isinstance(values, list) or len(values) != dimensions:
        raise ValueError("Embedding dimensions do not match the configured model")
    for value in values:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Embedding contains invalid numbers")
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding must have a finite nonzero length")
    return [value / norm for value in values]
