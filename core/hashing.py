from __future__ import annotations

import hashlib
from urllib.parse import urlparse, urlunparse


def sha256_url_hash(url: str) -> str:
    """SHA256 of the URL with query string/fragment stripped, full hex digest.

    Exact-duplicate dedup layer — URL only (not URL+title, not URL+image),
    per standing rule: a channel only ever compares against its own history.
    """
    parsed = urlparse(url)
    clean = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors, in [-1, 1]."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
