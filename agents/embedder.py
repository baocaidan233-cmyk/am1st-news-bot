from __future__ import annotations

from core.config import AppConfig
from core.openai_client import create_openai_client


class Embedder:
    def __init__(self, config: AppConfig) -> None:
        self._client = create_openai_client(config)
        self._model = config.openai.embedding_model

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(model=self._model, input=text)
        return resp.data[0].embedding

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """One request for a whole list, in the order given.

        The embeddings endpoint has always taken a list; the callers here were
        written as `[await embed(t) for t in texts]`, which is the same tokens
        spread over one round trip each. Measured 2026-09-28, a publish cycle
        did forty of those in series just to compute the cadence's trending
        term — four seconds of latency per cycle, all of it waiting.

        An empty input returns an empty list without calling anything, and a
        blank string is embedded as a single space: the endpoint rejects empty
        strings, and dropping the entry silently would misalign the result
        with the input, which is the one thing a caller indexing by position
        cannot survive."""
        if not texts:
            return []
        resp = await self._client.embeddings.create(
            model=self._model, input=[t if t.strip() else " " for t in texts])
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
