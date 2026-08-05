from __future__ import annotations

from openai import AsyncOpenAI

from core.config import AppConfig


class Embedder:
    def __init__(self, config: AppConfig) -> None:
        self._client = AsyncOpenAI(api_key=config.openai.api_key)
        self._model = config.openai.embedding_model

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.embeddings.create(model=self._model, input=text)
        return resp.data[0].embedding
