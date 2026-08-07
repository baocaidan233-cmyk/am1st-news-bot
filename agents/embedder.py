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
