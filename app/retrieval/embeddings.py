from abc import ABC, abstractmethod

import httpx


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_query(self, query: str) -> list[float]:
        vectors = await self.embed_texts([query])

        if not vectors:
            raise RuntimeError("Embedding provider returned no vector.")

        return vectors[0]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = ("https://api.openai.com/v1"),
        timeout_seconds: float = 30.0,
    ):
        if not api_key:
            raise ValueError("OpenAI API key is required.")

        if not model:
            raise ValueError("Embedding model is required.")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        clean_texts = [text.strip() for text in texts if text and text.strip()]

        if not clean_texts:
            return []

        url = f"{self.base_url}/embeddings"

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": (f"Bearer {self.api_key}"),
                    "Content-Type": ("application/json"),
                },
                json={"model": self.model, "input": clean_texts},
            )

            if response.status_code >= 400:
                raise RuntimeError(
                    f"Embedding API error "
                    f"{response.status_code}: "
                    f"{response.text}"
                )

            payload = response.json()

        data = payload.get("data", [])

        if len(data) != len(clean_texts):
            raise RuntimeError("Embedding response size " "does not match input size.")

        ordered = sorted(data, key=lambda item: item["index"])

        vectors: list[list[float]] = []

        for item in ordered:
            embedding = item.get("embedding")

            if not embedding:
                raise RuntimeError("Embedding response " "contains an empty vector.")

            vectors.append(embedding)

        return vectors
