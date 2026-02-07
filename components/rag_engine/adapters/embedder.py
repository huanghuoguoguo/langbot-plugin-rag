"""LangBot Embedder Adapter - bridges LangRAG's BaseEmbedder to Host RPC."""

from typing import TYPE_CHECKING

from langrag.llm.embedder.base import BaseEmbedder

if TYPE_CHECKING:
    from langbot_plugin.api.definition.plugin import BasePlugin


class LangBotEmbedderAdapter(BaseEmbedder):
    """Adapter that bridges LangRAG's BaseEmbedder interface to LangBot Host RPC.

    This adapter delegates embedding operations to the Host's embedding service
    via RPC calls, allowing LangRAG components to work seamlessly with LangBot's
    embedding infrastructure.

    Example:
        embedder = LangBotEmbedderAdapter(plugin, kb_id="kb-123")
        vectors = await embedder.embed_async(["Hello world"])
    """

    def __init__(self, plugin: "BasePlugin", kb_id: str):
        """Initialize the embedder adapter.

        Args:
            plugin: The LangBot plugin instance for RPC calls.
            kb_id: Knowledge base ID used to select the embedding model on Host.
        """
        self.plugin = plugin
        self.kb_id = kb_id
        self._dimension: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Sync embedding - not supported, use embed_async instead.

        Raises:
            NotImplementedError: Always, as Host RPC is async-only.
        """
        raise NotImplementedError(
            "LangBotEmbedderAdapter only supports async operations. "
            "Use embed_async() instead."
        )

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via Host RPC.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (same order as input).
        """
        if not texts:
            return []

        vectors = await self.plugin.rag_embed_documents(self.kb_id, texts)

        # Detect dimension from first result
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])

        return vectors

    async def embed_query_async(self, query: str) -> list[float]:
        """Embed a single query text via Host RPC.

        Args:
            query: The query text to embed.

        Returns:
            The embedding vector for the query.
        """
        vector = await self.plugin.rag_embed_query(self.kb_id, query)

        # Detect dimension
        if vector and self._dimension is None:
            self._dimension = len(vector)

        return vector

    @property
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            Size of embedding vectors. Returns 0 if not yet detected.
        """
        return self._dimension or 0
