"""
HostEmbedderAdapter - Bridge LangRAG's BaseEmbedder to LangBot Host RPC.

This adapter implements LangRAG's BaseEmbedder interface by delegating
embedding requests to the LangBot Host via RPC calls.
"""

from typing import TYPE_CHECKING

from langrag.llm.embedder.base import BaseEmbedder

if TYPE_CHECKING:
    from langbot_plugin.api.definition.plugin import BasePlugin


class HostEmbedderAdapter(BaseEmbedder):
    """Adapter that uses LangBot Host's embedding capability.

    This adapter enables LangRAG to use the Host's configured embedding model
    without needing direct API keys or model configuration in the plugin.

    Example:
        embedder = HostEmbedderAdapter(plugin, kb_id="kb-123", dimension=1536)
        vectors = await embedder.embed_async(["Hello", "World"])
    """

    def __init__(
        self,
        plugin: "BasePlugin",
        kb_id: str,
        dimension: int = 1536,
    ):
        """Initialize the Host Embedder Adapter.

        Args:
            plugin: The plugin instance for accessing RPC methods.
            kb_id: Knowledge base ID to determine which embedding model to use.
            dimension: Expected embedding dimension (for validation).
        """
        self._plugin = plugin
        self._kb_id = kb_id
        self._dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings (sync version - not recommended for Host RPC).

        This method exists for interface compatibility but should not be used
        directly as it would block the event loop. Use embed_async() instead.

        Raises:
            RuntimeError: Always raises - use embed_async() instead.
        """
        raise RuntimeError(
            "HostEmbedderAdapter.embed() is not supported. "
            "Use embed_async() for async RPC calls to Host."
        )

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via Host RPC (async version).

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        return await self._plugin.rag_embed_documents(self._kb_id, texts)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension
