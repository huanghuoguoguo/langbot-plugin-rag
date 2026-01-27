"""
HostVectorAdapter - Bridge LangRAG's BaseVector to LangBot Host RPC.

This adapter implements LangRAG's BaseVector interface by delegating
vector operations to the LangBot Host via RPC calls.

The Host manages the actual vector database (e.g., DuckDB, ChromaDB),
and this adapter provides a unified interface for LangRAG to interact with it.
"""

from typing import TYPE_CHECKING, Any

from langrag.datasource.vdb.base import BaseVector
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document

if TYPE_CHECKING:
    from langbot_plugin.api.definition.plugin import BasePlugin


class HostVectorAdapter(BaseVector):
    """Adapter that uses LangBot Host's vector database capability.

    This adapter enables LangRAG to use the Host's configured vector store
    without needing to manage database connections or configurations.

    The Host handles:
    - Vector storage backend selection
    - Index management
    - Hybrid search (vector + keyword) if supported

    Example:
        dataset = Dataset(id="kb-123", name="MyKB", collection_name="kb_123")
        vector_store = HostVectorAdapter(plugin, dataset)
        await vector_store.add_texts_async(documents)
        results = await vector_store.search_async("query", query_vector, top_k=5)
    """

    def __init__(
        self,
        plugin: "BasePlugin",
        dataset: Dataset,
    ):
        """Initialize the Host Vector Adapter.

        Args:
            plugin: The plugin instance for accessing RPC methods.
            dataset: Dataset configuration containing collection info.
        """
        super().__init__(dataset)
        self._plugin = plugin

    # ==================== Sync Methods (Not Supported) ====================

    def create(self, texts: list[Document], **kwargs) -> None:
        """Create collection - handled by Host, not supported via sync call."""
        raise RuntimeError(
            "HostVectorAdapter.create() is not supported. "
            "Host manages collection lifecycle."
        )

    def add_texts(self, texts: list[Document], **kwargs) -> None:
        """Add texts - use add_texts_async() instead."""
        raise RuntimeError(
            "HostVectorAdapter.add_texts() is not supported. "
            "Use add_texts_async() for async RPC calls to Host."
        )

    def search(
        self,
        query: str,
        query_vector: list[float] | None,
        top_k: int = 4,
        **kwargs
    ) -> list[Document]:
        """Search - use search_async() instead."""
        raise RuntimeError(
            "HostVectorAdapter.search() is not supported. "
            "Use search_async() for async RPC calls to Host."
        )

    def delete_by_ids(self, ids: list[str]) -> None:
        """Delete by IDs - use delete_by_ids_async() instead."""
        raise RuntimeError(
            "HostVectorAdapter.delete_by_ids() is not supported. "
            "Use delete_by_ids_async() for async RPC calls to Host."
        )

    def delete(self) -> None:
        """Delete collection - handled by Host, not supported via sync call."""
        raise RuntimeError(
            "HostVectorAdapter.delete() is not supported. "
            "Host manages collection lifecycle."
        )

    # ==================== Async Methods ====================

    async def create_async(self, texts: list[Document], **kwargs) -> None:
        """Create collection and add texts.

        Note: Collection creation is typically handled by Host when KB is created.
        This method just adds texts to the existing collection.
        """
        if texts:
            await self.add_texts_async(texts, **kwargs)

    async def add_texts_async(self, texts: list[Document], **kwargs) -> None:
        """Add documents to the vector store via Host RPC.

        Args:
            texts: List of Document objects with vectors and content.
            **kwargs: Additional arguments (unused).
        """
        if not texts:
            return

        # Prepare data for Host RPC
        vectors = []
        ids = []
        metadata = []

        for doc in texts:
            if doc.vector is None:
                raise ValueError(f"Document {doc.id} has no vector")

            vectors.append(doc.vector)
            ids.append(doc.id)
            meta = dict(doc.metadata)
            meta["text"] = doc.page_content  # Store text in metadata for retrieval
            metadata.append(meta)

        await self._plugin.rag_vector_upsert(
            collection_id=self.collection_name,
            vectors=vectors,
            ids=ids,
            metadata=metadata,
        )

    async def search_async(
        self,
        query: str,
        query_vector: list[float] | None,
        top_k: int = 4,
        **kwargs
    ) -> list[Document]:
        """Search for documents via Host RPC.

        Args:
            query: Text query for keyword/hybrid search.
            query_vector: Vector for similarity search.
            top_k: Number of results to return.
            **kwargs: Additional arguments:
                - search_type: "similarity", "keyword", or "hybrid"
                - filter: Metadata filters

        Returns:
            List of Document objects with scores in metadata.
        """
        search_type = kwargs.get("search_type", "similarity")
        filters = kwargs.get("filter") or kwargs.get("filters")

        results = await self._plugin.rag_vector_search(
            collection_id=self.collection_name,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            query_text=query,
            search_type=search_type,
        )

        # Convert Host results to Document objects
        documents = []
        for res in results:
            meta = res.get("metadata", {})
            content = meta.pop("text", "")  # Extract stored text

            doc = Document(
                id=res.get("id", ""),
                page_content=content,
                metadata={
                    **meta,
                    "score": res.get("score", 0.0),
                },
            )
            documents.append(doc)

        return documents

    async def delete_by_ids_async(self, ids: list[str]) -> None:
        """Delete documents by their IDs via Host RPC.

        Args:
            ids: List of document IDs to delete.
        """
        if not ids:
            return

        await self._plugin.rag_vector_delete(
            collection_id=self.collection_name,
            ids=ids,
        )

    async def delete_async(self) -> None:
        """Delete the entire collection.

        Note: Collection deletion is typically handled by Host when KB is deleted.
        This method deletes all documents in the collection.
        """
        # Delete all by passing no filters (Host should handle this)
        await self._plugin.rag_vector_delete(
            collection_id=self.collection_name,
        )
