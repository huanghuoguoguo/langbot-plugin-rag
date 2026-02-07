"""LangBot Vector Adapter - bridges LangRAG's BaseVector to Host RPC."""

import logging
from typing import TYPE_CHECKING, Any

from langrag.datasource.vdb.base import BaseVector
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document, DocumentType

if TYPE_CHECKING:
    from langbot_plugin.api.definition.plugin import BasePlugin

logger = logging.getLogger(__name__)


class LangBotVectorAdapter(BaseVector):
    """Adapter that bridges LangRAG's BaseVector interface to LangBot Host RPC.

    This adapter delegates vector database operations to the Host's VDB service
    via RPC calls, allowing LangRAG components to work seamlessly with LangBot's
    vector storage infrastructure.

    Example:
        dataset = Dataset(name="my-kb", collection_name="kb-123")
        vector_store = LangBotVectorAdapter(plugin, dataset)
        results = await vector_store.search_async("query", query_vector, top_k=5)
    """

    def __init__(self, plugin: "BasePlugin", dataset: Dataset):
        """Initialize the vector adapter.

        Args:
            plugin: The LangBot plugin instance for RPC calls.
            dataset: The dataset configuration containing collection info.
        """
        super().__init__(dataset)
        self.plugin = plugin

    # ==================== Sync Methods (Not Supported) ====================

    def create(self, texts: list[Document], **kwargs) -> None:
        """Sync create - not supported."""
        raise NotImplementedError("Use create_async() instead.")

    def add_texts(self, texts: list[Document], **kwargs) -> None:
        """Sync add_texts - not supported."""
        raise NotImplementedError("Use add_texts_async() instead.")

    def search(
        self,
        query: str,
        query_vector: list[float] | None,
        top_k: int = 4,
        **kwargs
    ) -> list[Document]:
        """Sync search - not supported."""
        raise NotImplementedError("Use search_async() instead.")

    def delete_by_ids(self, ids: list[str]) -> None:
        """Sync delete_by_ids - not supported."""
        raise NotImplementedError("Use delete_by_ids_async() instead.")

    def delete(self) -> None:
        """Sync delete - not supported."""
        raise NotImplementedError("Use delete_async() instead.")

    # ==================== Async Methods ====================

    async def create_async(self, texts: list[Document], **kwargs) -> None:
        """Create collection and add documents via Host RPC.

        Note: Host manages collection creation automatically.
        This method just adds the documents.
        """
        await self.add_texts_async(texts, **kwargs)

    async def add_texts_async(self, texts: list[Document], **kwargs) -> None:
        """Add documents to the vector store via Host RPC.

        Args:
            texts: List of Document objects to add.
            **kwargs: Additional arguments (ignored).
        """
        if not texts:
            return

        ids = []
        vectors = []
        metadatas = []

        for doc in texts:
            if doc.vector is None:
                logger.warning(f"Document {doc.id} has no vector, skipping")
                continue

            ids.append(doc.id)
            vectors.append(doc.vector)

            # Build metadata for Host
            metadata = dict(doc.metadata)
            metadata["text"] = doc.page_content
            metadata["type"] = doc.type.value
            metadatas.append(metadata)

        if not ids:
            return

        await self.plugin.rag_vector_upsert(
            collection_id=self.collection_name,
            vectors=vectors,
            ids=ids,
            metadata=metadatas,
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
            query: The raw text query (used for hybrid/keyword search).
            query_vector: The embedded query vector.
            top_k: Number of results to return.
            **kwargs:
                - search_type: "similarity", "keyword", "hybrid"
                - filter: Metadata filters (not yet supported by Host)

        Returns:
            List of Document objects matching the query.
        """
        if query_vector is None:
            logger.warning("query_vector is None, cannot perform vector search")
            return []

        search_type = kwargs.get("search_type", "similarity")

        # Call Host RPC
        results = await self.plugin.rag_vector_search(
            collection_id=self.collection_name,
            query_vector=query_vector,
            top_k=top_k,
            # Future: pass search_type and query_text when Host supports it
            # search_type=search_type,
            # query_text=query,
        )

        # Convert Host results to LangRAG Documents
        documents = []
        for res in results:
            metadata = res.get("metadata", {})
            text = metadata.pop("text", "")
            doc_type_str = metadata.pop("type", DocumentType.CHUNK.value)

            try:
                doc_type = DocumentType(doc_type_str)
            except ValueError:
                doc_type = DocumentType.CHUNK

            doc = Document(
                id=res["id"],
                page_content=text,
                metadata=metadata,
                type=doc_type,
                vector=None,  # Don't store vector in result
            )

            # Store score in metadata for downstream use
            if "score" in res:
                doc.set_meta("score", res["score"])
            if "distance" in res:
                doc.set_meta("distance", res["distance"])

            documents.append(doc)

        return documents

    async def delete_by_ids_async(self, ids: list[str]) -> None:
        """Delete documents by their IDs via Host RPC.

        Args:
            ids: List of document IDs to delete.
        """
        if not ids:
            return

        await self.plugin.rag_vector_delete(
            collection_id=self.collection_name,
            ids=ids,
        )

    async def delete_async(self) -> None:
        """Delete the entire collection.

        Note: This is typically handled by Host when KB is deleted.
        """
        logger.warning(
            "delete_async called on LangBotVectorAdapter. "
            "Collection deletion should be handled by Host."
        )
