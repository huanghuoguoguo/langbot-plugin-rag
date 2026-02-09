"""Paragraph (standard) indexing strategy."""

import logging
from typing import TYPE_CHECKING

from langbot_plugin.api.entities.builtin.rag import IngestionContext
from langrag import RecursiveCharacterChunker, Document as LangRAGDocument

from ..constants import (
    IndexStrategy,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    EMBEDDING_BATCH_SIZE,
)
from .base import BaseIndexStrategy

if TYPE_CHECKING:
    from langbot_plugin.api.definition.plugin import BasePlugin

logger = logging.getLogger(__name__)


class ParagraphStrategy(BaseIndexStrategy):
    """Standard paragraph-based indexing using RecursiveCharacterChunker."""

    async def ingest(
        self,
        context: IngestionContext,
        source_doc: LangRAGDocument,
        collection_id: str,
        kb_id: str,
    ) -> int:
        """Process document using standard paragraph chunking."""
        doc_id = source_doc.metadata["document_id"]
        filename = source_doc.metadata["document_name"]

        # Get chunking parameters
        chunk_size = context.chunk_size or context.custom_settings.get("chunk_size") or DEFAULT_CHUNK_SIZE
        chunk_overlap = context.chunk_overlap or context.custom_settings.get("overlap") or DEFAULT_CHUNK_OVERLAP

        # Chunk using RecursiveCharacterChunker
        chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunk_docs = chunker.split([source_doc])

        if not chunk_docs:
            return 0

        # Embed in batches
        chunk_texts = [doc.page_content for doc in chunk_docs]
        vectors = await self._embed_in_batches(kb_id, chunk_texts)

        # Build metadata and upsert
        ids = [f"{doc_id}_{i}" for i in range(len(chunk_docs))]
        metadatas = [
            {
                "file_id": doc_id,
                "document_id": doc_id,
                "document_name": filename,
                "chunk_index": i,
                "text": chunk_doc.page_content,
                "index_strategy": IndexStrategy.PARAGRAPH.value,
            }
            for i, chunk_doc in enumerate(chunk_docs)
        ]

        await self.plugin.rag_vector_upsert(
            collection_id=collection_id,
            vectors=vectors,
            ids=ids,
            metadata=metadatas,
        )

        return len(chunk_docs)

    async def _embed_in_batches(self, kb_id: str, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches to avoid IPC timeouts."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            batch_vectors = await self.plugin.rag_embed_documents(kb_id, batch)
            vectors.extend(batch_vectors)
        return vectors
