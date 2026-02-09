"""Parent-Child hierarchical indexing strategy."""

import logging
from typing import TYPE_CHECKING

from langbot_plugin.api.entities.builtin.rag import IngestionContext, RetrievalResultEntry
from langrag import RecursiveCharacterChunker, Document as LangRAGDocument, DocumentType

from ..constants import (
    IndexStrategy,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_PARENT_CHUNK_SIZE,
    DEFAULT_CHILD_CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
)
from ..adapters import LangBotKVAdapter
from .base import BaseIndexStrategy

if TYPE_CHECKING:
    from langbot_plugin.api.definition.plugin import BasePlugin

logger = logging.getLogger(__name__)


class ParentChildStrategy(BaseIndexStrategy):
    """Parent-Child hierarchical indexing.

    Parents are stored in KV store, children are embedded and stored in VDB.
    Children reference their parent via 'parent_id' in metadata.
    """

    async def ingest(
        self,
        context: IngestionContext,
        source_doc: LangRAGDocument,
        collection_id: str,
        kb_id: str,
    ) -> int:
        """Process document using Parent-Child hierarchical chunking."""
        doc_id = source_doc.metadata["document_id"]
        filename = source_doc.metadata["document_name"]

        # Get chunking parameters
        parent_chunk_size = context.custom_settings.get("parent_chunk_size") or DEFAULT_PARENT_CHUNK_SIZE
        child_chunk_size = context.custom_settings.get("child_chunk_size") or DEFAULT_CHILD_CHUNK_SIZE
        chunk_overlap = context.custom_settings.get("overlap") or DEFAULT_CHUNK_OVERLAP

        # Create parent and child splitters
        parent_chunker = RecursiveCharacterChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        child_chunker = RecursiveCharacterChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap // 2,
        )

        # Split into parents
        parent_docs = parent_chunker.split([source_doc])
        if not parent_docs:
            return 0

        # Initialize KV adapter for parent storage
        kv_adapter = LangBotKVAdapter(self.plugin, namespace=f"pc:{kb_id}")

        # Process each parent
        all_children: list[LangRAGDocument] = []
        parents_kv_data: dict[str, str] = {}

        for parent_idx, parent_doc in enumerate(parent_docs):
            parent_id = f"{doc_id}_parent_{parent_idx}"
            parent_doc.id = parent_id
            parent_doc.type = DocumentType.PARENT

            # Store parent content for KV
            parents_kv_data[parent_id] = parent_doc.page_content

            # Split parent into children
            children = child_chunker.split([parent_doc])
            for child_idx, child in enumerate(children):
                child.id = f"{doc_id}_child_{parent_idx}_{child_idx}"
                child.type = DocumentType.CHUNK
                child.metadata["parent_id"] = parent_id
                child.metadata["document_id"] = doc_id
                child.metadata["document_name"] = filename
                child.metadata["file_id"] = doc_id
                all_children.append(child)

        if not all_children:
            return 0

        # Store parents in KV
        await kv_adapter.mset_async(parents_kv_data)
        logger.info(f"Stored {len(parents_kv_data)} parent chunks in KV store")

        # Embed children in batches
        child_texts = [c.page_content for c in all_children]
        vectors = await self._embed_in_batches(kb_id, child_texts)

        # Build metadata and upsert children to VDB
        ids = [c.id for c in all_children]
        metadatas = [
            {
                "file_id": doc_id,
                "document_id": doc_id,
                "document_name": filename,
                "parent_id": child.metadata["parent_id"],
                "text": child.page_content,
                "index_strategy": IndexStrategy.PARENT_CHILD.value,
            }
            for child in all_children
        ]

        await self.plugin.rag_vector_upsert(
            collection_id=collection_id,
            vectors=vectors,
            ids=ids,
            metadata=metadatas,
        )

        logger.info(f"Parent-Child indexing: {len(parent_docs)} parents, {len(all_children)} children")
        return len(all_children)

    async def retrieve(
        self,
        child_results: list[dict],
        kb_id: str,
    ) -> list[RetrievalResultEntry]:
        """Retrieve parent content for Parent-Child indexed results.

        Children are used for precise matching, but we return parent content
        for complete context. Deduplicates parents when multiple children match.
        """
        # Initialize KV adapter
        kv_adapter = LangBotKVAdapter(self.plugin, namespace=f"pc:{kb_id}")

        # Collect unique parent IDs while preserving order
        parent_ids_seen: set[str] = set()
        parent_ids_ordered: list[str] = []
        child_scores: dict[str, tuple[float, float]] = {}  # parent_id -> (best_score, distance)

        for res in child_results:
            parent_id = res.get("metadata", {}).get("parent_id")
            if parent_id and parent_id not in parent_ids_seen:
                parent_ids_seen.add(parent_id)
                parent_ids_ordered.append(parent_id)
                # Use first child's score as parent score (best match)
                child_scores[parent_id] = (res.get("score"), res.get("distance"))

        # Fetch parent content from KV
        parent_contents = await kv_adapter.mget_async(parent_ids_ordered)

        # Build result entries
        entries: list[RetrievalResultEntry] = []
        for i, parent_id in enumerate(parent_ids_ordered):
            parent_content = parent_contents[i]
            if parent_content is None:
                # Fallback: parent not found in KV, use child content
                logger.warning(f"Parent {parent_id} not found in KV, using child content")
                for res in child_results:
                    if res.get("metadata", {}).get("parent_id") == parent_id:
                        parent_content = res.get("metadata", {}).get("text", "")
                        break

            score, distance = child_scores.get(parent_id, (None, None))

            entries.append(
                RetrievalResultEntry(
                    id=parent_id,
                    content=[{"type": "text", "text": parent_content or ""}],
                    metadata={
                        "parent_id": parent_id,
                        "index_strategy": IndexStrategy.PARENT_CHILD.value,
                        "text": parent_content or "",
                    },
                    score=score,
                    distance=distance,
                )
            )

        return entries

    async def cleanup(self, kb_id: str, document_id: str) -> None:
        """Clean up Parent-Child KV data for a deleted document."""
        try:
            kv_adapter = LangBotKVAdapter(self.plugin, namespace=f"pc:{kb_id}")
            # Try to delete up to 100 potential parent keys
            parent_keys = [f"{document_id}_parent_{i}" for i in range(100)]
            await kv_adapter.delete_async(parent_keys)
            logger.debug(f"Cleaned up Parent-Child KV data for document {document_id}")
        except Exception as e:
            logger.warning(f"Failed to clean up Parent-Child KV data: {e}")

    async def _embed_in_batches(self, kb_id: str, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches to avoid IPC timeouts."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            batch_vectors = await self.plugin.rag_embed_documents(kb_id, batch)
            vectors.extend(batch_vectors)
        return vectors
