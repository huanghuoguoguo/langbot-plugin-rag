import logging
from typing import Any

from langbot_plugin.api.definition.components.rag_engine import RAGEngine, RAGEngineCapability
from langbot_plugin.api.entities.builtin.rag import (
    IngestionContext,
    IngestionResult,
    RetrievalContext,
    RetrievalResponse,
    RetrievalResultEntry,
    DocumentStatus,
)

from .parser import FileParser

logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
# Batch size for embedding API calls to avoid IPC timeouts
EMBEDDING_BATCH_SIZE = 10


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks.

    Uses a simple sliding-window approach. For production use cases requiring
    smarter splitting (by sentence/paragraph boundaries), consider using
    langchain_text_splitters or similar libraries.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        logger.warning(
            f"chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size}), "
            "clamping overlap to chunk_size - 1"
        )
        chunk_overlap = chunk_size - 1

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        # Stop once we've captured the tail
        if start + chunk_size >= len(text):
            break
    return chunks


class LangRAG(RAGEngine):
    """Official LangBot RAG Engine implementation using Plugin IPC.

    This is the default RAG engine shipped with LangBot, providing:
    - Document ingestion with parsing, chunking, embedding, and vector storage
    - Vector-based retrieval
    - Full integration with Host's embedding models and vector database
    """

    @classmethod
    def get_capabilities(cls) -> list[str]:
        """Declare supported capabilities."""
        return [RAGEngineCapability.DOC_INGESTION]

    # ========== Lifecycle Hooks ==========

    async def on_knowledge_base_create(self, kb_id: str, config: dict) -> None:
        logger.info(f"Knowledge base created: {kb_id} with config: {config}")
        # Host handles vector collection creation; hook available for engine-specific init.

    async def on_knowledge_base_delete(self, kb_id: str) -> None:
        logger.info(f"Knowledge base deleted: {kb_id}")
        # Host handles vector collection cleanup; hook available for engine-specific cleanup.

    # ========== Core Methods ==========

    async def ingest(self, context: IngestionContext) -> IngestionResult:
        """Handle document ingestion: Read -> Parse -> Chunk -> Embed -> Store."""
        doc_id = context.file_object.metadata.document_id
        filename = context.file_object.metadata.filename
        collection_id = context.get_collection_id()

        logger.info(f"Ingesting file: {filename} (doc={doc_id}) into collection: {collection_id}")

        # 1. Get file content from Host
        try:
            content_bytes = await self.plugin.rag_get_file_stream(context.file_object.storage_path)
        except Exception as e:
            logger.error(f"Failed to get file content: {e}")
            return IngestionResult(
                document_id=doc_id,
                status=DocumentStatus.FAILED,
                error_message=f"Could not read file: {e}",
            )

        try:
            # 2. Parse file content
            parser = FileParser()
            text_content = await parser.parse(content_bytes, filename)

            if not text_content:
                logger.warning(f"No text content extracted from file: {filename}")
                return IngestionResult(
                    document_id=doc_id,
                    status=DocumentStatus.COMPLETED,
                    chunks_created=0,
                )

            # 3. Chunk with overlap
            chunk_size = context.chunk_size or context.custom_settings.get("chunk_size") or DEFAULT_CHUNK_SIZE
            chunk_overlap = context.chunk_overlap or context.custom_settings.get("overlap") or DEFAULT_CHUNK_OVERLAP
            chunks = _chunk_text(text_content, chunk_size, chunk_overlap)

            if not chunks:
                return IngestionResult(
                    document_id=doc_id,
                    status=DocumentStatus.COMPLETED,
                    chunks_created=0,
                )

            # 4. Embed in batches to avoid IPC timeouts
            vectors: list[list[float]] = []
            for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
                batch_vectors = await self.plugin.rag_embed_documents(context.knowledge_base_id, batch)
                vectors.extend(batch_vectors)

            # 5. Build metadata and upsert to vector store
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "file_id": doc_id,       # Required by Host's delete_by_file_id
                    "document_id": doc_id,    # Application-level reference
                    "document_name": filename,
                    "chunk_index": i,
                    "text": chunk,
                }
                for i, chunk in enumerate(chunks)
            ]

            await self.plugin.rag_vector_upsert(
                collection_id=collection_id,
                vectors=vectors,
                ids=ids,
                metadata=metadatas,
            )

            return IngestionResult(
                document_id=doc_id,
                status=DocumentStatus.COMPLETED,
                chunks_created=len(chunks),
            )

        except Exception as e:
            logger.error(f"Ingestion failed for {filename}: {e}")
            return IngestionResult(
                document_id=doc_id,
                status=DocumentStatus.FAILED,
                error_message=str(e),
            )

    async def retrieve(self, context: RetrievalContext) -> RetrievalResponse:
        """Retrieve relevant content: Embed query -> Vector search -> Format results."""
        query = context.query
        top_k = context.get_top_k()
        collection_id = context.get_collection_id()

        # 1. Embed query (Host selects the embedding model by KB ID)
        query_vector = await self.plugin.rag_embed_query(context.knowledge_base_id, query)

        # 2. Vector search
        results = await self.plugin.rag_vector_search(
            collection_id=collection_id,
            query_vector=query_vector,
            top_k=top_k,
        )

        # 3. Format results
        entries: list[RetrievalResultEntry] = []
        for res in results:
            content_text = res.get("metadata", {}).get("text", "")
            # 'score' from mgr.py is distance (lower = more similar)
            raw_score = res.get("score")
            distance = res.get("distance", raw_score)

            entries.append(
                RetrievalResultEntry(
                    id=res["id"],
                    content=[{"type": "text", "text": content_text}],
                    metadata=res.get("metadata", {}),
                    score=raw_score,
                    distance=distance,
                )
            )

        return RetrievalResponse(results=entries, total_found=len(entries))

    async def delete_document(self, kb_id: str, document_id: str) -> bool:
        """Delete a document's vectors by file_id.

        Uses ids parameter (treated as file_id by Host's delete_by_file_id)
        instead of filters (which are not supported by Core).
        """
        count = await self.plugin.rag_vector_delete(
            collection_id=kb_id,
            ids=[document_id],
        )
        # rag_vector_delete returns int (number of deleted items)
        return count > 0

    # ========== Schema Definitions ==========

    def get_creation_settings_schema(self) -> list[dict]:
        return [
            {
                "name": "embedding_model_uuid",
                "label": {"en_US": "Embedding Model", "zh_Hans": "嵌入模型"},
                "description": {
                    "en_US": "Select embedding model for text vectorization",
                    "zh_Hans": "选择用于文本向量化的嵌入模型",
                },
                "type": "embedding-model-selector",
                "required": True,
                "default": "",
            },
            {
                "name": "chunk_size",
                "label": {"en_US": "Chunk Size", "zh_Hans": "分块大小"},
                "type": "integer",
                "required": False,
                "default": DEFAULT_CHUNK_SIZE,
            },
            {
                "name": "overlap",
                "label": {"en_US": "Chunk Overlap", "zh_Hans": "分块重叠"},
                "type": "integer",
                "required": False,
                "default": DEFAULT_CHUNK_OVERLAP,
            },
        ]

    def get_retrieval_settings_schema(self) -> list[dict]:
        return [
            {
                "name": "top_k",
                "label": {"en_US": "Top K", "zh_Hans": "召回数量"},
                "type": "integer",
                "required": False,
                "default": 5,
            }
        ]
