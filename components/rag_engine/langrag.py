"""LangRAG Engine - Official LangBot RAG Engine implementation.

This module provides the main RAG engine class that orchestrates:
- Document ingestion with configurable indexing strategies
- Vector-based retrieval with strategy-aware post-processing
- Full integration with Host's embedding models and vector database
"""

import logging

from langbot_plugin.api.definition.components.rag_engine import RAGEngine, RAGEngineCapability
from langbot_plugin.api.entities.builtin.rag import (
    IngestionContext,
    IngestionResult,
    RetrievalContext,
    RetrievalResponse,
    RetrievalResultEntry,
    DocumentStatus,
)
from langrag import Document as LangRAGDocument, DocumentType

from .constants import IndexStrategy
from .schemas import get_creation_settings_schema, get_retrieval_settings_schema
from .parser import FileParser
from .strategies import ParagraphStrategy, ParentChildStrategy, QAStrategy

logger = logging.getLogger(__name__)


class LangRAG(RAGEngine):
    """Official LangBot RAG Engine implementation using Plugin IPC.

    This is the default RAG engine shipped with LangBot, providing:
    - Document ingestion with parsing, chunking, embedding, and vector storage
    - Multiple indexing strategies: Paragraph, Parent-Child, QA
    - Vector-based retrieval with strategy-aware post-processing
    - Full integration with Host's embedding models and vector database
    """

    @classmethod
    def get_capabilities(cls) -> list[str]:
        """Declare supported capabilities."""
        return [RAGEngineCapability.DOC_INGESTION]

    # ========== Lifecycle Hooks ==========

    async def on_knowledge_base_create(self, kb_id: str, config: dict) -> None:
        logger.info(f"Knowledge base created: {kb_id} with config: {config}")

    async def on_knowledge_base_delete(self, kb_id: str) -> None:
        logger.info(f"Knowledge base deleted: {kb_id}")

    # ========== Core Methods ==========

    async def ingest(self, context: IngestionContext) -> IngestionResult:
        """Handle document ingestion with configurable indexing strategy.

        Supports three strategies:
        - PARAGRAPH: Standard recursive character chunking (default)
        - PARENT_CHILD: Hierarchical indexing with parents in KV, children in VDB
        - QA: LLM-generated questions indexed for better semantic matching
        """
        doc_id = context.file_object.metadata.document_id
        filename = context.file_object.metadata.filename
        collection_id = context.get_collection_id()
        kb_id = context.knowledge_base_id

        # Get index strategy from settings
        index_strategy = context.custom_settings.get("index_strategy", IndexStrategy.PARAGRAPH.value)
        logger.info(f"Ingesting file: {filename} (doc={doc_id}) with strategy: {index_strategy}")

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

            # 3. Create source document
            source_doc = LangRAGDocument(
                page_content=text_content,
                metadata={
                    "file_id": doc_id,
                    "document_id": doc_id,
                    "document_name": filename,
                },
                type=DocumentType.ORIGINAL,
            )

            # 4. Dispatch to appropriate indexing strategy
            strategy = self._get_strategy(index_strategy)
            chunks_created = await strategy.ingest(context, source_doc, collection_id, kb_id)

            logger.info(f"Ingestion complete: {chunks_created} chunks created for {filename}")
            return IngestionResult(
                document_id=doc_id,
                status=DocumentStatus.COMPLETED,
                chunks_created=chunks_created,
            )

        except Exception as e:
            logger.error(f"Ingestion failed for {filename}: {e}")
            return IngestionResult(
                document_id=doc_id,
                status=DocumentStatus.FAILED,
                error_message=str(e),
            )

    async def retrieve(self, context: RetrievalContext) -> RetrievalResponse:
        """Retrieve relevant content with strategy-aware post-processing.

        For Parent-Child indexed content:
        - Searches children in VDB (precise matching)
        - Returns parent content from KV store (complete context)
        - Deduplicates if multiple children share same parent
        """
        query = context.query
        top_k = context.get_top_k()
        collection_id = context.get_collection_id()
        kb_id = context.knowledge_base_id

        # 1. Embed query
        query_vector = await self.plugin.rag_embed_query(kb_id, query)

        # 2. Vector search
        results = await self.plugin.rag_vector_search(
            collection_id=collection_id,
            query_vector=query_vector,
            top_k=top_k,
        )

        if not results:
            return RetrievalResponse(results=[], total_found=0)

        # 3. Check if results are from Parent-Child indexing
        first_result_strategy = results[0].get("metadata", {}).get("index_strategy")

        if first_result_strategy == IndexStrategy.PARENT_CHILD.value:
            strategy = ParentChildStrategy(self.plugin)
            entries = await strategy.retrieve(results, kb_id)
        else:
            # Standard retrieval (PARAGRAPH or QA)
            entries = self._format_standard_results(results)

        return RetrievalResponse(results=entries, total_found=len(entries))

    async def delete_document(self, kb_id: str, document_id: str) -> bool:
        """Delete a document's vectors and associated data.

        For Parent-Child indexed documents, also cleans up parent data from KV store.
        """
        # Delete vectors from VDB
        count = await self.plugin.rag_vector_delete(
            collection_id=kb_id,
            ids=[document_id],
        )

        # Best-effort cleanup of Parent-Child KV data
        strategy = ParentChildStrategy(self.plugin)
        await strategy.cleanup(kb_id, document_id)

        return count > 0

    # ========== Schema Definitions ==========

    def get_creation_settings_schema(self) -> list[dict]:
        return get_creation_settings_schema()

    def get_retrieval_settings_schema(self) -> list[dict]:
        return get_retrieval_settings_schema()

    # ========== Private Helpers ==========

    def _get_strategy(self, strategy_name: str):
        """Get the appropriate indexing strategy instance."""
        if strategy_name == IndexStrategy.PARENT_CHILD.value:
            return ParentChildStrategy(self.plugin)
        elif strategy_name == IndexStrategy.QA.value:
            return QAStrategy(self.plugin)
        else:
            return ParagraphStrategy(self.plugin)

    def _format_standard_results(self, results: list[dict]) -> list[RetrievalResultEntry]:
        """Format standard retrieval results (PARAGRAPH or QA)."""
        entries: list[RetrievalResultEntry] = []
        for res in results:
            content_text = res.get("metadata", {}).get("text", "")
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
        return entries
