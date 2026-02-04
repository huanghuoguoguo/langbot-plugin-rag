import logging
from typing import Any, List

from langbot_plugin.api.definition.components.rag_engine import RAGEngine, RAGEngineCapability
from langbot_plugin.api.entities.builtin.rag import (
    IngestionContext,
    IngestionResult,
    RetrievalContext,
    RetrievalResponse,
    RetrievalResultEntry,
    DocumentStatus
)

logger = logging.getLogger(__name__)

class LangRAG(RAGEngine):
    """
    Official LangBot RAG Engine implementation using Plugin IPC.

    This is the default RAG engine shipped with LangBot, providing:
    - Document ingestion with chunking
    - Vector-based retrieval
    - Integration with Host's embedding models and vector database
    """

    @classmethod
    def get_capabilities(cls) -> list[str]:
        """Declare supported capabilities."""
        return [RAGEngineCapability.DOC_INGESTION]

    async def on_knowledge_base_create(self, kb_id: str, config: dict) -> None:
        """
        Called when a knowledge base using this engine is created.

        Args:
            kb_id: The UUID of the newly created knowledge base
            config: Creation settings from get_creation_settings_schema()
        """
        logger.info(f"Knowledge base created: {kb_id} with config: {config}")
        # The Host handles collection creation in the vector database.
        # This hook can be used for engine-specific initialization if needed.

    async def on_knowledge_base_delete(self, kb_id: str) -> None:
        """
        Called when a knowledge base using this engine is deleted.

        Args:
            kb_id: The UUID of the knowledge base being deleted
        """
        logger.info(f"Knowledge base deleted: {kb_id}")
        # The Host handles collection cleanup in the vector database.
        # This hook can be used for engine-specific cleanup if needed.

    async def ingest(self, context: IngestionContext) -> IngestionResult:
        """
        Handle document ingestion: Read -> Parse -> Chunk -> Embed -> Store.
        """
        logger.info(f"Ingesting file: {context.file_object.metadata.filename} into KB: {context.knowledge_base_id}")

        # 1. Get file content from Host
        storage_path = context.file_object.storage_path
        filename = context.file_object.metadata.filename
        content_bytes = b""
        try:
            # SDK's rag_get_file_stream already decodes base64 and returns bytes
            content_bytes = await self.plugin.rag_get_file_stream(storage_path)

        except Exception as e:
            logger.error(f"Failed to get file content: {e}")
            return IngestionResult(
                document_id=context.file_object.metadata.document_id,
                status=DocumentStatus.FAILED,
                error_message=f"Could not read file: {str(e)}"
            )

        try:
            # 2. Parse file content using FileParser
            from .core.parser import FileParser
            parser = FileParser()
            text_content = await parser.parse(content_bytes, filename)
            
            if not text_content:
                logger.warning(f"No text content extracted from file: {filename}")
                return IngestionResult(
                    document_id=context.file_object.metadata.document_id,
                    status=DocumentStatus.COMPLETED,
                    chunks_created=0
                )
            
            # 3. Chunk the text content
            chunk_size = context.chunk_size or 512
            chunks = [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]
            
            if not chunks:
                 return IngestionResult(
                    document_id=context.file_object.metadata.document_id,
                    status=DocumentStatus.COMPLETED,
                    chunks_created=0
                )

            # 3. Embed
            # self.plugin.rag_embed_documents(kb_id, texts)
            vectors = await self.plugin.rag_embed_documents(context.knowledge_base_id, chunks)
            
            # 4. Store
            doc_id = context.file_object.metadata.document_id
            # Generate deterministic IDs for chunks if possible, or random
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            
            metadatas = [{
                "document_id": doc_id,
                "chunk_index": i,
                "text": chunk
            } for i, chunk in enumerate(chunks)]
            
            # upsert(collection_id, vectors, ids, metadata)
            # Use knowledge_base_id as collection_id for isolation
            await self.plugin.rag_vector_upsert(
                collection_id=context.knowledge_base_id,
                vectors=vectors,
                ids=ids,
                metadata=metadatas
            )
            
            return IngestionResult(
                document_id=doc_id,
                status=DocumentStatus.COMPLETED,
                chunks_created=len(chunks)
            )
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return IngestionResult(
                document_id=context.file_object.metadata.document_id,
                status=DocumentStatus.FAILED,
                error_message=str(e)
            )

    async def retrieve(self, context: RetrievalContext) -> RetrievalResponse:
        """
        Retrieval logic.
        """
        query = context.query
        top_k = context.get_top_k()
        
        # 1. Embed query (Host handles model lookup by KB ID)
        query_vector = await self.plugin.rag_embed_query(context.knowledge_base_id, query)
        
        # 2. Search
        results = await self.plugin.rag_vector_search(
            collection_id=context.knowledge_base_id,
            query_vector=query_vector,
            top_k=top_k,
            filters=context.filters
        )
        
        # 3. Format results
        entries = []
        for res in results:
            # metadata usually contains 'text' if we stored it there
            content_text = res.get('metadata', {}).get('text', '')
            entries.append(RetrievalResultEntry(
                id=res['id'],
                content=[{"type": "text", "text": content_text}],
                metadata=res.get('metadata', {}),
                score=res.get('score'),
                distance=0.0
            ))
            
        return RetrievalResponse(
            results=entries,
            total_found=len(entries)
        )

    async def delete_document(self, kb_id: str, document_id: str) -> bool:
        # Delete by filter on document_id metadata
        result = await self.plugin.rag_vector_delete(
            collection_id=kb_id,
            filters={"document_id": document_id}
        )
        # Result is like {'count': N}
        return isinstance(result, dict) and result.get('count', 0) > 0

    def get_creation_settings_schema(self) -> list[dict]:
        return [
            {
                "name": "embedding_model_uuid",
                "label": {"en_US": "Embedding Model", "zh_Hans": "嵌入模型"},
                "description": {"en_US": "Select embedding model for text vectorization", "zh_Hans": "选择用于文本向量化的嵌入模型"},
                "type": "embedding-model-selector",
                "required": True,
                "default": "",
            },
            {
                "name": "chunk_size",
                "label": {"en_US": "Chunk Size", "zh_Hans": "分块大小"},
                "type": "integer",
                "required": False,
                "default": 512,
            },
            {
                "name": "overlap",
                "label": {"en_US": "Chunk Overlap", "zh_Hans": "分块重叠"},
                "type": "integer",
                "required": False,
                "default": 50,
            }
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
