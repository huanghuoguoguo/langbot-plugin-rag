import logging
from typing import Any, List

from langbot_plugin.api.definition.components.knowledge_retriever import RAGEngine
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
    """

    async def ingest(self, context: IngestionContext) -> IngestionResult:
        """
        Handle document ingestion: Read -> Parse -> Chunk -> Embed -> Store.
        """
        logger.info(f"Ingesting file: {context.file_object.metadata.filename} into KB: {context.knowledge_base_id}")

        # 1. Get file content from Host
        storage_path = context.file_object.storage_path
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
            # 2. Parse and Chunk
            text_content = ""
            try:
                text_content = content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                text_content = "Binary content placeholder" 
            
            # Simple chunking
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

    def get_creation_settings_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "chunk_size": {
                    "type": "integer",
                    "default": 512,
                    "title": "Chunk Size"
                },
                "overlap": {
                    "type": "integer",
                    "default": 50,
                    "title": "Chunk Overlap"
                }
            }
        }

    def get_retrieval_settings_schema(self) -> dict:
        return {
             "type": "object",
            "properties": {
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "title": "Top K"
                }
            }
        }
