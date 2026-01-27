"""
LangRAG - Official LangBot RAG Engine powered by the LangRAG library.

This RAG engine uses the LangRAG library for advanced RAG capabilities:
- Multi-format document parsing (PDF, DOCX, MD, HTML)
- Smart chunking (recursive character, semantic)
- Hybrid search (vector + keyword with RRF fusion)
- Query rewriting and reranking

All infrastructure (LLM, Embedding, VDB) is provided by the LangBot Host
through adapter implementations.
"""

import logging
from typing import Any

from langbot_plugin.api.definition.components.rag_engine import RAGEngine, RAGEngineCapability
from langbot_plugin.api.entities.builtin.rag import (
    IngestionContext,
    IngestionResult,
    RetrievalContext,
    RetrievalResponse,
    RetrievalResultEntry,
    DocumentStatus
)

# LangRAG library imports
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.index_processor.splitter.providers.recursive_character import RecursiveCharacterChunker
from langrag.index_processor.extractor.factory import ParserFactory

# Host adapters
from .adapters import HostEmbedderAdapter, HostVectorAdapter

logger = logging.getLogger(__name__)


class LangRAG(RAGEngine):
    """
    Official LangBot RAG Engine implementation powered by LangRAG library.

    This engine provides advanced RAG capabilities:
    - Document ingestion with smart chunking
    - Hybrid search (vector + keyword)
    - Integration with Host's embedding models and vector database

    Architecture:
        LangBot Host <--RPC--> Plugin <--Adapters--> LangRAG Library
    """

    @classmethod
    def get_capabilities(cls) -> list[str]:
        """Declare supported capabilities."""
        return [
            RAGEngineCapability.DOC_INGESTION,
            RAGEngineCapability.CHUNKING_CONFIG,
            RAGEngineCapability.HYBRID_SEARCH,
        ]

    async def on_knowledge_base_create(self, kb_id: str, config: dict) -> None:
        """Called when a knowledge base using this engine is created."""
        logger.info(f"Knowledge base created: {kb_id} with config: {config}")

    async def on_knowledge_base_delete(self, kb_id: str) -> None:
        """Called when a knowledge base using this engine is deleted."""
        logger.info(f"Knowledge base deleted: {kb_id}")

    async def ingest(self, context: IngestionContext) -> IngestionResult:
        """
        Handle document ingestion using LangRAG's processing pipeline.

        Pipeline: Read -> Parse -> Chunk -> Embed -> Store
        """
        doc_id = context.file_object.metadata.document_id
        filename = context.file_object.metadata.filename
        kb_id = context.knowledge_base_id

        logger.info(f"Ingesting file: {filename} into KB: {kb_id}")

        try:
            # 1. Get file content from Host
            storage_path = context.file_object.storage_path
            content_bytes = await self.plugin.rag_get_file_stream(storage_path)

            # 2. Parse document using LangRAG's parser
            parser = self._get_parser(filename)
            documents = self._parse_content(parser, content_bytes, filename)
            logger.info(f"Parsed {len(documents)} documents from {filename}")

            if not documents:
                return IngestionResult(
                    document_id=doc_id,
                    status=DocumentStatus.COMPLETED,
                    chunks_created=0
                )

            # 3. Chunk using LangRAG's chunker
            chunk_size = context.custom_settings.get("chunk_size", context.chunk_size) or 512
            chunk_overlap = context.custom_settings.get("overlap", context.chunk_overlap) or 50

            chunker = RecursiveCharacterChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = chunker.split(documents)
            logger.info(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")

            if not chunks:
                return IngestionResult(
                    document_id=doc_id,
                    status=DocumentStatus.COMPLETED,
                    chunks_created=0
                )

            # 4. Embed using Host's embedding model via adapter
            embedder = HostEmbedderAdapter(self.plugin, kb_id=kb_id)
            texts = [c.page_content for c in chunks]
            vectors = await embedder.embed_async(texts)

            for chunk, vector in zip(chunks, vectors):
                chunk.vector = vector
                chunk.metadata["document_id"] = doc_id
                chunk.id = f"{doc_id}_{chunk.metadata.get('chunk_index', chunks.index(chunk))}"

            logger.info(f"Embedded {len(chunks)} chunks")

            # 5. Store using Host's vector database via adapter
            dataset = Dataset(
                id=kb_id,
                name=kb_id,
                collection_name=kb_id,
            )
            vector_store = HostVectorAdapter(self.plugin, dataset)
            await vector_store.add_texts_async(chunks)
            logger.info(f"Stored {len(chunks)} chunks")

            return IngestionResult(
                document_id=doc_id,
                status=DocumentStatus.COMPLETED,
                chunks_created=len(chunks)
            )

        except Exception as e:
            logger.exception(f"Ingestion failed: {e}")
            return IngestionResult(
                document_id=doc_id,
                status=DocumentStatus.FAILED,
                error_message=str(e)
            )

    async def retrieve(self, context: RetrievalContext) -> RetrievalResponse:
        """
        Retrieve relevant content using LangRAG's search pipeline.

        Supports: similarity, keyword, and hybrid search.
        """
        query = context.query
        kb_id = context.knowledge_base_id
        top_k = context.get_top_k()

        # Get search type from config
        search_type = "similarity"
        if context.config and context.config.custom_settings:
            search_type = context.config.custom_settings.get("search_type", "similarity")

        logger.info(f"Retrieving from KB {kb_id}: query='{query[:50]}...', top_k={top_k}, type={search_type}")

        try:
            # 1. Embed query using Host's embedding model
            query_vector = await self.plugin.rag_embed_query(kb_id, query)

            # 2. Search using Host's vector database via adapter
            dataset = Dataset(
                id=kb_id,
                name=kb_id,
                collection_name=kb_id,
            )
            vector_store = HostVectorAdapter(self.plugin, dataset)

            docs = await vector_store.search_async(
                query=query,
                query_vector=query_vector,
                top_k=top_k,
                search_type=search_type,
                filters=context.filters,
            )

            # 3. Format results
            entries = []
            for doc in docs:
                entries.append(RetrievalResultEntry(
                    id=doc.id,
                    content=[{"type": "text", "text": doc.page_content}],
                    metadata=doc.metadata,
                    score=doc.metadata.get("score"),
                    distance=0.0
                ))

            return RetrievalResponse(
                results=entries,
                total_found=len(entries),
                metadata={
                    "search_type": search_type,
                    "query_length": len(query),
                }
            )

        except Exception as e:
            logger.exception(f"Retrieval failed: {e}")
            return RetrievalResponse(
                results=[],
                total_found=0,
                metadata={"error": str(e)}
            )

    async def delete_document(self, kb_id: str, document_id: str) -> bool:
        """Delete a document and its chunks from the knowledge base."""
        try:
            result = await self.plugin.rag_vector_delete(
                collection_id=kb_id,
                filters={"document_id": document_id}
            )
            deleted = isinstance(result, dict) and result.get("count", 0) > 0
            logger.info(f"Deleted document {document_id} from KB {kb_id}: {deleted}")
            return deleted
        except Exception as e:
            logger.exception(f"Delete failed: {e}")
            return False

    def get_creation_settings_schema(self) -> dict:
        """JSON Schema for knowledge base creation settings."""
        return {
            "type": "object",
            "properties": {
                "chunk_size": {
                    "type": "integer",
                    "default": 512,
                    "minimum": 100,
                    "maximum": 4000,
                    "title": "Chunk Size",
                    "description": "Maximum size of text chunks in characters"
                },
                "overlap": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 0,
                    "maximum": 500,
                    "title": "Chunk Overlap",
                    "description": "Overlap between consecutive chunks"
                }
            }
        }

    def get_retrieval_settings_schema(self) -> dict:
        """JSON Schema for retrieval runtime settings."""
        return {
            "type": "object",
            "properties": {
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 100,
                    "title": "Top K",
                    "description": "Number of results to retrieve"
                },
                "search_type": {
                    "type": "string",
                    "enum": ["similarity", "keyword", "hybrid"],
                    "default": "similarity",
                    "title": "Search Type",
                    "description": "Type of search to perform"
                }
            }
        }

    # ==================== Helper Methods ====================

    def _get_parser(self, filename: str):
        """Get appropriate parser based on file extension."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        ext_map = {
            "md": "markdown", "markdown": "markdown",
            "htm": "html", "html": "html",
            "doc": "docx", "docx": "docx",
            "pdf": "pdf",
            "txt": "simple_text",
        }
        parser_type = ext_map.get(ext, "simple_text")
        try:
            return ParserFactory.create(parser_type)
        except ValueError:
            return ParserFactory.create("simple_text")

    def _parse_content(self, parser, content_bytes: bytes, filename: str) -> list[Document]:
        """Parse content bytes into Document objects."""
        import tempfile
        import os

        # Write content to temp file for parser
        ext = filename.rsplit(".", 1)[-1] if "." in filename else "txt"
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
            f.write(content_bytes)
            temp_path = f.name

        try:
            documents = parser.parse(temp_path)
            # Add source metadata
            for doc in documents:
                doc.metadata["source"] = filename
            return documents
        except Exception as e:
            logger.warning(f"Parser failed, falling back to text: {e}")
            # Fallback: treat as plain text
            try:
                text = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = content_bytes.decode("utf-8", errors="replace")

            return [Document(
                page_content=text,
                metadata={"source": filename}
            )]
        finally:
            os.unlink(temp_path)
