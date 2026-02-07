import logging
from enum import StrEnum
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

# LangRAG imports
from langrag import RecursiveCharacterChunker, Document as LangRAGDocument, DocumentType

from .parser import FileParser
from .adapters import (
    LangBotEmbedderAdapter,
    LangBotVectorAdapter,
    LangBotLLMAdapter,
    LangBotKVAdapter,
)

logger = logging.getLogger(__name__)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_PARENT_CHUNK_SIZE = 2000
DEFAULT_CHILD_CHUNK_SIZE = 400
# Batch size for embedding API calls to avoid IPC timeouts
EMBEDDING_BATCH_SIZE = 10


class IndexStrategy(StrEnum):
    """Available indexing strategies."""
    PARAGRAPH = "paragraph"      # Standard recursive character chunking
    PARENT_CHILD = "parent_child"  # Hierarchical: children in VDB, parents in KV
    QA = "qa"                    # LLM-generated questions indexed


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
            if index_strategy == IndexStrategy.PARENT_CHILD.value:
                chunks_created = await self._ingest_parent_child(context, source_doc, collection_id, kb_id)
            elif index_strategy == IndexStrategy.QA.value:
                chunks_created = await self._ingest_qa(context, source_doc, collection_id, kb_id)
            else:
                # Default: PARAGRAPH strategy
                chunks_created = await self._ingest_paragraph(context, source_doc, collection_id, kb_id)

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

    async def _ingest_paragraph(
        self,
        context: IngestionContext,
        source_doc: LangRAGDocument,
        collection_id: str,
        kb_id: str,
    ) -> int:
        """Standard paragraph-based indexing using RecursiveCharacterChunker."""
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

    async def _ingest_parent_child(
        self,
        context: IngestionContext,
        source_doc: LangRAGDocument,
        collection_id: str,
        kb_id: str,
    ) -> int:
        """Parent-Child hierarchical indexing.

        Parents are stored in KV store, children are embedded and stored in VDB.
        Children reference their parent via 'parent_id' in metadata.
        """
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

    async def _ingest_qa(
        self,
        context: IngestionContext,
        source_doc: LangRAGDocument,
        collection_id: str,
        kb_id: str,
    ) -> int:
        """QA-based indexing using LLM to generate questions.

        Generates questions from each chunk and indexes questions.
        The original chunk content is stored in 'answer' metadata.
        """
        doc_id = source_doc.metadata["document_id"]
        filename = source_doc.metadata["document_name"]

        # Check LLM model configuration
        llm_model_uuid = context.custom_settings.get("llm_model_uuid")
        if not llm_model_uuid:
            raise ValueError("LLM model UUID is required for QA indexing strategy")

        # Get chunking parameters
        chunk_size = context.chunk_size or context.custom_settings.get("chunk_size") or DEFAULT_CHUNK_SIZE
        chunk_overlap = context.chunk_overlap or context.custom_settings.get("overlap") or DEFAULT_CHUNK_OVERLAP

        # Chunk the document first
        chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunk_docs = chunker.split([source_doc])

        if not chunk_docs:
            return 0

        # Initialize LLM adapter
        llm_adapter = LangBotLLMAdapter(self.plugin, llm_model_uuid)

        # Generate questions for each chunk
        qa_documents: list[dict] = []
        failed_count = 0

        for chunk_idx, chunk in enumerate(chunk_docs):
            try:
                question = await self._generate_question(llm_adapter, chunk.page_content)
                if question:
                    qa_documents.append({
                        "id": f"{doc_id}_qa_{chunk_idx}",
                        "question": question,
                        "answer": chunk.page_content,
                        "chunk_idx": chunk_idx,
                    })
                else:
                    failed_count += 1
            except Exception as e:
                logger.warning(f"Failed to generate question for chunk {chunk_idx}: {e}")
                failed_count += 1

        if not qa_documents:
            logger.warning(f"No QA pairs generated for {filename}")
            return 0

        if failed_count > 0:
            logger.warning(f"QA generation: {failed_count}/{len(chunk_docs)} chunks failed")

        # Embed questions (not answers - questions are what we search by)
        question_texts = [qa["question"] for qa in qa_documents]
        vectors = await self._embed_in_batches(kb_id, question_texts)

        # Build metadata and upsert
        ids = [qa["id"] for qa in qa_documents]
        metadatas = [
            {
                "file_id": doc_id,
                "document_id": doc_id,
                "document_name": filename,
                "question": qa["question"],
                "text": qa["answer"],  # Store answer as text for retrieval
                "is_qa": True,
                "index_strategy": IndexStrategy.QA.value,
            }
            for qa in qa_documents
        ]

        await self.plugin.rag_vector_upsert(
            collection_id=collection_id,
            vectors=vectors,
            ids=ids,
            metadata=metadatas,
        )

        logger.info(f"QA indexing: {len(qa_documents)} QA pairs created for {filename}")
        return len(qa_documents)

    async def _embed_in_batches(self, kb_id: str, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches to avoid IPC timeouts."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            batch_vectors = await self.plugin.rag_embed_documents(kb_id, batch)
            vectors.extend(batch_vectors)
        return vectors

    async def _generate_question(self, llm: LangBotLLMAdapter, text: str) -> str | None:
        """Generate a question from text using LLM."""
        prompt = f"""You are a helpful assistant. Generate 1 concise question that can be answered by the following text.
Only output the question, nothing else.

Text:
{text}

Question:"""
        try:
            response = await llm.chat_async([{"role": "user", "content": prompt}])
            question = response.strip()
            return question if question else None
        except Exception as e:
            logger.error(f"LLM question generation failed: {e}")
            return None

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

        # 1. Embed query (Host selects the embedding model by KB ID)
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
            entries = await self._retrieve_parent_child(results, kb_id)
        else:
            # Standard retrieval (PARAGRAPH or QA)
            entries = self._format_standard_results(results)

        return RetrievalResponse(results=entries, total_found=len(entries))

    async def _retrieve_parent_child(
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
        child_to_parent: dict[str, str] = {}  # child_id -> parent_id
        child_scores: dict[str, tuple[float, float]] = {}  # parent_id -> (best_score, distance)

        for res in child_results:
            parent_id = res.get("metadata", {}).get("parent_id")
            if parent_id and parent_id not in parent_ids_seen:
                parent_ids_seen.add(parent_id)
                parent_ids_ordered.append(parent_id)
                # Use first child's score as parent score (best match)
                child_scores[parent_id] = (res.get("score"), res.get("distance"))
            child_to_parent[res["id"]] = parent_id

        # Fetch parent content from KV
        parent_contents = await kv_adapter.mget_async(parent_ids_ordered)

        # Build result entries
        entries: list[RetrievalResultEntry] = []
        for i, parent_id in enumerate(parent_ids_ordered):
            parent_content = parent_contents[i]
            if parent_content is None:
                # Fallback: parent not found in KV, use child content
                logger.warning(f"Parent {parent_id} not found in KV, using child content")
                # Find first child with this parent_id
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

    async def delete_document(self, kb_id: str, document_id: str) -> bool:
        """Delete a document's vectors and associated data.

        For Parent-Child indexed documents, also cleans up parent data from KV store.
        Uses ids parameter (treated as file_id by Host's delete_by_file_id).

        Note: Parent cleanup relies on predictable key pattern ({doc_id}_parent_{idx}).
        If the pattern changes, orphaned parents may remain in KV store.
        """
        # Delete vectors from VDB
        count = await self.plugin.rag_vector_delete(
            collection_id=kb_id,
            ids=[document_id],
        )

        # Best-effort cleanup of Parent-Child KV data
        # Parents are stored with keys like: {doc_id}_parent_0, {doc_id}_parent_1, ...
        # Since plugin_storage doesn't support pattern deletion, we try common indices
        try:
            kv_adapter = LangBotKVAdapter(self.plugin, namespace=f"pc:{kb_id}")
            # Try to delete up to 100 potential parent keys (should cover most documents)
            parent_keys = [f"{document_id}_parent_{i}" for i in range(100)]
            await kv_adapter.delete_async(parent_keys)
            logger.debug(f"Cleaned up Parent-Child KV data for document {document_id}")
        except Exception as e:
            logger.warning(f"Failed to clean up Parent-Child KV data: {e}")
            # Don't fail the delete operation if KV cleanup fails

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
                "name": "index_strategy",
                "label": {"en_US": "Index Strategy", "zh_Hans": "索引策略"},
                "description": {
                    "en_US": "Choose indexing strategy: Paragraph (standard), Parent-Child (hierarchical), or QA (LLM-generated questions)",
                    "zh_Hans": "选择索引策略：段落（标准）、父子层级（层级式）或问答（LLM生成问题）",
                },
                "type": "select",
                "required": False,
                "default": IndexStrategy.PARAGRAPH.value,
                "options": [
                    {"value": IndexStrategy.PARAGRAPH.value, "label": {"en_US": "Paragraph (Standard)", "zh_Hans": "段落（标准）"}},
                    {"value": IndexStrategy.PARENT_CHILD.value, "label": {"en_US": "Parent-Child (Hierarchical)", "zh_Hans": "父子层级"}},
                    {"value": IndexStrategy.QA.value, "label": {"en_US": "QA (LLM Questions)", "zh_Hans": "问答（LLM生成）"}},
                ],
            },
            {
                "name": "llm_model_uuid",
                "label": {"en_US": "LLM Model (for QA)", "zh_Hans": "LLM 模型（用于问答）"},
                "description": {
                    "en_US": "LLM model for generating questions (required for QA strategy)",
                    "zh_Hans": "用于生成问题的 LLM 模型（问答策略必需）",
                },
                "type": "llm-model-selector",
                "required": False,
                "default": "",
                "visible_when": {"index_strategy": IndexStrategy.QA.value},
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
            {
                "name": "parent_chunk_size",
                "label": {"en_US": "Parent Chunk Size", "zh_Hans": "父块大小"},
                "description": {
                    "en_US": "Size of parent chunks for Parent-Child strategy",
                    "zh_Hans": "父子层级策略中父块的大小",
                },
                "type": "integer",
                "required": False,
                "default": DEFAULT_PARENT_CHUNK_SIZE,
                "visible_when": {"index_strategy": IndexStrategy.PARENT_CHILD.value},
            },
            {
                "name": "child_chunk_size",
                "label": {"en_US": "Child Chunk Size", "zh_Hans": "子块大小"},
                "description": {
                    "en_US": "Size of child chunks for Parent-Child strategy",
                    "zh_Hans": "父子层级策略中子块的大小",
                },
                "type": "integer",
                "required": False,
                "default": DEFAULT_CHILD_CHUNK_SIZE,
                "visible_when": {"index_strategy": IndexStrategy.PARENT_CHILD.value},
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
