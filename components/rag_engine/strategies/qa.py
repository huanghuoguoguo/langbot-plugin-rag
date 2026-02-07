"""QA (Question-Answer) indexing strategy."""

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
from ..adapters import LangBotLLMAdapter
from .base import BaseIndexStrategy

if TYPE_CHECKING:
    from langbot_plugin.api.definition.plugin import BasePlugin

logger = logging.getLogger(__name__)


class QAStrategy(BaseIndexStrategy):
    """QA-based indexing using LLM to generate questions.

    Generates questions from each chunk and indexes questions.
    The original chunk content is stored as 'answer' in metadata.
    """

    async def ingest(
        self,
        context: IngestionContext,
        source_doc: LangRAGDocument,
        collection_id: str,
        kb_id: str,
    ) -> int:
        """Process document by generating QA pairs."""
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

    async def _embed_in_batches(self, kb_id: str, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches to avoid IPC timeouts."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            batch_vectors = await self.plugin.rag_embed_documents(kb_id, batch)
            vectors.extend(batch_vectors)
        return vectors
