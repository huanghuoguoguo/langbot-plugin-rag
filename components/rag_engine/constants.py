"""Constants for the RAG engine."""

from enum import StrEnum


# Default chunking parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_PARENT_CHUNK_SIZE = 2000
DEFAULT_CHILD_CHUNK_SIZE = 400

# Batch size for embedding API calls to avoid IPC timeouts
EMBEDDING_BATCH_SIZE = 10


class IndexStrategy(StrEnum):
    """Available indexing strategies."""
    PARAGRAPH = "paragraph"        # Standard recursive character chunking
    PARENT_CHILD = "parent_child"  # Hierarchical: children in VDB, parents in KV
    QA = "qa"                      # LLM-generated questions indexed
