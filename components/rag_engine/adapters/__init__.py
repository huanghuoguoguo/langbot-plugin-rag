"""
Host Adapters for LangRAG.

These adapters bridge LangRAG's abstract interfaces to LangBot Host's RPC capabilities.
This allows LangRAG's algorithms to run on LangBot's infrastructure (LLM, Embedding, VDB).
"""

from .embedder import HostEmbedderAdapter
from .vector import HostVectorAdapter
from .llm import HostLLMAdapter

__all__ = [
    "HostEmbedderAdapter",
    "HostVectorAdapter",
    "HostLLMAdapter",
]
