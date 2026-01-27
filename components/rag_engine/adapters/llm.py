"""
HostLLMAdapter - Bridge LangRAG's BaseLLM to LangBot Host RPC.

This adapter implements LangRAG's BaseLLM interface by delegating
LLM operations to the LangBot Host via RPC calls.

This is useful for LangRAG features that require LLM capabilities:
- Query rewriting
- Agentic routing
- Answer generation
"""

from typing import TYPE_CHECKING, AsyncIterator

from langrag.llm.base import BaseLLM

if TYPE_CHECKING:
    from langbot_plugin.api.definition.plugin import BasePlugin


class HostLLMAdapter(BaseLLM):
    """Adapter that uses LangBot Host's LLM capability.

    This adapter enables LangRAG to use the Host's configured LLM models
    for tasks like query rewriting, routing, and answer generation.

    Example:
        llm = HostLLMAdapter(plugin, model_uuid="gpt-4")
        answer = await llm.chat_async([{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        plugin: "BasePlugin",
        model_uuid: str | None = None,
        kb_id: str | None = None,
    ):
        """Initialize the Host LLM Adapter.

        Args:
            plugin: The plugin instance for accessing RPC methods.
            model_uuid: Optional LLM model UUID. If None, uses default.
            kb_id: Optional knowledge base ID for context-aware model selection.
        """
        self._plugin = plugin
        self._model_uuid = model_uuid
        self._kb_id = kb_id

    # ==================== Sync Methods (Not Supported) ====================

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents - use HostEmbedderAdapter instead."""
        raise RuntimeError(
            "HostLLMAdapter.embed_documents() is not supported. "
            "Use HostEmbedderAdapter for embedding operations."
        )

    def embed_query(self, text: str) -> list[float]:
        """Embed query - use HostEmbedderAdapter instead."""
        raise RuntimeError(
            "HostLLMAdapter.embed_query() is not supported. "
            "Use HostEmbedderAdapter for embedding operations."
        )

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Chat completion - use chat_async() instead."""
        raise RuntimeError(
            "HostLLMAdapter.chat() is not supported. "
            "Use chat_async() for async RPC calls to Host."
        )

    def stream_chat(self, messages: list[dict], **kwargs):
        """Stream chat - use stream_chat_async() instead."""
        raise RuntimeError(
            "HostLLMAdapter.stream_chat() is not supported. "
            "Use stream_chat_async() for async RPC calls to Host."
        )

    # ==================== Async Methods ====================

    async def embed_documents_async(self, texts: list[str]) -> list[list[float]]:
        """Embed documents via Host RPC.

        Note: For embedding, prefer using HostEmbedderAdapter directly.
        This method is provided for interface compatibility.
        """
        if not self._kb_id:
            raise ValueError("kb_id is required for embedding operations")
        return await self._plugin.rag_embed_documents(self._kb_id, texts)

    async def embed_query_async(self, text: str) -> list[float]:
        """Embed query via Host RPC.

        Note: For embedding, prefer using HostEmbedderAdapter directly.
        This method is provided for interface compatibility.
        """
        if not self._kb_id:
            raise ValueError("kb_id is required for embedding operations")
        return await self._plugin.rag_embed_query(self._kb_id, text)

    async def chat_async(self, messages: list[dict], **kwargs) -> str:
        """Chat completion via Host RPC.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional arguments passed to the LLM.

        Returns:
            Generated text response.
        """
        # Convert messages to SDK format
        from langbot_plugin.api.entities.builtin.provider.message import Message

        sdk_messages = [
            Message(role=m["role"], content=m["content"])
            for m in messages
        ]

        # Get model UUID - use provided or try to get default
        model_uuid = self._model_uuid
        if not model_uuid:
            # Try to get available models and use the first one
            models = await self._plugin.get_llm_models()
            if models:
                model_uuid = models[0]
            else:
                raise ValueError("No LLM models available and no model_uuid specified")

        # Invoke LLM
        response = await self._plugin.invoke_llm(
            llm_model_uuid=model_uuid,
            messages=sdk_messages,
            funcs=[],
            extra_args=kwargs,
        )

        return response.content or ""

    async def chat_dict_async(self, messages: list[dict], **kwargs) -> dict:
        """Chat completion returning full message dict.

        Args:
            messages: List of message dicts.
            **kwargs: Additional arguments.

        Returns:
            Response message dict with 'role' and 'content'.
        """
        content = await self.chat_async(messages, **kwargs)
        return {"role": "assistant", "content": content}

    async def stream_chat_async(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        """Stream chat completion.

        Note: Streaming is not yet supported via Host RPC.
        This method falls back to non-streaming chat.

        Args:
            messages: List of message dicts.
            **kwargs: Additional arguments.

        Yields:
            Full response as a single chunk (no actual streaming).
        """
        # TODO: Implement actual streaming when Host supports it
        response = await self.chat_async(messages, **kwargs)
        yield response
