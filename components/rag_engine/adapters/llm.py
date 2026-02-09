"""LangBot LLM Adapter - bridges LangRAG's BaseLLM to Host RPC."""

import logging
from typing import TYPE_CHECKING, AsyncIterator

from langrag.llm.base import BaseLLM

if TYPE_CHECKING:
    from langbot_plugin.api.definition.plugin import BasePlugin

logger = logging.getLogger(__name__)


class LangBotLLMAdapter(BaseLLM):
    """Adapter that bridges LangRAG's BaseLLM interface to LangBot Host RPC.

    This adapter delegates LLM operations to the Host's LLM service via RPC calls,
    allowing LangRAG components (QA Indexing, Query Rewriting, etc.) to work
    seamlessly with LangBot's LLM infrastructure.

    Example:
        llm = LangBotLLMAdapter(plugin, llm_model_uuid="model-123")
        response = await llm.chat_async([{"role": "user", "content": "Hello"}])
    """

    def __init__(self, plugin: "BasePlugin", llm_model_uuid: str):
        """Initialize the LLM adapter.

        Args:
            plugin: The LangBot plugin instance for RPC calls.
            llm_model_uuid: The UUID of the LLM model to use on Host.
        """
        self.plugin = plugin
        self.llm_model_uuid = llm_model_uuid

    # ==================== Sync Methods (Not Supported) ====================

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Sync embed_documents - not supported, use LangBotEmbedderAdapter instead."""
        raise NotImplementedError(
            "Use LangBotEmbedderAdapter for embedding operations."
        )

    def embed_query(self, text: str) -> list[float]:
        """Sync embed_query - not supported, use LangBotEmbedderAdapter instead."""
        raise NotImplementedError(
            "Use LangBotEmbedderAdapter for embedding operations."
        )

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Sync chat - not supported."""
        raise NotImplementedError("Use chat_async() instead.")

    def stream_chat(self, messages: list[dict], **kwargs):
        """Sync stream_chat - not supported."""
        raise NotImplementedError("Use stream_chat_async() instead.")

    # ==================== Async Methods ====================

    async def embed_documents_async(self, texts: list[str]) -> list[list[float]]:
        """Async embed_documents - not supported, use LangBotEmbedderAdapter instead."""
        raise NotImplementedError(
            "Use LangBotEmbedderAdapter for embedding operations."
        )

    async def embed_query_async(self, text: str) -> list[float]:
        """Async embed_query - not supported, use LangBotEmbedderAdapter instead."""
        raise NotImplementedError(
            "Use LangBotEmbedderAdapter for embedding operations."
        )

    async def chat_async(self, messages: list[dict], **kwargs) -> str:
        """Chat completion via Host RPC.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional arguments (temperature, max_tokens, etc.)

        Returns:
            The assistant's response text.
        """
        # Convert messages to LangBot format
        langbot_messages = self._convert_messages(messages)

        # Call Host RPC
        response = await self.plugin.invoke_llm(
            llm_model_uuid=self.llm_model_uuid,
            messages=langbot_messages,
            **kwargs,
        )

        # Extract text from response
        return response.readable_str() if hasattr(response, 'readable_str') else str(response)

    async def chat_dict_async(self, messages: list[dict], **kwargs) -> dict:
        """Chat completion returning full message dict via Host RPC.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional arguments.

        Returns:
            Dict with 'role' and 'content' keys.
        """
        content = await self.chat_async(messages, **kwargs)
        return {"role": "assistant", "content": content}

    async def stream_chat_async(self, messages: list[dict], **kwargs) -> AsyncIterator[str]:
        """Stream chat completion via Host RPC.

        Note: Streaming may not be fully supported by Host RPC.
        Falls back to non-streaming if not available.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional arguments.

        Yields:
            Tokens as they are generated.
        """
        # For now, fall back to non-streaming
        # TODO: Implement streaming when Host supports it
        logger.warning("Streaming not yet supported, falling back to non-streaming")
        response = await self.chat_async(messages, **kwargs)
        yield response

    def _convert_messages(self, messages: list[dict]) -> list:
        """Convert LangRAG message format to LangBot Message format.

        Args:
            messages: List of dicts with 'role' and 'content'.

        Returns:
            List of LangBot Message objects.
        """
        from langbot_plugin.api.entities.builtin.message import Message

        langbot_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map roles
            if role == "system":
                langbot_messages.append(Message.from_text(role="system", text=content))
            elif role == "assistant":
                langbot_messages.append(Message.from_text(role="assistant", text=content))
            else:  # user or other
                langbot_messages.append(Message.from_text(role="user", text=content))

        return langbot_messages
