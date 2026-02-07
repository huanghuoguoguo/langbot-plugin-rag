"""LangBot KV Store Adapter - bridges LangRAG's BaseKVStore to Host plugin_storage."""

import json
import logging
from typing import TYPE_CHECKING, Any

from langrag.datasource.kv.base import BaseKVStore

if TYPE_CHECKING:
    from langbot_plugin.api.definition.plugin import BasePlugin

logger = logging.getLogger(__name__)


class LangBotKVAdapter(BaseKVStore):
    """Adapter that bridges LangRAG's BaseKVStore interface to LangBot plugin_storage.

    This adapter uses Host's plugin_storage RPC for key-value operations,
    enabling Parent-Child indexing where parent documents are stored in KV
    and child chunks are stored in the vector database.

    Keys are namespaced with a prefix to avoid collisions with other plugin data.

    Example:
        kv_store = LangBotKVAdapter(plugin, namespace="kb-123")
        kv_store.mset({"parent_1": "Parent document content..."})
        content = kv_store.get("parent_1")
    """

    def __init__(self, plugin: "BasePlugin", namespace: str = "langrag"):
        """Initialize the KV adapter.

        Args:
            plugin: The LangBot plugin instance for RPC calls.
            namespace: Namespace prefix for keys to avoid collisions.
        """
        self.plugin = plugin
        self.namespace = namespace

    def _make_key(self, key: str) -> str:
        """Create a namespaced key.

        Args:
            key: The original key.

        Returns:
            Namespaced key string.
        """
        return f"{self.namespace}:kv:{key}"

    def mget(self, keys: list[str]) -> list[Any | None]:
        """Get multiple values synchronously.

        Note: This wraps the async implementation for sync compatibility.
        For async contexts, use mget_async() instead.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # Already in async context, need to use a different approach
            raise RuntimeError("Use mget_async() in async context")
        except RuntimeError:
            # No running loop, safe to use run
            return asyncio.run(self.mget_async(keys))

    def mset(self, data: dict[str, Any]) -> None:
        """Set multiple values synchronously.

        Note: This wraps the async implementation for sync compatibility.
        For async contexts, use mset_async() instead.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError("Use mset_async() in async context")
        except RuntimeError:
            asyncio.run(self.mset_async(data))

    def delete(self, keys: list[str]) -> None:
        """Delete multiple keys synchronously.

        Note: This wraps the async implementation for sync compatibility.
        For async contexts, use delete_async() instead.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError("Use delete_async() in async context")
        except RuntimeError:
            asyncio.run(self.delete_async(keys))

    # ==================== Async Methods ====================

    async def mget_async(self, keys: list[str]) -> list[Any | None]:
        """Get multiple values via Host plugin_storage.

        Args:
            keys: List of keys to retrieve.

        Returns:
            List of values (or None for missing keys).
        """
        results = []
        for key in keys:
            namespaced_key = self._make_key(key)
            try:
                data = await self.plugin.get_plugin_storage(namespaced_key)
                if data:
                    # Deserialize JSON
                    results.append(json.loads(data.decode('utf-8')))
                else:
                    results.append(None)
            except Exception as e:
                logger.warning(f"Failed to get key {key}: {e}")
                results.append(None)
        return results

    async def mset_async(self, data: dict[str, Any]) -> None:
        """Set multiple values via Host plugin_storage.

        Args:
            data: Dict of key-value pairs to store.
        """
        for key, value in data.items():
            namespaced_key = self._make_key(key)
            try:
                # Serialize to JSON bytes
                json_bytes = json.dumps(value).encode('utf-8')
                await self.plugin.set_plugin_storage(namespaced_key, json_bytes)
            except Exception as e:
                logger.error(f"Failed to set key {key}: {e}")
                raise

    async def delete_async(self, keys: list[str]) -> None:
        """Delete multiple keys via Host plugin_storage.

        Args:
            keys: List of keys to delete.
        """
        for key in keys:
            namespaced_key = self._make_key(key)
            try:
                # Set to empty/None to "delete"
                # Note: Host may not have explicit delete, so we set to empty
                await self.plugin.set_plugin_storage(namespaced_key, b'')
            except Exception as e:
                logger.warning(f"Failed to delete key {key}: {e}")

    async def get_async(self, key: str) -> Any | None:
        """Get single value asynchronously.

        Args:
            key: The key to retrieve.

        Returns:
            The value or None if not found.
        """
        results = await self.mget_async([key])
        return results[0] if results else None

    async def set_async(self, key: str, value: Any) -> None:
        """Set single value asynchronously.

        Args:
            key: The key to set.
            value: The value to store.
        """
        await self.mset_async({key: value})
