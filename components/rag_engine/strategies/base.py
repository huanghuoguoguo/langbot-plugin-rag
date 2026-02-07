"""Base class for indexing strategies."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from langbot_plugin.api.entities.builtin.rag import IngestionContext
from langrag import Document as LangRAGDocument

if TYPE_CHECKING:
    from langbot_plugin.api.definition.plugin import BasePlugin


class BaseIndexStrategy(ABC):
    """Abstract base class for indexing strategies."""

    def __init__(self, plugin: "BasePlugin"):
        """Initialize the strategy.

        Args:
            plugin: The LangBot plugin instance for RPC calls.
        """
        self.plugin = plugin

    @abstractmethod
    async def ingest(
        self,
        context: IngestionContext,
        source_doc: LangRAGDocument,
        collection_id: str,
        kb_id: str,
    ) -> int:
        """Process and index a document.

        Args:
            context: The ingestion context containing settings.
            source_doc: The parsed source document.
            collection_id: The vector collection ID.
            kb_id: The knowledge base ID.

        Returns:
            Number of chunks/items created.
        """
        pass
