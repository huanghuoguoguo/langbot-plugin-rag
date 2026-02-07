"""Schema definitions for the RAG engine settings."""

from .constants import (
    IndexStrategy,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_PARENT_CHUNK_SIZE,
    DEFAULT_CHILD_CHUNK_SIZE,
)


def get_creation_settings_schema() -> list[dict]:
    """Return schema for knowledge base creation settings."""
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


def get_retrieval_settings_schema() -> list[dict]:
    """Return schema for retrieval settings."""
    return [
        {
            "name": "top_k",
            "label": {"en_US": "Top K", "zh_Hans": "召回数量"},
            "type": "integer",
            "required": False,
            "default": 5,
        }
    ]
