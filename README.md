# langbot-plugin-rag

LangBot 官方 RAG (Retrieval-Augmented Generation) 引擎插件。

本插件为 LangBot 提供开箱即用的文档摄入与检索能力，基于宿主（LangBot Core）提供的 Embedding 模型和向量数据库基础设施运行，插件本身无状态。

## 功能

- **多格式文档解析** — 支持 PDF、DOCX、Markdown、HTML、TXT
- **可配置分块** — 滑动窗口分块，支持自定义分块大小和重叠
- **向量检索** — 嵌入查询 → 向量相似度搜索 → 返回 Top-K 结果
- **文档管理** — 按文档维度删除已索引的向量数据
- **动态配置表单** — 创建知识库和检索时通过 Schema 动态渲染配置项

## 架构

```
┌─────────────────────────────────┐
│         LangBot Core            │
│  (Embedding / VDB / Storage)    │
└──────────┬──────────────────────┘
           │ RPC (IPC)
┌──────────▼──────────────────────┐
│      langbot-plugin-rag         │
│  ┌───────────────────────────┐  │
│  │       LangRAG Engine      │  │
│  │  Parse → Chunk → Embed    │  │
│  │       → Store / Search    │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

插件通过 RPC 调用宿主提供的基础设施：

| 能力 | Host RPC 接口 |
|------|--------------|
| 文本嵌入 | `rag_embed_documents` / `rag_embed_query` |
| 向量存储与检索 | `rag_vector_upsert` / `rag_vector_search` / `rag_vector_delete` |
| 文件读取 | `rag_get_file_stream` |

## 配置项

### 创建知识库时

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `embedding_model_uuid` | 嵌入模型 | 必选 |
| `chunk_size` | 分块大小（字符数） | 512 |
| `overlap` | 分块重叠（字符数） | 50 |

### 检索时

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `top_k` | 召回数量 | 5 |

## 开发

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 复制环境变量文件
cp .env.example .env
```

### 调试

配置 `.env` 中的 `DEBUG_RUNTIME_WS_URL` 和 `PLUGIN_DEBUG_KEY`，然后通过 VS Code 的 launch.json 启动调试。

## 路线图

详见 [Roadmap Issue](https://github.com/huanghuoguoguo/langbot-plugin-rag/issues/1)，后续计划引入 [LangRAG](https://github.com/huanghuoguoguo/LangRAG) 内核，逐步支持：

- 递归字符分块（句子/段落感知）
- Parent-Child 层级索引
- QA 索引（LLM 提取问答对）
- 混合检索（向量 + BM25 全文 + RRF 融合）
- 查询改写 & 重排序
- 语义缓存
- RAPTOR 层级摘要索引

## 相关项目

- [LangBot](https://github.com/langbot-app/LangBot) — 宿主平台
- [langbot-plugin-sdk](https://github.com/langbot-app/langbot-plugin-sdk) — 插件 SDK
- [LangRAG](https://github.com/huanghuoguoguo/LangRAG) — RAG 内核（计划集成）
