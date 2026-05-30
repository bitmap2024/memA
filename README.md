# memA — 长程对话记忆系统

memA 是一个面向LLM的ai助手记忆系统
**记忆抽取（extract_mem）→ 离线合并整理（update_mem）→ 在线混合检索（retrieval_mem）**。

底层组件：
- **LLM**：DeepSeek（兼容 OpenAI 协议）
- **Embedding**：bge-m3（dense + sparse 双向量）
- **文本压缩**：LLMLingua-2-bert-base-multilingual-cased-meetingbank
- **向量库**：Qdrant Cloud（dense 命名向量 + sparse 命名向量）
- **关系库**：MySQL（本地默认走 SQLite，结构等价）
- **文档库**：阿里云 OSS（本地兜底目录）

配置分两层：

- **`.env`** —— 只放敏感信息（API Key、密码、私有 endpoint），不进 git。
- **`start.sh`** —— 只放业务参数（模型名、维度、阈值、top_k、端口、本地路径等）。
- `bash start.sh` 一键启动 Web 服务（Streamlit），其它命令请见 [启动 / CLI](#4-启动--cli)。

---

## 1. 目录结构

```
memA/
├── config/
│   ├── setting.py                # 统一配置：.env 解析
│   ├── memory_item.py            # MemoryItem dataclass
│   └── prompt/memory_prompt/chat_bot/prompt_zh/
│       ├── profile.py            # Profile 抽取 prompt
│       ├── episodic.py           # Episodic 抽取 prompt
│       ├── state.py              # State 抽取 prompt
│       ├── in-context.py         # Core Memory 蒸馏 prompt（markdown）
│       └── update_merge.py       # update / merge / category 文档生成 prompt
│
├── src/
│   ├── db/
│   │   ├── base.py                # MemoryStoreClient 抽象接口 + 共享序列化
│   │   ├── __init__.py            # create_memory_store(cfg) 工厂（按 USE_SQLITE 路由）
│   │   ├── mysql/
│   │   │   ├── mysql_client.py    # MysqlMemoryStore（PyMySQL，生产）
│   │   │   └── sql                # MySQL DDL
│   │   ├── sqllite/
│   │   │   ├── sqllite_client.py  # SqliteMemoryStore（本地 / mock）
│   │   │   └── sql                # SQLite DDL
│   │   └── qdrant/qdrant_memory_client.py # QdrantMemoryClient（dense + sparse）
│   ├── embeddings/bgem3_emb_client.py
│   ├── llm/openai_llm.py
│   ├── extract_mem/
│   │   ├── memory_extract_pipeline.py   # 端到端抽取管线
│   │   ├── memory_extractor.py          # 并发 profile/episodic/state 抽取
│   │   ├── text_compressor.py
│   │   └── topic_segment.py
│   ├── update_mem/
│   │   ├── sleep_mode_update.py         # sleep mode 聚类合并 + lineage
│   │   └── category_doc_builder.py      # 生成 category.md → OSS
│   ├── retrieval_mem/
│   │   ├── hybrid_retrieval.py          # dense + sparse + bm25 + RRF + 时间衰减 + MMR
│   │   ├── bm25_search.py
│   │   ├── time_decay.py
│   │   └── mmr_search.py
│   ├── reranker/
│   │   ├── rrf_reranker.py
│   │   ├── bge_reranker.py              # bge-reranker-v2-m3
│   │   └── llm_reranker.py
│   ├── services/                        # facade 层
│   ├── utils/oss_manage.py              # OSS 文档存储
│   └── main.py                          # CLI 入口
│
└── deployment/
    ├── requirements.txt
    └── sample_history.json              # 抽取入参样例
```

---

## 2. 数据流（与 README 要求对齐）

### 2.1 extract_mem 管线

输入对话格式（详见 `deployment/sample_history.json`）：

```json
{
  "user_id": "user_001",
  "session_id": "session_001",
  "conversation_date_time": "2026-05-18T13:21:00+08:00",
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

执行步骤：
1. **格式校验**：去掉空消息、非法 role。
2. **文本压缩**：用 LLMLingua-2 对每条 `content` 做压缩。
3. **主题分割**：用 bge-m3 dense 向量计算相邻轮次余弦相似度，
   总 token ≤ 512 视为单主题，否则按相似度阈值切分。
4. **格式化**：每个主题输出 `[i] role: content` 行。
5. **并发抽取**：profile / episodic / state 三个 prompt 并发执行，
   prompt 已强制要求输出 `importance` 与 `confidence`。
6. **持久化**：
   - bge-m3 同时产出 dense + sparse 向量 → Qdrant 命名向量；
   - MemoryItem 写入 MySQL `memories` 表；
   - 会话来源写入 `memory_session_sources`；
   - `content_hash` 用于去重。

Qdrant payload 与项目目标对齐：

```json
{
  "memory_id": "mem_...",
  "user_id": "user_001",
  "memory_content": "...",
  "memory_type": "profile|episodic|state",
  "memory_category": "fact|preference|...",
  "status": "active",
  "importance": 0.92,
  "confidence": 0.88,
  "content_hash": "...",
  "created_at": "...",
  "updated_at": "..."
}
```

### 2.2 update_mem（sleep mode）

按 `user_id` 拉取活跃记忆，按 `(memory_type, memory_category)` 分桶，
在桶内用 dense 向量构建相似度矩阵 → 并查集聚类 → 调用
`MERGE_MEMORY_PROMPT` 生成 canonical 记忆：

- 新 memory 的 `derived_from_memory_ids` 记录来源；
- 旧 memory `status` 置为 `archived`；
- 写入 `memory_lineage_edges` 维护血缘；
- Qdrant payload 同步更新。

随后 `CategoryDocBuilder` 按 (type, category) 汇总活跃记忆，
调用 `CATEGORY_DOC_PROMPT` 生成 Markdown，写入 OSS：

```
chat_bot_memory/
└── <user_id>/
    ├── Profile Memory/
    │   ├── fact.md
    │   ├── preference.md
    │   ├── ...
    │   └── communication_style.md
    ├── Episodic Memory/
    │   └── ...
    ├── State Memory/
    │   └── ...
    └── Core Memory/
        └── ...   (蒸馏层，由 in-context.py 生成)
```

OSS 不可达时自动落到本地目录 `var/oss_local/`（路径由 `oss_local_fallback_dir` 控制）。

### 2.3 retrieval_mem（混合检索）

`HybridRetrieval.retrieve()` 的流水线：

1. bge-m3 编码 query：dense + sparse；
2. 三路召回：
   - Qdrant **dense** 检索
   - Qdrant **sparse** 检索（bge-m3 lexical weights）
   - **BM25**（基于用户级倒排）
3. **RRF** 融合 → `rrf_score`；
4. **时间衰减**（`exp(-Δt/τ)`，τ 默认 30 天）+ `importance` 加权 + `confidence` 加权
   → `final_score` 排序；
5. **bge-rerank**（cross-encoder，可关）；
6. **LLM rerank**（可选，关闭以省成本）；
7. **MMR** 多样性筛选 → top-k。

命中结果会通过 `MemoryStoreClient.bump_retrieval` 自动更新
`retrieval_count` 与 `last_retrieved_at`。

---

## 3. 配置说明

### 3.1 `.env`（只放敏感信息）

```ini
# DeepSeek (LLM)
deepseek_api_key=sk-xxxxxxxx
deepseek_base_url=https://api.deepseek.com

# Qdrant
qdrant_url=https://<your-cluster>.qdrant.io
qdrant_api_key=xxxxxxxx

# MySQL
mysql_host=127.0.0.1
mysql_port=3306
mysql_user=root
mysql_password=xxxx
mysql_database=memA

# Aliyun OSS
oss_bucket_name=memA
oss_endpoint=https://oss-cn-hangzhou.aliyuncs.com
oss_access_key_id=xxxx
oss_access_key_secret=xxxx
```

### 3.2 `start.sh`（业务参数，按 `config/setting.py` 期望的 key 名导出）

`start.sh` 把所有非敏感参数显式 `export`，覆盖 `.env`。常用项：

| 维度 | 变量 | 说明 |
| --- | --- | --- |
| 路由 | `mysql_use_sqlite` | `true` 本地 SQLite；`false` 走 .env 的 MySQL |
| LLM | `deepseek_model` `deepseek_temperature` `deepseek_max_tokens` | 业务调参 |
| Qdrant | `qdrant_collection_name` `qdrant_dense_dim` `qdrant_distance` | 集合 / 向量维度 |
| 检索 | `retrieval_top_k` `retrieval_dense_threshold` `merge_similarity_threshold` `rrf_rank_constant` `mmr_lambda` `time_decay_tau_days` | 召回 / 融合 / 去重 |
| 切片 | `topic_token_threshold` `topic_similarity_threshold` | 主题切片 |
| 压缩 | `compressor_model_path` `compressor_rate` | LLMLingua-2 |
| Rerank | `bge_rerank_path` `bge_rerank_device` | bge-reranker-v2-m3 |
| Embedding | `embedding_host` `embedding_port` `embedding_model` | bge-m3 gRPC |
| Web | `web_port` `service_host` | Streamlit 监听 |

`bash start.sh show-config` 可打印当前生效配置（含 mask 后的密钥）。

---

## 4. 启动 / CLI

### 4.1 一键启动 Web 服务

```bash
bash start.sh                  # 等价于 bash start.sh web
# 默认在 0.0.0.0:8501 开 Streamlit，浏览器访问即可对话 + 看记忆
```

### 4.2 CLI 子命令（同一份 .env / start.sh 环境）

```bash
# 抽取一段对话的记忆
bash start.sh extract --user-id user_001 --history-file deployment/sample_history.json

# 对一个用户跑 sleep mode（聚类合并 + 生成 category.md → OSS）
bash start.sh update  --user-id user_001

# 在线混合检索
bash start.sh retrieve --user-id user_001 --query "用户喜欢什么口味" --top-k 5

# 查看当前配置
bash start.sh show-config
```

> 这些子命令本质上等价于 `python -m src.main <sub> ...`，只是 `start.sh` 帮你把
> `.env` 加载 + 业务参数导出 + `PYTHONPATH` 都一次性处理好了。

也可以直接调用 `MemoryService`：

```python
from src.services.memory_service import MemoryService

svc = MemoryService()

# 抽取
svc.extract_memory(
    user_id="user_001",
    session_id="session_001",
    conversation=[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
)

# 离线合并 + 写 OSS
svc.update_user_memory(user_id="user_001")

# 在线检索
hits = svc.retrieve_memory(user_id="user_001", query="用户最近压力大吗", top_k=5)
```

---

## 5. MemoryItem schema（与代码一致）

```python
@dataclass
class MemoryItem:
    id: str
    user_id: str
    memory_content: str
    memory_type: str            # profile / episodic / state / core
    memory_category: str        # fact / preference / ... / unresolved_event / ...
    status: str = "active"      # active / archived / conflicted / outdated / deleted
    created_at: str
    updated_at: str
    importance: float           # 0~1，LLM 抽取时强制输出
    confidence: float           # 0~1，LLM 抽取时强制输出
    embedding: List[float]
    retrieval_count: int
    last_retrieved_at: str
    source_session_ids: List[str]
    derived_from_memory_ids: List[str]
    content_hash: str
    metadata: Dict
```

MySQL / SQLite DDL 见 `src/db/mysql/sql`。

---

## 6. 依赖

```bash
pip install -r deployment/requirements.txt
```

> bge-rerank、llmlingua、OSS、真实 MySQL 都做了软依赖：
> 没安装对应包或路径不可达时，对应能力会自动降级（保证管线不中断），
> 但建议生产环境装齐。
