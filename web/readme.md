# Memory Chat Web UI

`web` 目录提供一个基于 Streamlit 的记忆对话调试界面。它把 `ChatService`、LLM 调用和在线记忆检索串在一起，方便用指定的 `child_id` / `agent_id` 进行多轮对话、观察本轮命中的记忆，并查看该用户在 Qdrant 中的记忆数据。

## 功能

- 多轮对话：保留当前页面会话内的上下文历史。
- 记忆增强：发送消息前会按用户输入检索相关记忆，并注入到 LLM 上下文中。
- 流式输出：使用 `chat_stream()` 实时展示模型回复。
- 用户切换：左侧内置多组 `child_id` / `agent_id` 预设，也支持手动输入。
- 记忆查看：可在侧边栏加载当前用户的全部记忆，并按记忆类型分组展示。

## 目录结构

```text
web/
├── __init__.py
├── app.py              # Streamlit 页面入口
├── chat_service.py     # 对话服务，封装 LLM、历史对话和记忆检索
├── requirements.txt    # Web UI 额外依赖
├── start.sh            # Linux/macOS 启动脚本
└── readme.md
```

## 环境要求

需要先准备好项目主体依赖，以及 Web UI 额外依赖：

```bash
cd code/horse
pip install -r deployment/requirements.base.txt
pip install -r deployment/requirements.txt
pip install -r web/requirements.txt
```

Web UI 会依赖以下外部服务或配置：

- LLM：通过 OpenAI 兼容协议调用，配置来源为 `LLM_*` 环境变量或 `config/config.py` 默认值。
- Qdrant：用于读取和检索记忆，配置来源为 `QDRANT_*`。
- Embedding 服务：用于在线记忆召回，配置来源为 `EMBEDDING_*`。

建议优先使用环境变量覆盖敏感配置，不要把真实密钥写入文档或提交到仓库：

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.example.com/v1"
export LLM_MODEL="your-model"

export QDRANT_URL="http://localhost:6333"
export QDRANT_COLLECTION="light_memory_rag"

export EMBEDDING_HOST="127.0.0.1"
export EMBEDDING_PORT="80"
```

## 启动方式

方式一：使用启动脚本。

```bash
cd code/horse
./web/start.sh
```

也可以指定端口：

```bash
./web/start.sh 8080
```

方式二：直接启动 Streamlit。

```bash
cd code/horse
export PYTHONPATH=$PWD:$PYTHONPATH
streamlit run web/app.py --server.port 8501 --server.address 0.0.0.0
```

Windows PowerShell 示例：

```powershell
cd d:\aiworks\code\horse
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
streamlit run web/app.py --server.port 8501 --server.address 0.0.0.0
```

启动后访问 `http://localhost:8501`。

## 使用说明

1. 在左侧「用户配置」里选择或输入 `child_id` 和 `agent_id`。
2. 按需勾选「启用记忆功能」。
3. 点击「初始化」，创建当前用户的对话服务实例。
4. 在底部输入框发送消息，主区域会流式显示回复。
5. 左侧「当前对话记忆」会展示本轮对话检索到并注入的记忆。
6. 左侧「用户记忆库」点击「加载记忆」，可查看当前用户的全部记忆。

记忆分组目前支持以下类型的展示映射：

- 事实记忆
- 偏好记忆
- 能力与发展
- 社会关系
- 性格与画像

## 程序化调用

如果不需要 Streamlit 页面，可以直接使用 `ChatService`：

```python
from web.chat_service import create_chat_service

service = create_chat_service(
    child_id="user_123",
    agent_id="agent_001",
    use_memory=True,
)

response = service.chat("你好")
print(response)

for chunk in service.chat_stream("给我讲个小故事"):
    print(chunk, end="", flush=True)

print(service.get_current_memories())
print(service.get_all_user_memories())
```

## 实现说明

`app.py` 负责页面和 `st.session_state` 管理；每次初始化会创建新的 `ChatService`，并清空页面内历史消息。

`chat_service.py` 负责核心对话流程：

1. 根据用户输入调用 `RetrievalMemoryService.get_relate_memory()` 检索相关记忆。
2. 将记忆拼接到当前用户输入中。
3. 调用 `LLMApi.chat_stream()` 或 `LLMApi.chat()` 获取模型回复。
4. 将用户输入和助手回复写入当前会话的 `conversation_history`。

## 常见问题

### 记忆功能没有生效

先确认「启用记忆功能」已勾选，然后检查 Qdrant、Embedding 服务和 `child_id` / `agent_id` 是否正确。若记忆服务初始化失败，`ChatService` 会自动降级为无记忆对话。

### 页面可以打开，但发送消息报错

通常是 LLM 配置或网络问题。检查 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 是否可用，并确认模型服务兼容 OpenAI Chat Completions 接口。

### 点击「加载记忆」没有数据

确认当前 `child_id` / `agent_id` 在 Qdrant 中确实有数据，并且 `QDRANT_COLLECTION`、向量维度和索引环境与写入记忆时保持一致。
