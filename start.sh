#!/usr/bin/env bash
# ============================================================
# memA 服务启动脚本
#
#   密钥 / 私有 endpoint -> .env
#   业务参数（阈值、维度、模型路径、端口等） -> 本文件
#
# 用法：
#   bash start.sh                  # 默认启动 Streamlit Web 服务
#   bash start.sh web              # 显式启动 Streamlit Web 服务
#   bash start.sh cli <sub> ...    # 走 CLI: extract / update / retrieve / show-config
#   bash start.sh show-config      # 打印当前配置
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------
# PYTHONPATH
# ------------------------------------------------------------
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# ------------------------------------------------------------
# 1) 加载 .env（敏感信息）
#    使用 set -a 让 source 后的所有变量自动 export
# ------------------------------------------------------------
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
else
  echo "[warn] .env not found at $SCRIPT_DIR/.env" >&2
fi

# ============================================================
# 2) 业务参数（非敏感，所有人都能在这里调整）
#    凡是带密码 / API Key / 私有 endpoint 的，请放到 .env
# ============================================================

# ---- 服务监听 / Web ----
export service_host="0.0.0.0"
export service_port="51666"
export service_workers="8"
export service_debug="false"
export web_port="${web_port:-8501}"

# ---- LLM 业务调参（API_KEY / BASE_URL 走 .env）----
export deepseek_model="deepseek-chat"
export deepseek_temperature="0.4"
export deepseek_max_tokens="4096"
export deepseek_top_p="0.95"
export deepseek_timeout="60"

# ---- Qdrant 业务参数（URL / API_KEY 走 .env）----
export qdrant_collection_name="memA"
export qdrant_dense_dim="1024"
export qdrant_distance="cosine"
export qdrant_timeout="30"

# ---- 关系型存储：MySQL or SQLite 路由 ----
# true  -> 本地 SQLite（无需任何外部依赖）
# false -> 走 .env 里的 mysql_* 配置
export mysql_use_sqlite="false"
export mysql_sqlite_path="$SCRIPT_DIR/var/memA.sqlite3"

# ---- OSS 业务路径（bucket / endpoint / key 全部走 .env）----
export oss_prefix="chat_bot_memory"
export oss_local_fallback_dir="$SCRIPT_DIR/var/oss_local"

# ---- bge-m3 Embedding gRPC（内网无密码，可写本文件）----
export embedding_host="127.0.0.1"
export embedding_port="50051"
export embedding_model="bge-m3"
export embedding_pool_size="5"
export embedding_timeout="30"

# ---- LLMLingua-2 文本压缩 ----
export compressor_model_path="/root/chendong/hf_models/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
export compressor_rate="0.5"
export compressor_max_tokens="512"

# ---- 切片 / 检索 / 重排 业务阈值 ----
export topic_token_threshold="512"
export topic_similarity_threshold="0.55"
export retrieval_top_k="10"
export retrieval_dense_threshold="0.4"
export merge_similarity_threshold="0.9"
export rrf_rank_constant="60"
export mmr_lambda="0.5"
export time_decay_tau_days="30"

# ---- BGE Reranker ----
export bge_rerank_path="/root/chendong/hf_models/BAAI/bge-reranker-v2-m3"
export bge_rerank_device="cpu"

# ------------------------------------------------------------
# 3) 本地目录准备
# ------------------------------------------------------------
mkdir -p "$SCRIPT_DIR/var/oss_local"
if [ "$mysql_use_sqlite" = "true" ]; then
  mkdir -p "$(dirname "$mysql_sqlite_path")"
fi

# ------------------------------------------------------------
# 4) 选择 python 解释器
# ------------------------------------------------------------
if command -v python >/dev/null 2>&1; then
  PY="python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "[error] python not found in PATH" >&2
  exit 127
fi

# ------------------------------------------------------------
# 5) 启动横幅
# ------------------------------------------------------------
echo "============================================================"
echo "  memA: starting"
echo "  PROJECT_DIR : $SCRIPT_DIR"
echo "  PYTHON      : $($PY -V 2>&1)"
echo "  USE_SQLITE  : $mysql_use_sqlite"
echo "  LLM Model   : $deepseek_model"
echo "  Qdrant      : ${qdrant_url:-<missing>}  (collection=$qdrant_collection_name)"
echo "  Web Port    : $web_port"
echo "============================================================"

# ------------------------------------------------------------
# 6) 分发命令
# ------------------------------------------------------------
case "${1:-web}" in
  web)
    # Streamlit 不会自动 load .env；前面已经 export，所以子进程能继承
    exec "$PY" -m streamlit run web/app.py \
      --server.address "$service_host" \
      --server.port    "$web_port"
    ;;

  cli)
    shift
    exec "$PY" -m src.main "$@"
    ;;

  extract|update|retrieve|show-config)
    exec "$PY" -m src.main "$@"
    ;;

  *)
    echo "Usage: bash start.sh [web | cli <sub> | extract | update | retrieve | show-config]" >&2
    exit 2
    ;;
esac
