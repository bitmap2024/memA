#!/bin/bash
# @File    : start.sh
# Memory 服务启动脚本

set -x

CURRENT_DIR=$(pwd)
export CURRENT_DIR
export PYTHONPATH="$CURRENT_DIR:$CURRENT_DIR/protos"


# ==================== 服务配置 ====================
export SERVICE_HOST="0.0.0.0"
export SERVICE_PORT="51666"
export SERVICE_WORKERS="100"
export SERVICE_DEBUG="false"


# 设置 WhaleSettings 相关环境变量
export WHALE_HOST="http://172.16.0.9:8000"

# ==================== LLM 配置 ====================
export LLM_API_KEY="sk-b7d48e7fc9ef44b4b33c80470402d52a"
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-chat"
export LLM_TEMPERATURE="0.7"
export LLM_MAX_TOKENS="2048"
export LLM_TOP_P="0.95"
export LLM_STREAM="false"
export LLM_TIMEOUT="60"

# ==================== Elasticsearch 配置 ====================
# es测试环境
# export ES_ADDRESS="http://es-cn-3ic48eimi0004zvpu.elasticsearch.aliyuncs.com:9200"
# export ES_USERNAME="elastic"
# export ES_PASSWORD="PEGBRK_dcMBh+5vU8M"
# es线上环境
export ES_ADDRESS="http://es-cn-zp5471vy60004ply7.elasticsearch.aliyuncs.com:9200"
export ES_USERNAME="elastic"
export ES_PASSWORD="5d2rcs_ES_elastic"
export ES_MEMORY_INDEX="light_memory_rag"

# 设置 EmbeddingConfig 相关环境变量
export EMBEDDING_HOST="172.16.2.124"
export EMBEDDING_PORT=80
export EMBEDDING_POOL_SIZE=5
export EMBEDDING_TIMEOUT=100

# ==================== PreCompressor 配置 ====================
export COMPRESSOR_MODEL_NAME="llmlingua-2"
export PRE_COMPRESSOR_MODEL_PATH="/root/chendong/hf_models/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
export PRE_COMPRESSOR_CONFIGS="{}"

# # ==================== TopicSegmenter 配置 ====================
# export TOPIC_SEGMENTER_MODEL_NAME="llmlingua-2"
# export TOPIC_SEGMENTER_MODEL_PATH="/root/chendong/hf_models/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
# export TOPIC_SEGMENTER_CONFIGS="{}"

# ==================== Memory 配置 ====================
export MEMORY_OFFLINE_HOUR_RANGE="24"
export MEMORY_SENSORY_BUFFER_SIZE="10"
export MEMORY_SHORT_TERM_MAX_SIZE="100"
export MEMORY_SIMILARITY_THRESHOLD="0.8"

# ==================== 启动服务 ====================
echo "Starting Memory Service..."
echo "Host: ${SERVICE_HOST}"
echo "Port: ${SERVICE_PORT}"

# 开发模式
python main.py



