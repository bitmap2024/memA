#!/bin/bash
# Memory Chat Web UI 启动脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 切换到项目目录
cd "$PROJECT_DIR"

# 设置 Python 路径
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# 默认端口
PORT=${1:-8501}

echo "============================================"
echo "  Memory Chat Web UI (Streamlit)"
echo "============================================"
echo "项目目录: $PROJECT_DIR"
echo "端口: $PORT"
echo "============================================"

# 检查依赖
echo "检查依赖..."
pip install streamlit -q 2>/dev/null || echo "请确保已安装 streamlit: pip install streamlit"

# 启动服务
echo "启动 Web UI..."
streamlit run web/app.py --server.port $PORT --server.address 0.0.0.0
