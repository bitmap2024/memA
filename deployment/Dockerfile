FROM registry-vpc.cn-beijing.aliyuncs.com/benepal/benepel-mem-base:0.0.1

WORKDIR /app

# 复制额外的依赖文件并安装
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制整个项目代码
COPY . /app

# 如果需要暴露端口，取消下面的注释
# EXPOSE 8000

# 启动命令（根据实际需要调整）
# CMD ["python", "app.py"]
