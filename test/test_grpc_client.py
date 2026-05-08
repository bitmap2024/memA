#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Memory gRPC 服务测试客户端
测试 MemoryService 的所有 gRPC 接口
"""

import sys
import os
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'protos'))

import grpc
from protos import mem_pb2
from protos import mem_pb2_grpc


class MemoryGrpcClient:
    """Memory 服务 gRPC 客户端"""
    
    def __init__(self, host: str = "localhost", port: int = 51666):
        self.address = f"{host}:{port}"
        self.channel = grpc.insecure_channel(self.address)
        self.stub = mem_pb2_grpc.MemoryServiceStub(self.channel)
        print(f"已连接到 Memory 服务: {self.address}")
    
    def close(self):
        """关闭连接"""
        self.channel.close()
    
    def get_active_interaction_memory(self, child_id: str, child_name: str = "", agent_id: str = None, limit: int = None):
        """获取活跃交互记忆（最近30天）"""
        request = mem_pb2.MemRequest(
            child_id=child_id,
            child_name=child_name
        )
        if agent_id:
            request.agent_id = agent_id
        if limit:
            request.limit = limit
        
        try:
            response = self.stub.GetActiveInteractionMemory(request, timeout=30)
            return response
        except grpc.RpcError as e:
            print(f"RPC 错误: {e.code()} - {e.details()}")
            return None
    
    def get_ican_memory(self, child_id: str, child_name: str = "", agent_id: str = None):
        """获取 ICan 相关记忆（能力记忆）"""
        request = mem_pb2.MemRequest(
            child_id=child_id,
            child_name=child_name
        )
        if agent_id:
            request.agent_id = agent_id
        
        try:
            response = self.stub.GetICanMemory(request, timeout=30)
            return response
        except grpc.RpcError as e:
            print(f"RPC 错误: {e.code()} - {e.details()}")
            return None
    
    def get_agent_all_memory(self, child_id: str, agent_id: str, child_name: str = ""):
        """获取指定 Agent 的所有记忆"""
        request = mem_pb2.MemRequest(
            child_id=child_id,
            child_name=child_name,
            agent_id=agent_id
        )
        
        try:
            response = self.stub.GetAgentAllMemory(request, timeout=30)
            return response
        except grpc.RpcError as e:
            print(f"RPC 错误: {e.code()} - {e.details()}")
            return None
    
    def get_child_all_memory(self, child_id: str, child_name: str = ""):
        """获取指定用户的所有记忆（不限 agent）"""
        request = mem_pb2.MemRequest(
            child_id=child_id,
            child_name=child_name
        )
        
        try:
            response = self.stub.GetChildAllMemory(request, timeout=30)
            return response
        except grpc.RpcError as e:
            print(f"RPC 错误: {e.code()} - {e.details()}")
            return None
    
    def get_relate_memory(self, child_id: str, query: str, child_name: str = "", agent_id: str = None, intent: str = None, limit: int = 10):
        """根据 query 检索相关记忆"""
        request = mem_pb2.MemRequest(
            child_id=child_id,
            child_name=child_name,
            query=query,
            limit=limit
        )
        if agent_id:
            request.agent_id = agent_id
        if intent:
            request.intent = intent
        
        try:
            response = self.stub.GetRelateMemory(request, timeout=30)
            print(response)
            return response
        except grpc.RpcError as e:
            print(f"RPC 错误: {e.code()} - {e.details()}")
            return None


def print_response(title: str, response):
    """打印响应结果"""
    print("\n" + "=" * 60)
    print(f"【{title}】")
    print("=" * 60)
    
    if response is None:
        print("请求失败，无响应")
        return
    
    print(f"状态码: {response.code}")
    print(f"状态信息: {response.message}")
    print(f"记忆数量: {len(response.data)}")
    
    if response.data:
        print("-" * 40)
        for i, content in enumerate(response.data[:5]):  # 只显示前5条
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"\n[{i+1}] {content}")
        
        if len(response.data) > 5:
            print(f"\n... 还有 {len(response.data) - 5} 条记忆未显示")


def test_all_apis(client: MemoryGrpcClient, child_id: str, child_name: str = "", agent_id: str = None):
    """测试所有 API"""
    
    # 1. 测试获取活跃交互记忆
    response = client.get_active_interaction_memory(child_id, child_name, agent_id, limit=10)
    print_response("GetActiveInteractionMemory - 活跃交互记忆（最近30天）", response)
    
    # 2. 测试获取 ICan 记忆
    response = client.get_ican_memory(child_id, child_name, agent_id)
    print_response("GetICanMemory - 能力记忆", response)
    
    # 3. 测试获取指定 Agent 的所有记忆
    if agent_id:
        response = client.get_agent_all_memory(child_id, agent_id, child_name)
        print_response("GetAgentAllMemory - Agent所有记忆", response)
    else:
        print("\n跳过 GetAgentAllMemory 测试（未提供 agent_id）")
    
    # 4. 测试获取用户所有记忆
    response = client.get_child_all_memory(child_id, child_name)
    print_response("GetChildAllMemory - 用户所有记忆", response)
    
    # 5. 测试相关记忆检索
    query = "学习数学"
    response = client.get_relate_memory(child_id, query, child_name, agent_id, limit=5)
    print_response(f"GetRelateMemory - 相关记忆检索 (query='{query}')", response)


def interactive_mode(client: MemoryGrpcClient):
    """交互模式"""
    print("\n" + "=" * 60)
    print("Memory gRPC 服务交互测试")
    print("=" * 60)
    
    while True:
        print("\n请选择要测试的接口：")
        print("1. GetActiveInteractionMemory - 获取活跃交互记忆")
        print("2. GetICanMemory - 获取能力记忆")
        print("3. GetAgentAllMemory - 获取 Agent 所有记忆")
        print("4. GetChildAllMemory - 获取用户所有记忆")
        print("5. GetRelateMemory - 相关记忆检索")
        print("6. 测试所有接口")
        print("0. 退出")
        
        choice = input("\n请输入选项 (0-6): ").strip()
        
        if choice == "0":
            print("退出测试")
            break
        
        child_id = input("请输入 child_id: ").strip()
        if not child_id:
            print("child_id 不能为空")
            continue
        child_name = input("请输入 child_name (可选，回车跳过): ").strip()
        
        if choice == "1":
            agent_id = input("请输入 agent_id (可选，回车跳过): ").strip() or None
            limit = input("请输入 limit (可选，回车跳过): ").strip()
            limit = int(limit) if limit else None
            response = client.get_active_interaction_memory(child_id, child_name, agent_id, limit)
            print_response("GetActiveInteractionMemory", response)
        
        elif choice == "2":
            agent_id = input("请输入 agent_id (可选，回车跳过): ").strip() or None
            response = client.get_ican_memory(child_id, child_name, agent_id)
            print_response("GetICanMemory", response)
        
        elif choice == "3":
            agent_id = input("请输入 agent_id (必填): ").strip()
            if not agent_id:
                print("agent_id 不能为空")
                continue
            response = client.get_agent_all_memory(child_id, agent_id, child_name)
            print_response("GetAgentAllMemory", response)
        
        elif choice == "4":
            response = client.get_child_all_memory(child_id, child_name)
            print_response("GetChildAllMemory", response)
        
        elif choice == "5":
            query = input("请输入查询内容 (query): ").strip()
            if not query:
                print("query 不能为空")
                continue
            agent_id = input("请输入 agent_id (可选，回车跳过): ").strip() or None
            intent = input("请输入 intent (可选，回车跳过): ").strip() or None
            limit = input("请输入 limit (默认10): ").strip()
            limit = int(limit) if limit else 10
            response = client.get_relate_memory(child_id, query, child_name, agent_id, intent, limit)
            print_response("GetRelateMemory", response)
        
        elif choice == "6":
            agent_id = input("请输入 agent_id (可选，回车跳过): ").strip() or None
            test_all_apis(client, child_id, child_name, agent_id)
        
        else:
            print("无效选项，请重新输入")


def main():
    parser = argparse.ArgumentParser(description="Memory gRPC 服务测试客户端")
    parser.add_argument("--host", default="localhost", help="服务地址 (默认: localhost)")
    parser.add_argument("--port", type=int, default=51666, help="服务端口 (默认: 51666)")
    parser.add_argument("--child-id", help="child_id 参数")
    parser.add_argument("--child-name", default="", help="child_name 参数（可选）")
    parser.add_argument("--agent-id", help="agent_id 参数（可选）")
    parser.add_argument("--query", help="查询内容（用于 GetRelateMemory）")
    parser.add_argument("--intent", help="意图（用于 GetRelateMemory，可选）")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--test-all", "-a", action="store_true", help="测试所有接口")
    
    args = parser.parse_args()
    
    client = MemoryGrpcClient(host=args.host, port=args.port)
    
    try:
        if args.interactive:
            interactive_mode(client)
        elif args.test_all and args.child_id:
            test_all_apis(client, args.child_id, args.child_name, args.agent_id)
        elif args.child_id:
            if args.query:
                response = client.get_relate_memory(args.child_id, args.query, args.child_name, args.agent_id, args.intent)
                print_response("GetRelateMemory", response)
            else:
                response = client.get_child_all_memory(args.child_id, args.child_name)
                print_response("GetChildAllMemory", response)
        else:
            print("使用方法:")
            print("  交互模式: python test_grpc_client.py -i")
            print("  测试所有接口: python test_grpc_client.py -a --child-id <id>")
            print("  查询相关记忆: python test_grpc_client.py --child-id <id> --query '学习数学'")
            print("  获取用户记忆: python test_grpc_client.py --child-id <id>")
            print("\n示例:")
            print("  python test_grpc_client.py -i --host localhost --port 51666")
            print("  python test_grpc_client.py -a --child-id test_child_001 --agent-id agent_001")
    finally:
        client.close()


if __name__ == "__main__":
    main()


# # 进入项目目录
# cd /root/chendong/code/benepel-mem

# # 交互模式（推荐，可以逐个测试接口）
# python test/test_grpc_client.py -i

# # 测试所有接口
# python test/test_grpc_client.py -a --child-id test_child_001

# # 测试所有接口（带 agent_id）
# python test/test_grpc_client.py -a --child-id test_child_001 --agent-id agent_001

# # 查询相关记忆
# python test/test_grpc_client.py --child-id 69307b75e30c4d85873070f47ad354c3 --query "我喜欢什么"

# # 指定服务地址和端口
# python test/test_grpc_client.py -i --host localhost --port 51666