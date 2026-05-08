#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Memory Chat Web UI - 基于 Streamlit 的多轮对话界面

功能:
1. 多轮对话支持
2. 记忆检索与展示
3. 用户配置
4. ES 记忆查看（左侧边栏）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from typing import List, Dict
from loguru import logger

from web.chat_service import ChatService, create_chat_service


# 预设的用户配置列表
USER_PRESETS = [
    {"child_id": "69307b75e30c4d85873070f47ad354c3", "agent_id": "01000201328b00000000000000053e19"},
    {"child_id": "bb58dcf78ba54a58ba5529bbb2839908", "agent_id": "01000201328b00000000000000054847"},
    {"child_id": "4de37cb5f618499d8f5875c69bd31b86", "agent_id": "01000201328b000000000000000590fa"},
    {"child_id": "95b4e40594554f12906aa748dba563e2", "agent_id": "01000201329800000000000000d4f40f"},
    {"child_id": "494dab06c868402186d1ef233c3aea9f", "agent_id": "01000201328b00000000000000052045"},
    {"child_id": "f51bf9f9453e445589e34213c5278a15", "agent_id": "01000201328b0000000000000004acb6"},
    {"child_id": "af1e650d7e6b4bae88f25dfddfeb2996", "agent_id": "01000201329800000000000000d49e6f"},
    {"child_id": "348b4dc857524aefae1cda5dc0b9e480", "agent_id": "01000201329800000000000000d577d2"},
    {"child_id": "a54d595e7ff34734bdda45bf6f1322ea", "agent_id": "01000201328b0000000000000004e294"},
    {"child_id": "600e02d18d2e45429834de3fafe7806c", "agent_id": "01000101328b0000000000000000c443"},
    {"child_id": "339c3b36a71043b99849bfc0bae5f9c1", "agent_id": "01000201329800000000000000d59e9b"},
    {"child_id": "e61553b2aae64cc8a302e3044c284f9e", "agent_id": "01000201329800000000000000d7a56d"},
    {"child_id": "2a6c9c6bd913485f9013b0f0c5753e8c", "agent_id": "01000201329800000000000000cab762"},
    {"child_id": "94e4841b95bc4092946e556c0fd4dcd9", "agent_id": "01000201328b000000000000000890ba"},
    {"child_id": "ddb1556b9a89460a89d056794169000e", "agent_id": "01000201329800000000000000d568bf"},
]

def init_session_state():
    """初始化 session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_service" not in st.session_state:
        st.session_state.chat_service = None
    if "current_memories" not in st.session_state:
        st.session_state.current_memories = []
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "all_memories" not in st.session_state:
        st.session_state.all_memories = []
    if "preset_index" not in st.session_state:
        st.session_state.preset_index = 0


def init_chat_service(child_id: str, agent_id: str, use_memory: bool) -> bool:
    """初始化对话服务"""
    try:
        st.session_state.chat_service = create_chat_service(
            child_id=child_id or "default_user",
            agent_id=agent_id or "default_agent",
            use_memory=use_memory
        )
        st.session_state.initialized = True
        st.session_state.messages = []
        st.session_state.current_memories = []
        st.session_state.all_memories = []
        return True
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        st.error(f"初始化失败: {str(e)}")
        return False


def get_all_memories() -> List[Dict]:
    """获取用户所有记忆"""
    if st.session_state.chat_service is None:
        return []
    try:
        memories = st.session_state.chat_service.get_all_user_memories()
        return memories
    except Exception as e:
        logger.warning(f"获取记忆失败: {e}")
        return []


def display_memories_by_type(memories: List[Dict]):
    """按类型分组显示记忆"""
    if not memories:
        st.info("暂无记忆数据")
        return
    
    type_names = {
        "factual": "📌 事实记忆",
        "preference": "❤️ 偏好记忆",
        "ability": "🎯 能力与发展",
        "relationship": "👥 社会关系",
        "portrait": "🎭 性格与画像",
        "事实记忆": "📌 事实记忆",
        "偏好记忆": "❤️ 偏好记忆",
        "能力与发展": "🎯 能力与发展",
        "社会关系": "👥 社会关系",
        "性格与画像": "🎭 性格与画像",
    }
    
    grouped = {}
    for mem in memories:
        mem_type = mem.get("memory_type", "other")
        type_name = type_names.get(mem_type, f"📋 {mem_type}")
        if type_name not in grouped:
            grouped[type_name] = []
        grouped[type_name].append(mem)
    
    st.caption(f"共 {len(memories)} 条记忆")
    
    for type_name, mems in grouped.items():
        with st.expander(f"{type_name} ({len(mems)})", expanded=False):
            for mem in mems:
                content = mem.get("memory_content", "")
                updated_at = mem.get("updated_at", "")
                st.markdown(f"• {content}")
                if updated_at:
                    st.caption(f"  {updated_at}")


def main():
    """主函数"""
    st.set_page_config(
        page_title="Memory Chat",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 自定义样式
    st.markdown("""
    <style>
    .stChatMessage {
        padding: 0.5rem 1rem;
    }
    .main .block-container {
        padding-bottom: 100px;
    }
    section[data-testid="stSidebar"] {
        width: 350px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    # ========== 左侧边栏 ==========
    with st.sidebar:
        st.title("🧠 Memory Chat")
        
        # 用户配置
        with st.expander("⚙️ 用户配置", expanded=True):
            # 获取当前预设
            current_preset = USER_PRESETS[st.session_state.preset_index]
            
            # 标题行带刷新按钮
            title_col, refresh_col = st.columns([4, 1])
            with title_col:
                st.caption(f"配置 {st.session_state.preset_index + 1}/{len(USER_PRESETS)}")
            with refresh_col:
                if st.button("🔀", help="切换下一组配置"):
                    st.session_state.preset_index = (st.session_state.preset_index + 1) % len(USER_PRESETS)
                    st.rerun()
            
            child_id = st.text_input(
                "用户 ID",
                value=current_preset["child_id"],
                help="输入用户的唯一标识"
            )
            
            agent_id = st.text_input(
                "Agent ID",
                value=current_preset["agent_id"],
                help="输入 Agent 的唯一标识"
            )
            
            use_memory = st.checkbox("启用记忆功能", value=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 初始化", use_container_width=True):
                    if init_chat_service(child_id, agent_id, use_memory):
                        st.success("成功!")
                        st.rerun()
            
            with col2:
                if st.button("🗑️ 清空", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.current_memories = []
                    if st.session_state.chat_service:
                        st.session_state.chat_service.clear_history()
                    st.rerun()
            
            # 显示当前状态
            if st.session_state.initialized:
                st.success(f"✅ 用户: {st.session_state.chat_service.child_id}")
            else:
                st.warning("⚠️ 请先初始化")
        
        st.divider()
        
        # 当前对话使用的记忆
        with st.expander("📝 当前对话记忆", expanded=False):
            if st.session_state.current_memories:
                for i, mem in enumerate(st.session_state.current_memories, 1):
                    st.markdown(f"{i}. {mem}")
            else:
                st.info("发送消息后显示检索到的记忆")
        
        st.divider()
        
        # 记忆库
        with st.expander("🧠 用户记忆库", expanded=False):
            if st.button("🔍 加载记忆", use_container_width=True):
                if st.session_state.initialized:
                    with st.spinner("加载中..."):
                        st.session_state.all_memories = get_all_memories()
                else:
                    st.warning("请先初始化")
            
            if st.session_state.all_memories:
                display_memories_by_type(st.session_state.all_memories)
            else:
                st.info("点击上方按钮加载记忆")
    
    # ========== 主页面 - 对话区域 ==========
    st.header("💬 对话")
    
    # 对话历史容器
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.initialized:
            st.info("👈 请在左侧配置用户信息并点击「初始化」按钮开始对话")
        
        # 显示对话历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # 输入框（在底部）
    if prompt := st.chat_input("输入你的问题...", disabled=not st.session_state.initialized):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 生成回复
            with st.chat_message("assistant"):
                if st.session_state.chat_service:
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    try:
                        for chunk in st.session_state.chat_service.chat_stream(prompt):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                        
                        # 更新当前记忆
                        st.session_state.current_memories = st.session_state.chat_service.get_current_memories()
                        
                    except Exception as e:
                        full_response = f"抱歉，发生错误: {str(e)}"
                        message_placeholder.error(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("请先初始化对话服务")


if __name__ == "__main__":
    main()
