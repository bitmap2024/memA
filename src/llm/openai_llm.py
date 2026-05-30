

from loguru import logger
from typing import List, Dict, Optional, Union, Generator
from dataclasses import dataclass, field
from openai import OpenAI


@dataclass
class LLMConfig:
    """LLM 配置类"""
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    stream: bool = False
    timeout: int = 60
    extra_params: Dict = field(default_factory=dict)


class LLMApi:
    """
    基于 OpenAI 协议的通用 LLM API 类
    
    支持:
    - OpenAI (GPT-3.5, GPT-4, etc.)
    - 通义千问 (Qwen)
    - DeepSeek
    - 本地部署模型 (vLLM, Ollama, etc.)
    - 其他兼容 OpenAI 协议的服务
    
    使用示例:
    ```python
    # 方式1: 使用配置对象
    config = LLMConfig(
        api_key="your-api-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo"
    )
    llm = LLMApi(config)
    
    # 方式2: 直接传参
    llm = LLMApi.from_params(
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",
        model="gpt-3.5-turbo"
    )
    
    # 调用
    messages = [{"role": "user", "content": "你好"}]
    response = llm.chat(messages)
    
    # 流式调用
    for chunk in llm.chat_stream(messages):
        print(chunk, end="")
    ```
    """

    def __init__(self, config: LLMConfig):
        """
        初始化 LLM API
        
        Args:
            config: LLM 配置对象
        """
        self.config = config
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout
        )

    @classmethod
    def from_params(
        cls,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        stream: bool = False,
        timeout: int = 60,
        **extra_params
    ) -> "LLMApi":
        """
        通过参数直接创建 LLMApi 实例
        
        Args:
            api_key: API 密钥
            base_url: API 基础 URL
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            top_p: top_p 采样参数
            stream: 是否默认使用流式输出
            timeout: 请求超时时间（秒）
            **extra_params: 其他额外参数
            
        Returns:
            LLMApi 实例
        """
        config = LLMConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=stream,
            timeout=timeout,
            extra_params=extra_params
        )
        return cls(config)

    def _build_params(
        self,
        messages: List[Dict[str, str]],
        stream: Optional[bool] = None,
        response_format: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        enable_thinking: Optional[bool] = False,
        **kwargs
    ) -> Dict:
        """
        构建请求参数
        
        Args:
            messages: 消息列表
            stream: 是否流式输出（覆盖默认配置）
            response_format: 响应格式
            tools: 工具列表
            tool_choice: 工具选择策略
            **kwargs: 其他参数
            
        Returns:
            请求参数字典
        """
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "stream": stream if stream is not None else self.config.stream,
            "extra_body": {"enable_thinking": enable_thinking}
        }
        
        # 添加可选参数
        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice or "auto"
            
        # 合并额外参数
        params.update(self.config.extra_params)
        params.update(kwargs)
        
        return params

    def chat(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        非流式调用 LLM
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            response_format: 响应格式，如 {"type": "json_object"}
            tools: 工具列表（用于 function calling）
            tool_choice: 工具选择策略
            **kwargs: 其他传递给 API 的参数
            
        Returns:
            模型生成的文本内容
        """
        params = self._build_params(
            messages=messages,
            stream=False,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs
        )
        
        try:
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        buffer_size: int = 0,
        enable_thinking: Optional[bool] = False,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式调用 LLM
        
        Args:
            messages: 消息列表
            response_format: 响应格式
            tools: 工具列表
            tool_choice: 工具选择策略
            buffer_size: 缓冲区大小，0 表示不缓冲（逐字符输出）
            **kwargs: 其他参数
            
        Yields:
            生成的文本片段
        """
        params = self._build_params(
            messages=messages,
            stream=True,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            enable_thinking=enable_thinking,
            **kwargs,
        )
        
        try:
            response = self.client.chat.completions.create(**params)
            
            if buffer_size > 0:
                # 带缓冲的流式输出
                buffer = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        buffer += chunk.choices[0].delta.content
                        if len(buffer) >= buffer_size:
                            yield buffer
                            buffer = ""
                if buffer:
                    yield buffer
            else:
                # 不缓冲，直接输出每个 chunk
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                        
        except Exception as e:
            logger.error(f"LLM 流式调用失败: {e}")
            raise

    def chat_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        带重试机制的 LLM 调用
        
        Args:
            messages: 消息列表
            max_retries: 最大重试次数
            **kwargs: 其他参数
            
        Returns:
            模型生成的文本内容
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                return self.chat(messages, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"LLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                
        logger.error(f"LLM 调用在 {max_retries} 次重试后仍然失败")
        raise last_error

    def get_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        简化的单轮对话接口
        
        Args:
            prompt: 用户输入
            system_prompt: 系统提示词
            **kwargs: 其他参数
            
        Returns:
            模型生成的文本内容
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, **kwargs)

    def get_json_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        获取 JSON 格式的响应
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            JSON 格式的响应文本
        """
        return self.chat(
            messages,
            response_format={"type": "json_object"},
            **kwargs
        )


# 预定义的常用配置
class LLMPresets:
    """预定义的 LLM 配置"""
    
    @staticmethod
    def qwen(api_key: str, model: str = "qwen-turbo", **kwargs) -> LLMApi:
        """通义千问"""
        return LLMApi.from_params(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=model,
            **kwargs
        )
    
    @staticmethod
    def deepseek(api_key: str, model: str = "deepseek-chat", **kwargs) -> LLMApi:
        """DeepSeek"""
        return LLMApi.from_params(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            model=model,
            **kwargs
        )
    
    @staticmethod
    def openai(api_key: str, model: str = "gpt-3.5-turbo", **kwargs) -> LLMApi:
        """OpenAI"""
        return LLMApi.from_params(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            model=model,
            **kwargs
        )
    
    @staticmethod
    def local(base_url: str, model: str = "default", **kwargs) -> LLMApi:
        """本地部署模型（如 vLLM, Ollama）"""
        return LLMApi.from_params(
            api_key="EMPTY",  # 本地模型通常不需要 API key
            base_url=base_url,
            model=model,
            **kwargs
        )


if __name__ == "__main__":
    # 使用示例
    
    # 示例1: 使用预设配置
    # llm = LLMPresets.qwen(api_key="your-api-key")
    
    # 示例2: 自定义配置
    # llm = LLMApi.from_params(
    #     api_key="your-api-key",
    #     base_url="https://your-api-endpoint/v1",
    #     model="your-model",
    #     temperature=0.7
    # )
    
    # 示例3: 基本调用
    # response = llm.chat([{"role": "user", "content": "你好"}])
    # print(response)
    
    # 示例4: 流式调用
    # for chunk in llm.chat_stream([{"role": "user", "content": "讲一个故事"}]):
    #     print(chunk, end="", flush=True)
    
    # 示例5: 简化的单轮对话
    # response = llm.get_completion("你好", system_prompt="你是一个helpful助手")
    # print(response)
    
    pass
