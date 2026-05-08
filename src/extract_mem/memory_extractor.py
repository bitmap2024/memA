from api.llm_api import LLMApi, LLMConfig
from config.mem_prompt import EXTRACT_MEMORY_PROMPT
from config.config import Config
from loguru import logger

class LLmExtractor:
    def __init__(self):
        config = LLMConfig(api_key=Config.llm.API_KEY, base_url=Config.llm.BASE_URL, model=Config.llm.MODEL)
        self.llm = LLMApi(config)

    def extract(self, text: str) -> str:
        messages = [{"role": "user", "content": EXTRACT_MEMORY_PROMPT.format(text=text)}]
        # logger.debug(f"extract messages: {messages}")
        response = self.llm.chat(messages)
        return response
    
    
if __name__ == "__main__":
    # config = Config()
    extractor = LLmExtractor()
    text = """
    --- Topic 1 ---
    [2022-03-20T13:21:00.000, Sun] 0.User: 我想要去巴黎玩，给我推荐一些美食.
    [2022-03-20T13:21:00.500, Sun] 0.Assistant: 好的，给你推荐一些巴黎的美食，比如法式蜗牛、鹅肝、奶酪、红酒、法式面包等.
    [2022-03-20T13:21:00.1000, Sun]0.USer: 我不喜欢，我喜欢烤肉，有好吃的烤肉店吗？
    [2022-03-20T13:21:00.1500, Sun]0.Assistant: 好的，给你推荐一些巴黎的烤肉店，比如巴黎烤肉、巴黎烤肉、巴黎烤肉等.
    --- Topic 2 ---
    [2022-03-20T13:21:01.000, Sun] 4.User: 我想要去巴黎玩，给我推荐一些景点.
    [2022-03-20T13:21:01.500, Sun] 4.Assistant: 好的，给你推荐一些巴黎的景点，比如埃菲尔铁塔、卢浮宫、圣母院、凯旋门等.
       --- Topic 2 ---
    [2022-03-20T13:21:01.000, Sun] 4.User: 我妈妈给我买了最喜欢的奥特曼，我好开心.
    [2022-03-20T13:21:01.500, Sun] 4.Assistant: 真好啊，你有没有谢谢妈妈.
    
    """
    response = extractor.extract(text)
    print(response)
    
