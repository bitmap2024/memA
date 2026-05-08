from typing import List, Tuple, Dict, Any, Optional
from llmlingua import PromptCompressor
import torch

class TextCompressor:
    """基于 LLMLingua-2 的文本压缩器"""
    
    # DEFAULT_MODEL_PATH = "/root/chendong/hf_models/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    DEFAULT_FORCE_TOKENS = ['\n', '.', '!', '?', ',', '、']
    DEFAULT_CHUNK_END_TOKENS = ['.', '\n']
    
    def __init__(self, model_path, max_tokens: int = 512):
        """
        初始化压缩器
        
        Args:
            model_path: 模型路径，默认使用 DEFAULT_MODEL_PATH
            max_tokens: 压缩后的最大 token 数，默认 512
        """
        # self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.max_tokens = max_tokens
        self.compressor = PromptCompressor(
            model_name=model_path,
            use_llmlingua2=True,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu"
        )
    
    def compress(
        self,
        text: str,
        rate: float = 0.5,
        force_tokens: Optional[List[str]] = None,
        chunk_end_tokens: Optional[List[str]] = None,
        return_word_label: bool = True,
        drop_consecutive: bool = True
    ) -> Dict[str, Any]:
        """
        压缩文本
        
        Args:
            text: 原始文本
            rate: 压缩率 (0-1)，越小压缩越多
            force_tokens: 强制保留的 token 列表
            chunk_end_tokens: 分块结束 token 列表
            return_word_label: 是否返回词标签
            drop_consecutive: 是否丢弃连续重复
            
        Returns:
            包含压缩结果的字典
        """
        force_tokens = force_tokens or self.DEFAULT_FORCE_TOKENS
        chunk_end_tokens = chunk_end_tokens or self.DEFAULT_CHUNK_END_TOKENS
        
        results = self.compressor.compress_prompt_llmlingua2(
            text,
            rate=rate,
            force_tokens=force_tokens,
            chunk_end_tokens=chunk_end_tokens,
            return_word_label=return_word_label,
            drop_consecutive=drop_consecutive
        )
        return results
    
    def get_annotated_results(
        self,
        results: Dict[str, Any],
        word_sep: str = "\t\t|\t\t",
        label_sep: str = " "
    ) -> List[Tuple[str, str]]:
        """
        从压缩结果中提取标注结果
        
        Args:
            results: compress() 方法返回的结果
            word_sep: 词之间的分隔符
            label_sep: 词和标签之间的分隔符
            
        Returns:
            标注结果列表，每个元素为 (word, label) 元组，label 为 '+' 或 '-'
        """
        lines = results["fn_labeled_original_prompt"].split(word_sep)
        annotated_results = []
        for line in lines:
            word, label = line.split(label_sep)
            annotated_results.append((word, '+') if label == '1' else (word, '-'))
        return annotated_results
    
    def compress_and_annotate(
        self,
        messages: List[Dict[str, str]],
        rate: float = 0.5,
        **kwargs
    ) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
        """
        压缩文本并返回标注结果
        
        Args:
            text: 原始文本
            rate: 压缩率
            **kwargs: 传递给 compress() 的其他参数
            
        Returns:
            (压缩结果字典, 标注结果列表)
        """
        for mes in messages:
            content = mes["content"]
            if not content or not content.strip():
                continue
            
            results = self.compress(content, rate=rate, **kwargs)
            # print(f"results: {results}")
            while results["compressed_tokens"] > self.max_tokens:
                content = results["compressed_prompt"]
                results = self.compress(content, rate=rate, **kwargs)
            mes["content"] = results["compressed_prompt"]
        return messages


if __name__ == "__main__":
    # 使用示例
    import json
    compressor = TextCompressor(model_path="/root/chendong/hf_models/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
    turn_messages = [
        {"role": "user", "content": "咱们先把这些航班订好吧！接下来，您能否推荐一些坦帕的住宿，位置便利，方便游览景点，并且符合我单人入住的预算？", "time_stamp": "2022-03-20 13:21:00", "speaker_name": "User", "speaker_id": "speaker_a"},
        {"role": "assistant", "content": "绝佳选择！对于单人入住且注重便利性的人来说，坦帕市中心是理想之选。我推荐以下房源：\n- 位于纽约市的宽敞一卧公寓（整套公寓，每晚 547 美元，至少入住一晚，评分 5.0）\n- 主要信息：靠近河滨步道（步行即可到达众多景点），禁止吸烟/举办派对，可容纳两人（空间充足）——两晚总价：1,094 美元。\n这样，您的剩余预算为 1,800 美元 - 360 美元（机票）- 1,094 美元（住宿）= 346 美元，可用于餐饮、景点和当地交通。这个住宿方案符合您的需求吗？", "time_stamp": "2022-03-20 13:21:00", "speaker_name": "Assistant", "speaker_id": "speaker_b"}
    ]
    print(json.dumps(turn_messages, indent=4, ensure_ascii=False))
    turn_messages = compressor.compress_and_annotate(turn_messages, rate=0.5)

    print(f"Compressed turn_messages: {turn_messages}")
    print(json.dumps(turn_messages, indent=4, ensure_ascii=False))
        
# [{'role': 'user', 'content': 'Let’s lock in these flights! Next, could you recommend accommodations in Tampa that are convenient for attractions, and fit my single occupancy budget?', 'time_stamp': '2022-03-20 13:21:00', 'speaker_name': 'User', 'speaker_id': 'speaker_a'}, {'role': 'assistant', 'content': 'Great choice! For single occupancy and convenience, Tampa’s downtown area is ideal. I recommend:\n- Spacious 1+ bedroom apt which was located in NYC (Entire home/apt, $547/night, 1-night minimum, 5.0 review rate)\n- Key details: Near the Riverwalk (walkable to many attractions), no smoking/parties, fits 2 people (plenty of space for you) – total for 2 nights: $1,094.\nThis keeps your remaining budget at $1,800 - $360 (flights) - $1,094 (accommodations) = $346 for meals, attractions, and local transport. Does this accommodation work for you?', 'time_stamp': '2022-03-20 13:21:00', 'speaker_name': 'Assistant', 'speaker_id': 'speaker_b'}]

# [{'role': 'user', 'content': 'One last question: Is there a place to rent a bike near the Riverwalk for Day 1’s stroll?', 'time_stamp': '2022-03-20 13:21:00', 'speaker_name': 'User', 'speaker_id': 'speaker_a'}, {'role': 'assistant', 'content': 'Yes! Tampa Bay Bike Share has a station at 100 N Ashley Dr – 5-minute walk from your apartment (Riverwalk-near). Details:\n- $15/day pass, unlimited 30-minute rides\n- Booking link: [Tampa Bay Bike Share](https://tampabaybikeshare.org/)\n\nLast tips: The Florida Aquarium gets busy on weekends (arrive by 10 AM), and your apartment has a kitchenette (stock up on water at Publix to save). Enjoy your trip – let me know if you need last-minute tweaks!', 'time_stamp': '2022-03-20 13:21:00', 'speaker_name': 'Assistant', 'speaker_id': 'speaker_b'}]


