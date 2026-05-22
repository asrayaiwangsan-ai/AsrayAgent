from hook_base import BaseHook
from langchain_core.messages import ToolMessage
import copy

class ToolMessageCompressHook(BaseHook):
    """
    专门用于压缩旧的工具消息输出。
    它会保留最近 n 条消息不被压缩，确保模型对当前正在进行的对话保持完整感知。
    """
    def __init__(self, conf: dict = None):
        self.conf = conf or {}
        self.max_len = self.conf.get("max_tool_output_length", 500)
        # 头部保留长度
        self.head_len = self.conf.get("head_length", 100)
        # 尾部保留长度
        self.tail_len = self.conf.get("tail_length", 100)
        # 保留最近多少条消息不压缩 (n 轮)
        self.keep_n = self.conf.get("keep_recent_n", 10)

    async def run(self, msg_list: list, extra: dict, n_msg=None) -> list:
        # 如果消息太少，直接返回
        if len(msg_list) <= self.keep_n:
            return msg_list
            
        new_msg_list = []
        
        # 压缩分界线：索引小于此值的旧消息将被检查是否需要压缩
        cutoff_idx = len(msg_list) - self.keep_n

        for i, msg in enumerate(msg_list):
            # 只有 ToolMessage 且处于“旧消息”区域才需要考虑压缩
            if i < cutoff_idx and isinstance(msg, ToolMessage):
                content = msg.content
                
                # 处理纯文本内容
                if isinstance(content, str) and len(content) > self.max_len:
                    msg = copy.deepcopy(msg)
                    msg.content = (
                        content[:self.head_len] +
                        f"\n\n[... 已自动压缩 {len(content) - self.head_len - self.tail_len} 字符以节省空间 ...]\n\n" +
                        content[-self.tail_len:]
                    )
                
                # 处理多模态或列表格式的内容
                elif isinstance(content, list):
                    modified = False
                    new_content = []
                    for item in content:
                        if item.get("type") == "text" and len(item.get("text", "")) > self.max_len:
                            text = item["text"]
                            item = copy.deepcopy(item)
                            item["text"] = (
                                text[:self.head_len] +
                                f"\n\n[... 此处大段文本内容已折叠 ...]\n\n" +
                                text[-self.tail_len:]
                            )
                            modified = True
                        new_content.append(item)
                    if modified:
                        msg = copy.deepcopy(msg)
                        msg.content = new_content
            
            new_msg_list.append(msg)
            
        return new_msg_list
