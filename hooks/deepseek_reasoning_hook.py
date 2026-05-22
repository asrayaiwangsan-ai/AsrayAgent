import os
import copy
from hook_base import BaseHook
from langchain_core.messages import AIMessage, SystemMessage,ToolMessage,BaseMessage

class DeepSeekReasoningHook(BaseHook):

    async def run(self, msg_list: list, extra: dict, n_msg: BaseMessage = None) -> list:
        # 仅在 before_invoke 阶段且上一条是 ToolMessage 时执行
        if n_msg is not None or len(msg_list) < 2 or not isinstance(msg_list[-1], ToolMessage):
            return msg_list

        # DeepSeek 的规则：在包含 tool_calls 的 AIMessage 中，必须包含 reasoning_content
        # 否则模型可能会在后续对话中由于上下文结构问题导致输出异常
        
        # 寻找触发本次 ToolMessage 的 AIMessage
        # 通常结构是 [..., AIMessage(tool_calls), ToolMessage]
        llm = extra.get("_llm")
        
        if llm.model != "deepseek-v4-pro":
            return msg_list
        
        last_ai_msg = None
        for m in reversed(msg_list):
            if isinstance(m, AIMessage) and m.tool_calls:
                last_ai_msg = m
                break
        
        if last_ai_msg:


            last_ai_msg.additional_kwargs["set_add_reasoning"] = True
            print(f"[DeepSeekHook] 已恢复推理内容到 AIMessage (ID: {getattr(last_ai_msg, 'id', 'unknown')})")
            
        return msg_list
