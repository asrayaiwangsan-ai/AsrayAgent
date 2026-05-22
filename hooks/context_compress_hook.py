import os
import re
from hook_base import BaseHook
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage,BaseMessage
from hooks.multimodal_hook import MultimodalHook

class ContextCompressHook(BaseHook):
    """
    智能上下文压缩 Hook。当消息数量超过阈值时，调用 LLM 对旧消息进行总结并替换。
    """
    def __init__(self, conf:dict):
        self.threshold = conf.get("threshold",50)
        self.keep_recent = conf.get("keep_recent",10) 
        self.multimodal_converter = MultimodalHook()

    async def run(self, msg_list: list, extra: dict, n_msg: BaseMessage = None) -> list:
        # 仅在 after_invoke 阶段且消息数超过阈值时触发
        if len(msg_list) <= self.threshold or not isinstance(msg_list[-1], AIMessage):
            return msg_list

        llm = extra.get("_llm")
        if not llm:
            return msg_list

        # 1. 识别 SystemMessage
        system_msg = None
        start_idx = 0
        if len(msg_list) > 0 and isinstance(msg_list[0], SystemMessage):
            system_msg = msg_list[0]
            start_idx = 1
        
        # 2. 寻找切分点 (从后往前找最近的一个 AIMessage)
        split_idx = len(msg_list) - self.keep_recent
        while split_idx > start_idx:
            if isinstance(msg_list[split_idx], AIMessage):
                break
            split_idx -= 1
            
        # 如果没找到合适的 AIMessage 作为结束，则不进行压缩
        if split_idx <= start_idx or not isinstance(msg_list[split_idx], AIMessage):
            return msg_list

        print(f"\n[ContextCompressHook] 触发自动记忆闭环，当前消息数: {len(msg_list)}，压缩区截止索引: {split_idx}")

        # 3. 准备压缩数据
        to_compress = msg_list[start_idx : split_idx+1]
        # 添加引导信息确保 LLM 知道开始执行总结
        to_compress.append(HumanMessage(content="请开始对上述对话进行高保真压缩和长期记忆提取。"))
        
        to_compress = await self.multimodal_converter.run(to_compress, {}, None)

        active_msgs = msg_list[split_idx +1 :]

        # 4. 构造基于主题的智能压缩 Prompt
        prompt = (
            "你现在是高保真记忆架构师。你的任务是将一段漫长的历史对话“折叠”成结构化的语义记忆，确保未来的你（AI）能精准接续工作。\n\n"
            "请按以下逻辑进行压缩并输出：\n\n"
            "<summary>: 请基于对话的主题（Topic）进行分类总结：\n"
            "   - [话题名称]: 简述目标、执行过程、最终结论或当前状态（如：已完成、进行中、待继续）。\n"
            "   - [技术决策]: 明确记录已达成的技术共识、选定的方案及其理由。\n"
            "   - [关键细节]: 保留后续任务绝对需要的特定 ID、路径、选择器或环境变量。\n"
            "   - [上下文衔接]: 明确指出当前活跃对话正在处理的是哪个话题的哪个阶段。\n\n"
            "请注意，##对话主题可能有多个  ## 如果有更早的已经被压缩的记忆，请不要丢失信息"
            "格式要求：严格按以下格式输出，严禁开场白：\n"
            "<summary>\n### [主题 A]: ...\n### [主题 B]: ...\n</summary>\n"
        )
        
        # 补充更早的背景
        if system_msg and "--- 历史对话摘要 (已压缩) ---" in system_msg.content:
            earlier_mem = system_msg.content.split("--- 历史对话摘要 (已压缩) ---")[1].strip()
            prompt += f"\n\n以下是更早的已经被压缩的记忆：\n{earlier_mem}"
            
        try:
            msgs = [SystemMessage(content=prompt)] + to_compress
            summary_response = await llm.ainvoke(msgs)
            content = summary_response.content
            
            # 5. 解析并处理结果
            summary_match = re.search(r'<summary>(.*?)</summary>', content, re.DOTALL)
            insight_match = re.search(r'<user_insight>(.*?)</user_insight>', content, re.DOTALL)
            
            summary_text = summary_match.group(1).strip() if summary_match else content.strip()
            # user_insight = insight_match.group(1).strip() if insight_match else ""

            print(f"[ContextCompressHook] 总结完成。摘要长度: {len(summary_text)} , content{summary_text}")

            # 6. 自动更新长期记忆 (user.md)
            # if user_insight:
            #     print(f"[ContextCompressHook] 发现长期记忆洞察，自动同步中: {user_insight}...")
            #     from base_tools import update_user_memory
            #     try:
            #         # update_user_memory 是一个 async tool
            #         await update_user_memory.ainvoke({"new_insight": user_insight})
            #     except Exception as e:
            #         print(f"[ContextCompressHook] 自动同步长期记忆失败: {str(e)}")

        except Exception as e:
            print(f"[ContextCompressHook] 压缩过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return msg_list

        # 7. 构造新的消息列表
        new_msg_list = []
        
        # 提取原始 SystemMessage 内容（保留 soul.md 等原始内容）
        raw_system_content = system_msg.content if system_msg else ""
        if "--- 历史对话摘要 (已压缩) ---" in raw_system_content:
            raw_system_content = raw_system_content.split("--- 历史对话摘要 (已压缩) ---")[0].strip()
        
        # 注入新总结
        summary_tag = f"\n\n--- 历史对话摘要 (已压缩) ---\n{summary_text}\n--- 以上为历史背景 ---"
        new_system_msg = SystemMessage(content=raw_system_content + summary_tag)
        new_msg_list.append(new_system_msg)
        
        # 拼回活跃区消息
        new_msg_list.extend(active_msgs)

        # 8. 更新状态
        msg_list.clear()
        msg_list.extend(new_msg_list)

        return new_msg_list
