import os
import re
import asyncio
import base64
import mimetypes
import copy
from hook_base import BaseHook
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from asray_core import is_mutimodel, get_llm,get_defuat_mutimodel


def _encode_image(path):
    """将图片文件编码为 data URL 字符串"""
    mime_type, _ = mimetypes.guess_type(path)
    mime_type = mime_type or "image/jpeg"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return mime_type, encoded


class MultimodalHook(BaseHook):
    """
    自动将消息中的 image_ref 转换为正确的格式：
    - 多模态模型：转为 Base64 image_url
    - 非多模态模型：调用 muti_llm 将图片总结为文字描述，喂给 raw_llm
    """
    async def run(self, msg_list: list, extra: dict, n_msg: BaseMessage = None) -> list:
        if n_msg is not None:
            return msg_list

        raw_llm: ChatOpenAI = extra.get("_llm")
        model_name = raw_llm.model_name
        raw_is_multimodal = is_mutimodel(model_name)
        muti_llm = await get_llm(get_defuat_mutimodel())

        processed_messages = []
        for msg in msg_list:
            if not isinstance(msg.content, list):
                processed_messages.append(msg)
                continue

            # 收集 image_ref 路径和已有文本
            text_parts = []
            image_paths = []
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") == "image_ref":
                    path = part.get("image_ref", {}).get("path")
                    if path and os.path.exists(path):
                        image_paths.append(path)
                    else:
                        text_parts.append(f"[图片未找到: {path}]")
                elif isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))

            if not image_paths:
                processed_messages.append(msg)
                continue

            if raw_is_multimodal:
                # 多模态模型：直接嵌入 Base64 图片
                new_content = []
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "image_ref":
                        path = part.get("image_ref", {}).get("path")
                        if path and os.path.exists(path):
                            mime_type, encoded = _encode_image(path)
                            new_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{encoded}"}
                            })
                        else:
                            new_content.append({"type": "text", "text": f"[图片未找到: {path}]"})
                    else:
                        new_content.append(part)
                new_msg = copy.deepcopy(msg)
                new_msg.content = new_content
                processed_messages.append(new_msg)
            else:
                # 非多模态模型：逐张图片调用 muti_llm 描述，按原始位置插入文本流
                # 第一步：遍历 content，拆成「文本块」和「图片任务」两类，保留顺序
                content_parts = []  # ("text", str) 或 ("image", path, before_text, after_text)
                for i, part in enumerate(msg.content):
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_parts.append(("text", part.get("text", "")))
                    elif isinstance(part, dict) and part.get("type") == "image_ref":
                        path = part.get("image_ref", {}).get("path")
                        if path and os.path.exists(path):
                            before = ""
                            after = ""
                            for j in range(i - 1, -1, -1):
                                prev = msg.content[j]
                                if isinstance(prev, dict) and prev.get("type") == "text":
                                    before = prev.get("text", "")
                                    break
                            for j in range(i + 1, len(msg.content)):
                                nxt = msg.content[j]
                                if isinstance(nxt, dict) and nxt.get("type") == "text":
                                    after = nxt.get("text", "")
                                    break
                            content_parts.append(("image", path, before, after))
                        else:
                            content_parts.append(("text", f"[图片未找到: {path}]"))

                # 第二步：收集所有图片，按 10 张一组打包，每组一个 API 调用
                BATCH_SIZE = 10
                image_entries = []  # (global_index, path)
                for typ, *args in content_parts:
                    if typ == "image":
                        image_entries.append((len(image_entries) + 1, args[0]))

                batches = [
                    image_entries[i:i + BATCH_SIZE]
                    for i in range(0, len(image_entries), BATCH_SIZE)
                ]

                async def _describe_batch(batch):
                    """一次 API 调用描述一组（最多 BATCH_SIZE 张）图片"""
                    indices = [e[0] for e in batch]
                    start, end = indices[0], indices[-1]
                    blocks = [{
                        "type": "text",
                        "text": (
                            f"请按顺序详细描述以下{len(batch)}张图片中可见的所有信息，"
                            f"包括文字、图表、公式、结构、数据等。\n"
                            f"每张图片的描述必须以\"<imageN>:\"开头，"
                            f"其中 N 为对应图片的序号（从{start}到{end}）。"
                            f"各图片描述之间用空行分隔。"
                        )
                    }]
                    for _, path in batch:
                        mime_type, encoded = _encode_image(path)
                        blocks.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{encoded}"}
                        })
                    resp = await muti_llm.ainvoke([HumanMessage(content=blocks)])
                    return resp.content.strip()

                async def _describe_batch_with_indices(batch):
                    text = await _describe_batch(batch)
                    return [e[0] for e in batch], text

                batch_results = await asyncio.gather(
                    *[_describe_batch_with_indices(b) for b in batches]
                ) if batches else []

                # 第三步：解析所有批次的描述，建立 index -> description 映射
                all_descriptions = {}
                for indices, batch_text in batch_results:
                    parts = re.split(r'<image(\d+)>:\s*', batch_text)
                    parsed = False
                    for i in range(1, len(parts), 2):
                        idx = int(parts[i])
                        desc = parts[i + 1].strip() if i + 1 < len(parts) else ""
                        all_descriptions[idx] = desc
                        parsed = True
                    if not parsed:
                        # 回退：整段文本作为这批所有图的描述
                        for idx in indices:
                            all_descriptions[idx] = batch_text

                # 第四步：按原始顺序拼接文本，图片位置用 <imageN> 占位
                img_idx = 0
                final_parts = []
                for typ, *args in content_parts:
                    if typ == "text":
                        final_parts.append(args[0])
                    elif typ == "image":
                        img_idx += 1
                        final_parts.append(f"<image{img_idx}>")

                final_text = "\n\n".join(final_parts)

                # 第五步：末尾追加所有图片描述
                desc_lines = []
                for idx in sorted(all_descriptions.keys()):
                    desc_lines.append(f"<image{idx}>: {all_descriptions[idx]}")
                if desc_lines:
                    final_text += "\n\n---\n图片描述：\n" + "\n\n".join(desc_lines)

                new_msg = copy.deepcopy(msg)
                new_msg.content = final_text
                processed_messages.append(new_msg)
                
        msg_list.clear()
        msg_list.extend(processed_messages)

        return processed_messages
