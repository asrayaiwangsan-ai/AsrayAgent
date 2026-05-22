import threading
from langchain_core.runnables import RunnableConfig
import shlex
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
import os

from office_file_loader import pdf2text, doc2text, pptx2text, xlsx2text
from web_tools import browser_tools_map

from public_rag import get_manager


from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field
import asyncio

base_dir = os.getenv("ASRAY_BASE_DIR", "/home/ai/asraydata")


@tool
def rag_save_usermemory(text: str) -> str:
    """保存一条记忆到用户长期记忆库。
    :param text: 要保存的记忆内容。"""
    try:
        get_manager().add("user_memory", [text])
        return "记忆已保存。"
    except Exception as e:
        return f"保存记忆失败: {str(e)}"


@tool
def rag_load_user_memory(query: str, k: int = 5) -> str:
    """从用户长期记忆库中检索相关记忆。
    :param query: 用于检索的查询文本。
    :param k: 返回结果数量，默认5。"""
    try:
        ret = get_manager().search("user_memory", query, k=k)
        return str(ret)
    except Exception as e:
        return f"检索记忆失败: {str(e)}"


DANGEROUS = frozenset({"sudo", "su"})


def has_dangerous_cmd(cmd: str) -> bool:
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()

    for t in tokens:
        name = t.rsplit("/", 1)[-1]
        if name in DANGEROUS:
            return True
    return False


@tool
def bash(command: str, timeout: int = 30) -> str:
    """执行 bash 指令并返回输出结果。严禁使用sudo"""
    import subprocess
    dispatch_custom_event(name="custom_ret", data={
                          "chunk": {"content": f"Bash {command}"}})
    if (has_dangerous_cmd(command)):
        return f"指令执行失败，禁止使用sudo:"

    try:
        output = subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, timeout=timeout)
        return output
    except subprocess.TimeoutExpired:
        return f"指令执行超时（{timeout}s）"
    except subprocess.CalledProcessError as e:
        return f"指令执行失败，返回码: {e.returncode}\n输出: {e.output}"
    except Exception as e:
        return f"发生错误: {str(e)}"


@tool
def bash_windows(command: str, timeout: int = 30) -> str:
    """执行 Windows CMD 指令并返回输出结果。严禁使用sudo/runas"""
    import subprocess
    import platform
    dispatch_custom_event(name="custom_ret", data={
                          "chunk": {"content": f"BashWin {command}"}})
    if (has_dangerous_cmd(command)):
        return f"指令执行失败，禁止使用危险指令:"

    try:
        output = subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            text=True, timeout=timeout, encoding='utf-8', errors='replace')
        return output
    except subprocess.TimeoutExpired:
        return f"指令执行超时（{timeout}s）"
    except subprocess.CalledProcessError as e:
        return f"指令执行失败，返回码: {e.returncode}\n输出: {e.output}"
    except Exception as e:
        return f"发生错误: {str(e)}"


@tool
def grep_search(pattern: str, path: str = ".", include: str = "*", max_results: int = 30) -> str:
    """跨文件搜索。在指定目录中递归搜索匹配关键词的行。
    :param pattern: 搜索关键词或正则表达式。
    :param path: 搜索根目录。
    :param include: 文件名 glob 模式，如 '*.py' 或 '*.py,*.md'。默认 '*'。
    :param max_results: 最大返回条数。
    """
    import subprocess
    import shlex as _shlex

    includes = " ".join(f"'--include={g.strip()}'" for g in include.split(","))

    cmd = f"grep -rn {includes} -e {_shlex.quote(pattern)} {_shlex.quote(path)} 2>/dev/null | head -n {max_results}"

    try:
        output = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, timeout=10
        )
        return output if output.strip() else "未找到匹配结果。"
    except subprocess.TimeoutExpired:
        return "搜索超时。"
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            return "未找到匹配结果。"
        return f"搜索失败: {e.output}"
    except Exception as e:
        return f"发生错误: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """创建或覆盖指定路径的文件，并写入内容。"""
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"成功写入文件: {file_path}"
    except Exception as e:
        return f"写入文件失败: {str(e)}"


def _load_skill(name):
    file_path = base_dir + "/skills/"+name+"/SKILL.md"
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        try:
            content = f.read()
            return content
        except Exception as e:
            print(e)
            return None


@tool
def load_skill(skill_name: str) -> str:
    """加载并学习指定的技能包。
    这会将该技能的专家手册和专属 Python 工具注入到你的【当前会话】中，让你【亲自】具备处理该任务的能力。
    注意：这不同于委派任务给其他 Agent。加载技能是提升“你自身”的能力。
    如果你想将任务完全交给另一个独立的专家（Agent）去处理，请使用 'call_agents'。
    """
    import importlib.util
    import inspect
    from langchain_core.tools import BaseTool

    skill_info = _load_skill(skill_name)
    if not skill_info:
        return f"错误：未找到名为 '{skill_name}' 的技能。"

    content = skill_info
    tool_info_str = ""

    # 动态加载工具逻辑
    skill_dir = os.path.join(base_dir, "skills", skill_name)
    if os.path.exists(skill_dir):
        # 寻找目录下所有的 .py 文件
        for filename in os.listdir(skill_dir):
            if filename.endswith(".py"):
                file_path = os.path.join(skill_dir, filename)
                module_name = f"skills.{skill_name}.{filename[:-3]}"
                try:
                    spec = importlib.util.spec_from_file_location(
                        module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # 查找模块中的工具 (检查是否为 BaseTool 实例或带有 tool 装饰器的函数)
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)

                        # 检查是否是 LangChain Tool
                        if isinstance(attr, BaseTool):
                            # 注册到全局 tools_map
                            tools_map[attr.name] = attr

                            # 提取参数信息
                            args_info = ""
                            if hasattr(attr, "args") and attr.args:
                                for arg_name, arg_details in attr.args.items():
                                    arg_desc = arg_details.get(
                                        "description", "无描述")
                                    arg_type = arg_details.get(
                                        "type", "unknown")
                                    args_info += f"    - {arg_name} ({arg_type}): {arg_desc}\n"

                            tool_info_str += f"- 工具名: {attr.name}\n  描述: {attr.description}\n"
                            if args_info:
                                tool_info_str += f"  参数详情:\n{args_info}"
                        elif hasattr(attr, "tool") or (inspect.isfunction(attr) and hasattr(attr, "args_schema")):
                            # 处理某些装饰器可能不直接变成 BaseTool 的情况
                            from langchain_core.tools import Tool
                            # 如果是函数且被识别为 tool，尝试转换
                            if hasattr(attr, "name"):
                                tools_map[attr.name] = attr
                                # 尝试提取参数（对于装饰过的函数，通常在 args 属性中）
                                args_info = ""
                                if hasattr(attr, "args") and attr.args:
                                    for arg_name, arg_details in attr.args.items():
                                        arg_desc = arg_details.get(
                                            "description", "无描述")
                                        arg_type = arg_details.get(
                                            "type", "unknown")
                                        args_info += f"    - {arg_name} ({arg_type}): {arg_desc}\n"

                                tool_info_str += f"- 工具名: {attr.name}\n  描述: {attr.description if hasattr(attr, 'description') else '无描述'}\n"
                                if args_info:
                                    tool_info_str += f"  参数详情:\n{args_info}"

                except Exception as e:
                    print(f"加载技能工具失败 [{file_path}]: {e}")

    # 构造返回给 LLM 的最终内容
    ret = f"--- 技能内容: {skill_name} ---\n{content}\n"
    if tool_info_str:
        ret += "\n--- 检出配套工具 ---\n"
        ret += tool_info_str
        ret += "\n**注意**：上述工具是专门为本技能任务设计的插件，请仅在执行与本技能相关的任务时调用它们。\n"

    return ret


@tool
def replace(file_path: str, old_string: str, new_string: str) -> str:
    """替换文件中的特定文本。要求 old_string 必须在文件中唯一存在。"""
    try:
        if not os.path.exists(file_path):
            return f"错误：文件 {file_path} 不存在。"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        count = content.count(old_string)
        if count == 0:
            return f"错误：在文件 {file_path} 中未找到指定的文本。"
        if count > 1:
            return f"错误：在文件 {file_path} 中找到多处匹配，请提供更多上下文以确保唯一性。"

        new_content = content.replace(old_string, new_string)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        import difflib

        old_pos = content.find(old_string)
        start_line = content[:old_pos].count('\n') + 1

        old_lines = old_string.splitlines()
        new_lines = new_string.splitlines()

        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        diff_lines = []
        old_ln = start_line
        new_ln = start_line

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for k in range(i1, i2):
                    diff_lines.append(f"{old_ln}   {old_lines[k]}")
                    old_ln += 1
                    new_ln += 1
            elif tag == 'replace':
                for k in range(i1, i2):
                    diff_lines.append(f"{old_ln} - {old_lines[k]}")
                    old_ln += 1
                for k in range(j1, j2):
                    diff_lines.append(f"{new_ln} + {new_lines[k]}")
                    new_ln += 1
            elif tag == 'delete':
                for k in range(i1, i2):
                    diff_lines.append(f"{old_ln} - {old_lines[k]}")
                    old_ln += 1
            elif tag == 'insert':
                for k in range(j1, j2):
                    diff_lines.append(f"{new_ln} + {new_lines[k]}")
                    new_ln += 1

        diff_output = "\n".join(diff_lines)
        dispatch_custom_event(name="custom_ret", data={
                              "chunk": {"content": f"{file_path}\n{diff_output}"}})

        return f"成功更新文件: {file_path}"
    except Exception as e:
        return f"更新文件失败: {str(e)}"

# --- Read File Strategies ---


def _read_office_logic(file_path: str, multimodal: bool = True) -> list:
    """内部 Office/PDF 读取逻辑"""
    text = ""
    pic = []
    result = []

    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".docx", ".doc"]:
        text, pic = doc2text(file_path, multimodal, base_dir)
    elif ext == ".pdf":
        text, pic = pdf2text(file_path, multimodal, base_dir)
    elif ext in [".pptx", ".ppt"]:
        text, pic = pptx2text(file_path, multimodal, base_dir)
    elif ext in [".xlsx", ".xls"]:
        text, pic = xlsx2text(file_path, multimodal, base_dir)
    elif ext in [".jpg", ".jpeg", ".png"]:
        if multimodal:
            result.append(
                {"type": "image_ref", "image_ref": {"path": file_path}})

    # 使用 <image> 分割文本并重新构造返回列表
    parts = text.split("<image>")
    

    for i in range(len(parts)):
        if parts[i]:
            result.append({"type": "text", "text": parts[i]})

        # 如果还有对应的图片路径，则插入一个 image_ref
        if i < len(pic):
            result.append({"type": "image_ref", "image_ref": {"path": pic[i]}})

    return result 


@tool
def read_text_file(
    file_path: str,
    offset: int = 1,
    limit: int = -1
) -> str:
    """读取纯文本文件内容。
    支持格式：.txt, .py, .md, .json, .yaml, .c, .cpp, .h, .js, .ts, .sh, .html, .css 等。
    :param file_path: 文件路径
    :param offset: 开始行号（从1开始，默认为1）
    :param limit: 结束行号（包含该行，-1表示读取到文件末尾）
    """
    if not os.path.exists(file_path):
        return f"错误：文件 {file_path} 不存在。"
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        start = max(0, offset - 1)
        end = len(lines) if limit == -1 else limit

        selected_content = "".join(lines[start:end])
        return selected_content
    except Exception as e:
        return f"读取文件失败: {str(e)}"





import logging

@tool
async def read_doc_file(
    file_path: str,
    tool_call_id: Annotated[str, InjectedToolArg()],
    multimodal: Annotated[bool, InjectedToolArg()] = True
) -> ToolMessage:
   
    """
    读取富文档内容（PDF、Word、Excel、PPT）。
    支持格式：.pdf, .docx, .xlsx, .pptx, .doc, .xls, .ppt .jpg.png.jpeg等图片文件
    会自动提取文字和图片。
    """
    if not os.path.exists(file_path):
        return ToolMessage(
            content=[{"type": "text", "text": f"错误：文件 {file_path} 不存在。"}],
            tool_call_id=tool_call_id
        )
    try:
        content_list = _read_office_logic(file_path, multimodal)
        text_len = 0
        for c in content_list:
            if c.get("type") == "text":
                text_len = text_len + len(c.get("text"))
        if text_len > 30000:
            from asray_core import get_llm,load_models_config,get_defuat_mutimodel
            
            
            
            confs = load_models_config()
            mname = get_defuat_mutimodel()
            if mname==None:
                for name, info in confs.items():
                    mname = name
                    break
                
            llm = await get_llm(model_name=mname)

            # 第一步：遍历 content_list，按 ~30k 字切片
            chunks = []
            current_text = ""
            current_images = []

            for c in content_list:
                if c.get("type") == "text":
                    t = c.get("text", "")
                    current_text += t
                    while len(current_text) >= 30000:
                        # 在 30000 字符处附近找最近的换行符作为自然断点
                        split_at = 30000
                        nl_pos = current_text.rfind("\n", 0, split_at)
                        if nl_pos > 20000:  # 如果换行符不会让 chunk 太小，则以此断点
                            split_at = nl_pos + 1
                        chunks.append((current_text[:split_at], list(current_images)))
                        current_text = current_text[split_at:]
                        current_images = []
                elif c.get("type") == "image_ref":
                    img_path = c.get("image_ref", {}).get("path", "")
                    current_images.append(img_path)
                    current_text += f"\n[图片: {img_path}]\n"

            # 收尾最后一个 chunk
            if current_text.strip() or current_images:
                chunks.append((current_text, list(current_images)))

            # 第二步：将每个 chunk 写入文件，并发起 LLM 摘要
            import base64 as _base64
            import mimetypes as _mimetypes

            chunk_files = []
            summary_coros = []

            async def _summarize_one(idx, text, images):
                content_blocks = [
                    {
                        "type": "text",
                        "text": (
                            f"请对以下文档片段进行总结（这是第{idx + 1}个片段，"
                            f"共{len(chunks)}个片段），提取核心要点：\n\n{text}"
                        ),
                    }
                ]
                for img_path in images:
                    if os.path.exists(img_path):
                        try:
                            mime_type = _mimetypes.guess_type(img_path)[0] or "image/png"
                            with open(img_path, "rb") as f:
                                encoded = _base64.b64encode(f.read()).decode("utf-8")
                            content_blocks.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{encoded}"
                                    },
                                }
                            )
                        except Exception:
                            content_blocks.append(
                                {
                                    "type": "text",
                                    "text": f"\n[图片读取失败: {img_path}]",
                                }
                            )

                try:
                    from langchain_core.messages import HumanMessage

                    resp = await llm.ainvoke([HumanMessage(content=content_blocks)])
                    return idx, resp.content.strip()
                except Exception as e:
                    return idx, f"摘要生成失败: {str(e)}"

            for idx, (text, images) in enumerate(chunks):
                # 写入完整内容到 base_dir
                chunk_filename = (
                    f"doc_chunk_{idx}_{os.path.basename(file_path)}.txt"
                )
                cache_dir = os.path.join(base_dir, "cached_doc_file")
                os.makedirs(cache_dir, exist_ok=True)
                chunk_path = os.path.join(cache_dir, chunk_filename)
                with open(chunk_path, "w", encoding="utf-8") as f:
                    f.write(text)
                chunk_files.append(chunk_path)

                summary_coros.append(_summarize_one(idx, text, images))

            # 第三步：并发获取所有摘要
            results = await asyncio.gather(*summary_coros)

            # 第四步：构造 markdown 返回
            md_parts = [
                f"文档 `{os.path.basename(file_path)}` 过长（共 {text_len} 字），"
                f"已自动切片为 {len(chunks)} 段。以下是各部分摘要：\n"
            ]
            for idx, summary in sorted(results, key=lambda x: x[0]):
                md_parts.append(f"## 片段 {idx + 1}")
                md_parts.append(f"- **摘要**: {summary}")
                md_parts.append(f"- **完整内容位置**: `{chunk_files[idx]}`")
                md_parts.append("")

            return ToolMessage(
                content=[{"type": "text", "text": "\n".join(md_parts)}],
                name="read_doc_file",
                tool_call_id=tool_call_id,
            )

        else:
            return ToolMessage(content=content_list,  name="read_doc_file" ,tool_call_id=tool_call_id)
    except Exception as e:
        logging.exception(e)
        return ToolMessage(
            content=[{"type": "text", "text": f"读取文档失败: {str(e)}"}],
            name="read_doc_file",
            tool_call_id=tool_call_id,
        )







def read_graph(name):
    import json
    try:
        with open(f"{base_dir}/graphs/{name}.json", 'r') as f:
            data = f.read()
            ret = json.loads(data)
            return ret
    except Exception as e:

        print(f"read_graph e {e}")
        return None


async def _build_agent_state(agent_name: str, call_id: str):
    from asray_core import create_graph, shared_memory
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    graph_conf = read_graph(agent_name)
    if not graph_conf:
        return None, None, None, None

    sub_config = {"configurable": {"thread_id": call_id}}

    workflow = await create_graph(graph_conf)
    app = workflow.compile(checkpointer=shared_memory)

    current_snapshot = await app.aget_state(sub_config)
    state = current_snapshot.values if current_snapshot.values else {
        "messages": [], "extra": {}}

    if not current_snapshot.values:
        init_state = graph_conf.get("init_state")
        if init_state:
            init_msgs = init_state.get("messages") or init_state.get("message")
            if init_msgs:
                for m in init_msgs:
                    m_type = m.get("type", "human")
                    content = m.get("content", "")
                    if "path" in m:
                        p_path = os.path.join(base_dir, "prompts", m["path"])
                        if os.path.exists(p_path):
                            with open(p_path, "r", encoding="utf-8") as f:
                                content = f.read()
                    if m_type == "system":
                        state["messages"].append(
                            SystemMessage(content=content))
                    elif m_type == "human":
                        state["messages"].append(HumanMessage(content=content))
                    elif m_type == "ai":
                        state["messages"].append(AIMessage(content=content))
            if "extra" in init_state:
                state["extra"].update(init_state["extra"])

    return app, state, sub_config, graph_conf


async def _invoke_agent(app, state, sub_config, prompt: str):
    from langchain_core.messages import HumanMessage

    state["messages"].append(HumanMessage(content=prompt))
    final_state = await app.ainvoke(state, sub_config)
    final_msgs = final_state.get("messages", [])
    retmsg = final_msgs[-1].content if final_msgs else "子 Agent 执行完毕，未返回内容。"
    return retmsg


_async_task_queue = []
_async_results = {}
lock = threading.Lock()


async def _async_worker():
    while True:
        import time
        if (len(_async_task_queue) == 0):
            time.sleep(1)
            continue
        agent_name, prompt, call_id, thread_id = _async_task_queue.pop()
        try:
            app, state, sub_config, _ = await _build_agent_state(agent_name, call_id)
            if app is None:
                retmsg = f"异步任务失败：未找到 Agent '{agent_name}'"
            else:
                retmsg = await _invoke_agent(app, state, sub_config, prompt)
        except Exception as e:
            retmsg = f"异步任务异常: {str(e)}"

        with lock:

            _async_results[thread_id] = (call_id, retmsg)


async def pop_async_results(thread_id: str):
    with lock:
        results = _async_results.pop(thread_id, None)
    return results


@tool
async def call_agents(
    agent_name: str,
    prompt: str,
    call_id: str,
    is_async: bool = False,
    extra: dict = None,
) -> str:
    """将任务完全【委派】给指定的独立 Agent 去执行。
    :param agent_name: 目标 Agent 的名称。
    :param prompt: 发送给 Agent 的具体指令。
    :param call_id: 任务的唯一识别 ID（你可以自定义）。
    :param is_async: 是否异步执行。为 True 时立即返回提交成功状态，否则Agent 在后台执行，执行完成后会返回结果给你，返回的消息以[AsyncAgentResult]开头。

    使用说明：
    1. call_id 用于维持 Agent 的会话记忆。如果你使用相同的 call_id 再次调用同一个 Agent，它会记得之前的对话上下文。
    2. 如果子 Agent 的返回内容表明它需要更多信息或需要你确认，你可以根据它的反馈再次调用它（保持相同的 call_id）。
    3. 你应该使用 'manager_agent_state' 工具来记录和管理这些 call_id 及其对应的任务进度，以免遗忘。
    """
    dispatch_custom_event(name="custom_ret", data={
                          "chunk": {"content": f"\n[CallAgent] ID: {call_id} | Prompt: {prompt}\n"}})

    app, state, sub_config, graph_conf = await _build_agent_state(agent_name, call_id)
    if app is None:
        return f"错误：未找到名为 '{agent_name}' 的 agent 配置。"

    if is_async:
        if extra is None:
            return "错误：异步调用需要 extra 参数。"
        thread_id = extra.get("thread_id")
        if not thread_id:
            return "错误：extra 中缺少 thread_id。"
        _async_task_queue.append((agent_name, prompt, call_id, thread_id))
        return "异步任务已提交"

    try:
        retmsg = await _invoke_agent(app, state, sub_config, prompt)
        dispatch_custom_event(name="custom_ret", data={
                              "chunk": {"content": f"\n[CallAgent] Response: {retmsg}...\n"}})
        return retmsg
    except Exception as e:
        return f"调用子 Agent 失败: {str(e)}"


@tool
def manager_agent_state(
    op: str,
    call_id: str = None,
    state_desc: str = None,
    extra: dict = None
) -> str:
    """管理和追踪已调用的子 Agent 任务状态。
    :param op: 操作类型，可选：'list' (查看所有任务), 'set' (记录/更新任务状态), 'delete' (删除已完成任务)。
    :param call_id: 任务的识别 ID。
    :param state_desc: 对该任务当前状态的自然语言描述（如：“正在等待用户回复服务器地址”）。

    使用说明：
    1. 你应该像管理备忘录一样使用此工具，以记住你分配给了哪些 call_id 任务，以及它们进行到了哪一步。
    2. 只有你自己调用此工具，你才能在未来的对话中通过 'list' 查看到这些记录。
    """
    if extra is None:
        return "错误：无法访问系统元数据。"

    if "agent_tasks" not in extra:
        extra["agent_tasks"] = {}

    tasks = extra["agent_tasks"]

    if op == "list":
        if not tasks:
            return "当前没有任何记录的任务。"
        res = "当前任务状态列表：\n"
        for cid, info in tasks.items():
            res += f"- [ID: {cid}] Agent: {info.get('name', 'unknown')}, 状态: {info.get('state', '无描述')}\n"
        return res

    if op == "set":
        if not call_id:
            return "错误：设置状态需要提供 call_id。"
        # 尝试从上下文中自动识别名字（如果可能）或保持原样
        tasks[call_id] = {
            "name": tasks.get(call_id, {}).get("name", "active_task"),
            "state": state_desc or "运行中"
        }
        return f"任务 {call_id} 的状态已更新。"

    if op == "delete":
        if not call_id:
            return "错误：删除任务需要提供 call_id。"
        if call_id in tasks:
            del tasks[call_id]
            return f"任务 {call_id} 已从追踪列表中移除。"
        return f"未找到 ID 为 {call_id} 的任务。"

    return f"未知操作: {op}"


mcp_arg_map = {
    "WebSearch": {
        "transport": "sse",
        # "description": "提供实时互联网全栈信息检索。",
        # "isActive": True,
        # "name": "阿里云百炼_联网搜索",
        "url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse",
        "headers": {"Authorization": f"Bearer key"}
    },
    "antv-visualization-chart": {
        "transport": "sse",
        # "description": "基于 AntV 可视化解决方案封装的可视化图表生成 MCP 插件。",
        # "isActive": True,
        # "name": "阿里云百炼_AntV 可视化图表",
        "url": "https://dashscope.aliyuncs.com/api/v1/mcps/antv-visualization-chart/sse",
        "headers": {"Authorization": "Bearer  key"}
    },
}

mcp_tools_map = {}


@tool
async def update_user_memory(new_insight: str) -> str:
    """
    更新用户的核心画像、长期目标、全局偏好或当前最重要的项目状态。
    此信息将被注入 System Prompt，作为你时刻感知的核心记忆。
    仅在以下情况使用：
    1. 发现用户的新核心偏好（如：编程风格、交互语气）。
    2. 获取到用户最重要的身份信息或项目核心设定。
    3. 任务进入了关键的新阶段，需要更新全局状态。
    注意：一般的琐碎事实或非核心细节请使用 rag,以节省 System Prompt 空间。
    """
    from asray_core import get_llm, base_dir
    import os

    user_md_path = os.path.join(base_dir, "user.md")

    # 1. 读取旧内容
    old_content = ""
    if os.path.exists(user_md_path):
        with open(user_md_path, "r", encoding="utf-8") as f:
            old_content = f.read()

    if not old_content:
        old_content = "# 用户长期记忆 (User Profile)\n\n## 核心偏好\n\n## 重要事实\n\n## 当前目标\n"

    # 2. 准备管理员 Prompt
    merge_prompt = f"""你是一名专业的档案管理员。
当前用户档案内容：
---
{old_content}
---

新收到的信息：
{new_insight}

你的任务：将新信息整合进现有档案。
1. **去重与合并**：如果新信息已存在，则保持不变；如果与旧信息相关但有补充，请合并。
2. **冲突处理**：如果新信息与旧信息冲突，以新信息为准。
3. **结构化归类**：将信息放入合适的二级标题下。
4. **精简至上**：保持 Markdown 格式，删除陈旧或琐碎的描述，确保档案总长度控制在 800 字以内。
5. **严禁开场白**：直接输出更新后的完整 Markdown 内容。"""

    try:
        # 调用 LLM 进行融合（获取一个纯净的 LLM 实例）
        # 这里默认使用 qwen3，不带任何工具
        summary_llm = await get_llm(model_name="qwen3", temperature=0.1)
        response = await summary_llm.ainvoke(merge_prompt)
        new_content = response.content.strip()

        if not new_content:
            return "整合后的内容为空，更新取消。"

        # 3. 写回文件
        with open(user_md_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return "用户核心记忆已更新并完成结构化归档。"
    except Exception as e:
        return f"更新用户记忆失败: {str(e)}"


@tool
def list_all_agent() -> str:
    """列出所有已安装的 Agent（不受白名单限制，返回全量）。
    每个 Agent 是一个可委派任务的独立执行体，拥有自己的 DSL 配置和工具链。
    """
    import json

    graphs_dir = os.path.join(base_dir, "graphs")
    if not os.path.exists(graphs_dir):
        return "graphs 目录不存在。"

    ret = ""
    for f in sorted(os.listdir(graphs_dir)):
        if f.endswith(".json"):
            name = f[:-5]
            try:
                with open(os.path.join(graphs_dir, f), 'r') as fp:
                    data = json.loads(fp.read())
                    desc = data.get("description", "无描述")
            except Exception:
                desc = "读取失败"
            ret += f"- **{name}**: {desc}\n"
    return ret or "未找到任何 Agent。"


@tool
def list_all_skill() -> str:
    """列出所有已安装的 Skill（不受白名单限制，返回全量）。
    每个 Skill 是可加载的能力扩展包，包含专家手册和配套工具。
    """
    skills_dir = os.path.join(base_dir, "skills")
    if not os.path.exists(skills_dir):
        return "skills 目录不存在。"

    ret = ""
    for d in sorted(os.listdir(skills_dir)):
        skill_md = os.path.join(skills_dir, d, "SKILL.md")
        if os.path.exists(skill_md):
            try:
                with open(skill_md, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    b = content.find("description:")
                    e = content.find("---", b)
                    desc = content[b:e].replace(
                        "description:", "").strip() if b != -1 and e != -1 else "无描述"
            except Exception:
                desc = "读取失败"
            ret += f"- **{d}**: {desc}\n"
    return ret or "未找到任何 Skill。"



@tool
def search_remote_kbs(query , kb_name):
    """查找远程资料库中与query匹配度最高的资料的资料
    :param query: 查询内容，查询更具query的embedding与数据库中内容的embedding计算相似度返回
    :param kb_name: 数据库名"""
    import requests


    headers = {
        'Content-Type': 'application/json',  
        'token': "from_agent"
    }

    response = requests.post("http://127.0.0.1:8899/query", json={"query":query,"kb_name":kb_name},
                                                headers = headers,verify=False) 
    
    ret = response.json().get("data")
        
                                                   
    return ret


tools_map = {
    "bash": bash,
    "bash_windows": bash_windows,
    "grep_search": grep_search,
    "write_file": write_file,
    "replace": replace,
    "read_text_file": read_text_file,
    "read_doc_file": read_doc_file,
    "load_skill": load_skill,

    "call_agents": call_agents,
    "update_user_memory": update_user_memory,
    "manager_agent_state": manager_agent_state,

    "rag_save_usermemory": rag_save_usermemory,
    "rag_load_user_memory": rag_load_user_memory,

    "list_all_agent": list_all_agent,
    "list_all_skill": list_all_skill,
}
tools_map.update(browser_tools_map)
