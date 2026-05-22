import asyncio
import os
import argparse
import sys
from typing import Annotated, TypedDict
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.messages import BaseMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style as PromptStyle
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
# from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from asray_core import create_graph, BaseState, shared_memory

from db_conf import DB_URI
# 默认配置
t_config = {
    "start": "chatbot",
    "nodes": [
        {"name": "chatbot", "type": "llm_invoke", "conf": {
            "before_invoke": {
                "system_prompt_hook.SystemPromptHook":{},
                "multimodal_hook.MultimodalHook":{},
                "deepseek_reasoning_hook.DeepSeekReasoningHook":{}
            },
           "after_invoke": {"async_result_hook.AsyncResultHook":{} }
           }
        },
        {"name": "tools", "type": "base_tool_call"},
    ],
    "edges": {
        # "welcome": [{"type": "goto", "to": "chatbot"}],
        "chatbot": [{"type": "cond", "conf": {
            "type": "msg_find",
            "conf": {
                "<tool_call>": {"to": "tools"},
                "[AsyncAgentResult] call_id" : {"to": "chatbot"},
            }
        }}],
        "tools": [{"type": "goto", "to": "chatbot"}]
    },
    
    "llm_config": {
        "model_name": "deepseek-v4-pro",
        # "model_name": "qwen3.6-plus",
        "temperature": 0.3,
        "max_tokens": 52000,
        "top_p": 0.9,
        "tools": ["bash_windows", "write_file", "replace", "grep_search",
                  "read_text_file", "read_doc_file", 
                  "list_all_agent" , "list_all_skill","load_skill", "call_agents","manager_agent_state",
                  "update_user_memory", "rag_save_usermemory" , "rag_load_user_memory" ,
                  "browser_tool" , "web_fetch"]
    },
    "description": "Asray CLI Agent"
}

# 初始化 Rich Console
console = Console()
memory = shared_memory

# 定义提示符样式
prompt_style = PromptStyle.from_dict({
    'prompt': '#00ff00 bold',
})



from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
base_dir = os.getenv("ASRAY_BASE_DIR", "/home/ai/asraydata")

async def run_cli(thread_id: str, model_name: str = None, debug: bool = False):
    """
    运行增强版 CLI 交互循环
    """
    if model_name:
        t_config["llm_config"]["model_name"] = model_name

    # 调试模式：可在运行时通过 /debug 切换
    _debug = debug

    config = {"configurable": {"thread_id": thread_id,"recursion_limit": 50} ,"recursion_limit": 50}
    
    # 创建并编译图
    workflow = await create_graph(t_config)
    # async with AsyncPostgresSaver.from_conn_string(DB_URI) as pgcheckpointer:
    async with AsyncSqliteSaver.from_conn_string(f"{base_dir}/checkpoints.db") as pgcheckpointer:
        await pgcheckpointer.setup()
        
        app = workflow.compile(checkpointer=pgcheckpointer)
    
        session = PromptSession(history=InMemoryHistory())
    
        snapshot = await app.aget_state(config)
        if snapshot.values:
            state = snapshot.values
        else:
            state = {"messages": [], "extra": {"thread_id": thread_id}}

        console.print(Panel.fit(
            f"[bold blue]Asray Agent CLI[/bold blue]\n[green]Thread ID:[/green] {thread_id}\n[green]Model:[/green] {t_config['llm_config']['model_name']}\n[green]Debug:[/green] {_debug}\n\n指令: [bold magenta]/exit[/bold magenta] (退出), [bold magenta]/clear[/bold magenta] (清屏), [bold magenta]/show_state[/bold magenta] (查看状态), [bold magenta]/debug[/bold magenta] (切换调试), [bold magenta]/graph[/bold magenta] (图结构), [bold magenta]/checkpoint[/bold magenta] (检查点)",
            title="Welcome",
            border_style="blue"
        ))

        while True:
            try:
                # 获取用户输入
                user_input = await session.prompt_async(
                    [('class:prompt', '>>> ')], 
                    style=prompt_style
                )
                user_input = user_input.strip()
            
                if not user_input:
                    continue
                
                if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                    console.print("[yellow]Bye![/yellow]")
                    break
                
                if user_input.lower() == "/clear":
                    console.clear()
                    continue
                
                if user_input.lower() == "/show_state":
                    console.print("\n" + "="*20 + " CURRENT STATE " + "="*20, style="blue")
                    msgs = state.get("messages", [])
                    console.print(f"[dim]共 {len(msgs)} 条消息[/dim]\n")
                    # 打印消息历史
                    for i, m in enumerate(msgs):
                        if isinstance(m, ToolMessage):
                            role = f"Tool ({m.name})"
                            color = "magenta"
                            content_str = str(m.content)
                            # if len(content_str) > 300:
                            #     content_str = content_str[:300] + f"... [共 {len(str(m.content))} 字符]"
                            console.print(
                                f"[{i}] [bold magenta]🔧 Tool ({m.name}):[/bold magenta] {content_str}"
                            )
                            console.print(f"     [dim]tool_call_id: {m.tool_call_id}[/dim]")
                        else:
                            if isinstance(m, HumanMessage):
                                role, color = "Human", "green"
                            elif isinstance(m, AIMessage):
                                role, color = "AI", "cyan"
                                # 额外显示 tool_calls（如果有）
                                if hasattr(m, "tool_calls") and m.tool_calls:
                                    for tc in m.tool_calls:
                                        console.print(f"     [dim]🔨 tool_call: {tc.get('name', '?')}({str(tc.get('args', ''))[:100]})[/dim]")
                            elif isinstance(m, SystemMessage):
                                role, color = "System", "yellow"
                            else:
                                role, color = "Unknown", "white"
                            console.print(f"[{i}] [bold {color}]{role}:[/bold {color}] {m.content}")

                    # 打印 Extra 数据
                    if state.get("extra"):
                        console.print("\n[bold magenta]Extra Data:[/bold magenta]")
                        import json
                        serializable_extra = {k: v for k, v in state["extra"].items() if not k.startswith("_")}
                        console.print(serializable_extra)
                
                    console.print("="*55 + "\n", style="blue")
                    continue

                if user_input.lower() == "/debug":
                    _debug = not _debug
                    console.print(f"[yellow]调试模式:[/yellow] {'[green]ON[/green]' if _debug else '[red]OFF[/red]'}")
                    continue

                if user_input.lower() == "/graph":
                    console.print("\n" + "="*20 + " GRAPH STRUCTURE " + "="*20, style="blue")
                    console.print(f"[bold]入口节点:[/bold] {t_config['start']}")
                    console.print(f"\n[bold]节点:[/bold]")
                    for node in t_config["nodes"]:
                        extra_info = ""
                        if node["type"] == "llm_invoke" and node.get("conf"):
                            hooks = node["conf"].get("before_invoke", []) + node["conf"].get("after_invoke", [])
                            if hooks:
                                extra_info = f"  [dim]({len(hooks)} hooks)[/dim]"
                        console.print(f"  • {node['name']} ({node['type']}){extra_info}")
                    console.print(f"\n[bold]边:[/bold]")
                    for src, edges in t_config["edges"].items():
                        for e in edges:
                            if e["type"] == "goto":
                                console.print(f"  {src} → {e['to']}")
                            elif e["type"] == "cond":
                                rules = e["conf"]["conf"]
                                for k, v in rules.items():
                                    console.print(f"  {src} ─[{k}]→ {v['to']}")
                    console.print(f"\n[bold]工具:[/bold] {', '.join(t_config['llm_config']['tools'])}")
                    console.print("="*55 + "\n", style="blue")
                    continue

                if user_input.lower() == "/checkpoint":
                    console.print("\n" + "="*20 + " CHECKPOINT " + "="*20, style="blue")
                    snapshot = await app.aget_state(config)
                    if snapshot.values:
                        msg_count = len(snapshot.values.get("messages", []))
                        console.print(f"[green]Thread ID:[/green] {thread_id}")
                        console.print(f"[green]消息数:[/green] {msg_count}")
                        console.print(f"[green]下一步节点:[/green] {snapshot.next if snapshot.next else '(结束)'}")
                        console.print(f"[green]Checkpoint ID:[/green] {snapshot.config.get('configurable', {}).get('checkpoint_id', 'N/A')}")
                    else:
                        console.print("[yellow]无 checkpoint 数据[/yellow]")
                    console.print("="*45 + "\n", style="blue")
                    continue

                state["messages"].append(HumanMessage(content=user_input))
            
                # 流式处理输出
                last_role = None
                reasoning_text = ""
                content_text = ""
            
                console.print(f"[bold cyan]Agent:[/bold cyan] ", end="")
            
                # 使用 Live 更新输出
                with Live(Text(""), console=console, refresh_per_second=20, transient=False) as live:
                    reasoning_started = False
                    content_started = False
                    async for event in app.astream_events(state, config, version="v2"):
                        kind = event["event"]

                        if kind == "on_chat_model_stream":
                            chunk = event["data"].get("chunk")
                            if chunk:
                                # 处理思维链 (reasoning_content)
                                r_content = chunk.additional_kwargs.get("reasoning_content", "")
                                if r_content:
                                    if not reasoning_started:
                                        reasoning_started = True
                                    reasoning_text += r_content

                                # 处理正式回答内容
                                if chunk.content:
                                    if not content_started:
                                        content_started = True
                                    content_text += chunk.content

                                # 组装显示文本
                                display_parts = []
                                if reasoning_started:
                                    if content_started:
                                        display_parts.append((f"[已思考]\n", "dim"))
                                    else:
                                        display_parts.append((f"[思考中...]\n{reasoning_text}", "dim"))

                                if content_started:
                                    display_parts.append((content_text, ""))

                                live.update(Text.assemble(*display_parts))
                    
                        elif kind == "on_custom_event" and event.get("name") == "custom_ret":
                            # v2 中自定义数据在 event["data"]
                            custom_data = event["data"]
                            # 兼容处理不同的 chunk 嵌套
                            actual_data = custom_data.get("chunk", custom_data)
                            if "content" in actual_data:
                                content_text += f"\n[DEBUG] {actual_data['content']}\n"
                                live.update(Text.assemble((content_text, "")))

                # 打印一个换行，结束 Agent 输出
                console.print()
                await asyncio.sleep(0)
                # 更新内部状态
                snapshot = await app.aget_state(config)
                state = snapshot.values
            
            except KeyboardInterrupt:
                continue  # 按 Ctrl+C 不退出，仅取消当前输入
            except EOFError:
                console.print("\n[yellow]Exit via EOF[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
                if _debug:
                    console.print("\n[bold yellow]--- 调试上下文 ---[/bold yellow]")
                    msgs = state.get("messages", [])
                    console.print(f"[dim]消息总数: {len(msgs)}[/dim]")
                    # 打印最近 3 条消息
                    recent = msgs[-3:] if len(msgs) >= 3 else msgs
                    for i, m in enumerate(recent):
                        idx = len(msgs) - len(recent) + i
                        role = type(m).__name__
                        snippet = str(m.content)[:200] + "..." if len(str(m.content)) > 200 else str(m.content)
                        console.print(f"  [{idx}] {role}: {snippet}")
                    if state.get("extra"):
                        safe_extra = {k: v for k, v in state["extra"].items() if not k.startswith("_")}
                        console.print(f"[dim]extra: {safe_extra}[/dim]")
                    console.print("[bold yellow]--- 完整 traceback ---[/bold yellow]")
                import traceback
                console.print(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description="Asray Agent CLI Tool")
    parser.add_argument("--thread-id", type=str, default="gbhyte", help="Session thread ID")
    parser.add_argument("--model", type=str, help="Model name to use")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    
    args = parser.parse_args()
    
    try:
        import threading
        def aa():
            # task1 = run_cli(args.thread_id, args.model, args.debug)
           
            from base_tools import _async_worker
            asyncio.run(_async_worker()) 
      
            # results = await asyncio.gather(task1, task2)
        
        t = threading.Thread(target=aa,args=())
        t.start()
        asyncio.run(run_cli(args.thread_id, args.model, args.debug))
    except KeyboardInterrupt:
        pass




if __name__ == "__main__":
    main()
