
from typing import Annotated, TypedDict
from langgraph.graph import END, StateGraph, add_messages
from parse_lopenai import RChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage ,SystemMessage, convert_to_messages

from langchain.messages import HumanMessage
from base_tools import tools_map,mcp_arg_map,mcp_tools_map
import json
import os
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient
from hook_loader import load_hook_instance, execute_hooks





# 将 custom 目录加入路径
custom_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom")
if custom_dir not in sys.path:
    sys.path.append(custom_dir)


base_dir = os.getenv("ASRAY_BASE_DIR", "/home/ai/asraydata")

n_costom = base_dir + "/custom"
if n_costom not in sys.path:
    sys.path.append(n_costom)
    
    

class BaseState(TypedDict):
    messages: list
    extra: dict



_llmconfig = None
def load_models_config():
    global _llmconfig
    if _llmconfig!=None:
        return _llmconfig
    
    config_path = os.path.join(base_dir, "models_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                _llmconfig = json.load(f)
                return _llmconfig
            except Exception as e:
                print(f"Error loading models_config.json: {e}")
    return {}


def is_mutimodel(name):
    config = load_models_config()
    model_info = config.get(name) or config.get("default", {})
    return model_info.get("multimodel", True)


def get_model_info(name):
    config = load_models_config()
    return config.get(name) or config.get("default", {})


def get_defuat_mutimodel():
    config = load_models_config()
    for name, info in config.items():
        if info.get("multimodel", False):
            return name
    return None
    
        


async def get_llm(model_name = "qwen3",temperature = 0.1 , max_tokens = 12480, tools=[], mcp_tools=[],mcp_only=[] ,top_p = 0.8, extra_body=None):
    
    model_info = get_model_info(model_name)
    max_tokens = 10240 if max_tokens==None else max_tokens
    
    if extra_body == None:
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True}
        }
    
   
    llm = RChatOpenAI(temperature=temperature,  api_key=model_info.get("api_key"),
                      base_url=model_info.get("api_base_url"), model=model_name,
                      max_tokens=max_tokens,
                      top_p=top_p,
                      extra_body=extra_body
    )
    
    
    utools = []
    
    if len(tools)>0:
        for str_tool in tools:
            ut = tools_map.get(str_tool)
            if ut!=None:
                utools.append(ut)
         
    if len(mcp_tools)>0:
        for str_mcp_tool in mcp_tools:
            mtools = mcp_tools_map.get(str_mcp_tool)
            if mtools == None:
                mcp_client_arg = mcp_arg_map.get(str_mcp_tool)
                if mcp_client_arg!=None: 
                    client =  MultiServerMCPClient({str_mcp_tool:mcp_client_arg})
                    mtools = await client.get_tools()
                    mcp_tools_map[str_mcp_tool] = mtools
                    for t in mtools:
                        tools_map[t.name] = t
            if mtools!=None:
                for t in mtools:
                    if len(mcp_only)>0:
                        if t.name in mcp_only:
                            utools.append(t)
                    else:
                        utools.append(t)
                            
        
    if len(utools)>0:
        llm_with_tools = llm.bind_tools(utools)
        return llm_with_tools
    
    
    return llm




from hook_loader import load_hook_instance, execute_hooks

from langgraph.checkpoint.memory import MemorySaver

# 全局共享内存，用于主图和子图的持久化
shared_memory = MemorySaver()




def get_llm_invoke_closure(conf:dict , llm):
    
    # 预加载 Hook 实例（单例模式，避免重复实例化）
    before_hooks = []
    after_hooks = []
    bhs = conf.get("before_invoke", {})
    for k in bhs.keys():
        before_hooks.append(load_hook_instance(k , bhs.get(k,None)) )
    
    
    ahs = conf.get("after_invoke", {})
    for k in ahs.keys():
        after_hooks.append(load_hook_instance(k , ahs.get(k,None)) )
    
    

    async def llm_invoke(state:BaseState):
               
        msg_list = state["messages"]
        extra = state["extra"] 
        
        try:
            # 将 llm 引用放入 extra，方便需要调用 LLM 的 Hook（如压缩 Hook）使用
            extra["_llm"] = llm
                 
            llm_ready_messages = await execute_hooks(before_hooks, msg_list, extra)
            n_msg = await llm.ainvoke(llm_ready_messages)
            msg_list.append(n_msg)
            await execute_hooks(after_hooks, msg_list, extra, n_msg=n_msg)
        finally:
            # 必须在返回前移除不可序列化的对象，否则持久化（Checkpointing）会报错
            if "_llm" in extra:
                del extra["_llm"]
        
        return {"messages": msg_list, "extra": extra} 

    return llm_invoke




def get_tool_call_closure(model_name, node_conf: dict = {}):
    
    # 预加载 Hook 实例
    before_hooks = [load_hook_instance(h) for h in node_conf.get("before_tool_call", [])]
    after_hooks = [load_hook_instance(h) for h in node_conf.get("after_tool_call", [])]

    async def toolcall_invoke(state:BaseState):

        msg_list = state["messages"]
        extra = state["extra"] 

        # 执行 before_tool_call 管道
        msg_list = await execute_hooks(before_hooks, msg_list, extra)
        
        last_message = msg_list[-1]

        outputs = []
        calls = []

        for call in last_message.tool_calls:
            calls.append(call)


        for tool_call in calls:
        
            arg = tool_call.get("args")
            if arg == None:
                arg = tool_call.get("arguments")

            # 注入隐藏参数
            arg["tool_call_id"] = tool_call["id"]
            arg["multimodal"] = True if get_defuat_mutimodel()!=None else False

            # 关键修复：注入可序列化的副本，避免将 _llm (RunnableBinding) 传入工具导致持久化报错
            serializable_extra = {k: v for k, v in extra.items() if not k.startswith("_")}
            arg["extra"] = serializable_extra

            try:
                tool = tools_map.get(tool_call["name"])
                tool_result = await tool.ainvoke(arg)

                # 同步工具对 extra 的修改回原始 extra 对象
                if "extra" in arg:
                    for k, v in arg["extra"].items():
                        if not k.startswith("_"):
                            extra[k] = v

            except Exception as e:
                import logging
                logging.exception(e)
                
                tool_result = "Tools call 失败"

            if isinstance(tool_result ,ToolMessage):
                # 如果工具返回的是 ToolMessage，确保 ID 正确
                tool_result.tool_call_id = tool_call["id"]
                outputs.append(tool_result)
            else:
                outputs.append(
                    ToolMessage(
                        content=str(tool_result),
                        name=tool_call["name"],
                        tool_call_id=f"{tool_call.get("id")}",
                    )
                )

        nmsgs = msg_list + outputs

        # 执行 after_tool_call 管道
        nmsgs = await execute_hooks(after_hooks, nmsgs, extra,n_msg=outputs)

        return {"messages": nmsgs , "extra" : extra}


    return toolcall_invoke
        
        
        
    
    
def get_cond_router_closure(conf):
    
    def cond_router(state:BaseState):
        if conf["type"] == "msg_find":
            router =  conf["conf"]
            if router.get("nokey")==None:
                router["nokey"]=END
            msg = state["messages"][-1].content
            ks =router.keys()
            if "<tool_call>" in  ks:
                if isinstance(state["messages"][-1],AIMessage) and( state["messages"][-1].tool_calls or state["messages"][-1].invalid_tool_calls):
                    return router["<tool_call>"]["to"]
            for k in ks:
                if k =="nokey" or k ==  "<tool_call>":
                    continue
                if msg.find(k)!=-1:
                    return router[k]["to"]
            
            
            return router["nokey"]
        
    return  cond_router

   
async def create_graph(conf):
    workflow = StateGraph(BaseState)
    nodes =  conf["nodes"]
    edges =  conf["edges"]
    
    
    llm_config ={} if conf.get("llm_config") == None else conf.get("llm_config")
    llm = await get_llm(llm_config.get("model_name","qwen3") , llm_config.get("temperature",0.7) ,  
                  llm_config.get("max_tokens",10240) , llm_config.get("tools",[])   ,  llm_config.get("mcp_tools",[]), llm_config.get("mcp_only",[]) , llm_config.get("top_p") , llm_config.get("extra_body") )
    
    for node in nodes:
        if node["type"] == "llm_invoke":            
            fn = get_llm_invoke_closure(node["conf"] ,llm )
            workflow.add_node(node["name"] ,fn )
            
        if node["type"] == "base_tool_call":
            tool_node = get_tool_call_closure(llm_config.get("model_name"), node.get("conf", {}))
            workflow.add_node(node["name"] ,tool_node )
        if node["type"] == "custom":
            # 加载自定义节点类实例
            custom_instance = load_hook_instance(node["conf"]["class"], node["conf"])
            if custom_instance:
                workflow.add_node(node["name"], custom_instance)
            else:
                print(f"警告：无法加载自定义节点 {node['name']}")
            
            
    start =  conf["start"]
    workflow.set_entry_point(start)

            
    for fnode in edges.keys():
        edge = edges[fnode]
        for e in edge:
            if e["type"] == "goto":
                 workflow.add_edge(fnode ,e["to"] )       
            if e["type"] == "cond":
                fn = get_cond_router_closure(e["conf"])
                workflow.add_conditional_edges(fnode ,fn )
                
            if e["type"] == "custom_cond":
                # 加载自定义边类实例
                custom_edge_instance = load_hook_instance(e["conf"]["class"], e["conf"])
                if custom_edge_instance:
                    workflow.add_conditional_edges(fnode, custom_edge_instance)
                else:
                    print(f"警告：无法加载自定义边从节点 {fnode}")
                

    return workflow
                
                
                