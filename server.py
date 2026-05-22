from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import rich

from test import t_config
from asray_core import create_graph
import uuid

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from db_conf import DB_URI
from sse_starlette.sse import EventSourceResponse

from langchain.messages import HumanMessage,SystemMessage,AIMessage



app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




import json
import time

async def call_graph(graph , msg , tid = "1" ,init_state = None):
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    
    config  = {"configurable": {"thread_id": tid} ,"recursion_limit": 50}
    # rich.print(f" conf  {config}")
    async def stream_out(graph):
        # global memory
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as memory:
            await memory.setup()
            graph = graph.compile(checkpointer=memory)
            
            snapshot = await graph.aget_state(config)
            if snapshot.values:
                state = snapshot.values
            else:
                state = {"messages": [], "extra": {"thread_id": tid}}
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
                                state["messages"].append(SystemMessage(content=content))
                            elif m_type == "human":
                                state["messages"].append(HumanMessage(content=content))
                            elif m_type == "ai":
                                state["messages"].append(AIMessage(content=content))
                    if "extra" in init_state:
                        state["extra"].update(init_state["extra"])
                
            state["messages"].append(msg)
   
            async for event in graph.astream_events(state,config, version="v2"):
                data = event.get("data")
                chunk = None
                if data!=None:
                    chunk = data.get("chunk")
                if event.get("event") ==  "on_chat_model_stream":
                    if chunk!=None and hasattr(chunk,"content"):
                        text = chunk.content
                        r_text = chunk.additional_kwargs.get("reasoning_content")
                        
                        delta = {}
                        if text:
                            delta["content"] = text
                        if r_text:
                            delta["reasoning_content"] = r_text
                        
                        if delta:
                            ret = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": "asraychat",
                                "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                            }
                            yield json.dumps(ret)

                if event.get("event") ==  "on_custom_event":
                    if chunk!=None and chunk.get("content")!=None:
                        text = chunk["content"]
                        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
                            text = text[1:-1]
                            
                        ret = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": "asraychat",
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                        }
                        yield json.dumps(ret)
            
            # 发送结束标记
            yield json.dumps({
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": "asraychat",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            })
            yield "[DONE]"
                        
    return EventSourceResponse(stream_out(graph))

    


    
from db import with_session
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
import os

base_dir = os.getenv("ASRAY_BASE_DIR", "/home/ai/asraydata")


def process_file(ufs , msg:HumanMessage) :
    from base_tools import _read_office_logic
    
    TEXT_EXTENSIONS = {'.txt', '.py', '.cpp', '.c', '.h', '.js', '.ts', 
                       '.md', '.json', '.yaml', '.yml', '.xml', '.html', 
                       '.css', '.sh', '.log', '.csv', '.ini', '.cfg', '.toml'}
    
    if (not isinstance(ufs,list)):
        ufs = [ufs]
        
    for uf in ufs:
        str_path = base_dir + "/upload_file/" + uf
        fs = os.listdir(str_path)
        for f in fs:
            full_path = os.path.join(str_path, f)
            ret = _read_office_logic(full_path)
            if ret!=None:
                msg.content.extend(ret)
            else:
                # 尝试作为文本文件读取
                ext = os.path.splitext(f)[1].lower()
                if ext in TEXT_EXTENSIONS:
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as file:
                            content = file.read()
                        msg.content.append({"type": "text", "text": content})
                    except Exception as e:
                        msg.content.append({"type": "text", "text": f"读取文本文件失败 {f}: {str(e)}"}) 
    

@with_session
def upsert_conversation(user_name: str, conversation_id: str, content: str, session: Session = None):
    # 使用 PostgreSQL 特有的 ON CONFLICT 语法
    # 理由：原子性操作，在高并发下比 "先查询再插入" 更安全、性能更高
    sql = text("""
        INSERT INTO graph_conversation (conversation_id, user_name, content, archive, create_time)
        VALUES (:cid, :uname, :content, 0, CURRENT_TIMESTAMP)
        ON CONFLICT (conversation_id) 
        DO UPDATE SET content = EXCLUDED.content
    """)
    session.execute(sql, {"cid": conversation_id, "uname": user_name, "content": content})
    
    

    
    
    

async def graph_chat(body : dict ,return_graph = False ):
    
    if(body.get("username")  and   body.get("conversation_id")):
        # 调用插入函数记录对话状态
        upsert_conversation(
            user_name=body.get("username"),
            conversation_id=body.get("conversation_id"),
            content=body.get("messages")[-1].get("content", "")[:100] # 仅截取前100个字符用于显示
        )
    
    from base_tools import read_graph
    
    graph_conf = read_graph(body.get("graph_config" , "")) 
    if graph_conf ==None:
        graph_conf=t_config
        
        
    init_state = graph_conf.get("init_state")
    
    graph = await create_graph(graph_conf)
    
    msg = HumanMessage(
             content=[
                 {"type": "text", "text":  body.get("messages")[-1]["content"] },
             ])
           
    
    if(body.get("upload_file")!=None ):
        process_file(body.get("upload_file") , msg) 
    


    return await call_graph(graph, msg , body.get("conversation_id"), init_state)

    
    
    
    
    
@app.post("/v1/chat/completions")
async def chat_completions(    
    body: dict,request: Request):
    rich.print(body)
    return  await graph_chat(body)


@with_session
def _get_user_graph_history_internal(username: str, session: Session = None):
    sql = text("SELECT conversation_id, content, create_time FROM graph_conversation WHERE user_name = :uname AND (archive = 0 OR archive IS NULL) ORDER BY create_time DESC")
    res = session.execute(sql, {"uname": username}).fetchall()
    return [{"conversation_id": r[0], "content": r[1], "create_time": r[2]} for r in res]

@app.post("/agent/get_user_graph_history")
async def get_user_graph_history(body: dict):
    username = body.get("username")
    data = _get_user_graph_history_internal(username)
    return {"code": 200, "data": data}

@app.post("/agent/get_garph_history_by_cid")
async def get_graph_history_by_cid(body: dict):
    cid = body.get("cid")
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as memory:
        config = {"configurable": {"thread_id": cid}}
        gc = await memory.aget(config)
        if gc == None:
            return {"code": 200, "data":[]}
        ml = gc.get("channel_values")
        if ml == None:
            return  {"code": 200, "data":[]}

        return {"code": 200, "data": ml["messages"]}
            
    

@with_session
def _delete_graph_history_internal(cid: str, session: Session = None):
    sql = text("DELETE FROM graph_conversation WHERE conversation_id = :cid")
    session.execute(sql, {"cid": cid})
    return "success"

@app.post("/agent/delete_garph_history_by_cid")
async def delete_graph_history_by_cid(body: dict):
    cid = body.get("cid")
    _delete_graph_history_internal(cid)
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as memory:
        await memory.setup()
        await memory.adelete_thread(cid)
    return {"code": 200, "data": "success"}

@app.post("/agent/list_llm")
async def list_llm():
    # 这里可以根据 asray_core 中的模型配置返回
    from asray_core import load_models_config
    config = load_models_config()
    models = [{"name": k, "value": k} for k in config.keys()]
    return {"code": 200, "data": models}

@app.post("/agent/list_mcp")
async def list_mcp():
    from base_tools import mcp_arg_map
    mcps = [{"name": k, "value": k} for k in mcp_arg_map.keys()]
    return {"code": 200, "data": mcps}

@app.post("/agent/list_graph")
async def list_graph():
    ret = [{"Asray": "智能agent"}]
    import os, json
    graphs_dir = os.path.join(base_dir, "graphs")
    if os.path.exists(graphs_dir):
        for f in sorted(os.listdir(graphs_dir)):
            if f.endswith(".json"):
                name = f[:-5]
                desc = ""
                try:
                    with open(os.path.join(graphs_dir, f), 'r') as fp:
                        data = json.loads(fp.read())
                        desc = data.get("description", "")
                except Exception:
                    pass
                ret.append({name: desc})
    return {"code": 200, "data": ret}





if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8899)

# docker run -d --name t1   -v /home/ai/asray_client_data/admin:/app/workspace   -e  ASRAY_BASE_DIR=/app/workspace   -p 8899:8899   asraychat:v3.14   /bin/bash -c "python3 server.py"