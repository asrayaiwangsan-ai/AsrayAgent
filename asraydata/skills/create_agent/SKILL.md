---
name: create_agent
description: 通过编写JSON文件，用DSL创建具备特定逻辑的子 Agent
---

## 完整 JSON 结构

```json
{
  "description": "",
  "start": "",
  "nodes": [ ... ],
  "edges": { ... },
  "llm_config": { ... },
  "init_state": { ... }
}
```
##  JSON 存放目录
  将生存的json存放在你默认工作路径/graphs目录下
## DSL 与 LangGraph 的映射

| DSL 概念 | LangGraph 对应 |
|----------|---------------|
| `start` | `workflow.set_entry_point()` |
| `nodes[]` | `workflow.add_node()` 的节点标识 |
| `edges[]` | `workflow.add_edge()` |
| `init_state` | 初始 `BaseState`（`messages: list` + `extra: dict`） |
| `llm_config` | 构建ChatOpenAI的参数，构造 LLM 实例并绑定工具 |

所有 DSL 配置最终解析并编译为 LangGraph StateGraph并执行。

### 顶层字段一览

| 字段 | 必需 | 说明 |
|------|------|------|
| `description` | 是 | 功能描述，`list_agents` 工具读取此字段 |
| `start` | 是 | 入口节点名称，对应 LangGraph `entry_point` |
| `nodes` | 是 | 节点数组，定义所有处理步骤 |
| `edges` | 是 | 边配置，定义节点间流转规则 |
| `llm_config` | 否 | 模型参数与工具绑定，不填则使用默认值 |
| `init_state` | 否 | 初始消息和元数据，不填则为空 messages 和空 extra |
---

## 三、节点 (nodes)

### 3.1 通用结构

每个节点必须包含：
- `name`：节点名称（字符串，图内唯一）
- `type`：节点类型，决定执行逻辑
- `conf`：节点配置对象（可选，依赖 type）

### 3.2 节点类型详解

#### A. `llm_invoke` —— LLM 调用节点

调用 LLM 并返回 AI 消息。这是最核心的节点类型。

```json
{"name": "chatbot", "type": "llm_invoke", "conf": {
    "before_invoke": {
        "system_prompt_hook.SystemPromptHook":{},
        "multimodal_hook.MultimodalHook":{},
        "deepseek_reasoning_hook.DeepSeekReasoningHook":{}
    },
   "after_invoke": {"async_result_hook.AsyncResultHook":{} }
   }
 }
```
**conf 字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `before_invoke` | `dict` | 在 LLM 调用前执行的 Hook,key为类名，value为创建类的参数 |
| `after_invoke` | `dict` | 在 LLM 调用后执行的 Hook,key为类名，value为创建类的参数 |

**Hook 执行顺序：**
```
msg_list → 按key的顺序执行before_invoke→ LLM.ainvoke() → 按key的顺序after_invoke → 输出 n_msg
```

**可用的before hook：**
| 字段 | 说明 |
|------|------|
|`system_prompt_hook.SystemPromptHook`  | 读取soul.md,user.md并添加到system prompt，请注意你自己的system prompt就来自于这两个文件，请慎重使用 ,dict使用： key "usermd_path" ,"soulmd_path" 对应user.md和soul.md的相对路径，不填写为user.md,soul..md |
|`multimodal_hook.MultimodalHook`  | 处理多模态消息，对纯文本模型没有影响，可以默认添加 ,dict为空|
|`deepseek_reasoning_hook.DeepSeekReasoningHook` |  如果基座模型是deepseek，必须添加 ,dict为空|


**可用的after hook：**
| 字段 | 说明 |
|------|------|
|`async_result_hook.AsyncResultHook`  | 用于异步读取子agent的返回，如果你创建的子agent有call_graph tool则必须添加,dict为空 |
|`context_compress_hook.ContextCompressHook`  | 用于压缩历史上下文,除非你确定agent会有很多轮交互，否则不用添加 dict使用： "threshold","keep_recent" 对应压缩消息阈值与保留keep_recent条不压缩，默认50,10  |
|`tool_message_compress_hook.ToolMessageCompressHook` |  用于压缩历前n轮会话之前的调用tool返回的Toolsmessage,除非你确定agent会有很多轮交互，否则不用添加 ，dict使用： "max_tool_output_length" 超过设置的字符即触发压缩默认500，"head_length"保留头部字符数默认100，"tail_length"保留尾部字符数默认100，"keep_recent_n" 保留最近多少条消息不压缩默认7  |



#### 自定义hook

你可以自己编写hook，请存放在默认工作路径/hooks下 ，自定hook需要继承下面的父类，dsl中加入规则为:文件名.类名
```python
from abc import ABC, abstractmethod
from langchain_core.messages import AIMessage,BaseMessage

class BaseHook(ABC):
    """
    Hook 基类。所有自定义 Hook 必须继承此类。
    """
    
    @abstractmethod
    async def run(self, msg_list: list, extra: dict, n_msg: BaseMessage = None) -> list:
        """
        执行 Hook 逻辑。
        :param msg_list: 当前的消息列表
        :param extra: 共享的上下文字典
        :param n_msg: LLM 的回复消息 (仅在 after_invoke 阶段有效)
        :return: 处理后的消息列表
        
        请注意
        ## befor_hook中，llm执行的是的函数retuen：
        llm_ready_messages = await execute_hooks(before_hooks, msg_list, extra)
        n_msg = await llm.ainvoke(llm_ready_messages),修改msg_list会导致整个 graph的state被修改。如果不需要修改state请deepcopy后再进行处理
        
        ## after_hook中，return将被忽略
        """
        pass
```



#### B. `base_tool_call` —— 工具调用节点

解析上一条 AI 消息中的 `tool_calls`，逐条执行并返回 `ToolMessage`。

```json
{
  "name": "tools",
  "type": "base_tool_call"
}
```

**无需 conf。** 行为由 `llm_config.tools` 和 `llm_config.mcp_tools` 决定。

**如果你添加了工具，必须添加base_tool_call node**

#### C. `custom` —— 自定义节点

加载自定义 Python 类并执行。 请将你自己写的custom_node存放在默认工作路径/custom下，自定node需要继承下面的父类，dsl中加入规则为:文件名.类名
```python
class BaseCustomNode(ABC):
    """
    自定义节点基类。
    实现时需覆盖 __call__ 方法。
    """
    def __init__(self, conf: dict):
        self.conf = conf

    @abstractmethod
    async def __call__(self, state: dict) -> dict:
        """
        处理节点逻辑并返回状态更新字典。
        """
        pass
```


## 边 (edges)

edges 是 `{ 源节点名: 边规则数组 }` 的 dict。

### 4.1 `goto` —— 直接跳转

```json
{"type": "goto", "to": "chatbot"}
```

无条件从源节点跳转到目标节点。对应 `workflow.add_edge(from, to)`。

### 4.2 `cond` —— 条件跳转

```json
{
  "type": "cond",
  "conf": {
    "type": "msg_find",
    "conf": {
      "<tool_call>": {"to": "tools"},
      "关键词A": {"to": "node_a"},
      "关键词B": {"to": "node_b"},
      "nokey": "__end__"
    }
  }
}
```

**conf.type 当前支持：** `msg_find`（基于最后一条消息内容的关键词匹配）

**msg_find 匹配规则：**

1. **优先级最高：`<tool_call>`** — 如果最后一条消息包含 `tool_calls` 或 `invalid_tool_calls`，立即跳转到指定节点。这是工具调用路由的标准方式。
2. **关键词匹配** — 在最后一条消息的 `content` 中查找关键词（`msg.content.find(key) != -1`）
3. **默认路由：`nokey`** — 没有任何关键词匹配时的目标。如果不设置，默认为 `__end__`。

**重要：** 匹配顺序是固定的——先检查 `<tool_call>`，然后按 keys 遍历。一旦匹配就返回，不继续检查。



## 五、LLM 配置 (llm_config)

```json
{
  "model_name": "deepseek-v4-pro",
  "temperature": 0.6,
  "max_tokens": 52000,
  "top_p": 0.9,
  "extra_body": {
    "chat_template_kwargs": {"enable_thinking": true}
  },
  "tools": ["bash", "write_file", "read_text_file"],
  "mcp_tools": ["antv-visualization-chart"],
  "mcp_only": ["generate_line_chart", "generate_bar_chart"]
}
```

### 5.1 可用工具说明
tools中可以加入你自己的已有的工具
### 5.1 可用model_name说明
默认工作路径/models_config.json 中有可用的全部模型


## 六、初始状态 (init_state)

```json
{
  "init_state": {
    "message": [
      {"type": "system", "path": "my_prompt.md"},
      {"type": "human", "content": "初始用户消息（可选）"}
    ],
    "extra": {
      "custom_key": "custom_value"
    }
  }
}
```

### message 条目格式

| 字段 | 说明 |
|------|------|
| `type` | `"system"` / `"human"` / `"ai"` |
| `path` | Prompt 文件路径。默认基于 `默认工作路径/prompts/`。与 `content` 互斥 |
| `content` | 直接指定消息内容。与 `path` 互斥 |

**加载逻辑：**
- 如果指定 `path`，从文件中读取内容作为消息正文
- 如果指定 `content`，直接使用
- `SystemMessage` 会先于所有对话消息注入

### extra

初始元数据字典。在整个图执行过程中可以读写。

---

## 完整实例
```json
t_config = {
    "start": "chatbot",
    "nodes": [
        {"name": "welcome", "type": "custom", "conf": {"class": "test_node.WelcomeNode", "prefix": "TEST"}},
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
        "welcome": [{"type": "goto", "to": "chatbot"}],
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
        "temperature": 0.6,
        "max_tokens": 52000,
        "top_p": 0.9,
        "tools": ["bash", "write_file", "replace", "grep_search",
                  "read_text_file", "read_doc_file", 
                  "load_skill", "call_agents","manager_agent_state",
                  "update_user_memory", "rag_save_usermemory" , "rag_load_user_memory" ,
                  "browser_tool" , "web_fetch"],
        "mcp_tools":["WebSearch"]
    },
    "description": "Asray CLI Agent"
}
```



                




