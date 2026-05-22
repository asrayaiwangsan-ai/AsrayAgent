from abc import ABC, abstractmethod
from langchain_core.messages import AIMessage,BaseMessage

class BaseHook(ABC):
    """
    Hook 基类。所有自定义 Hook 必须继承此类。
    """
    def __init__(self, conf: dict):
        self.conf = conf
    
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
