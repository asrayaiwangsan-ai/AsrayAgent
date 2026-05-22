from langchain_openai import ChatOpenAI
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
import rich


from langchain_core.messages import AIMessage
class RChatOpenAI(ChatOpenAI):
    
    def _get_request_payload(self, messages, *args, **kwargs):
        # 1. 先调用父类方法，完成标准的 OpenAI 格式序列化
        payload = super()._get_request_payload(messages, *args, **kwargs)
        
        # 2. 遍历比对：拦截 payload["messages"] 数组，并将 reasoning_content 强行注回字典
        if "messages" in payload:
            for msg, msg_dict in zip(messages, payload["messages"]):
                # 只有原始对象是 AIMessage，且其 additional_kwargs 中确实存在 reasoning_content 时才处理
                if isinstance(msg, AIMessage) and "reasoning_content" in msg.additional_kwargs and "set_add_reasoning" in msg.additional_kwargs :
                    msg_dict["reasoning_content"] = msg.additional_kwargs["reasoning_content"]
                    
        return payload
    
    
    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: Optional[dict],
    ) -> Optional[ChatGenerationChunk]:
        ret =  super()._convert_chunk_to_generation_chunk(chunk,default_chunk_class,base_generation_info)
        
        message = ret.message
        choices = (
            chunk.get("choices", [])
            # from beta.chat.completions.stream
            or chunk.get("chunk", {}).get("choices", [])
        )
        if choices and len(choices)>0:
            choice = choices[0]
            delta = choice["delta"]
            # rich.print(chunk)
            
            if ret and message and choices and choice and delta:
                reasoning = cast(str, delta.get("reasoning_content") or "")
                if reasoning == "":
                    reasoning= cast(str, delta.get("reasoning") or "")
                message.additional_kwargs["reasoning_content" ] = reasoning
        
        return ret