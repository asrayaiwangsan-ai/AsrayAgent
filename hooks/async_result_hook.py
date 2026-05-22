from hook_base import BaseHook
from langchain_core.messages import HumanMessage,AIMessage


class AsyncResultHook(BaseHook):
    async def run(self, msg_list: list, extra: dict, n_msg=None) -> list:
        thread_id = extra.get("thread_id")
        if not thread_id:
            return msg_list

        from base_tools import pop_async_results

        results = await pop_async_results(thread_id)
        if not results or not isinstance(n_msg,AIMessage):
            return msg_list
        if n_msg.tool_calls or  n_msg.invalid_tool_calls:
            return msg_list
            
        msg_list.append(HumanMessage(
            content=f"[AsyncAgentResult] call_id: {results[0]}\n {results[1]}"
        ))

        return msg_list
