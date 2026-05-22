from custom_base import BaseCustomEdge
import datetime
from langchain_core.messages import HumanMessage

class AsyncAgentCallEdge(BaseCustomEdge):
    async def __call__(self, state: dict) :
       msg = state["messages"][-1]
       if isinstance(msg,HumanMessage):
           return "chatbot"
       return "end"