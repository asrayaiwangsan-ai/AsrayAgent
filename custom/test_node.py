from custom_base import BaseCustomNode
import datetime

class WelcomeNode(BaseCustomNode):
    async def __call__(self, state: dict) -> dict:
        prefix = self.conf.get("prefix", "INFO")
        print(f"\n[{prefix}] [WelcomeNode] 正在处理，准备注入欢迎信息...")
        
        extra = state.get("extra", {})
        # 在 extra 中注入一些测试数据
        extra["welcome_message"] = f"Hello from Custom Node! Time: {datetime.datetime.now()}"
        
        return {"extra": extra}
