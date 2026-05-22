from abc import ABC, abstractmethod

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

class BaseCustomEdge(ABC):
    """
    自定义条件边基类。
    实现时需覆盖 __call__ 方法。
    """
    def __init__(self, conf: dict):
        self.conf = conf

    @abstractmethod
    async def __call__(self, state: dict) -> str:
        """
        根据状态返回下一个节点的名称。
        """
        pass
