import importlib
import sys
import os

# 将当前目录和 hooks 目录加入路径，确保反射加载正常
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
hooks_dir = os.path.join(current_dir, "hooks")
if hooks_dir not in sys.path:
    sys.path.append(hooks_dir)
    
base_dir = os.getenv("ASRAY_BASE_DIR", "/home/ai/asraydata")
n_costom = base_dir + "/hooks"
if n_costom not in sys.path:
    sys.path.append(n_costom)

def load_hook_instance(hook_path: str, conf: dict = None):
    """
    根据 '文件名.类名' 动态加载类实例
    例如: 'multimodal_hook.MultimodalHook'
    如果提供了 conf，则将其传递给构造函数。
    """
    try:
        module_name, class_name = hook_path.rsplit(".", 1)
        # 动态导入模块
        module = importlib.import_module(module_name)
        # 获取类
        hook_class = getattr(module, class_name)
        # 实例化
        if conf is not None:
            return hook_class(conf)
        return hook_class()
    except Exception as e:
        print(f"加载类失败 [{hook_path}]: {str(e)}")
        return None

async def execute_hooks(hooks: list, msg_list: list, extra: dict, n_msg=None):
    """
    管道式顺序执行 Hook 列表
    """
    current_msgs = msg_list
    for hook in hooks:
        if hook:
            current_msgs = await hook.run(current_msgs, extra, n_msg=n_msg)
    return current_msgs
