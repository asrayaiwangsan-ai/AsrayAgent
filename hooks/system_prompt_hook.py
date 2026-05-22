import os
import json
import copy
from hook_base import BaseHook
from langchain_core.messages import AIMessage, SystemMessage, BaseMessage


base_dir = os.getenv("ASRAY_BASE_DIR", "/home/ai/asraydata")


skill_map = {}


graphs_map = {}


def read_graph(name):
    try:
        with open(f"{base_dir}/graphs/{name}.json", 'r') as f:
            data = f.read()
            ret = json.loads(data)
            return ret
    except Exception as e:
        print(f"read_graph e {e}")
        return None


def _load_skill_whitelist():
    try:
        with open(f"{base_dir}/system_load_skill.json", 'r') as f:
            return json.loads(f.read())
    except Exception as e:
        print(f"load skill whitelist failed: {e}")
        return []


def _load_agent_whitelist():
    try:
        with open(f"{base_dir}/system_load_agent.json", 'r') as f:
            return json.loads(f.read())
    except Exception as e:
        print(f"load agent whitelist failed: {e}")
        return []


def list_skill() -> str:
    whitelist = _load_skill_whitelist()
    for s in whitelist:
        if s not in skill_map:
            file_path = base_dir + "/skills/" + s + "/SKILL.md"
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    try:
                        content = f.read()
                        b_description = content.find("description:")
                        e_description = content.find("---", b_description)
                        description = content[b_description:e_description].replace('description:', "")
                        sc = content[e_description + len("---"):]
                        skill_map[s] = {"description": description, "content": sc}
                    except Exception as e:
                        print(e)
    ret = ""
    for s in whitelist:
        if s in skill_map:
            ret = ret + f"- **{s}**: {skill_map.get(s).get('description')}\n"
    return ret


def list_agents() -> str:
    whitelist = _load_agent_whitelist()
    for s in whitelist:
        if graphs_map.get(s) is None:
            graph = read_graph(s)
            if graph is None:
                continue
            graphs_map[s] = {"description": graph.get("description"), "content": graph}

    ret = ""
    for s in whitelist:
        if s in graphs_map:
            ret = ret + f"- **{s}**: {graphs_map.get(s).get('description')}\n"
    return ret


class SystemPromptHook(BaseHook):

    def __init__(self, conf: dict):
        self.usermd_path = conf.get("usermd_path", "user.md") if conf is not None else "user.md"
        self.soulmd_path = conf.get("soulmd_path", "soul.md") if conf is not None else "soul.md"

    async def run(self, msg_list: list, extra: dict, n_msg: BaseMessage = None) -> list:
        if n_msg is not None:
            return msg_list

        base_dir = os.getenv("ASRAY_BASE_DIR", "/home/ai/asraydata")

        soul_content = ""
        soul_path = os.path.join(base_dir, self.soulmd_path)
        if os.path.exists(soul_path):
            with open(soul_path, "r", encoding="utf-8") as f:
                soul_content = f.read()

        user_content = ""
        user_path = os.path.join(base_dir, self.usermd_path)
        if os.path.exists(user_path):
            with open(user_path, "r", encoding="utf-8") as f:
                user_content = f.read()

        full_content = f"{soul_content}\n\n{user_content}\n\n "

        full_content = full_content + f"\n- #默认工作路径：`{base_dir}`。所有非绝对路径的文件操作（包括读写、加载 Prompt）都将基于此目录进行。"
        full_content = full_content + f"\n- #可以直接通过load_skill使用的Skill：\n {list_skill()}"
        full_content = full_content + f"\n- #可以直接通过call_agent使用的Agent：\n {list_agents()}"
        full_content = full_content + f"\n- #并非所有Agent和Skill都在上面的列表中，你可以使用list_all_agent/list_all_skill查看全部agent/skill"

        system_index = -1
        for i, msg in enumerate(msg_list):
            if isinstance(msg, SystemMessage):
                system_index = i
                break

        if system_index != -1:
            msg_list[system_index] = SystemMessage(content=full_content)
        else:
            msg_list.insert(0, SystemMessage(content=full_content))

        return msg_list
