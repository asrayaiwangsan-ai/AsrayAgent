\# AsrayAgent

An AI agent based on langgraph that converts JSON-defined DSL into agents. In fact, 95% of this agent's code was written by deepseek-v4 running on this agent itself, and was also submitted to GitHub by itself.

## How to Run

conda create -n your_name python=3.12

pip install -r requirements.txt

Before running, please configure `api_key` in `models_config.json`. All models compatible with the OpenAI API can be used.

export ASRAY_BASE_DIR your_path
(Note: `asraydata` is a reference value for `ASRAY_BASE_DIR`.)

python test.py  --thread-id = your_id 

## You can continue the conversation through your_id

## It can run on Windows, but has not been fully tested.

---

# AsrayAgent

一个基于langgraph的 ai agent ，将json定义的DSL转换为agent，事实上此agent 95%的代码是deepseek-v4 运行在这个agent上自己写的，也是自己提交到github的。

\## 运行方式

conda create -n your\_name python=3.12

pip install -r requirements.txt

运行前请在 `models_config.json` 中配置 `api_key`，所有兼容 OpenAI API 的模型均可使用。

export ASRAY\_BASE\_DIR your\_path
（注：`asraydata` 为 `ASRAY_BASE_DIR` 的参考路径。）

python test.py  --thread-id = your\_id 



\## 你可以通过your\_id继续对话

\## windows 下可以运行，但未进行全面测试







