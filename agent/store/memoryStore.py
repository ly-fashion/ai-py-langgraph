import ast
import getpass
import os
import uuid
import typing_extensions
from langchain_openai import OpenAI
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

from config import config
from langchain_openai import ChatOpenAI

in_memory_store = InMemoryStore()
memory = MemorySaver()

llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=config.TEMPERATURE,
    openai_api_key=config.OPENAI_API_KEY,
    openai_api_base=config.OPENAI_BASE_URL,
)


# 定义状态模型
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 消息列表


# 定义对话节点,访问记忆并在模型中调用
def call_model(state: MessagesState, *, config: RunnableConfig, store: BaseStore):
    # 获取用户id
    user_id = config["configurable"]["user_id"]

    # 定义命名空间
    namespace = ("memories", user_id)

    # 更具用户id检索记忆
    memories = store.search(namespace)
    info = "\n".join([d.value["data"] for d in memories])

    # 存储记忆
    last_message = state["messages"][-1]
    store.put(namespace, str(uuid.uuid4()), {"data": last_message.content})

    system_message = f"你是一个有用的助手，以下是用户的相关记忆：\n{info}\n请根据这些记忆和用户的输入进行回复。"

    response = llm.invoke(
        [{"type": "system", "content": system_message}] + state["messages"]
    )

    store.put(namespace, str(uuid.uuid4()), {"data": response.content})
    return {"messages": response}


# 构建状态图
graph = StateGraph(State)

# 添加节点
graph.add_node(
    "call_model",
    call_model,
)

# 定义边
graph.add_edge(START, "call_model")
graph.add_edge("call_model", END)

# 编译图
graph_compile = graph.compile(checkpointer=memory, store=in_memory_store)

# 可视化 — 保存为图片文件
os.makedirs("dist/image", exist_ok=True)
png_data = graph_compile.get_graph().draw_mermaid_png()
with open("dist/image/graph.png", "wb") as f:
    f.write(png_data)
print("图已保存到 dist/image/graph.png")

config_mine = {"configurable": {"thread_id": "33", "user_id": "110"}}

config_mine_two = {"configurable": {"thread_id": "33", "user_id": "111"}}


async def one():
    async for chunk in graph_compile.astream(
        {"messages": ["你好,我是林北"]}, config=config_mine, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()


async def two():
    async for chunk in graph_compile.astream(
        {"messages": ["你知道我叫什么吗?"]},
        config=config_mine_two,
        stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()


import asyncio

asyncio.run(one())
asyncio.run(two())

for memory in in_memory_store.search(("memories", "111")):
    print("User 111 Memory:", memory.value)
