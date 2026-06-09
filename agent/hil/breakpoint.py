import getpass
import os
from typing import TypedDict
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver


from config import config

llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=config.TEMPERATURE,
    openai_api_key=config.OPENAI_API_KEY,
    openai_api_base=config.OPENAI_BASE_URL,
)


# 定义状态模式
class State(TypedDict):
    user_input: str
    model_response: str
    user_approval: str


# 定义大模型交互节点
def call_model(state):
    messages = state["user_input"]
    if "删除" in state["user_input"]:
        state["user_approval"] = (
            f"用户请求删除数据，内容为：{state['user_input']}，请确认是否批准该请求？"
        )
        state["model_response"] = AIMessage(content="")
    else:
        response = llm.invoke(messages)
        state["user_approval"] = "直接执行"
        state["model_response"] = response
    return state


# 定义人工介入的breakpoint内部的执行逻辑
def execute_users(state):
    if state["user_approval"] == "是":
        response = "你的删除请求已批准，正在执行删除操作。"
        return {"model_response": AIMessage(content=response)}
    elif state["user_approval"] == "否":
        response = "你的删除请求已被拒绝，操作已取消。"
        return {"model_response": AIMessage(content=response)}
    else:
        return state


# 翻译提示词
def translate_message(state: State):
    system_prompt = """
    你是一个翻译助手，负责将用户输入的文本翻译成英文。请确保翻译准确且自然。
    """

    model_response = state["model_response"]
    content = (
        model_response.content
        if hasattr(model_response, "content")
        else str(model_response)
    )

    messages = [SystemMessage(content=system_prompt)] + [HumanMessage(content=content)]
    response = llm.invoke(messages)
    return {"model_response": response}


# 构建状态图
builder = StateGraph(State)

# 添加节点
builder.add_node(
    "call_model",
    call_model,
)

builder.add_node(
    "execute_users",
    execute_users,
)

builder.add_node(
    "translate_message",
    translate_message,
)

# 定义边
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "execute_users")
builder.add_edge("execute_users", "translate_message")
builder.add_edge("translate_message", END)

# 设置checkpointer,使用内存存储
memory = MemorySaver()

# 编译图,并且添加短期记忆,使用interrupt_before参数在execute_users节点前设置断点,等待用户审批
graph = builder.compile(checkpointer=memory, interrupt_before=["execute_users"])

# 可视化 — 保存为图片文件
os.makedirs("dist/image", exist_ok=True)
png_data = graph.get_graph().draw_mermaid_png()
image_name = "graph_with_breakpoint.png"
with open(f"dist/image/{image_name}", "wb") as f:
    f.write(png_data)
print(f"图已保存到 dist/image/{image_name}")

run_config = {"configurable": {"thread_id": "1"}}


# async def app():
#     async for chunk in graph.astream(
#         {"user_input": f"我要删除dist/image目录下的{image_name}文件"},
#         config=run_config,
#         stream_mode="values",
#     ):
#         print(chunk)


# import asyncio

# asyncio.run(app())

# snapshot = graph.get_state(run_config)
# print("当前状态快照:", snapshot)

# snapshot.values["user_approval"] = "否"
# graph.update_state(run_config, snapshot.values)

# print("-----------------")


# async def next():
#     async for chunk in graph.astream(
#         {"user_input": f"请你介绍一下你自己"},
#         config=run_config,
#         stream_mode="values",
#     ):
#         print(chunk)


# asyncio.run(next())


# 下面就来创建一个函数来封装对话逻辑
def run_dialogue(graph, config, all_chunks=[]):
    while True:
        # 接受用户输入
        user_point = input(
            "请输入你的请求（例如：我要删除dist/image目录下的graph_with_breakpoint.png文件）,也可以输入'退出'或'quit'结束对话："
        )

        if user_point.lower() in ["退出", "quit"]:
            print("对话结束。")
            break

        # 运行图,直至断点的节点
        for chunk in graph.stream(
            {"user_input": user_point}, config=config, stream_mode="values"
        ):
            # print(chunk)
            all_chunks.append(chunk)

        # 处理可能的审批请求
        last_chunk = all_chunks[-1]
        if (
            last_chunk["user_approval"]
            == f"用户输入的指令是:{last_chunk["user_input"]},请人工确认是否执行!"
        ):
            user_approval = input(
                f"当前用户的输入是:{last_chunk["user_input"]},请人工确认是否执行? (是/否): "
            )
            graph.update_state(config, {"user_approval": user_approval})

        # 继续执行图
        for chunk in graph.stream(None, config, stream_mode="values"):
            all_chunks.append(chunk)

        # 显示最终模型响应
        print("AI:", all_chunks[-1]["model_response"].content)


run_dialogue_config = {"configurable": {"thread_id": "90"}}

run_dialogue(graph, run_dialogue_config)
