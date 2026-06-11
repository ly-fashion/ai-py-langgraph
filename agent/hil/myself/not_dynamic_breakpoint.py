"""
我来练习一下,非动态断点实践(非动态断点,静态断点)
非动态断点（静态断点）的缺点：
1. 编译时固定：interrupt_before/interrupt_after 在 compile 时写死，无法根据运行时条件决定是否中断
2. 灵活性差：所有请求都会在断点处暂停，即使是无需人工审核的普通请求
3. 无法按条件跳过：不能根据输入内容动态判断是否需要中断，必须统一处理
4. 维护成本高：如果业务逻辑变化（如新增敏感词），需要重新修改图结构并重新编译
"""

import os
import sys
import json

# 将项目根目录加入 sys.path，解决直接运行脚本时的导入问题
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import MessagesState, START
from langgraph.prebuilt import ToolNode

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from config import config

from agent.tools.index import fetch_real_time_info, get_weather

tools = [fetch_real_time_info, get_weather]
tool_node = ToolNode(tools)

llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=config.TEMPERATURE,
    openai_api_key=config.OPENAI_API_KEY,
    openai_api_base=config.OPENAI_BASE_URL,
)

llm = llm.bind_tools(tools)


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # 如果没有工具调用,直接终点
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    # 如果还有子任务需要继续执行工具调用的话,继续等待执行
    else:
        return "continue"


def call_model(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.add_edge(START, "agent")

# 条件边添加
workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "action", "end": END}
)

workflow.add_edge("action", "agent")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory, interrupt_before=["action"])

# 生成可视化图片
os.makedirs("dist/image", exist_ok=True)
png_data = graph.get_graph().draw_mermaid_png()
image_name = "dynamic_breakpoint_self.png"
with open(f"dist/image/{image_name}", "wb") as f:
    f.write(png_data)
print(f"图已保存到 dist/image/{image_name}")


config_mine = {"configurable": {"thread_id": "4"}}

# 第一次运行：会在 action 节点前暂停
print("=== 第一次运行（到断点暂停）===")
for chunk in graph.stream(
    {"messages": "请帮我查一下天津和上海的天气"},
    config=config_mine,
    stream_mode="values",
):
    print(chunk["messages"][-1])

# 检查是否在断点处暂停
snapshot = graph.get_state(config_mine)
print(f"\n暂停在: {snapshot.next}")

# 恢复执行：继续运行工具调用
print("\n=== 恢复执行（执行工具）===")
from langgraph.types import Command

for chunk in graph.stream(
    Command(resume=True), config=config_mine, stream_mode="values"
):
    print(chunk["messages"][-1])
