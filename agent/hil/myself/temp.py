"""
我来练习一下,动态断点实践
"""

import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from config import config

# 导入工具（相对导入）
from ...tools.index import fetch_real_time_info, get_weather

llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=config.TEMPERATURE,
    openai_api_key=config.OPENAI_API_KEY,
    openai_api_base=config.OPENAI_BASE_URL,
)

# 绑定工具
llm = llm.bind_tools([fetch_real_time_info, get_weather])


# 定义状态
class State(MessagesState):
    is_sensitive: bool = False


# 敏感词检测节点
def detect_sensitive(state: State):
    """检测最后一条消息是否包含敏感操作"""
    sensitive_keywords = ["删除", "修改密码", "转账", "支付", "退款"]
    last_message = state["messages"][-1].content
    is_sensitive = any(kw in last_message for kw in sensitive_keywords)
    return {"is_sensitive": is_sensitive}


# Agent 节点
def agent_node(state: State):
    """调用 LLM，可能返回工具调用或直接回复"""
    if state.get("is_sensitive"):
        # 动态断点：敏感操作需确认
        human_feedback = interrupt(
            f"检测到敏感操作，内容: {state['messages'][-1].content}，是否继续？(是/否)"
        )
        if human_feedback != "是":
            return {"messages": [AIMessage(content="操作已取消。")]}

    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# 工具执行节点
def tool_node(state: State):
    """执行工具调用"""
    from langgraph.prebuilt import ToolNode

    tools = [fetch_real_time_info, get_weather]
    tool_executor = ToolNode(tools)
    return tool_executor.invoke(state)


# 判断是否需要调用工具
def should_continue(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# 构建图
builder = StateGraph(State)

builder.add_node("detect_sensitive", detect_sensitive)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "detect_sensitive")
builder.add_edge("detect_sensitive", "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 可视化（网络不可用时跳过）
os.makedirs("dist/image", exist_ok=True)
try:
    png_data = graph.get_graph().draw_mermaid_png()
    with open("dist/image/graph_dynamic_tools.png", "wb") as f:
        f.write(png_data)
    print("图已保存到 dist/image/graph_dynamic_tools.png")
except Exception as e:
    print(f"(跳过图片生成: {e})")


# 交互式运行
def run():
    run_config = {"configurable": {"thread_id": "dynamic-tools-1"}}

    while True:
        user_input = input("\n请输入请求（输入'退出'结束）: ")
        if user_input.lower() in ["退出", "quit"]:
            print("对话结束。")
            break

        # 运行图
        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=run_config,
        )

        # 检查是否有待处理的中断
        snapshot = graph.get_state(run_config)
        while snapshot.next:
            interrupt_value = None
            for task in snapshot.tasks:
                if task.interrupts:
                    interrupt_value = task.interrupts[0].value

            if interrupt_value:
                print(f"\n⚠️  {interrupt_value}")
                user_confirm = input("请输入确认信息: ")

                from langgraph.types import Command

                result = graph.invoke(Command(resume=user_confirm), config=run_config)

            snapshot = graph.get_state(run_config)

        # 输出最终结果
        if result and "messages" in result:
            last_msg = result["messages"][-1]
            content = (
                last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            )
            print(f"\nAI: {content}")


if __name__ == "__main__":
    run()
