"""
动态断点
与静态断点（interrupt_before/interrupt_after 编译时固定）不同，
动态断点使用 interrupt() 函数在运行时根据条件决定是否中断。
优点：
1. 运行时判断：根据输入内容动态决定是否需要人工介入，普通请求可直接通过
2. 灵活性高：中断逻辑写在节点函数中，业务变化时只需修改判断条件，无需改动图结构
3. 精准控制：可以对不同类型的敏感操作设置不同的确认流程
4. 节省资源：非敏感请求跳过中断，减少不必要的等待，提升用户体验
"""

import os
import sys

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from config import config
from agent.tools.index import fetch_real_time_info, get_weather

# 初始化 LLM 并绑定工具
tools = [fetch_real_time_info, get_weather]
llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=config.TEMPERATURE,
    openai_api_key=config.OPENAI_API_KEY,
    openai_api_base=config.OPENAI_BASE_URL,
).bind_tools(tools)


# ===== 节点函数 =====

def detect_sensitive(state: MessagesState):
    """敏感词检测节点：判断是否需要人工介入"""
    sensitive_keywords = ["删除", "修改密码", "转账", "支付", "退款"]
    last_message = state["messages"][-1].content
    is_sensitive = any(kw in last_message for kw in sensitive_keywords)

    if is_sensitive:
        # 动态断点：运行时触发中断，等待用户确认
        human_feedback = interrupt(
            f"检测到敏感操作：「{last_message}」，请确认是否继续？(是/否)"
        )
        if human_feedback != "是":
            return {"messages": [AIMessage(content="操作已取消。")]}

    # 非敏感请求或已确认，继续执行
    return {}


def call_model(state: MessagesState):
    """Agent 节点：调用 LLM"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: MessagesState):
    """条件判断：是否有工具调用"""
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    return "continue"


def call_tools(state: MessagesState):
    """工具执行节点"""
    from langgraph.prebuilt import ToolNode
    tool_node = ToolNode(tools)
    return tool_node.invoke(state)


# ===== 构建图 =====

builder = StateGraph(MessagesState)

builder.add_node("detect_sensitive", detect_sensitive)
builder.add_node("agent", call_model)
builder.add_node("tools", call_tools)

builder.add_edge(START, "detect_sensitive")
builder.add_edge("detect_sensitive", "agent")
builder.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 可视化
os.makedirs("dist/image", exist_ok=True)
try:
    png_data = graph.get_graph().draw_mermaid_png()
    with open("dist/image/dynamic_breakpoint.png", "wb") as f:
        f.write(png_data)
    print("图已保存到 dist/image/dynamic_breakpoint.png")
except Exception as e:
    print(f"(跳过图片生成: {e})")


# ===== 交互式运行 =====

def run():
    run_config = {"configurable": {"thread_id": "dynamic-1"}}

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

        # 检查是否有待处理的中断（动态断点触发）
        snapshot = graph.get_state(run_config)
        while snapshot.next:
            interrupt_value = None
            for task in snapshot.tasks:
                if task.interrupts:
                    interrupt_value = task.interrupts[0].value

            if interrupt_value:
                print(f"\n⚠️  {interrupt_value}")
                user_confirm = input("请输入确认信息: ")

                # 恢复执行，传入用户确认值
                result = graph.invoke(
                    Command(resume=user_confirm), config=run_config
                )

            snapshot = graph.get_state(run_config)

        # 输出最终结果
        if result and "messages" in result:
            last_msg = result["messages"][-1]
            content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            print(f"\nAI: {content}")


if __name__ == "__main__":
    run()
