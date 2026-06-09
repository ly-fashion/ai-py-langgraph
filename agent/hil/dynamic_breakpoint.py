"""
动态断点:
与静态断点（interrupt_before/interrupt_after 在编译时固定）不同，
动态断点根据运行时条件决定是否中断，使用 interrupt() 函数实现。
适用于：敏感操作确认、异常检测暂停、条件性人工审核等场景。
"""

import os
from typing import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from config import config
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=config.TEMPERATURE,
    openai_api_key=config.OPENAI_API_KEY,
    openai_api_base=config.OPENAI_BASE_URL,
)


# 定义状态
class State(TypedDict):
    user_input: str
    model_response: str
    is_sensitive: bool


# 敏感词检测节点
def detect_sensitive(state: State):
    """动态判断是否需要人工介入"""
    sensitive_keywords = ["删除", "修改密码", "转账", "支付", "退款"]
    user_input = state["user_input"]

    is_sensitive = any(kw in user_input for kw in sensitive_keywords)
    return {"is_sensitive": is_sensitive}


# 处理请求节点（含动态断点）
def process_request(state: State):
    """如果检测到敏感操作，动态触发中断等待确认"""
    if state["is_sensitive"]:
        # 动态断点：运行时触发，等待用户提供确认信息
        human_feedback = interrupt(
            f"检测到敏感操作：「{state['user_input']}」，请确认是否继续？(是/否)"
        )

        if human_feedback != "是":
            return {"model_response": AIMessage(content="操作已取消。")}

    # 正常调用 LLM
    response = llm.invoke(state["user_input"])
    return {"model_response": response}


# 结果展示节点
def show_result(state: State):
    content = (
        state["model_response"].content
        if hasattr(state["model_response"], "content")
        else str(state["model_response"])
    )
    # print(f"\n 状态:{state}")
    print(f"\n最终结果: {content}")
    return {}


# 构建图
builder = StateGraph(State)

builder.add_node("detect_sensitive", detect_sensitive)
builder.add_node("process_request", process_request)
builder.add_node("show_result", show_result)

builder.add_edge(START, "detect_sensitive")
builder.add_edge("detect_sensitive", "process_request")
builder.add_edge("process_request", "show_result")
builder.add_edge("show_result", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 可视化（网络不可用时跳过）
os.makedirs("dist/image", exist_ok=True)
try:
    png_data = graph.get_graph().draw_mermaid_png()
    with open("dist/image/graph_dynamic_breakpoint.png", "wb") as f:
        f.write(png_data)
    print("图已保存到 dist/image/graph_dynamic_breakpoint.png")
except Exception as e:
    print(f"(跳过图片生成: {e})")


# 交互式运行
def run():
    run_config = {"configurable": {"thread_id": "dynamic-1"}}

    while True:
        user_input = input("\n请输入请求（输入'退出'结束）: ")
        if user_input.lower() in ["退出", "quit"]:
            print("对话结束。")
            break

        # 第一次运行，可能在 interrupt 处暂停
        result = graph.invoke({"user_input": user_input}, config=run_config)
        print(f"\n result:{result}")

        # 检查是否有待处理的中断
        snapshot = graph.get_state(run_config)
        while snapshot.next:
            # interrupt 产生的值在 snapshot.tasks 中
            interrupt_value = None
            for task in snapshot.tasks:
                if task.interrupts:
                    interrupt_value = task.interrupts[0].value

            if interrupt_value:
                print(f"\n⚠️  {interrupt_value}")
                user_confirm = input("请输入确认信息: ")

                # 通过 Command 恢复执行，传入用户确认值
                from langgraph.types import Command

                result = graph.invoke(Command(resume=user_confirm), config=run_config)

            snapshot = graph.get_state(run_config)

        # 输出最终结果
        if result and "model_response" in result:
            content = (
                result["model_response"].content
                if hasattr(result["model_response"], "content")
                else str(result["model_response"])
            )
            print(f"\nAI: {content}")


if __name__ == "__main__":
    run()
