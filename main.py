"""AI Agent 主入口

基于 LangGraph 的智能 Agent，支持工具调用和多轮对话。

使用方式:
    python main.py              # 交互式聊天
    python main.py --once "问题"  # 单次问答
    python main.py --stream "问题" # 流式输出
"""

import argparse
import sys

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import create_agent_graph, create_agent_with_checkpointer
from config import config


def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("  🤖 LangGraph AI Agent")
    print("  基于 LangGraph 构建的智能助手")
    print("=" * 60)
    print(f"  模型: {config.MODEL_NAME}")
    print(f"  温度: {config.TEMPERATURE}")
    print("=" * 60)
    print("  输入消息开始对话，输入 'quit' 或 'exit' 退出")
    print("  输入 'clear' 清除对话历史")
    print("=" * 60)
    print()


def extract_ai_content(result: dict) -> str:
    """从结果中提取 AI 回复内容"""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return "（无回复）"


def chat_mode():
    """交互式聊天模式"""
    print_banner()

    graph = create_agent_with_checkpointer()
    thread_id = "default"

    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        if user_input.lower() == "clear":
            graph = create_agent_with_checkpointer()
            print("✅ 对话历史已清除\n")
            continue

        # 调用 Agent
        try:
            print("🤖 AI: ", end="", flush=True)

            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": thread_id}},
            )

            content = extract_ai_content(result)
            print(content)
            print()

        except Exception as e:
            print(f"\n❌ 错误: {e}\n")


def once_mode(question: str):
    """单次问答模式"""
    graph = create_agent_graph()

    try:
        result = graph.invoke({"messages": [HumanMessage(content=question)]})
        content = extract_ai_content(result)
        print(content)
    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        sys.exit(1)


def stream_mode(question: str):
    """流式输出模式"""
    graph = create_agent_graph()

    try:
        print("🤖 AI: ", end="", flush=True)

        for event in graph.stream(
            {"messages": [HumanMessage(content=question)]},
            stream_mode="values",
        ):
            messages = event.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.content:
                    print(msg.content, flush=True)

        print()

    except Exception as e:
        print(f"\n❌ 错误: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LangGraph AI Agent")
    parser.add_argument(
        "--once",
        type=str,
        help="单次问答模式，传入问题",
    )
    parser.add_argument(
        "--stream",
        type=str,
        help="流式输出模式，传入问题",
    )

    args = parser.parse_args()

    # 验证配置
    try:
        config.validate()
    except ValueError as e:
        print(f"❌ 配置错误: {e}", file=sys.stderr)
        print("请复制 .env.example 为 .env 并填写配置", file=sys.stderr)
        sys.exit(1)

    # 根据参数选择模式
    if args.once:
        once_mode(args.once)
    elif args.stream:
        stream_mode(args.stream)
    else:
        chat_mode()


if __name__ == "__main__":
    main()
