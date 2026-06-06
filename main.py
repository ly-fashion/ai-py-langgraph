"""AI Agent 主入口

基于 LangGraph 的智能 Agent，支持工具调用和多轮对话。

使用方式:
    python main.py              # 交互式聊天（流式输出）
    python main.py --once "问题"  # 单次问答
    python main.py --no-stream   # 交互式聊天（非流式）
"""

import argparse
import sys
from typing import Any

# Windows 终端 UTF-8 修复
if sys.platform == "win32":
    import os
    os.system("chcp 65001 >nul 2>&1")
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from agent.graph import create_agent_graph, create_agent_with_checkpointer
from config import config


def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("  LangGraph AI Agent")
    print("  基于 LangGraph 构建的智能助手")
    print("=" * 60)
    print(f"  模型: {config.MODEL_NAME}")
    print(f"  温度: {config.TEMPERATURE}")
    print("=" * 60)
    print("  输入消息开始对话，输入 quit 或 exit 退出")
    print("  输入 clear 清除对话历史")
    print("=" * 60)
    print()


def extract_ai_content(result: dict) -> str:
    """从结果中提取 AI 回复内容"""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return "（无回复）"


def _stream_agent(graph, input_messages, config=None):
    """流式调用 Agent，逐 token 输出

    使用 stream_mode="messages" 获取 token 级别的流式输出。
    自动处理工具调用的中间过程（显示思考/执行状态）。

    Returns:
        最终的完整响应内容
    """
    full_content = ""
    tool_calling = False

    stream_kwargs: dict[str, Any] = {
        "input": {"messages": input_messages},
        "stream_mode": "messages",
    }
    if config:
        stream_kwargs["config"] = config

    for event in graph.stream(**stream_kwargs):
        # event 是 (message, metadata) 元组
        if not isinstance(event, tuple) or len(event) < 2:
            continue

        message, metadata = event

        # 处理工具调用状态提示
        if isinstance(message, AIMessageChunk):
            # 检测工具调用开始
            if message.tool_calls and not tool_calling:
                tool_calling = True
                tool_names = [tc["name"] for tc in message.tool_calls]
                print(f"\n  [调用工具: {', '.join(tool_names)}] ", flush=True)

            # 检测工具调用完成
            if not message.tool_calls and tool_calling:
                tool_calling = False
                print("  [工具执行完成]", flush=True)
                print("\n[AI]", end="", flush=True)

            # 输出文本内容（逐 token）
            if message.content and not tool_calling:
                try:
                    text = str(message.content)
                    print(text, end="", flush=True)
                    full_content += text
                except UnicodeEncodeError:
                    pass

    return full_content


def chat_mode(stream: bool = True):
    """交互式聊天模式

    Args:
        stream: 是否启用流式输出
    """
    print_banner()

    graph = create_agent_with_checkpointer()
    thread_id = "default"

    while True:
        try:
            user_input = input("[You]").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见!")
            break

        if user_input.lower() == "clear":
            graph = create_agent_with_checkpointer()
            print("对话历史已清除\n")
            continue

        try:
            if stream:
                print("[AI]", end="", flush=True)
                _stream_agent(
                    graph,
                    [HumanMessage(content=user_input)],
                    config={"configurable": {"thread_id": thread_id}},
                )
                print("\n")
            else:
                print("[AI]", end="", flush=True)
                result = graph.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                print(extract_ai_content(result))
                print()

        except Exception as e:
            print(f"\n[ERROR] {e}\n")


def once_mode(question: str):
    """单次问答模式（非流式）"""
    graph = create_agent_graph()

    try:
        result = graph.invoke({"messages": [HumanMessage(content=question)]})
        content = extract_ai_content(result)
        print(content)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


def stream_mode(question: str):
    """单次流式输出模式"""
    graph = create_agent_graph()

    try:
        print("[AI]", end="", flush=True)
        _stream_agent(graph, [HumanMessage(content=question)])
        print("\n")

    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LangGraph AI Agent")
    parser.add_argument(
        "--once",
        type=str,
        help="单次问答模式（非流式），传入问题",
    )
    parser.add_argument(
        "--stream",
        type=str,
        help="单次流式输出模式，传入问题",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="交互模式下关闭流式输出",
    )

    args = parser.parse_args()

    # 验证配置
    try:
        config.validate()
    except ValueError as e:
        print(f"[ERROR] 配置错误: {e}", file=sys.stderr)
        print("请复制 .env.example 为 .env 并填写配置", file=sys.stderr)
        sys.exit(1)

    # 根据参数选择模式
    if args.once:
        once_mode(args.once)
    elif args.stream:
        stream_mode(args.stream)
    else:
        chat_mode(stream=not args.no_stream)


if __name__ == "__main__":
    main()
