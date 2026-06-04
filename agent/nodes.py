"""Agent 图节点定义

定义 LangGraph 中使用的各个节点。
"""

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.tools import TOOLS
from config import config

# 系统提示词
SYSTEM_PROMPT = """你是一个智能助手，擅长使用工具来帮助用户解决问题。

你可以使用以下工具：
- calculator: 计算数学表达式
- get_current_time: 获取当前时间
- text_analyzer: 分析文本统计信息
- unit_converter: 单位转换

请根据用户的问题，决定是否需要使用工具。如果需要使用工具，请调用相应的工具；
如果不需要工具，直接回答用户的问题即可。

回答时请使用中文，保持友好和专业的语气。
"""


def create_llm() -> ChatOpenAI:
    """创建 LLM 实例（支持 OpenAI 兼容 API）"""
    return ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        openai_api_key=config.OPENAI_API_KEY,
        openai_api_base=config.OPENAI_BASE_URL,
    ).bind_tools(TOOLS)


def agent_node(state: AgentState) -> dict:
    """Agent 主节点 - 调用 LLM 处理用户输入

    这是图的核心节点，负责：
    1. 接收用户消息
    2. 调用 LLM（带工具绑定）
    3. 返回 LLM 的响应（可能包含工具调用）
    """
    messages = state["messages"]

    # 如果没有系统消息，添加系统提示
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    llm = create_llm()
    response = llm.invoke(messages)

    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """工具执行节点 - 执行 LLM 请求的工具调用

    遍历最后一条 AI 消息中的工具调用，执行它们，并返回工具结果。
    """
    messages = state["messages"]
    last_message = messages[-1]

    # 如果最后一条消息没有工具调用，返回空
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}

    # 执行所有工具调用
    tool_results = []
    tools_by_name = {t.name: t for t in TOOLS}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name in tools_by_name:
            tool_func = tools_by_name[tool_name]
            try:
                result = tool_func.invoke(tool_args)
            except Exception as e:
                result = f"工具执行错误: {e}"
        else:
            result = f"未知工具: {tool_name}"

        from langchain_core.messages import ToolMessage
        tool_results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": tool_results}


def should_continue(state: AgentState) -> str:
    """条件边 - 判断是否需要继续执行工具调用

    检查最后一条消息：
    - 如果是 AI 消息且包含工具调用 -> 转到工具节点
    - 否则 -> 结束
    """
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"
