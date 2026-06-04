"""Agent 状态定义"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Agent 的状态结构

    使用 LangGraph 的 add_messages reducer 来管理消息列表。
    新消息会追加到现有列表中，而不是替换。
    """

    messages: Annotated[list[BaseMessage], add_messages]
