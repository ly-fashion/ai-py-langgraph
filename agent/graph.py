"""LangGraph 图定义

构建 Agent 的状态图，定义节点和边的连接关系。

图结构:
    START -> agent_node
                |
                v
          should_continue
           /         \
        "tools"      "end"
          |            |
          v            v
      tool_node      END
          |
          v
      agent_node (循环)
"""

from langgraph.graph import END, START, StateGraph

from agent.nodes import agent_node, should_continue, tool_node
from agent.state import AgentState
from config import config


def create_agent_graph() -> StateGraph:
    """创建并编译 Agent 图

    Returns:
        编译后的 LangGraph 图，可以直接调用
    """
    # 创建状态图
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # 添加边
    # START -> agent
    graph.add_edge(START, "agent")

    # agent -> 条件判断
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # 需要执行工具
            "end": END,         # 任务完成
        },
    )

    # tools -> agent (工具执行完后回到 agent 节点)
    graph.add_edge("tools", "agent")

    # 编译图
    compiled = graph.compile()

    return compiled


def create_agent_with_checkpointer():
    """创建带记忆功能的 Agent 图

    使用 MemorySaver 实现对话历史持久化（仅在会话内有效）。
    """
    from langgraph.checkpoint.memory import MemorySaver

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")

    # 使用内存检查点
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled
