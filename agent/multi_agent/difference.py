# 情况二: 智能体之间有不同的状态

from typing import TypedDict
import os
import dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
import asyncio

from config import config
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    temperature=os.getenv("TEMPERATURE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
)


# 定义父图的状态
class ParentState(TypedDict):
    user_input: str
    final_answer: str


def parent_node(state: ParentState):
    response = llm.invoke(state["user_input"])
    return {"final_answer": response}


# 定义子图的状态
class SubgraphState(TypedDict):
    summary_answer: str
    response_answer: str
    score: str


def subgraph_node(state: SubgraphState):
    system_prompt = """ 
      请描述你的内容.字数不多于50字
    """

    messages = state["response_answer"]
    messages = [SystemMessage(content=system_prompt)] + [
        HumanMessage(content=messages.content)
    ]
    response = llm.invoke(messages)
    return {"summary_answer": response}


def subgraph_node2(state: SubgraphState):
    messages = f""" 
    完整回答:{state["response_answer"]}\n
    概要回答:{state["summary_answer"]}\n
    请给这个概要打分:1-10分
    """
    response = llm.invoke([HumanMessage(content=messages)])
    return {"score": response.content}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node)
subgraph_builder.add_node(subgraph_node2)
subgraph_builder.add_edge(START, "subgraph_node")
subgraph_builder.add_edge("subgraph_node", "subgraph_node2")
subgraph = subgraph_builder.compile()


def translate_node(state: ParentState):
    # 1. 父图状态 → 子图输入：手动转换 key
    response = subgraph.invoke({"response_answer": state["final_answer"]})
    # 2. 子图输出 → 父图状态：手动提取 key
    return {"final_answer": response["score"]}


builder = StateGraph(ParentState)
builder.add_node("parent_node", parent_node)
builder.add_node("translate_node", translate_node)
builder.add_edge(START, "parent_node")
builder.add_edge("parent_node", "translate_node")
graph = builder.compile()

# from IPython.display import Image, display

# display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

all_chunk = []


async def run():
    async for chunk in graph.astream(
        {"user_input": "我现在想要学习大模型,应该关注哪些技术?"},
        stream_mode="values",
        subgraphs=True,
    ):
        all_chunk.append(chunk)
        print(chunk)


asyncio.run(run())
all_chunk[-1][1]["final_answer"]
