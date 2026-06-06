"""Agent 工具集

在这里定义 Agent 可以使用的工具。每个工具使用 @tool 装饰器定义。
"""

import math
from datetime import datetime

from langchain_core.tools import tool

from agent.store.inMemoryStore40 import save_memory, search_memories


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。

    支持基本运算（+, -, *, /, **）和常用数学函数。
    示例: "2 + 3 * 4", "sqrt(16)", "sin(pi/2)"

    Args:
        expression: 要计算的数学表达式
    """
    # 安全的数学环境
    safe_dict = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "log": math.log,
        "log10": math.log10,
        "ceil": math.ceil,
        "floor": math.floor,
    }
    try:
        result = eval(expression, safe_dict)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_current_time() -> str:
    """获取当前日期和时间。"""
    now = datetime.now()
    return now.strftime("当前时间: %Y年%m月%d日 %H:%M:%S (星期%w)")


@tool
def text_analyzer(text: str) -> str:
    """分析文本的基本统计信息。

    Args:
        text: 要分析的文本
    """
    chars = len(text)
    words = len(text.split())
    lines = text.count("\n") + 1 if text else 0
    chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")

    return (
        f"文本分析结果:\n"
        f"  - 总字符数: {chars}\n"
        f"  - 单词数: {words}\n"
        f"  - 行数: {lines}\n"
        f"  - 中文字符数: {chinese_chars}"
    )


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """单位转换工具。支持常见单位转换。

    支持的单位:
    - 长度: km, m, cm, mm, mile, inch, ft
    - 重量: kg, g, mg, lb, oz
    - 温度: celsius, fahrenheit, kelvin

    Args:
        value: 要转换的数值
        from_unit: 源单位
        to_unit: 目标单位
    """
    # 转换为标准单位（米、克、摄氏度）
    length_to_m = {
        "km": 1000, "m": 1, "cm": 0.01, "mm": 0.001,
        "mile": 1609.344, "inch": 0.0254, "ft": 0.3048,
    }
    weight_to_g = {
        "kg": 1000, "g": 1, "mg": 0.001,
        "lb": 453.592, "oz": 28.3495,
    }

    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    # 长度转换
    if from_unit in length_to_m and to_unit in length_to_m:
        result = value * length_to_m[from_unit] / length_to_m[to_unit]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"

    # 重量转换
    if from_unit in weight_to_g and to_unit in weight_to_g:
        result = value * weight_to_g[from_unit] / weight_to_g[to_unit]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"

    # 温度转换
    if from_unit in ("celsius", "fahrenheit", "kelvin") and to_unit in (
        "celsius", "fahrenheit", "kelvin"
    ):
        # 先转为摄氏度
        if from_unit == "celsius":
            c = value
        elif from_unit == "fahrenheit":
            c = (value - 32) * 5 / 9
        else:
            c = value - 273.15

        # 从摄氏度转为目标
        if to_unit == "celsius":
            result = c
        elif to_unit == "fahrenheit":
            result = c * 9 / 5 + 32
        else:
            result = c + 273.15

        return f"{value} {from_unit} = {result:.2f} {to_unit}"

    return f"不支持的单位转换: {from_unit} -> {to_unit}"


# ============ 长期记忆工具 ============

# 默认用户ID（可通过配置或会话传入）
DEFAULT_USER_ID = "default"


@tool
def memory_save(content: str, category: str = "memories") -> str:
    """保存一条长期记忆。当用户提到重要信息（如偏好、习惯、事实等）时使用。

    Args:
        content: 要保存的记忆内容，应该是简洁明确的陈述句
        category: 记忆类别，如 preferences（偏好）、facts（事实）、conversations（对话要点）
    """
    memory_id = save_memory(
        user_id=DEFAULT_USER_ID,
        content=content,
        category=category,
    )
    return f"已保存记忆 (ID: {memory_id}): {content}"


@tool
def memory_search(query: str, category: str = "memories") -> str:
    """搜索长期记忆。当需要回忆用户之前提到的信息时使用。

    Args:
        query: 搜索关键词
        category: 记忆类别（可选）
    """
    memories = search_memories(
        user_id=DEFAULT_USER_ID,
        query=query,
        category=category,
        limit=5,
    )

    if not memories:
        return f"未找到与 '{query}' 相关的记忆。"

    result = f"找到 {len(memories)} 条相关记忆：\n"
    for i, mem in enumerate(memories, 1):
        content = mem["value"].get("content", str(mem["value"]))
        result += f"{i}. {content}\n"

    return result


# 工具列表 - 在 graph 中使用
TOOLS = [calculator, get_current_time, text_analyzer, unit_converter, memory_save, memory_search]
