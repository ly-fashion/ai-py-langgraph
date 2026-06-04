# 🤖 LangGraph AI Agent

基于 **Python + LangGraph** 构建的智能 AI Agent 项目，支持工具调用和多轮对话。

## 项目结构

```
ai-py-langgraph/
├── main.py              # 主入口（CLI 界面）
├── config.py            # 配置管理
├── requirements.txt     # 依赖列表
├── .env.example         # 环境变量模板
├── .gitignore
├── README.md
└── agent/               # Agent 核心模块
    ├── __init__.py
    ├── graph.py         # LangGraph 图定义
    ├── nodes.py         # 图节点（Agent、工具执行）
    ├── state.py         # 状态定义
    └── tools.py         # 工具集
```

## 架构说明

### LangGraph 图结构

```
START
  │
  ▼
agent_node (LLM 推理)
  │
  ▼
should_continue (条件判断)
  │
  ├─ 需要工具 ──▶ tool_node ──▶ agent_node (循环)
  │
  └─ 任务完成 ──▶ END
```

这是一个标准的 **ReAct (Reasoning + Acting)** 模式：
1. **Agent 节点**: 调用 LLM 分析用户输入，决定是否需要工具
2. **条件边**: 检查 LLM 响应中是否有工具调用请求
3. **工具节点**: 执行工具并返回结果给 Agent

### 内置工具

| 工具 | 功能 |
|------|------|
| `calculator` | 数学表达式计算 |
| `get_current_time` | 获取当前时间 |
| `text_analyzer` | 文本统计分析 |
| `unit_converter` | 单位转换（长度/重量/温度） |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. 运行

```bash
# 交互式聊天模式
python main.py

# 单次问答
python main.py --once "帮我计算 123 * 456"

# 流式输出
python main.py --stream "现在几点了？"
```

## 使用示例

```
👤 你: 帮我计算 (15 * 23 + 47) / 8
🤖 AI: 计算结果: (15 * 23 + 47) / 8 = 49.25

👤 你: 100华氏度是多少摄氏度？
🤖 AI: 100 fahrenheit = 37.78 celsius

👤 你: 现在几点了？
🤖 AI: 当前时间: 2026年06月04日 15:30:45 (星期3)
```

## 扩展开发

### 添加新工具

在 `agent/tools.py` 中添加新工具：

```python
@tool
def my_new_tool(param: str) -> str:
    """工具描述（LLM 会根据这个描述决定何时使用）

    Args:
        param: 参数说明
    """
    # 实现逻辑
    return "结果"

# 不要忘记将工具添加到 TOOLS 列表
TOOLS = [..., my_new_tool]
```

### 自定义系统提示

编辑 `agent/nodes.py` 中的 `SYSTEM_PROMPT` 变量。

### 修改 LLM 模型

在 `.env` 文件中修改：
```
MODEL_NAME=gpt-4o
TEMPERATURE=0.5
```

## 技术栈

- **LangGraph** - 状态图编排框架
- **LangChain** - LLM 应用开发框架
- **OpenAI GPT** - 大语言模型
- **Python 3.12+**

## License

MIT
