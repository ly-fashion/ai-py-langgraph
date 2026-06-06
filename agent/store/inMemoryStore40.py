"""
长期记忆 :基于 LangGraph Store 来实现

使用 InMemoryStore 在内存中存储用户记忆，支持：
- 按命名空间（用户ID + 类别）组织记忆
- 保存/搜索/删除记忆
- 在对话中自动注入相关记忆到系统提示
"""

import uuid
from typing import Any

from langgraph.store.memory import InMemoryStore

# 全局内存存储实例
memory_store = InMemoryStore()

# 默认命名空间前缀
DEFAULT_NAMESPACE_KEY = "memories"


def _get_namespace(user_id: str, category: str = DEFAULT_NAMESPACE_KEY) -> tuple:
    """获取命名空间

    Args:
        user_id: 用户ID
        category: 记忆类别

    Returns:
        命名空间元组
    """
    return (user_id, category)


def save_memory(user_id: str, content: str, category: str = DEFAULT_NAMESPACE_KEY, metadata: dict[str, Any] | None = None) -> str:
    """保存一条记忆

    Args:
        user_id: 用户ID
        content: 记忆内容
        category: 记忆类别（如 preferences, facts, conversations）
        metadata: 额外元数据

    Returns:
        记忆ID
    """
    namespace = _get_namespace(user_id, category)
    memory_id = str(uuid.uuid4())

    memory_data = {
        "content": content,
        "category": category,
        **(metadata or {}),
    }

    memory_store.put(namespace, memory_id, memory_data)
    return memory_id


def search_memories(user_id: str, query: str | None = None, category: str = DEFAULT_NAMESPACE_KEY, limit: int = 10) -> list[dict]:
    """搜索记忆

    Args:
        user_id: 用户ID
        query: 搜索关键词（可选，用于过滤内容）
        category: 记忆类别
        limit: 返回数量上限

    Returns:
        记忆列表
    """
    namespace = _get_namespace(user_id, category)

    results = memory_store.search(namespace, query=query, limit=limit)

    memories = []
    for item in results:
        # 如果有查询关键词，过滤包含关键词的记忆
        if query:
            content = item.value.get("content", str(item.value))
            # 支持多关键词搜索（空格分隔）
            keywords = query.lower().split()
            if not any(kw in content.lower() for kw in keywords):
                continue

        memories.append({
            "id": item.key,
            "value": item.value,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
        })

    return memories[:limit]


def delete_memory(user_id: str, memory_id: str, category: str = DEFAULT_NAMESPACE_KEY) -> bool:
    """删除一条记忆

    Args:
        user_id: 用户ID
        memory_id: 记忆ID
        category: 记忆类别

    Returns:
        是否删除成功
    """
    namespace = _get_namespace(user_id, category)

    try:
        memory_store.delete(namespace, memory_id)
        return True
    except Exception:
        return False


def get_all_memories(user_id: str, category: str = DEFAULT_NAMESPACE_KEY) -> list[dict]:
    """获取用户所有记忆

    Args:
        user_id: 用户ID
        category: 记忆类别

    Returns:
        记忆列表
    """
    return search_memories(user_id, category=category, limit=100)


def format_memories_for_prompt(user_id: str, limit: int = 5) -> str:
    """格式化记忆用于系统提示

    搜索所有类别的记忆并格式化。

    Args:
        user_id: 用户ID
        limit: 记忆数量上限

    Returns:
        格式化的记忆文本
    """
    all_memories = []

    # 搜索所有常见类别
    categories = ["memories", "preferences", "facts", "conversations"]
    for cat in categories:
        namespace = _get_namespace(user_id, cat)
        results = memory_store.search(namespace, limit=limit)
        for item in results:
            all_memories.append(item.value.get("content", str(item.value)))

    if not all_memories:
        return ""

    memory_text = "\n\n## 用户长期记忆\n以下是关于该用户的重要信息，请在回答时参考：\n"
    for i, content in enumerate(all_memories[:limit], 1):
        memory_text += f"{i}. {content}\n"

    return memory_text
