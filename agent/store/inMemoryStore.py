""" 
这是一个实验,我是用来测试在内存中存储数据的功能的,目前还没有什么实际的功能,只是一个简单的示例,以后可能会添加更多的功能,比如说可以设置过期时间,或者是可以设置一些标签等等,总之就是一个简单的内存存储的实现,希望能够帮助到大家.

"""

from langgraph.store.memory import InMemoryStore

# class MyInMemoryStore(InMemoryStore):
#     """一个简单的内存存储实现"""

#     def __init__(self):
#         super().__init__()
#         self.data = {}

#     def set(self, key: str, value: str):
#         """设置键值对"""
#         self.data[key] = value

#     def get(self, key: str) -> str:
#         """获取键对应的值"""
#         return self.data.get(key, None)

#     def delete(self, key: str):
#         """删除键值对"""
#         if key in self.data:
#             del self.data[key]

#     def clear(self):
#         """清空所有数据"""
#         self.data.clear()

in_Memory_store = InMemoryStore()

userId = '1'
namespace_for_memory = (userId, 'memories')

import uuid 
memory_id = str(uuid.uuid4())
memory = {'user':'你好,我是林北'}
in_Memory_store.put(namespace_for_memory, memory_id, memory)
memories = in_Memory_store.search(namespace_for_memory)
print(memories[-1].dict())