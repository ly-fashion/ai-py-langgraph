# GraphRAG

> 查看微软开源项目: [GraphRAG](https://github.com/microsoft/graphrag)

一些相关项目:

- [GraphRAG](https://github.com/microsoft/graphrag),构建知识图谱,模型能力要求高,成本高,但是准确率好
- [LightRAG](https://github.com/HKUDS/LightRAG),轻量,低成本
- [Fast-GraphRAG](https://github.com/circlemind-ai/fast-graphrag)

## 原理

### 经典RAG流程

原始数据(pdf,doc,html等不同文件格式内容数据) ->文件解析 -> 文件 -> 文件分割(分块大小) -> chunks -> 向量化 -> Embedding(嵌入模型) -> 入库(向量数据库)

prompt -> embedding -> 检索算法 -> 查向量数据库(召回) -> chunks -> 重排序(ReLank) -> chunks -> 增强 -> prompt -> llm - > 回答

适合场景: 文本片段,文章,FAQ,员工手册,企业规章制度

#### 缺点

- 问题宏观,跨文档查询,效果不好.例子: 去年技术部门团队成果是什么
- 没有全局视角(graphrag query --method global --query 文档里涉及了几家公司)

> 为了解决以上问题,引入 RAG + 知识图谱

### 知识图谱

实体和关系(就是一个有向图)

## Init

> 初始化 GraphRAG 配置文件

配置.env以及settings.yaml

## Index流程详解

> 构建 GraphRAG 的索引

## Query流程详解

> 检索GraphRAG 的查询,global query 和 local query

## Update流程

> 更新索引,就是增量更新

## 源码分析以及二次化开发
