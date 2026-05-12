# 检索评测记录

## 当前结论

当前项目已接入 8 份本地 Markdown 语料，覆盖简历、personal_collection、thought_tracker 和脑电语义解析 demo。使用 20 条面试场景问题做 Top-K 检索评测，当前命中率为：

```txt
retrieval hit rate: 20/20 = 100%
```

## 语料规模

```txt
documents: 8
chunks: 31
```

## Embedding 调整

第一版使用 `sentence-transformers/all-MiniLM-L6-v2`。在中文问题和中文长文档场景下，脑电项目相关问题召回不稳定，20 条评测中有 5 条未命中目标文档。

当前切换为：

```txt
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

切换后中文项目文档召回更稳定，20 条评测全部命中。

## 评测范围

评测问题覆盖：

- personal_collection 的 Agent 架构、local-first 边界和项目表达
- thought_tracker 的项目背景、核心功能、LLM 重排、关键词提取和工程限制
- 脑电语义解析 demo 的数据输入、预处理、模型结构、运行方式和简历表达
- 多项目整体归纳与面试准备表达

## 当前限制

这套评测只验证检索结果是否包含预期来源文档，不等同于完整回答质量评测。后续可以继续补充：

- answer faithfulness 检查
- 引用来源准确性检查
- 多轮问答测试
- 不相关问题拒答测试
