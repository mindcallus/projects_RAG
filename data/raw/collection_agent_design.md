# personal_collection Agent 设计说明

personal_collection 的 Agent 架构围绕 collection 级知识空间设计。系统先把用户选择的本地资料组织成 collection，然后在 collection 范围内执行检索、上下文组装和回答生成。

核心链路包括：

- collection-scoped retriever：只在当前 collection 的文档范围内检索，避免跨项目污染。
- planner / executor 思路：先根据问题判断需要检索的资料，再把命中的片段交给 LLM 生成回答。
- 引用来源输出：回答需要保留命中文档来源，方便用户回查原始材料。

这个设计适合面试中解释为：personal_collection 更偏本地优先的 Agent/RAG 产品架构，强调用户资料的组织边界、检索范围控制和可追溯回答。
