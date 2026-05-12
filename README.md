# project-interview-rag

一个面向实习面试准备的个人项目知识库问答助手。它读取本地项目文档，例如简历、项目 README、架构文档、Agent 设计文档和 demo 说明，通过标准 RAG 链路回答面试相关问题，并输出引用来源。

## 技术栈

- Python 3.11+
- LangChain
- Chroma
- sentence-transformers multilingual embedding
- OpenAI-compatible Chat Completions LLM
- python-dotenv

## RAG 流程

```txt
文档加载 -> 文本切分 -> Embedding -> 向量库 -> Retriever -> LLM 回答 -> 引用来源 -> 简单检索评测
```

## 准备文档

把需要问答的 Markdown 或 TXT 文档放到：

```txt
data/raw/
```

示例：

```txt
data/raw/
  resume.md
  personal_collection_readme.md
  architecture.md
  collection_agent_design.md
  thought_tracker_project_profile.md
  eeg_semantic_parsing_project_profile.md
```

第一版暂不解析 PDF。

仓库里已经放入简历、personal_collection、thought_tracker 和脑电语义解析 demo 的项目说明文档。当前语料规模为 8 个文档、31 个 chunks。

## 安装依赖

建议使用 Python 3.11+ 虚拟环境：

```bash
pip install -r requirements.txt
```

复制环境变量模板：

```bash
cp .env.example .env
```

填写：

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=
```

如果使用 OpenAI 官方接口，`OPENAI_BASE_URL` 可以留空；如果使用 DeepSeek 等 OpenAI-compatible endpoint，则填写对应 base URL。

未配置 `OPENAI_API_KEY` 时，CLI 会返回检索式结果和引用来源；配置模型后会生成完整的 LLM 引用式回答。

## 构建索引

```bash
python -m src.build_index
```

脚本会重建 `data/chroma/` 下的 Chroma 向量库，并打印文档数量和 chunk 数量。

## 运行问答

```bash
python -m src.cli
```

交互示例：

```txt
Ask> personal_collection 的 Agent 架构是什么？
Answer> ...
Sources:
- collection_agent_design.md
- architecture.md
```

输入 `exit` 或 `quit` 退出。

## 运行检索评测

评测问题放在：

```txt
eval/questions.jsonl
```

格式：

```jsonl
{"question":"personal_collection 的 Agent 架构是什么？","expected_source":"collection_agent_design.md"}
{"question":"personal_collection 为什么是 local-first？","expected_source":"architecture.md"}
```

运行：

```bash
python -m src.eval_retrieval
```

输出示例：

```txt
PASS personal_collection 的 Agent 架构是什么？
PASS thought_tracker 用了哪些后端技术？
retrieval hit rate: 20/20 = 100%
```

## 示例问题

- personal_collection 的 Agent 架构是什么？
- personal_collection 为什么是 local-first？
- thought_tracker 解决了什么问题？
- thought_tracker 的关键词提取与高亮是怎么实现的？
- 脑电语义解析 demo 的核心流程是什么？
- 脑电语义解析 demo 的模型结构有哪些？
- 这个项目最适合在面试中如何介绍？

## 简历描述

> 基于 LangChain + Chroma 构建个人项目知识库 RAG，围绕简历、项目 README、架构设计和研究 demo 文档完成本地知识库构建；实现文档切分、多语言向量检索、LLM 引用式回答和 20 条检索评测样例。
