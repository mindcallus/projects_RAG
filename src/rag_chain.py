from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    RETRIEVER_TOP_K,
)


SYSTEM_PROMPT = """你是一个个人项目知识库问答助手，用于实习面试准备。

要求：
1. 只能根据给定 context 回答项目事实。
2. 如果 context 中没有明确依据，回答“资料中没有明确体现”。
3. 回答要简洁、具体，优先面向面试表达。
4. 回答最后必须包含“引用来源：”，列出使用到的文件名。
"""


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


@lru_cache(maxsize=1)
def get_retriever():
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill your API key."
        )

    kwargs = {
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_MODEL,
        "temperature": 0,
    }
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL

    return ChatOpenAI(**kwargs)


def format_context(docs) -> str:
    blocks = []
    for index, doc in enumerate(docs, start=1):
        file_name = doc.metadata.get("file_name") or doc.metadata.get("source", "unknown")
        source = doc.metadata.get("source", file_name)
        blocks.append(
            f"[{index}] source={source} file_name={file_name}\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def unique_sources(docs) -> list[str]:
    sources = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("file_name") or doc.metadata.get("source")
        if source and source not in seen:
            sources.append(source)
            seen.add(source)
    return sources


def cited_sources_from_answer(answer: str, candidates: list[str]) -> list[str]:
    cited = [source for source in candidates if source in answer]
    return cited or candidates


def get_fallback_docs(docs):
    primary_source = docs[0].metadata.get("file_name") or docs[0].metadata.get("source")
    return [
        doc
        for doc in docs
        if (doc.metadata.get("file_name") or doc.metadata.get("source")) == primary_source
    ]


def format_retrieval_fallback(docs) -> str:
    focused_docs = get_fallback_docs(docs)

    lines = [
        "未配置 OPENAI_API_KEY，当前返回检索式结果；配置模型后会生成完整引用式回答。",
        "",
        "相关内容摘录：",
    ]
    for index, doc in enumerate(focused_docs, start=1):
        file_name = doc.metadata.get("file_name") or doc.metadata.get("source", "unknown")
        snippet = " ".join(doc.page_content.split())
        if len(snippet) > 260:
            snippet = f"{snippet[:260]}..."
        lines.append(f"{index}. [{file_name}] {snippet}")

    lines.append("")
    lines.append("引用来源：")
    for source in unique_sources(focused_docs):
        lines.append(f"- {source}")
    return "\n".join(lines)


def ask(question: str) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(question)
    sources = unique_sources(docs)

    if not docs:
        return {
            "answer": "资料中没有明确体现。\n\n引用来源：无",
            "sources": [],
        }

    context = format_context(docs)
    user_prompt = f"""context:
{context}

question:
{question}
"""

    if not OPENAI_API_KEY:
        fallback_docs = get_fallback_docs(docs)
        return {
            "answer": format_retrieval_fallback(fallback_docs),
            "sources": unique_sources(fallback_docs),
        }

    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    answer = response.content
    return {
        "answer": answer,
        "sources": cited_sources_from_answer(str(answer), sources),
    }
